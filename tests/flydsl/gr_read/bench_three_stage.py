"""Matched graph and ordinary Python eager measurements on real HC weights."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from combined_host import (
    CombinedHostGRRead,
    CombinedPaddedGRRead,
    CombinedThreeStageGRRead,
)
from kernel import GRRead
from register_gate import RegisterGateGRRead
from support import (
    BASELINE_PATH,
    TOLERANCES,
    capture,
    checkpoint_pairs,
    load_flydsl_baseline,
    load_triton_baseline,
    reference,
    time_graph,
    torch_mix,
)
from three_stage import ThreeStageGRRead

RESULTS_DIR = Path("/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results")


def time_eager(calls, samples, min_calls):
    repeats = math.ceil(min_calls / len(calls))
    count = repeats * len(calls)
    for _ in range(3):
        for call in calls:
            call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    events, walls = [], []
    for _ in range(samples):
        begin = time.perf_counter()
        start.record()
        for _ in range(repeats):
            for call in calls:
                call()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - begin) * 1e6 / count)
        events.append(start.elapsed_time(end) * 1000 / count)
    return {"event_span_samples_us": events, "wall_samples_us": walls, "calls_per_sample": count}


class ReferenceReader:
    def __init__(self, call):
        self.call = call
        self.output = None

    def __call__(self, x):
        self.output = self.call(x)
        return self.output


def verify(outputs, refs, enforce=True):
    maximum = 0.0
    passed = 0
    tol = TOLERANCES[torch.bfloat16]
    for output, ref in zip(outputs, refs):
        value = output.double()
        if enforce:
            torch.testing.assert_close(value, ref, **tol)
        error = ((value - ref).abs() / (tol["atol"] + tol["rtol"] * ref.abs())).max().item()
        maximum = max(maximum, error)
        passed += error <= 1.0
    return {"max_scaled_error": maximum, "passed_pairs": passed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 4, 8, 16, 17, 24])
    parser.add_argument("--weights", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--register-gate", action="store_true")
    parser.add_argument("--combined-host", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--prefetch-kernel", type=Path)
    parser.add_argument("--hidden-pad-sweep", action="store_true")
    parser.add_argument("--hidden-pad", type=int, default=0)
    parser.add_argument("--reduce-threads", type=int, default=64)
    parser.add_argument("--reduce-vec", type=int, default=4)
    parser.add_argument("--eager-min-calls", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        any(not 1 <= t <= 24 for t in args.rows)
        or min(args.weights, args.rounds, args.samples, args.eager_min_calls) < 1
    ):
        raise ValueError("rows must be 1..24 and benchmark counts must be positive")
    if (args.hidden_pad or args.hidden_pad_sweep) and not args.prefetch_kernel:
        raise ValueError("LDS padding options require --prefetch-kernel")
    options = (
        [(64, 1), (64, 2), (64, 4), (128, 1), (128, 2), (256, 1)]
        if args.sweep
        else [(args.reduce_threads, args.reduce_vec)]
    )
    previous_path = RESULTS_DIR / "e10_full_checkpoint_compensated.kernel.py"
    previous = load_flydsl_baseline(previous_path)
    prefetch = load_flydsl_baseline(args.prefetch_kernel) if args.prefetch_kernel else None
    triton_baseline = torch_compiled = None
    if args.baselines:
        if (
            hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
            != "647a90bf2622e8e145e2b9784059afb9ed9593287f7ce54b980eb54e6b669854"
        ):
            raise RuntimeError("Triton baseline differs from the verified 8cf5501b source")
        triton_baseline = load_triton_baseline()
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.fail_on_recompile_limit_hit = True
        torch_compiled = torch.compile(torch_mix, dynamic=False, fullgraph=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as log:
        props = torch.cuda.get_device_properties(0)
        meta = {
            "type": "environment",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": props.name,
            "arch": props.gcnArchName,
            "compute_units": props.multi_processor_count,
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "flydsl": importlib.metadata.version("flydsl"),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "input": "actual BF16 checkpoint weights; seeded synthetic normalized BF16 inputs",
            "graph": "2 repetitions of all weight pairs; 3 replays/sample; HIP events; full GR read",
            "eager": "ordinary prepared Python readers, synchronized at batch boundaries, no capture or per-call sync",
            "eager_caveat": "wall and event span include host-path effects; differences are not pure hardware launch cost",
            "order": "backend and mode shuffled together in each round",
            "sources": {},
        }
        source_paths = {
            name: Path(__file__).with_name(name)
            for name in (
                "kernel.py",
                "three_stage.py",
                "register_gate.py",
                "combined_host.py",
                "support.py",
                "bench_three_stage.py",
            )
        }
        source_paths["previous_kernel.py"] = previous_path
        if args.baselines:
            source_paths["hc_mix_triton.py"] = BASELINE_PATH
        if args.prefetch_kernel:
            source_paths["prefetch_kernel.py"] = args.prefetch_kernel
        for name, path in source_paths.items():
            content = path.read_bytes()
            meta["sources"][name] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
            with args.output.with_name(args.output.stem + "." + name).open("xb") as snapshot:
                snapshot.write(content)
        pairs = list(checkpoint_pairs(limit=args.weights))
        if len(pairs) != args.weights:
            raise ValueError("requested more checkpoint pairs than available")
        meta["weight_names"] = [name for name, _, _ in pairs]
        log.write(json.dumps(meta) + "\n")
        log.flush()
        rng = random.Random(args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        for rows in args.rows:
            print(f"T={rows}: preparing {len(pairs)} real weight pairs", flush=True)
            xs = [torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in pairs]
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            implementations = {
                "v1": [previous.GRRead(rows, wd, wu) for _, wd, wu in pairs],
                "current": [GRRead(rows, wd, wu) for _, wd, wu in pairs],
            }
            for threads, vec in options:
                implementations[f"three_t{threads}_v{vec}"] = [
                    ThreeStageGRRead(rows, wd, wu, threads, vec) for _, wd, wu in pairs
                ]
            if args.register_gate:
                for ordered in (True, False):
                    name = "register_ordered" if ordered else "register_pairwise"
                    implementations[name] = [RegisterGateGRRead(rows, wd, wu, ordered=ordered) for _, wd, wu in pairs]
            if args.combined_host:
                implementations["combined_pair"] = [CombinedHostGRRead(rows, wd, wu) for _, wd, wu in pairs]
                implementations["combined_three"] = [
                    CombinedThreeStageGRRead(rows, wd, wu, args.reduce_threads, args.reduce_vec) for _, wd, wu in pairs
                ]
                if args.hidden_pad:
                    implementations["combined_padded"] = [
                        CombinedPaddedGRRead(rows, wd, wu, args.hidden_pad) for _, wd, wu in pairs
                    ]
            if prefetch:
                implementations["prefetch_up"] = [prefetch.GRRead(rows, wd, wu) for _, wd, wu in pairs]
                implementations["prefetch_control"] = [
                    prefetch.GRRead(rows, wd, wu, replace(prefetch.default_config(rows), prefetch_low=False))
                    for _, wd, wu in pairs
                ]
                if args.hidden_pad_sweep or args.hidden_pad:
                    for padding in ((4, 8, 16, 32) if args.hidden_pad_sweep else (args.hidden_pad,)):
                        config = replace(prefetch.default_config(rows), hidden_pad=padding, prefetch_low=False)
                        implementations[f"lds_pad{padding}"] = [
                            prefetch.GRRead(rows, wd, wu, config) for _, wd, wu in pairs
                        ]
            if args.baselines:
                implementations["torch_compile"] = [
                    ReferenceReader(lambda x, wd=wd, wu=wu: torch_compiled(x, wd, wu)) for _, wd, wu in pairs
                ]
                if rows <= 16:
                    implementations["tuned_triton"] = [
                        ReferenceReader(lambda x, wd=wd, wu=wu: triton_baseline.fused_hc_mix(x, wd, wu, 4, 2560))
                        for _, wd, wu in pairs
                    ]
            calls, graphs, graph_outputs, results = {}, {}, {}, {}
            for name, readers in implementations.items():
                calls[name] = [lambda r=r, x=x: r(x) for r, x in zip(readers, xs)]
                graphs[name] = capture(calls[name], repeats=2)
                graph_outputs[name] = [r.output for r in readers]
                graphs[name][0].replay()
                results[name] = {
                    "config": asdict(readers[0].config) if hasattr(readers[0], "config") else {"backend": name},
                    **verify(graph_outputs[name], refs, enforce=name not in ("torch_compile", "tuned_triton")),
                    "graph_rounds": [],
                    "eager_rounds": [],
                }
                if name.startswith("three_"):
                    results[name]["reduce_threads"] = readers[0].reduce_threads
                    results[name]["reduce_vec"] = readers[0].reduce_vec
                if name.startswith("register_"):
                    results[name]["ordered_gate_reduce"] = readers[0].ordered_gate_reduce
                if name.startswith("combined_"):
                    results[name]["compiled_host_entry"] = True
            for round_id in range(args.rounds):
                order = [(name, mode) for name in implementations for mode in ("graph", "eager")]
                rng.shuffle(order)
                for name, mode in order:
                    if mode == "graph":
                        graph, count = graphs[name]
                        timing = time_graph(graph, count, samples=args.samples, replay_per_sample=3)
                    else:
                        timing = time_eager(calls[name], args.samples, args.eager_min_calls)
                    results[name][mode + "_rounds"].append({"round": round_id, "order": order, **timing})
            for x in xs:
                x.mul_(0.99).add_(0.015625)
            changed_refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            for name, readers in implementations.items():
                for _ in range(20):
                    graphs[name][0].replay()
                result = results[name]
                changed = verify(
                    graph_outputs[name], changed_refs, enforce=name not in ("torch_compile", "tuned_triton")
                )
                result["changed_input_max_scaled_error"] = changed["max_scaled_error"]
                result["changed_input_passed_pairs"] = changed["passed_pairs"]
                result["graph_median_us"] = statistics.median(
                    v for sample in result["graph_rounds"] for v in sample["samples_us"]
                )
                for metric in ("wall", "event_span"):
                    result["eager_" + metric + "_median_us"] = statistics.median(
                        v for sample in result["eager_rounds"] for v in sample[metric + "_samples_us"]
                    )
            record = {"type": "result", "rows": rows, "weight_pairs": len(pairs), "results": results}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(
                json.dumps(
                    {
                        name: {
                            "graph_us": round(result["graph_median_us"], 3),
                            "eager_wall_us": round(result["eager_wall_median_us"], 3),
                        }
                        for name, result in results.items()
                    }
                ),
                flush=True,
            )
            del implementations, calls, graphs, graph_outputs, readers, refs, changed_refs, xs, graph


if __name__ == "__main__":
    main()
