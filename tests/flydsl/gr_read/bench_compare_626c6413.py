"""Full-call CUDA Graph comparison: selected FlyDSL vs upstream 626c6413.

T=1..16 only. Synthetic weights by default; no SGLang installation or model
is required. The colleague's public three-launch wrapper is used unchanged.
"""

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
import random
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import triton

if __package__:
    from .combined_host import CombinedPaddedGRRead
    from .support import TOLERANCES, capture, checkpoint_pairs, load_triton_baseline, reference, synthetic, time_graph
else:
    from combined_host import CombinedPaddedGRRead
    from support import TOLERANCES, capture, checkpoint_pairs, load_triton_baseline, reference, synthetic, time_graph

SOURCE_COMMIT = "626c64132d7f197de2053232264aeabb37ad156b"
SOURCE_SHA256 = "4a36c9cb92cfcccdd2232ffa6e3a9a5c086361c03d7f30fc3c83bd653739a394"
SOURCE_PATH = Path(__file__).with_name("baselines") / "hc_mix_triton_626c6413.py"


class ColleagueReader:
    def __init__(self, module, wd, wu):
        self.module, self.wd, self.wu = module, wd, wu
        self.output = None

    def __call__(self, x):
        self.output = self.module.fused_hc_mix(x, self.wd, self.wu, 4, 2560)
        return self.output


def verify(outputs, refs, names):
    tol = TOLERANCES[torch.bfloat16]
    failures = []
    maximum = 0.0
    for output, expected, name in zip(outputs, refs, names):
        actual = output.double()
        finite = bool(torch.isfinite(actual).all())
        passed = torch.allclose(actual, expected, **tol)
        scaled = (actual - expected).abs() / (tol["atol"] + tol["rtol"] * expected.abs())
        error = scaled.max().item() if finite else None
        if error is not None:
            maximum = max(maximum, error)
        if not passed:
            failures.append({"weight": name, "finite": finite, "max_scaled_error": error})
    return {"passed_pairs": len(refs) - len(failures), "total_pairs": len(refs),
            "max_scaled_error": maximum, "failures": failures}


def trace_graph(graph, count, backend, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = collections.Counter(e["name"] for e in events if e.get("cat") == "kernel")
    expected = ("down_partial", "up_gate") if backend == "flydsl_selected" else (
        "_hc_mix_down_kernel", "_hc_mix_reduce_kernel", "_hc_mix_up_kernel"
    )
    assert sum(kernels.values()) == count * len(expected), kernels
    for token in expected:
        assert sum(n for name, n in kernels.items() if token in name) == count, kernels
    return {"path": str(path), "calls": count, "kernels_per_call": len(expected), "kernel_counts": dict(kernels)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--weights", type=int, default=100)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--model-path", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-dir", type=Path, help="optional dispatch-count traces, collected after timing")
    parser.add_argument("--trace-rows", type=int, nargs="+", default=[1, 16])
    args = parser.parse_args()
    if any(not 1 <= t <= 16 for t in args.rows):
        parser.error("this comparison covers only T=1..16")
    if min(args.weights, args.rounds, args.samples) < 1:
        parser.error("weights, rounds and samples must be positive")
    if args.model_path and not (args.model_path / "model.safetensors.index.json").is_file():
        parser.error("checkpoint index not found; omit --model-path for synthetic weights")
    if hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("the bundled 626c6413 source has changed; restore the exact upstream copy")
    if torch.version.hip is None:
        parser.error("this comparison requires ROCm PyTorch")
    module = load_triton_baseline(SOURCE_PATH)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=False)
    with args.output.open("x") if args.output else nullcontext() as log:
        def emit(record):
            if log is not None:
                log.write(json.dumps(record) + "\n")
                log.flush()

        props = torch.cuda.get_device_properties(0)
        meta = {
            "type": "environment", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "gpu": props.name, "arch": props.gcnArchName, "compute_units": props.multi_processor_count,
            "torch": torch.__version__, "hip": torch.version.hip, "triton": triton.__version__,
            "flydsl": importlib.metadata.version("flydsl"), "hip_visible_devices": os.getenv("HIP_VISIBLE_DEVICES"),
            "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256,
            "weight_source": "checkpoint" if args.model_path else "synthetic",
            "timer": "support.capture/time_graph; 2 passes/graph, 3 replays/sample; full GR, no stage-time summation",
            "allocation": "FlyDSL prepares weights/P/Y before capture; upstream public wrapper unchanged, temporaries allocated during capture from graph pool",
            "fp64_tolerance": TOLERANCES[torch.bfloat16],
            "accuracy_policy": "FlyDSL failures abort; upstream failures are reported without hiding its timing",
            "not_measured": "eager host time, weight packing, hc_norm, serving throughput",
            "sources": {},
        }
        for name in ("bench_compare_626c6413.py", "combined_host.py", "prefetch_up.py", "kernel.py", "support.py"):
            path = Path(__file__).with_name(name)
            content = path.read_bytes()
            meta["sources"][name] = hashlib.sha256(content).hexdigest()
            if log is not None:
                with args.output.with_name(args.output.stem + "." + name).open("xb") as snapshot:
                    snapshot.write(content)
        meta["sources"][SOURCE_PATH.name] = SOURCE_SHA256
        if log is not None:
            with args.output.with_name(args.output.stem + "." + SOURCE_PATH.name).open("xb") as snapshot:
                snapshot.write(SOURCE_PATH.read_bytes())
        if args.model_path:
            pairs = list(checkpoint_pairs(args.weights, args.model_path))
        else:
            pairs = [(f"synthetic_{i}", *synthetic(1, seed=args.seed + i)[1:]) for i in range(args.weights)]
        if len(pairs) != args.weights:
            parser.error("requested more weight pairs than the checkpoint contains")
        names = [name for name, _, _ in pairs]
        meta["weight_names"] = names
        emit(meta)
        print(f"{props.name}, {props.gcnArchName}, source={meta['weight_source']}, weights={len(pairs)}", flush=True)
        print("Graph us = complete GR call. Upstream source is unchanged; FP64 failures remain visible.", flush=True)
        rng = random.Random(args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        for rows in args.rows:
            print(f"T={rows}: preparing and capturing both implementations", flush=True)
            xs = [torch.randn(rows, 10240, dtype=torch.bfloat16, device="cuda", generator=generator) for _ in pairs]
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            readers = {
                "flydsl_selected": [CombinedPaddedGRRead(rows, wd, wu) for _, wd, wu in pairs],
                "triton_626c6413": [ColleagueReader(module, wd, wu) for _, wd, wu in pairs],
            }
            graphs, counts, outputs, results = {}, {}, {}, {}
            down_cfg, reduce_cfg, up_cfg = module._select_configs(
                props=props, rows_pad=16, dtype=torch.bfloat16, hc=4, hs=2560, lowrank=320, k=10240
            )
            for name, group in readers.items():
                calls = [lambda reader=reader, x=x: reader(x) for reader, x in zip(group, xs)]
                graphs[name], counts[name] = capture(calls, repeats=2)
                outputs[name] = [reader.output for reader in group]
                graphs[name].replay()
                initial = verify(outputs[name], refs, names)
                if name == "flydsl_selected":
                    assert not initial["failures"], initial
                config = asdict(group[0].config) if name == "flydsl_selected" else {
                    "rows_pad": 16, "down": down_cfg, "reduce": reduce_cfg, "up": up_cfg,
                    "k_chunks": triton.cdiv(10240, down_cfg["BLOCK_K"]), "activation_dtype": "BF16",
                }
                results[name] = {"config": config, "graph_rounds": [], "initial_fp64": initial}
            for round_id in range(args.rounds):
                order = list(readers)
                rng.shuffle(order)
                for name in order:
                    timing = time_graph(graphs[name], counts[name], args.samples, replay_per_sample=3)
                    results[name]["graph_rounds"].append({"round": round_id, "order": order, **timing})
            for x in xs:
                x.mul_(0.99).add_(0.015625)
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            print(f"{'Backend':<22} {'Graph us':>10} {'FP64 initial':>14} {'FP64 replay':>14}", flush=True)
            for name in readers:
                for _ in range(20):
                    graphs[name].replay()
                changed = verify(outputs[name], refs, names)
                if name == "flydsl_selected":
                    assert not changed["failures"], changed
                result = results[name]
                result["changed_input_fp64"] = changed
                result["graph_median_us"] = statistics.median(v for r in result["graph_rounds"] for v in r["samples_us"])
                initial = result["initial_fp64"]
                print(f"{name:<22} {result['graph_median_us']:10.3f} "
                      f"{initial['passed_pairs']:>9}/{len(pairs):<4} {changed['passed_pairs']:>9}/{len(pairs):<4}", flush=True)
                if args.trace_dir and rows in args.trace_rows:
                    result["trace"] = trace_graph(graphs[name], counts[name], name, args.trace_dir / f"t{rows}_{name}.json")
            speedup = results["flydsl_selected"]["graph_median_us"] / results["triton_626c6413"]["graph_median_us"]
            emit({"type": "result", "rows": rows, "weight_pairs": len(pairs), "results": results,
                  "colleague_speedup_over_flydsl": speedup})
            print(f"  Colleague speedup over FlyDSL: {speedup:.3f}x (>1 favors colleague)", flush=True)
            del graphs, readers, outputs, refs, xs, calls, group


if __name__ == "__main__":
    main()
