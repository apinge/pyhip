"""Matched large-T down experiments. Synthetic default; no SGLang dependency.

Example: HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python3 bench_large_down.py
Every candidate shares the control's exact weight pointers. Down/up timings
are isolated graph repeats, not a decomposition of the full-call latency.
"""

import argparse
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

if __package__:
    from .combined_host import CombinedPaddedGRRead
    from .large_down import DownConfig, LargeDownGRRead, default_config
    from .support import TOLERANCES, capture, checkpoint_pairs, reference, synthetic, time_graph, torch_mix
else:
    from combined_host import CombinedPaddedGRRead
    from large_down import DownConfig, LargeDownGRRead, default_config
    from support import TOLERANCES, capture, checkpoint_pairs, reference, synthetic, time_graph, torch_mix


class TorchReader:
    def __init__(self, call, wd, wu):
        self.call, self.wd, self.wu = call, wd, wu
        self.output = None

    def __call__(self, x):
        self.output = self.call(x, self.wd, self.wu)
        return self.output


def time_eager(calls, samples):
    repeats = max(1, (200 + len(calls) - 1) // len(calls))
    for _ in range(3):
        for call in calls:
            call()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(repeats):
            for call in calls:
                call()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1e6 / (len(calls) * repeats))
    return {"wall_samples_us": values, "calls_per_sample": len(calls) * repeats}


def verify(readers, refs, outputs=None, enforce=True):
    tol = TOLERANCES[torch.bfloat16]
    maximum, passed = 0.0, 0
    outputs = [reader.output for reader in readers] if outputs is None else outputs
    for index, (output, expected) in enumerate(zip(outputs, refs)):
        actual = output.double()
        close = torch.allclose(actual, expected, **tol)
        passed += close
        if enforce:
            assert close, f"pair {index}: FP64 mismatch"
        maximum = max(maximum, ((actual - expected).abs() / (tol['atol'] + tol['rtol'] * expected.abs())).max().item())
    return {"passed_pairs": passed, "total_pairs": len(refs), "max_scaled_error": maximum, "enforced": enforce}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[17, 24, 32])
    parser.add_argument("--block-k", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--block-m", type=int, nargs="+", default=[16])
    parser.add_argument("--waves", type=int, nargs="+", default=[4])
    parser.add_argument("--prefetch", type=int, choices=[0, 1], nargs="+", default=[0, 1])
    parser.add_argument("--global-split", type=int, nargs="+", default=[1])
    parser.add_argument("--prefetch-unroll", type=int, nargs="+", default=[1])
    parser.add_argument("--selected", action="store_true", help="use the recommended per-T candidate instead of the tuning grid")
    parser.add_argument("--contiguous-k", action="store_true")
    parser.add_argument("--weights", type=int, default=100)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--model-path", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--eager", action="store_true", help="also time full-call eager host wall time")
    parser.add_argument("--torch-baseline", action="store_true", help="also measure the current torch.compile fallback")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not 17 <= t <= 32 for t in args.rows):
        parser.error("only T=17..32; T<=16 is frozen")
    if min(args.weights, args.rounds, args.samples) < 1:
        parser.error("weights, rounds and samples must be positive")
    if args.model_path and not (args.model_path / "model.safetensors.index.json").is_file():
        parser.error("checkpoint index not found; omit --model-path for synthetic weights")
    configs = [DownConfig(m, k, w, bool(p), not args.contiguous_k, s, u)
               for m in args.block_m for k in args.block_k for w in args.waves
               for p in args.prefetch for s in args.global_split for u in args.prefetch_unroll]
    if args.selected:
        configs = []
    for config in configs:
        try:
            config.validate()
        except ValueError as error:
            parser.error(str(error))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") if args.output else nullcontext() as log:
        def emit(record):
            if log:
                log.write(json.dumps(record) + "\n")
                log.flush()

        props = torch.cuda.get_device_properties(0)
        meta = {
            "type": "environment", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "gpu": props.name, "arch": props.gcnArchName, "compute_units": props.multi_processor_count,
            "torch": torch.__version__, "hip": torch.version.hip,
            "flydsl": importlib.metadata.version("flydsl"), "hip_visible_devices": os.getenv("HIP_VISIBLE_DEVICES"),
            "timer": "2 passes/graph, 3 replays/sample; randomized backend and stage order each round",
            "weight_source": "checkpoint" if args.model_path else "synthetic", "sources": {},
            "fp64_tolerance": TOLERANCES[torch.bfloat16],
            "baseline_accuracy": "torch.compile FP64 errors are recorded, not enforced; FlyDSL is enforced",
        }
        for name in ("large_down.py", "bench_large_down.py", "combined_host.py", "prefetch_up.py", "support.py"):
            path = Path(__file__).with_name(name)
            data = path.read_bytes()
            meta["sources"][name] = hashlib.sha256(data).hexdigest()
            if log:
                with args.output.with_name(args.output.stem + "." + name).open("xb") as snapshot:
                    snapshot.write(data)
        if args.model_path:
            pairs = list(checkpoint_pairs(args.weights, args.model_path))
        else:
            pairs = [(f"synthetic_{i}", *synthetic(1, seed=args.seed + i)[1:]) for i in range(args.weights)]
        if len(pairs) != args.weights:
            parser.error("requested more weight pairs than the checkpoint contains")
        meta["weight_names"] = [name for name, _, _ in pairs]
        emit(meta)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        rng = random.Random(args.seed)
        compiled = None
        if args.torch_baseline:
            torch._dynamo.config.recompile_limit = 64
            torch._dynamo.config.fail_on_recompile_limit_hit = True
            compiled = torch.compile(torch_mix, dynamic=False, fullgraph=True)
        for rows in args.rows:
            print(f"T={rows}: {len(pairs)} {meta['weight_source']} pairs, compiling candidates", flush=True)
            xs = [torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in pairs]
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            controls = [CombinedPaddedGRRead(rows, wd, wu) for _, wd, wu in pairs]
            readers = {"control": controls}
            for config in ([default_config(rows)] if args.selected else configs):
                readers[config.name] = [LargeDownGRRead(base, config) for base in controls]
            if compiled:
                readers["torch_compile"] = [TorchReader(compiled, wd, wu) for _, wd, wu in pairs]
            graphs, counts, checks, timings, eager, graph_outputs = {}, {}, {}, {}, {}, {}
            for name, group in readers.items():
                for base, reader, x in zip(controls, group, xs):
                    if name != "torch_compile":
                        assert reader.w_down.data_ptr() == base.w_down.data_ptr()
                        assert reader.w_up.data_ptr() == base.w_up.data_ptr()
                    reader(x)
                    if name != "torch_compile":
                        assert torch.isfinite(reader.partial).all(), f"{name}: nonfinite workspace"
                        assert torch.count_nonzero(reader.partial.view(-1, 32, 320)[:, rows:]) == 0, f"{name}: padded rows"
                checks[name] = {"initial": verify(group, refs, enforce=name != "torch_compile")}
                eager[name] = []
                for stage in (("full",) if name == "torch_compile" else ("full", "down", "up")):
                    calls = [lambda r=r, x=x, stage=stage: r(x) if stage == "full" else getattr(r, "run_" + stage)(x)
                             for r, x in zip(group, xs)]
                    key = (name, stage)
                    graphs[key], counts[key] = capture(calls, repeats=2)
                    if stage == "full":
                        graph_outputs[name] = [reader.output for reader in group]
                    timings[key] = []
            for _ in range(args.rounds):
                order = list(graphs)
                rng.shuffle(order)
                for key in order:
                    timings[key].append(time_graph(graphs[key], counts[key], args.samples, replay_per_sample=3))
                if args.eager:
                    names = list(readers)
                    rng.shuffle(names)
                    for name in names:
                        calls = [lambda r=r, x=x: r(x) for r, x in zip(readers[name], xs)]
                        eager[name].append(time_eager(calls, args.samples))
            for x in xs:
                x.mul_(0.99).add_(0.015625)
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            print(f"{'Backend':<28} {'Full graph':>12} {'Down only':>12} {'Up only':>12} {'Eager wall':>12}", flush=True)
            for name, group in readers.items():
                for _ in range(3):
                    graphs[(name, "full")].replay()
                checks[name]["changed_replay"] = verify(
                    group, refs, graph_outputs[name], enforce=name != "torch_compile"
                )
                medians = {stage: statistics.median([v for r in timings[(name, stage)] for v in r['samples_us']])
                           for stage in ("full", "down", "up") if (name, stage) in timings}
                eager_us = statistics.median([v for r in eager[name] for v in r['wall_samples_us']]) if args.eager else None
                emit({"type": "result", "rows": rows, "backend": name,
                      "config": asdict(group[0].config) if name != "torch_compile" else None,
                      "median_us": medians, "rounds": {s: timings[(name, s)] for s in medians}, "checks": checks[name],
                      "eager_rounds": eager[name], "eager_wall_median_us": eager_us,
                      "weight_pointers_shared": name != "torch_compile"})
                values = [medians.get(s) for s in ("full", "down", "up")] + [eager_us]
                print(f"{name:<28}" + "".join(f"{v:12.3f}" if v is not None else f"{'-':>12}" for v in values), flush=True)
                if name == "torch_compile":
                    for phase, result in checks[name].items():
                        if result['passed_pairs'] != result['total_pairs']:
                            print(f"  torch_compile {phase}: FP64 tolerance passed {result['passed_pairs']}/{result['total_pairs']}", flush=True)
            del graphs, readers, controls, refs, xs


if __name__ == "__main__":
    main()
