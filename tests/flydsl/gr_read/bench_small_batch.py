"""T1..16 tuning with immutable weights; --selected uses measured defaults."""

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
    from .bench_compare_626c6413 import ColleagueReader, SOURCE_COMMIT, SOURCE_PATH, SOURCE_SHA256, verify
    from .bench_large_down import time_eager
    from .combined_host import CombinedPaddedGRRead
    from .large_down import DownConfig
    from .small_batch import SmallBatchGRRead, SmallConfig, default_config
    from .support import capture, checkpoint_pairs, load_triton_baseline, reference, synthetic, time_graph
else:
    from bench_compare_626c6413 import ColleagueReader, SOURCE_COMMIT, SOURCE_PATH, SOURCE_SHA256, verify
    from bench_large_down import time_eager
    from combined_host import CombinedPaddedGRRead
    from large_down import DownConfig
    from small_batch import SmallBatchGRRead, SmallConfig, default_config
    from support import capture, checkpoint_pairs, load_triton_baseline, reference, synthetic, time_graph


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=list(range(1, 17)))
    parser.add_argument("--down-k", nargs="+", type=int, default=[128])
    parser.add_argument("--down-waves", nargs="+", type=int, default=[4])
    parser.add_argument("--split", nargs="+", type=int, default=[4])
    parser.add_argument("--prefetch", nargs="+", type=int, choices=[0, 1], default=[1])
    parser.add_argument("--unroll", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--up-n", nargs="+", type=int, default=[128])
    parser.add_argument("--up-waves", nargs="+", type=int, default=[4])
    parser.add_argument("--selected", action="store_true", help="use the measured per-T config instead of the tuning grid")
    parser.add_argument("--weights", type=int, default=100)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--model-path", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--stages", action="store_true", help="also measure isolated FlyDSL down/up graphs")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not 1 <= t <= 16 for t in args.rows) or min(args.weights, args.rounds, args.samples) < 1:
        parser.error("rows must be 1..16 and counts must be positive")
    if args.model_path and not (args.model_path / "model.safetensors.index.json").is_file():
        parser.error("checkpoint index not found; omit --model-path for synthetic weights")
    configs = [SmallConfig(DownConfig(block_k=k, waves=w, global_split=s, prefetch=bool(p), prefetch_unroll=u), n, uw)
               for k in args.down_k for w in args.down_waves for s in args.split for p in args.prefetch
               for u in args.unroll for n in args.up_n for uw in args.up_waves]
    if args.selected:
        configs = []
    for cfg in configs:
        try:
            cfg.validate()
        except ValueError as error:
            parser.error(f"{cfg.name}: {error}")
    if hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("626c6413 baseline has changed")
    upstream = load_triton_baseline(SOURCE_PATH)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") if args.output else nullcontext() as log:
        def emit(value):
            if log:
                log.write(json.dumps(value) + "\n")
                log.flush()

        props = torch.cuda.get_device_properties(0)
        meta = {"type": "environment", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "gpu": props.name, "arch": props.gcnArchName, "compute_units": props.multi_processor_count,
                "torch": torch.__version__, "hip": torch.version.hip,
                "flydsl": importlib.metadata.version("flydsl"), "triton": importlib.metadata.version("triton"),
                "hip_visible_devices": os.getenv("HIP_VISIBLE_DEVICES"), "baseline_commit": SOURCE_COMMIT,
                "source_sha256": {}, "weight_source": "checkpoint" if args.model_path else "synthetic",
                "timer": "200 full calls/graph for 100 weights; 3 replays/sample; shuffled backends/stages per round",
                "contract": "All FlyDSL candidates share control packed weight pointers; original reorder/preshuffle unchanged",
                "fp64": "rtol=1e-2, atol=5e-3; FlyDSL enforced, upstream accuracy only recorded"}
        paths = [Path(__file__).with_name(n) for n in (
            "small_batch.py", "large_down.py", "prefetch_up.py", "combined_host.py", "support.py",
            "bench_small_batch.py", "bench_compare_626c6413.py", "bench_large_down.py")]
        paths.append(SOURCE_PATH)
        for path in paths:
            content = path.read_bytes()
            meta["source_sha256"][path.name] = hashlib.sha256(content).hexdigest()
            if log:
                with args.output.with_name(args.output.stem + "." + path.name).open("xb") as f:
                    f.write(content)
        pairs = list(checkpoint_pairs(args.weights, args.model_path)) if args.model_path else [
            (f"synthetic_{i}", *synthetic(1, seed=args.seed + i)[1:]) for i in range(args.weights)]
        if len(pairs) != args.weights:
            parser.error("requested more checkpoint pairs than available")
        names = [n for n, _, _ in pairs]
        meta["weight_names"] = names
        emit(meta)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        rng = random.Random(args.seed)
        for rows in args.rows:
            print(f"T={rows}: preparing {len(pairs)} {meta['weight_source']} pairs", flush=True)
            xs = [torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in pairs]
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            controls = [CombinedPaddedGRRead(rows, wd, wu) for _, wd, wu in pairs]
            groups = {"frozen": controls, "triton3stage": [ColleagueReader(upstream, wd, wu) for _, wd, wu in pairs]}
            for cfg in ([default_config(rows)] if args.selected else configs):
                name = "ours_optimized" if args.selected else cfg.name
                groups[name] = [SmallBatchGRRead(control, cfg) for control in controls]
            graphs, counts, outputs, timings, checks, eager = {}, {}, {}, {}, {}, {}
            for name, group in groups.items():
                for reader, control in zip(group, controls):
                    if isinstance(reader, SmallBatchGRRead):
                        assert reader.w_down.data_ptr() == control.w_down.data_ptr()
                        assert reader.w_up.data_ptr() == control.w_up.data_ptr()
                for stage in (("full", "down", "up") if args.stages and name != "triton3stage" else ("full",)):
                    calls = [lambda r=r, x=x, s=stage: r(x) if s == "full" else getattr(r, "run_" + s)(x)
                             for r, x in zip(group, xs)]
                    key = (name, stage)
                    graphs[key], counts[key] = capture(calls, repeats=2)
                    timings[key] = []
                    if stage == "full":
                        outputs[name] = [r.output for r in group]
                        graphs[key].replay()
                        checks[name] = {"initial": verify(outputs[name], refs, names)}
                        if name != "triton3stage":
                            assert not checks[name]["initial"]["failures"], checks[name]
                eager[name] = []
            for _ in range(args.rounds):
                order = list(graphs)
                rng.shuffle(order)
                for key in order:
                    timings[key].append(time_graph(graphs[key], counts[key], args.samples, replay_per_sample=3))
                if args.eager:
                    order = list(groups)
                    rng.shuffle(order)
                    for name in order:
                        calls = [lambda r=r, x=x: r(x) for r, x in zip(groups[name], xs)]
                        eager[name].append(time_eager(calls, args.samples))
            for x in xs:
                x.mul_(0.99).add_(0.015625)
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            print(f"{'Backend':<43} {'Full graph':>11} {'Down only':>11} {'Up only':>11} {'Eager wall':>11}", flush=True)
            for name, group in groups.items():
                for _ in range(20):
                    graphs[(name, "full")].replay()
                checks[name]["changed"] = verify(outputs[name], refs, names)
                if name != "triton3stage":
                    assert not checks[name]["changed"]["failures"], checks[name]
                medians = {s: statistics.median(v for r in timings[(name, s)] for v in r["samples_us"])
                           for s in ("full", "down", "up") if (name, s) in timings}
                wall = statistics.median(v for r in eager[name] for v in r["wall_samples_us"]) if args.eager else None
                emit({"type": "result", "rows": rows, "backend": name,
                      "config": asdict(group[0].config) if name != "triton3stage" else None,
                      "median_us": medians, "rounds": {s: timings[(name, s)] for s in medians},
                      "eager_rounds": eager[name], "eager_wall_us": wall, "checks": checks[name],
                      "packed_weight_pointers_shared": isinstance(group[0], SmallBatchGRRead)})
                vals = [medians.get(s) for s in ("full", "down", "up")] + [wall]
                print(f"{name:<43}" + "".join(f"{v:11.3f}" if v is not None else f"{'-':>11}" for v in vals), flush=True)
            del graphs, outputs, groups, controls, refs, xs, calls, group, reader, control


if __name__ == "__main__":
    main()
