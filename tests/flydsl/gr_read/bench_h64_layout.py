"""Same-address old/new-layout GR comparison. Synthetic by default; no SGLang.

Runtime X/WD/WU/P/Y addresses are shared by all measured implementations.
Only this benchmark keeps old/new source copies and switches WU before timing;
the H64 reader itself consumes one prepacked pair without conversion.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import statistics
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import torch

if __package__:
    from .bench_large_down import time_eager
    from .combined_host import CombinedPaddedGRRead
    from .h64_layout import H64GRRead, LAYOUT_ID, default_configs, prepare_up_weight
    from .large_down import LargeDownGRRead
    from .small_batch import SmallBatchGRRead
    from .support import TOLERANCES, capture, checkpoint_pairs, reference, synthetic, time_graph
else:
    from bench_large_down import time_eager
    from combined_host import CombinedPaddedGRRead
    from h64_layout import H64GRRead, LAYOUT_ID, default_configs, prepare_up_weight
    from large_down import LargeDownGRRead
    from small_batch import SmallBatchGRRead
    from support import TOLERANCES, capture, checkpoint_pairs, reference, synthetic, time_graph


def idle_preflight():
    hip, cuda = os.getenv("HIP_VISIBLE_DEVICES"), os.getenv("CUDA_VISIBLE_DEVICES")
    if hip and cuda and hip != cuda:
        raise ValueError("HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES must match")
    physical = hip or cuda or "0"
    if not physical.isdigit() or os.getenv("ROCR_VISIBLE_DEVICES"):
        raise ValueError("select one physical GPU with HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES")
    env = {k: v for k, v in os.environ.items() if k not in (
        "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
    result = subprocess.run(
        ["rocm-smi", "-d", physical, "--showuse", "--showmemuse", "--json"],
        env=env, text=True, capture_output=True, check=True, timeout=30,
    )
    card = json.loads(result.stdout)[f"card{physical}"]
    if int(card["GPU use (%)"]) != 0 or int(card["GPU Memory Allocated (VRAM%)"]) != 0:
        raise RuntimeError(f"GPU{physical} occupied; wait and retry, do not benchmark: {card}")
    return {"physical_gpu": int(physical), "rocm_smi": card,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def verify(group, refs):
    maximum = 0.0
    tol = TOLERANCES[torch.bfloat16]
    for reader, expected in zip(group, refs):
        actual = reader.output.double()
        assert torch.allclose(actual, expected, **tol), "FP64 mismatch"
        maximum = max(maximum, ((actual - expected).abs() / (tol["atol"] + tol["rtol"] * expected.abs())).max().item())
    return {"passed_pairs": len(group), "max_scaled_error": maximum}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=list(range(1, 33)))
    parser.add_argument("--selected", action="store_true", help="compare the fixed H64 candidate selection; no tuning sweep")
    parser.add_argument("--check-no-regression", action="store_true", help="fail after all rows if a selected full-graph median exceeds baseline")
    parser.add_argument("--preload-weights", type=int, nargs="+", choices=[0, 1], help="selected-path override; default uses the per-T choice")
    parser.add_argument("--up-n", type=int, nargs="+", help="explicit tile sweep; otherwise use the selected per-T tile")
    parser.add_argument("--up-waves", type=int, nargs="+", default=[4])
    parser.add_argument("--up-k", type=int, nargs="+", choices=[32, 64, 160, 320], default=[64])
    parser.add_argument("--skip-padding", type=int, nargs="+", choices=[0, 1], default=[1])
    parser.add_argument("--strategy", nargs="+", choices=["remap", "plane", "register"], default=["remap"])
    parser.add_argument("--weight-copy-bits", type=int, nargs="+", choices=[64, 128], default=[64])
    parser.add_argument("--b-first", type=int, nargs="+", choices=[0, 1], default=[0])
    parser.add_argument("--weights", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--eager", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not 1 <= t <= 32 for t in args.rows) or min(args.weights, args.rounds, args.samples) < 1:
        parser.error("rows must be 1..32 and counts must be positive")
    if args.model_path and not (args.model_path / "model.safetensors.index.json").is_file():
        parser.error("checkpoint index not found; use --synthetic without a model")
    if not args.selected and (args.preload_weights is not None or args.check_no_regression):
        parser.error("--preload-weights/--check-no-regression require --selected")
    preflight = idle_preflight()
    print(f"GPU{preflight['physical_gpu']} rocm-smi idle preflight PASS", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") if args.output else nullcontext() as log:
        def emit(record):
            if log:
                log.write(json.dumps(record) + "\n")
                log.flush()

        props = torch.cuda.get_device_properties(0)
        meta = {"type": "environment", "preflight": preflight,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "gpu": props.name, "arch": props.gcnArchName, "compute_units": props.multi_processor_count,
                "torch": torch.__version__, "hip": torch.version.hip,
                "flydsl": importlib.metadata.version("flydsl"), "layout": LAYOUT_ID,
                "weight_source": "checkpoint" if args.model_path else "synthetic",
                "timing": "2 passes/graph, 3 replays/sample, shuffled stage/backend order; layout copies outside timing",
                "same_runtime_addresses": "X, WD, WU, P, Y; comparisons serialized", "sources": {}}
        for name in ("h64_layout.py", "bench_h64_layout.py", "small_batch.py", "large_down.py",
                     "prefetch_up.py", "combined_host.py", "support.py", "bench_large_down.py"):
            content = Path(__file__).with_name(name).read_bytes()
            meta["sources"][name] = hashlib.sha256(content).hexdigest()
            if log:
                with args.output.with_name(args.output.stem + "." + name).open("xb") as snapshot:
                    snapshot.write(content)
        pairs = list(checkpoint_pairs(args.weights, args.model_path)) if args.model_path else [
            (f"synthetic_{i}", *synthetic(1, seed=args.seed + i)[1:]) for i in range(args.weights)]
        if len(pairs) != args.weights:
            parser.error("requested more HC pairs than available")
        meta["weight_names"] = [name for name, _, _ in pairs]
        emit(meta)
        rng = random.Random(args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        regressions = []
        for rows in args.rows:
            print(f"T={rows}: preparing {len(pairs)} {meta['weight_source']} pairs", flush=True)
            xs = [torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in pairs]
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            prepared = [CombinedPaddedGRRead(rows, wd, wu) for _, wd, wu in pairs]
            old = [SmallBatchGRRead(p) if rows <= 16 else LargeDownGRRead(p) for p in prepared]
            old_up = [r.w_up for r in old]
            new_up = [prepare_up_weight(wu) for _, _, wu in pairs]
            shared_up = [w.clone() for w in new_up]
            _, selected = default_configs(rows)
            groups = {"baseline_selected": old}
            for n in [] if args.selected else args.up_n or [selected.up_n]:
                for waves, bk, skip in product(args.up_waves, args.up_k, args.skip_padding):
                    cfg = replace(selected, up_n=n, waves=waves, block_k=bk, skip_padding=bool(skip))
                    for strategy in args.strategy:
                        if strategy == "register" and n != 64 * waves:
                            continue
                        for bits in args.weight_copy_bits:
                            for b_first in args.b_first:
                                name = f"h64_{strategy}_n{n}w{waves}_b{bits}_f{b_first}_k{bk}_s{skip}"
                                groups[name] = [H64GRRead(rows, r.w_down, wu, up_config=cfg, strategy=strategy,
                                                        weight_copy_bits=bits, b_first=bool(b_first))
                                                for r, wu in zip(old, shared_up)]
            if args.selected:
                for preload in args.preload_weights if args.preload_weights is not None else [None]:
                    name = "h64_selected" if preload is None else "h64_selected_preload" if preload else "h64_selected_nopreload"
                    groups[name] = [H64GRRead(rows, r.w_down, wu, preload_weights=None if preload is None else bool(preload))
                                    for r, wu in zip(old, shared_up)]
            if len(groups) == 1:
                parser.error("no compatible H64 configurations selected")
            # Benchmark-only aliasing: no two graphs execute concurrently.
            for group in groups.values():
                for r, baseline, wu in zip(group, old, shared_up):
                    r.w_up, r.partial, r.output = wu, baseline.partial, baseline.output
                    assert r.w_down.data_ptr() == baseline.w_down.data_ptr()
                    assert r.w_up.data_ptr() == baseline.w_up.data_ptr()
                    assert r.partial.data_ptr() == baseline.partial.data_ptr()
                    assert r.output.data_ptr() == baseline.output.data_ptr()
            active_layout = None

            def switch(name):
                nonlocal active_layout
                layout = "old" if name == "baseline_selected" else "new"
                if layout != active_layout:
                    torch._foreach_copy_(shared_up, old_up if layout == "old" else new_up)
                    active_layout = layout

            graphs, counts, timings, checks, eager = {}, {}, {}, {}, {}
            for name, group in groups.items():
                switch(name)
                for stage in ("full", "down", "up") if args.stages else ("full",):
                    calls = [lambda r=r, x=x, s=stage: r(x) if s == "full" else getattr(r, "run_" + s)(x)
                             for r, x in zip(group, xs)]
                    key = (name, stage)
                    graphs[key], counts[key] = capture(calls, repeats=2)
                    timings[key] = []
                    if stage == "full":
                        graphs[key].replay()
                        checks[name] = {"initial": verify(group, refs)}
                eager[name] = []
            for _ in range(args.rounds):
                order = list(graphs)
                rng.shuffle(order)
                for key in order:
                    switch(key[0])
                    timings[key].append(time_graph(graphs[key], counts[key], args.samples, replay_per_sample=3))
                if args.eager:
                    order = list(groups)
                    rng.shuffle(order)
                    for name in order:
                        switch(name)
                        calls = [lambda r=r, x=x: r(x) for r, x in zip(groups[name], xs)]
                        eager[name].append(time_eager(calls, args.samples))
            for x in xs:
                x.mul_(0.99).add_(0.015625)
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            baseline_us = statistics.median(v for r in timings[("baseline_selected", "full")] for v in r["samples_us"])
            print(f"{'Backend':<24} {'Full graph':>12} {'Delta %':>10} {'Down':>10} {'Up':>10} {'Eager':>10}", flush=True)
            for name, group in groups.items():
                switch(name)
                for _ in range(20):
                    graphs[(name, "full")].replay()
                checks[name]["changed"] = verify(group, refs)
                medians = {s: statistics.median(v for r in timings[(name, s)] for v in r["samples_us"])
                           for s in ("full", "down", "up") if (name, s) in timings}
                wall = statistics.median(v for r in eager[name] for v in r["wall_samples_us"]) if args.eager else None
                delta = (medians["full"] / baseline_us - 1) * 100
                if name != "baseline_selected" and delta > 0:
                    regressions.append({"rows": rows, "backend": name, "delta_pct": delta})
                r = group[0]
                up = r.up_config
                down = r.down_config if isinstance(r, H64GRRead) else (r.config.down if rows <= 16 else r.config)
                emit({"type": "result", "rows": rows, "backend": name,
                      "down_config": asdict(down), "up_config": asdict(up),
                      "strategy": getattr(r, "strategy", "baseline"),
                      "weight_copy_bits": getattr(r, "weight_copy_bits", 64),
                      "b_first": getattr(r, "b_first", False),
                      "preload_weights": getattr(r, "preload_weights", False),
                      "median_us": medians, "delta_pct": delta, "checks": checks[name],
                      "rounds": {s: timings[(name, s)] for s in medians},
                      "eager_wall_us": wall, "eager_rounds": eager[name], "same_addresses": True})
                values = [medians.get("down"), medians.get("up"), wall]
                print(f"{name:<24}{medians['full']:12.3f}{delta:10.2f}" +
                      "".join(f"{v:10.3f}" if v is not None else f"{'-':>10}" for v in values), flush=True)
            del graphs, groups, old, prepared, refs, xs, calls, group, r, baseline, new_up, old_up, shared_up
        emit({"type": "summary", "rows": args.rows, "regressions": regressions,
              "measured_no_regression": not regressions})
        if args.check_no_regression:
            assert not regressions, f"full-graph regressions: {regressions}"
            print("Full-graph no-regression check PASS for every requested row", flush=True)


if __name__ == "__main__":
    main()
