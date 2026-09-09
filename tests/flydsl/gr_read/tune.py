"""Screen explicit configurations on rotating real HC weights."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from kernel import Config, GRRead
from support import TOLERANCES, capture, checkpoint_pairs, reference, time_graph


def configurations(rows, family):
    base = Config(split_k=8, down_n=32, waves=2, block_m=32 if rows > 16 else 16)
    if family == "compensated":
        base = replace(base, waves=4, down_n=64, split_k=16, compensate_hidden=True)
        return [
            ("compensated", base),
            ("comp_other_m", replace(base, block_m=16 if rows > 16 else 32)),
            ("comp_un64", replace(base, up_n=64)),
            ("comp_s32", replace(base, split_k=32)),
        ]
    if family == "refine":
        base = replace(base, waves=4, down_n=64, split_k=16)
        return [
            ("w4_fast", base),
            ("w4_strict", replace(base, fast_math=False)),
            ("w4_s8", replace(base, split_k=8)),
            ("w4_s32", replace(base, split_k=32)),
            ("w4_un64", replace(base, up_n=64)),
            ("w4_un256", replace(base, up_n=256)),
            ("w4_other_m", replace(base, block_m=16 if rows > 16 else 32)),
            ("w4_bk32", replace(base, block_k=32)),
        ]
    base = replace(base, fast_math=False)
    return [
        ("base", base),
        ("split4", replace(base, split_k=4)),
        ("split16", replace(base, split_k=16)),
        ("split32", replace(base, split_k=32)),
        ("dn64_s16", replace(base, down_n=64, split_k=16)),
        ("dn16_w1_s4", replace(base, down_n=16, waves=1, split_k=4)),
        ("dn16_w1_s8", replace(base, down_n=16, waves=1, split_k=8)),
        ("dn32_w1_s8", replace(base, waves=1)),
        ("un64", replace(base, up_n=64)),
        ("un256", replace(base, up_n=256)),
        ("bk32", replace(base, block_k=32)),
        ("w4", replace(base, down_n=64, waves=4, split_k=16)),
        ("no_padding_skip", replace(base, skip_padding=False)),
        ("other_m_tile", replace(base, block_m=16 if rows > 16 else 32)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 8, 16, 24])
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--family", choices=["initial", "refine", "compensated"], default="initial")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as log:
        pairs = list(checkpoint_pairs(limit=args.pairs))
        header = {
            "type": "environment",
            "time": time.time(),
            "seed": args.seed,
            "flydsl": importlib.metadata.version("flydsl"),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "physical_gpu": "HIP_VISIBLE_DEVICES",
            "pairs": [p[0] for p in pairs],
            "rows": args.rows,
            "kernel_sha256": hashlib.sha256(Path(__file__).with_name("kernel.py").read_bytes()).hexdigest(),
        }
        log.write(json.dumps(header) + "\n")
        log.flush()
        rng = random.Random(args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        for rows in args.rows:
            xs = [torch.randn(rows, 10240, dtype=torch.bfloat16, device="cuda", generator=generator) for _ in pairs]
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            configs = configurations(rows, args.family)
            timings = {name: [] for name, _ in configs}
            for round_id in range(args.rounds):
                rng.shuffle(configs)
                for name, config in configs:
                    start = time.perf_counter()
                    print(f"T={rows} round={round_id} config={name}", flush=True)
                    record = {"type": "sample", "rows": rows, "round": round_id, "name": name, "config": asdict(config)}
                    try:
                        kernels = [GRRead(rows, wd, wu, config) for _, wd, wu in pairs]
                        max_error = 0.0
                        for k, x, ref in zip(kernels, xs, refs):
                            out = k(x)
                            torch.testing.assert_close(out.double(), ref, **TOLERANCES[x.dtype])
                            max_error = max(max_error, (out.double() - ref).abs().max().item())
                        calls = [lambda k=k, x=x: k(x) for k, x in zip(kernels, xs)]
                        graph, n = capture(calls, repeats=4)
                        graph.replay()
                        for k, ref in zip(kernels, refs):
                            torch.testing.assert_close(k.output.double(), ref, **TOLERANCES[k.dtype])
                        result = time_graph(graph, n, samples=args.samples)
                        record.update(result, max_abs_error=max_error, status="passed")
                        timings[name].extend(result["samples_us"])
                        print(f"  {result['median_us']:.3f} us, max error {max_error:.6f}", flush=True)
                        del graph, calls, kernels
                    except (ValueError, AssertionError, RuntimeError) as exc:
                        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        print(record["error"], flush=True)
                        if "HIP error" in str(exc) or "illegal memory" in str(exc):
                            raise
                    record["setup_and_measure_seconds"] = time.perf_counter() - start
                    log.write(json.dumps(record) + "\n")
                    log.flush()
            ordered = sorted((statistics.median(v), name) for name, v in timings.items() if v)
            summary = {"type": "summary", "rows": rows, "ranked": ordered}
            print(json.dumps(summary), flush=True)
            log.write(json.dumps(summary) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
