"""Final GR read effective bandwidth; synthetic by default, no SGLang required.

Uses cudaPerf's decimal GB/s formula with explicit logical byte accounting.
The denominator is the existing full-call graph timer, not host wall time.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

if __package__:
    from .combined_host import CombinedPaddedGRRead
    from .prefetch_up import HS, MAX_ROWS, K, R
    from .support import (
        TOLERANCES,
        capture,
        checkpoint_pairs,
        reference,
        synthetic,
        time_graph,
    )
else:
    from combined_host import CombinedPaddedGRRead
    from prefetch_up import HS, MAX_ROWS, K, R
    from support import (
        TOLERANCES,
        capture,
        checkpoint_pairs,
        reference,
        synthetic,
        time_graph,
    )


def external_byte_counts(rows):
    """Count each BF16 external input once; exclude workspace and repeated loads."""
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"rows must be in 1..{MAX_ROWS}")
    x = rows * K * 2
    wd = R * K * 2
    wu = K * R * 2
    y = rows * HS * 2
    return {"x": x, "w_down": wd, "w_up": wu, "y": y, "read": x + wd + wu, "io": x + wd + wu + y}


def bandwidth_gbs(byte_count, latency_us):
    """Same units as cudaPerf: bytes * 1e-6 / milliseconds, or bytes / us / 1000."""
    if byte_count < 0 or not math.isfinite(latency_us) or latency_us <= 0:
        raise ValueError("byte count must be nonnegative and latency must be finite and positive")
    return byte_count / latency_us / 1000


def check_outputs(readers, xs, pairs):
    maximum = 0.0
    tol = TOLERANCES[torch.bfloat16]
    for reader, x, (name, wd, wu) in zip(readers, xs, pairs):
        expected = reference(x, wd, wu)
        actual = reader.output.double()
        assert torch.allclose(actual, expected, **tol), f"T={x.shape[0]}, {name}: FP64 mismatch"
        maximum = max(maximum, ((actual - expected).abs() / (tol["atol"] + tol["rtol"] * expected.abs())).max().item())
    return {"passed_pairs": len(pairs), "max_scaled_error": maximum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=list(range(1, MAX_ROWS + 1)))
    parser.add_argument("--weights", type=int, default=100)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--synthetic", action="store_true", help="random BF16 weights (default when --model-path is absent)"
    )
    source.add_argument(
        "--model-path", type=Path, help="use real HC weights from this checkpoint instead of synthetic weights"
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--output", type=Path, help="optionally save raw samples, byte counts and source snapshots")
    args = parser.parse_args()
    if any(not 1 <= rows <= MAX_ROWS for rows in args.rows):
        parser.error(f"--rows must be in 1..{MAX_ROWS}")
    if min(args.weights, args.rounds, args.samples) < 1:
        parser.error("weight, round and sample counts must be positive")
    if args.model_path and not (args.model_path / "model.safetensors.index.json").is_file():
        parser.error("checkpoint index not found; correct --model-path or omit it for synthetic weights")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    weight_source = "checkpoint" if args.model_path else "synthetic"
    with args.output.open("x") if args.output else nullcontext() as log:
        metadata = {
            "type": "environment",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backend": "combined_padded",
            "weight_source": weight_source,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "gpu": torch.cuda.get_device_name(),
            "arch": torch.cuda.get_device_properties(0).gcnArchName,
            "compute_units": torch.cuda.get_device_properties(0).multi_processor_count,
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "flydsl": importlib.metadata.version("flydsl"),
            "timer": "support.time_graph: 2 passes through all weights; 3 replays/sample; graph GPU event time per full GR read",
            "byte_model": "read=X+W_down+W_up once each; io=read+Y; BF16 external tensors",
            "bandwidth_formula": "decimal GB/s = bytes / latency_us / 1000; same as pyhip.cudaPerf",
            "not_counted": "workspace traffic, CTA/wave duplicate loads, cache-line traffic, LDS, packing, reference; not measured HBM bytes",
            "sources": {},
        }
        if log is not None:
            for name in ("bench_bandwidth.py", "combined_host.py", "prefetch_up.py", "kernel.py", "support.py"):
                path = Path(__file__).with_name(name)
                content = path.read_bytes()
                metadata["sources"][name] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
                with args.output.with_name(args.output.stem + "." + name).open("xb") as snapshot:
                    snapshot.write(content)
        if args.model_path:
            pairs = list(checkpoint_pairs(limit=args.weights, model_path=args.model_path))
            if len(pairs) != args.weights:
                parser.error("requested more checkpoint weight pairs than available")
        else:
            pairs = []
            for i in range(args.weights):
                _, wd, wu = synthetic(1, seed=args.seed + i, scale=0.02)
                pairs.append((f"synthetic_{i}", wd, wu))
        metadata["weight_names"] = [name for name, _, _ in pairs]
        if log is not None:
            log.write(json.dumps(metadata) + "\n")
            log.flush()
        print(f"combined_padded, source={weight_source}, weight_pairs={len(pairs)}", flush=True)
        print("Logical effective bandwidth, not measured HBM traffic. Read=X+Wd+Wu; I/O=Read+Y.", flush=True)
        print(
            f"{'T':>3} {'Graph us':>10} {'Read MB':>10} {'Read GB/s':>12} {'I/O GB/s':>12} {'Scratch KiB':>12}",
            flush=True,
        )
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        for rows in args.rows:
            xs = [torch.randn(rows, K, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in pairs]
            readers = [CombinedPaddedGRRead(rows, wd, wu) for _, wd, wu in pairs]
            counts = external_byte_counts(rows)
            assert counts["w_down"] == readers[0].w_down.numel() * readers[0].w_down.element_size()
            assert counts["w_up"] == readers[0].w_up.numel() * readers[0].w_up.element_size()
            assert counts["x"] == xs[0].numel() * xs[0].element_size()
            assert counts["y"] == readers[0].output.numel() * readers[0].output.element_size()
            calls = [lambda r=r, x=x: r(x) for r, x in zip(readers, xs)]
            graph, call_count = capture(calls, repeats=2)
            graph.replay()
            initial = check_outputs(readers, xs, pairs)
            rounds = [
                time_graph(graph, call_count, samples=args.samples, replay_per_sample=3) for _ in range(args.rounds)
            ]
            samples = [value for result in rounds for value in result["samples_us"]]
            latency = statistics.median(samples)
            for x in xs:
                x.mul_(0.99).add_(0.015625)
            for _ in range(20):
                graph.replay()
            changed = check_outputs(readers, xs, pairs)
            scratch = readers[0].partial.numel() * readers[0].partial.element_size()
            record = {
                "type": "result",
                "rows": rows,
                "weight_pairs": len(pairs),
                "weight_source": weight_source,
                "config": asdict(readers[0].config),
                "bytes": counts,
                "scratch_capacity_bytes": scratch,
                "graph_median_us": latency,
                "read_gbs": bandwidth_gbs(counts["read"], latency),
                "io_gbs": bandwidth_gbs(counts["io"], latency),
                "graph_rounds": rounds,
                "initial_fp64": initial,
                "changed_input_fp64": changed,
            }
            if log is not None:
                log.write(json.dumps(record) + "\n")
                log.flush()
            print(
                f"{rows:3d} {latency:10.3f} {counts['read'] / 1e6:10.3f} {record['read_gbs']:12.1f} {record['io_gbs']:12.1f} {scratch / 1024:12.0f}",
                flush=True,
            )
            del graph, calls, readers, xs


if __name__ == "__main__":
    main()
