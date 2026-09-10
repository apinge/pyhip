"""Small rotating-weight workload for rocprof discovery, ATT and separate PMC jobs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import prefetch_up
import torch
from combined_host import CombinedHostGRRead, CombinedThreeStageGRRead
from kernel import Config, GRRead
from register_gate import RegisterGateGRRead
from support import TOLERANCES, capture, checkpoint_pairs, reference
from three_stage import ThreeStageGRRead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument(
        "--kind",
        choices=["v1", "current", "wave", "three", "register", "combined_pair", "combined_three", "padded"],
        default="current",
    )
    parser.add_argument("--weights", type=int, default=100)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rows <= 24 or min(args.weights, args.passes) < 1:
        raise ValueError("rows must be 1..24 and counts positive")
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    if args.metadata.exists():
        raise FileExistsError(args.metadata)
    pairs = list(checkpoint_pairs(limit=args.weights))
    if len(pairs) != args.weights:
        raise ValueError("wrong checkpoint weight count")
    generator = torch.Generator(device="cuda").manual_seed(101)
    xs = [torch.randn(args.rows, 10240, dtype=torch.bfloat16, device="cuda", generator=generator) for _ in pairs]
    constructors = {
        "v1": lambda wd, wu: GRRead(args.rows, wd, wu, Config(compensate_hidden=True)),
        "current": lambda wd, wu: GRRead(args.rows, wd, wu),
        "wave": lambda wd, wu: GRRead(
            args.rows, wd, wu, Config(down_mode="wave_splitk", down_n=16, down_block_k=512, compensate_hidden=True)
        ),
        "three": lambda wd, wu: ThreeStageGRRead(args.rows, wd, wu, 128, 1),
        "register": lambda wd, wu: RegisterGateGRRead(args.rows, wd, wu),
        "combined_pair": lambda wd, wu: CombinedHostGRRead(args.rows, wd, wu),
        "combined_three": lambda wd, wu: CombinedThreeStageGRRead(args.rows, wd, wu, 128, 1),
        "padded": lambda wd, wu: prefetch_up.GRRead(
            args.rows, wd, wu, replace(prefetch_up.default_config(args.rows), hidden_pad=4, prefetch_low=False)
        ),
    }
    readers = [constructors[args.kind](wd, wu) for _, wd, wu in pairs]
    calls = [lambda r=r, x=x: r(x) for r, x in zip(readers, xs)]
    if args.graph:
        graph, _ = capture(calls, repeats=1, warmup=1)
        for _ in range(args.passes):
            graph.replay()
    else:
        for _ in range(args.passes):
            for call in calls:
                call()
    torch.cuda.synchronize()
    # Keep correctness checks after the selected profiling dispatches.
    for index in (0, len(pairs) - 1):
        _, wd, wu = pairs[index]
        torch.testing.assert_close(
            readers[index].output.double(), reference(xs[index], wd, wu), **TOLERANCES[torch.bfloat16]
        )
    sources = {
        name: Path(__file__).with_name(name)
        for name in (
            "kernel.py",
            "three_stage.py",
            "register_gate.py",
            "combined_host.py",
            "prefetch_up.py",
            "support.py",
            "profile_case.py",
        )
    }
    metadata = {
        "args": vars(args) | {"metadata": str(args.metadata)},
        "gpu": torch.cuda.get_device_name(),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "flydsl": importlib.metadata.version("flydsl"),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "debug_info_env": os.environ.get("FLYDSL_DEBUG_ENABLE_DEBUG_INFO"),
        "cache_env": os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE"),
        "config": asdict(readers[0].config),
        "weight_names": [name for name, _, _ in pairs],
        "reference": "seed-101 synthetic BF16 normalized inputs, actual checkpoint weights",
        "note": "instrumented workload; use unprofiled benchmark for performance conclusions",
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
    }
    with args.metadata.open("x") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")
    print(
        "Profile workload completed", args.kind, args.rows, args.weights, "graph" if args.graph else "eager", flush=True
    )


if __name__ == "__main__":
    main()
