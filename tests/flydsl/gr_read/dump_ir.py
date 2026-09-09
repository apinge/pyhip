"""Dump FlyDSL and the exact tuned Triton baseline for the same gfx942 shape."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = str(args.output / "flydsl")
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

    import torch
    import triton
    from kernel import Config, GRRead, default_config
    from support import BASELINE_PATH, load_triton_baseline, synthetic

    config = Config(**json.loads(args.config)) if args.config else default_config(args.rows)
    x, wd, wu = synthetic(args.rows)
    gr = GRRead(args.rows, wd, wu, config=config)
    gr(x)
    torch.cuda.synchronize()
    metadata = {
        "rows": args.rows,
        "config": vars(config),
        "flydsl": importlib.metadata.version("flydsl"),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "kernel_sha256": hashlib.sha256(Path(__file__).with_name("kernel.py").read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest(),
    }
    (args.output / "kernel.py").write_bytes(Path(__file__).with_name("kernel.py").read_bytes())
    (args.output / "hc_mix_triton.py").write_bytes(BASELINE_PATH.read_bytes())
    if args.rows <= 16:
        baseline = load_triton_baseline()
        padded = max(2, triton.next_power_of_2(args.rows))
        scratch = torch.empty((padded, 320), device="cuda", dtype=torch.float32)
        out = torch.empty((args.rows, 2560), device="cuda", dtype=torch.bfloat16)
        counters = torch.zeros(3, device="cuda", dtype=torch.int32)
        compiled = baseline._hc_mix_persistent_kernel.warmup(
            x,
            wd,
            wu,
            scratch,
            out,
            counters,
            10240,
            320,
            2560,
            args.rows,
            80,
            0.25,
            ROWS=padded,
            HC=4,
            **baseline._GFX942_MIX_CONFIG,
            grid=(80,),
        )
        directory = args.output / "triton"
        directory.mkdir()
        for name, value in compiled.asm.items():
            path = directory / f"hc_mix.{name}"
            if isinstance(value, str):
                path.write_text(value)
            elif isinstance(value, bytes):
                path.write_bytes(value)
        metadata["triton_metadata"] = compiled.metadata._asdict()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(json.dumps(metadata, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
