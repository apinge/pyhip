"""Dump standalone experimental kernels; preparation may compile unused baseline kernels."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from dataclasses import asdict, replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument(
        "--kind", choices=["three", "register", "combined_pair", "combined_three", "padded"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = str(args.output / "flydsl")
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

    import prefetch_up
    import torch
    from combined_host import CombinedHostGRRead, CombinedThreeStageGRRead
    from register_gate import RegisterGateGRRead
    from support import TOLERANCES, reference, synthetic
    from three_stage import ThreeStageGRRead

    x, wd, wu = synthetic(args.rows, seed=97)
    constructors = {
        "three": lambda: ThreeStageGRRead(args.rows, wd, wu, 128, 1),
        "register": lambda: RegisterGateGRRead(args.rows, wd, wu),
        "combined_pair": lambda: CombinedHostGRRead(args.rows, wd, wu),
        "combined_three": lambda: CombinedThreeStageGRRead(args.rows, wd, wu, 128, 1),
        "padded": lambda: prefetch_up.GRRead(
            args.rows, wd, wu, replace(prefetch_up.default_config(args.rows), hidden_pad=4, prefetch_low=False)
        ),
    }
    reader = constructors[args.kind]()
    torch.testing.assert_close(reader(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
    metadata = {
        "rows": args.rows,
        "kind": args.kind,
        "config": asdict(reader.config),
        "flydsl": importlib.metadata.version("flydsl"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "note": "constructors compile a baseline up during preparation; it is replaced before execution",
        "runtime_up_input": "completed FP32 activation" if "three" in args.kind else "reader partial buffer",
        "sources": {},
    }
    for name in (
        "kernel.py",
        "three_stage.py",
        "register_gate.py",
        "combined_host.py",
        "prefetch_up.py",
        "dump_experiment_ir.py",
    ):
        content = Path(__file__).with_name(name).read_bytes()
        with (args.output / name).open("xb") as snapshot:
            snapshot.write(content)
        metadata["sources"][name] = hashlib.sha256(content).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
