"""Reproduce a full-checkpoint reference failure without changing its tolerance."""

import argparse
import json
from pathlib import Path

import torch
from benchmark import output_error
from kernel import Config, GRRead
from support import checkpoint_pairs, load_triton_baseline, reference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    pairs = list(checkpoint_pairs())
    for rows in range(1, args.rows):
        for _ in pairs:
            torch.randn(rows, 10240, dtype=torch.bfloat16, device="cuda", generator=generator)
    xs = [torch.randn(args.rows, 10240, dtype=torch.bfloat16, device="cuda", generator=generator) for _ in pairs]
    baseline = load_triton_baseline()
    failures = []
    for i, (x, (name, wd, wu)) in enumerate(zip(xs, pairs)):
        ref = reference(x, wd, wu)
        # Recreate the single-BF16 activation rounding failure, not the fixed default.
        gr = GRRead(args.rows, wd, wu, config=Config(compensate_hidden=False))
        outputs = {"flydsl": gr(x)}
        if args.rows <= 16:
            outputs["tuned_triton"] = baseline.fused_hc_mix(x, wd, wu, 4, 2560)
        for backend, out in outputs.items():
            error = output_error(out, ref)
            if error["max_scaled"] > 1:
                record = {"index": i, "weight": name, "backend": backend, **error}
                print(json.dumps(record), flush=True)
                failures.append(record)
                torch.save(
                    {"x": x.cpu(), "wd": wd.cpu(), "wu": wu.cpu(), "out": out.cpu(), "ref": ref.cpu()},
                    args.output / f"{i}_{backend}.pt",
                )
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2))
    print("Failure cases:", len(failures), flush=True)
