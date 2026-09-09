"""Single-weight screening; final results must use rotating checkpoint weights."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from kernel import Config, GRRead
from support import (
    TOLERANCES,
    capture,
    load_triton_baseline,
    reference,
    synthetic,
    time_graph,
    torch_mix,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16, 24])
    parser.add_argument("--split-k", type=int, default=8)
    parser.add_argument("--down-n", type=int, default=32)
    parser.add_argument("--up-n", type=int, default=128)
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--no-skip-padding", action="store_true")
    parser.add_argument("--no-preshuffle", action="store_true")
    parser.add_argument("--strict-math", action="store_true")
    parser.add_argument("--compensate-hidden", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    config = Config(
        split_k=args.split_k,
        block_m=args.block_m,
        down_n=args.down_n,
        up_n=args.up_n,
        waves=args.waves,
        block_k=args.block_k,
        skip_padding=not args.no_skip_padding,
        preshuffle=not args.no_preshuffle,
        fast_math=not args.strict_math,
        compensate_hidden=args.compensate_hidden,
    )
    baseline = load_triton_baseline()
    compiled = torch.compile(torch_mix, dynamic=False)
    results = []
    for rows in args.rows:
        x, wd, wu = synthetic(rows)
        kernel = GRRead(rows, wd, wu, config=config)
        torch.testing.assert_close(kernel(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
        candidates = {"flydsl": lambda: kernel(x)}
        if args.stages:
            candidates["down_only"] = lambda: kernel.down(
                x.view(-1), kernel.w_down, kernel.partial, torch.cuda.current_stream()
            )
            candidates["up_only"] = lambda: kernel.up(
                x.view(-1), kernel.w_up, kernel.partial, kernel.output.view(-1), torch.cuda.current_stream()
            )
        if not args.skip_baselines:
            candidates["torch_compile"] = lambda: compiled(x, wd, wu)
            if rows <= 16:
                candidates["tuned_triton"] = lambda: baseline.fused_hc_mix(x, wd, wu, 4, 2560)
        result = {"rows": rows, "config": asdict(config), "cache": "single synthetic weight pair"}
        for name, call in candidates.items():
            print("Preparing", rows, name, flush=True)
            graph, calls = capture([call], repeats=100)
            result[name] = time_graph(graph, calls)
        print(json.dumps(result), flush=True)
        results.append(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(results, f, indent=2)
