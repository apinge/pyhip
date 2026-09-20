"""IR/ISA dump or rocprof workload. Instrumented timings are not benchmarks."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--kind", choices=["h64", "baseline"], default="h64")
    parser.add_argument("--up-n", type=int)
    parser.add_argument("--up-k", type=int)
    parser.add_argument("--skip-padding", type=int, choices=[0, 1])
    parser.add_argument("--b-first", type=int, choices=[0, 1], default=0)
    parser.add_argument("--strategy", choices=["plane", "remap", "register"], default="plane")
    parser.add_argument("--preload-weights", type=int, choices=[0, 1])
    parser.add_argument("--weights", type=int, default=1)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=False)
        os.environ["FLYDSL_DUMP_IR"] = "1"
        os.environ["FLYDSL_DUMP_DIR"] = str(args.dump_dir)
        os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

    import torch
    import flydsl.compiler as flyc
    import prefetch_up
    from bench_h64_layout import idle_preflight
    from h64_layout import H64GRRead, default_configs, prepare_weights
    from large_down import pair_launcher
    from support import TOLERANCES, reference, synthetic
    from test_graph_buckets import pack_original_weights, selected_configs

    class BaselineRead:
        # Compile only the actual selected baseline, not preparation controls.
        def __init__(self, x, wd, wu):
            self.pd, self.pu = pack_original_weights(wd, wu)
            down, self.up_config = selected_configs(args.rows)
            self.partial = torch.empty(4 * ((args.rows + 15) // 16 * 16) * 320, device=x.device, dtype=torch.float32)
            self.output = torch.empty((args.rows, 2560), device=x.device, dtype=x.dtype)
            self.stream = torch.cuda.current_stream()
            inputs = (x.view(-1), self.pd, self.pu, self.partial, self.output.view(-1), self.stream)
            self.dispatch = flyc.compile(pair_launcher(args.rows, down, self.up_config), *inputs)
            _, up = prefetch_up._launchers(args.rows, x.dtype, self.up_config)
            self.up = flyc.compile(up, x.view(-1), self.pu, self.partial, self.output.view(-1), self.stream)

        def __call__(self, x):
            self.dispatch(x.view(-1), self.pd, self.pu, self.partial, self.output.view(-1), self.stream)

        def run_up(self, x):
            self.up(x.view(-1), self.pu, self.partial, self.output.view(-1), self.stream)

    preflight = idle_preflight()
    cases = []
    for i in range(args.weights):
        x, wd, wu = synthetic(args.rows, seed=101 + i)
        if args.kind == "h64":
            pd, pu = prepare_weights(wd, wu)
            _, up = default_configs(args.rows)
            overrides = {k: v for k, v in (("up_n", args.up_n), ("block_k", args.up_k),
                         ("skip_padding", None if args.skip_padding is None else bool(args.skip_padding))) if v is not None}
            up = replace(up, **overrides)
            reader = H64GRRead(args.rows, pd, pu, up_config=up, strategy=args.strategy,
                               weight_copy_bits=128, b_first=bool(args.b_first), preload_weights=args.preload_weights)
        else:
            reader = BaselineRead(x, wd, wu)
        reader(x)
        cases.append((reader, x, wd, wu))
    for _ in range(args.passes):
        for reader, x, _, _ in cases:
            reader.run_up(x)
    torch.cuda.synchronize()
    for reader, x, wd, wu in cases:
        assert torch.allclose(reader.output.double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("x") as output:
        json.dump({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "preflight": preflight, "up_config": asdict(reader.up_config),
                   "note": "instrumented workload, not performance evidence", "source_sha256": {
                       n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                       for n in ("h64_layout.py", "profile_h64_layout.py", "prefetch_up.py")}}, output, indent=2)
    print("Profile workload correctness PASS", flush=True)


if __name__ == "__main__":
    main()
