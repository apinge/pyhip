"""Plain Python assertions for large_down; no pytest or SGLang required."""

import argparse

import torch

if __package__:
    from .combined_host import CombinedPaddedGRRead
    from .large_down import DownConfig, LargeDownGRRead
    from .support import TOLERANCES, capture, reference, synthetic
else:
    from combined_host import CombinedPaddedGRRead
    from large_down import DownConfig, LargeDownGRRead
    from support import TOLERANCES, capture, reference, synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[17, 24, 25, 31, 32])
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--global-split", type=int, default=1)
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--prefetch-unroll", type=int, default=1)
    parser.add_argument("--selected", action="store_true")
    args = parser.parse_args()
    config = DownConfig(args.block_m, args.block_k, args.waves, not args.no_prefetch, True, args.global_split, args.prefetch_unroll)
    config.validate()
    for rows in (1, 8, 16):
        x, wd, wu = synthetic(rows, seed=101)
        control = CombinedPaddedGRRead(rows, wd, wu)
        assert torch.allclose(control(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
        try:
            LargeDownGRRead(control, config)
        except ValueError:
            pass
        else:
            raise AssertionError("small-T path must be rejected")
    for rows in args.rows:
        for seed in (101, 202):
            x, wd, wu = synthetic(rows, seed=seed)
            control = CombinedPaddedGRRead(rows, wd, wu)
            reader = LargeDownGRRead(control) if args.selected else LargeDownGRRead(control, config)
            if args.selected:
                assert reader.config.prefetch_unroll == (2 if rows <= 25 else 1)
            assert reader.w_down.data_ptr() == control.w_down.data_ptr()
            assert reader.w_up.data_ptr() == control.w_up.data_ptr()
            assert reader.partial.data_ptr() != control.partial.data_ptr()
            assert reader.output.data_ptr() != control.output.data_ptr()
            reader.partial.fill_(float("nan"))
            reader.run_down(x)
            assert torch.isfinite(reader.partial).all()
            assert torch.count_nonzero(reader.partial.view(-1, 32, 320)[:, rows:]) == 0
            reader.run_up(x)
            expected = reference(x, wd, wu)
            assert torch.allclose(reader.output.double(), expected, **TOLERANCES[x.dtype])
            graph, _ = capture([lambda: reader(x)], repeats=2)
            graph.replay()
            assert torch.allclose(reader.output.double(), expected, **TOLERANCES[x.dtype])
            x.mul_(0.99).add_(0.015625)
            reader.partial.fill_(float("nan"))
            for _ in range(5):
                graph.replay()
            assert torch.allclose(reader.output.double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
            x.zero_()
            graph.replay()
            assert torch.count_nonzero(reader.output) == 0
            assert torch.count_nonzero(reader.partial) == 0
        print(f"T={rows}, {reader.config.name}: FP64 / stages / changed replay / zero / padding / shared weights PASS", flush=True)
    print("T=1/8/16 unchanged-control checks and candidate rejection PASS", flush=True)


if __name__ == "__main__":
    main()
