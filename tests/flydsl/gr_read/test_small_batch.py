"""Plain Python correctness, immutable-weight, and graph checks for T1..16."""

import argparse

import torch

if __package__:
    from .combined_host import CombinedPaddedGRRead
    from .small_batch import SmallBatchGRRead, SmallConfig, default_config
    from .support import TOLERANCES, capture, reference, synthetic
else:
    from combined_host import CombinedPaddedGRRead
    from small_batch import SmallBatchGRRead, SmallConfig, default_config
    from support import TOLERANCES, capture, reference, synthetic


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=list(range(1, 17)))
    parser.add_argument("--up-n", nargs="+", type=int, default=[64, 128])
    args = parser.parse_args()
    if any(not 1 <= rows <= 16 for rows in args.rows):
        parser.error("rows must be 1..16")
    for rows in args.rows:
        for seed in (101, 202):
            x, wd, wu = synthetic(rows, seed=seed)
            control = CombinedPaddedGRRead(rows, wd, wu)
            packed_down, packed_up = control.w_down.clone(), control.w_up.clone()
            for up_n in [None] + args.up_n:
                reader = SmallBatchGRRead(control, SmallConfig(up_n=up_n)) if up_n is not None else SmallBatchGRRead(control)
                if up_n is None:
                    assert reader.config.up_n == (128 if rows <= 6 or rows == 16 else 64)
                assert reader.w_down.data_ptr() == control.w_down.data_ptr()
                assert reader.w_up.data_ptr() == control.w_up.data_ptr()
                assert reader.partial.numel() == 4 * 16 * 320
                assert reader.partial.data_ptr() != control.partial.data_ptr()
                assert reader.output.data_ptr() != control.output.data_ptr()
                reader.partial.fill_(float("nan"))
                reader.run_down(x)
                assert torch.isfinite(reader.partial).all()
                assert torch.count_nonzero(reader.partial.view(4, 16, 320)[:, rows:]) == 0
                reader.run_up(x)
                assert torch.allclose(reader.output.double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
                graph, _ = capture([lambda: reader(x)], repeats=2)
                for _ in range(3):
                    x.mul_(0.99).add_(0.015625)
                    reader.partial.fill_(float("nan"))
                    graph.replay()
                    assert torch.allclose(reader.output.double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
                x.zero_()
                graph.replay()
                assert torch.count_nonzero(reader.output) == 0
                assert torch.count_nonzero(reader.partial) == 0
                assert torch.equal(reader.w_down, packed_down)
                assert torch.equal(reader.w_up, packed_up)
                x.normal_()
        print(f"T={rows}: FP64, replay, padding, independent scratch, unchanged packed weights PASS", flush=True)
    for rows in (-1, 0, 17, 32):
        try:
            default_config(rows)
        except ValueError:
            pass
        else:
            raise AssertionError("small candidate must not claim rows outside 1..16")
    for rows in (0, 17, 32):
        x, wd, wu = synthetic(rows, seed=173)
        control = CombinedPaddedGRRead(rows, wd, wu)
        try:
            SmallBatchGRRead(control)
        except ValueError:
            pass
        else:
            raise AssertionError("small candidate must not claim rows outside 1..16")
        assert torch.allclose(control(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
    print("Out-of-range rejection and original control checks PASS", flush=True)


if __name__ == "__main__":
    main()
