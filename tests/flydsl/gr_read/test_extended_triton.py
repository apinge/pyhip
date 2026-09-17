"""Plain Python checks for the opt-in Triton ROWS=32 experiment."""

import argparse

import torch

if __package__:
    from .experimental_triton import ExtendedTritonGRRead
    from .support import TOLERANCES, load_triton_baseline, reference, synthetic
else:
    from experimental_triton import ExtendedTritonGRRead
    from support import TOLERANCES, load_triton_baseline, reference, synthetic


def check_case(rows):
    baseline = load_triton_baseline()
    assert baseline._FUSED_MIX_MAX_ROWS == 16
    x, wd, wu = synthetic(rows, seed=149)
    reader = ExtendedTritonGRRead(rows, wd, wu, baseline)
    for _ in range(3):
        output = reader(x)
    assert torch.allclose(output.double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
    assert torch.all(reader.counters == 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = reader(x)
    for _ in range(3):
        x.mul_(0.9).add_(0.03125)
        for _ in range(20):
            graph.replay()
        assert torch.allclose(output.double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
        assert torch.all(reader.counters == 0)
        assert torch.all(reader.partial[rows:] == 0)
    print(f"T={rows}: extended Triton FP64, changed-input replay and counters passed; {reader.resources}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[17, 24, 25, 31, 32])
    args = parser.parse_args()
    if any(not 17 <= rows <= 32 for rows in args.rows):
        parser.error("--rows must be in 17..32")
    for rows in args.rows:
        check_case(rows)
