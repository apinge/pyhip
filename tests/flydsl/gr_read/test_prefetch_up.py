from dataclasses import replace
from pathlib import Path

import pytest
import torch

from .prefetch_up import MAX_ROWS, GRRead, _launchers, default_config
from .support import TOLERANCES, reference, synthetic


@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1))
def test_prefetch_up_fp64(rows):
    x, wd, wu = synthetic(rows, seed=103)
    reader = GRRead(rows, wd, wu)
    torch.testing.assert_close(reader(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])


@pytest.mark.parametrize("rows", [1, 17, 24, 25, 31, 32])
def test_prefetch_up_changed_graph_inputs(rows):
    x, wd, wu = synthetic(rows, seed=107)
    reader = GRRead(rows, wd, wu)
    for _ in range(3):
        reader(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = reader(x)
    for _ in range(4):
        x.mul_(0.95).add_(0.03125)
        for _ in range(30):
            graph.replay()
        torch.testing.assert_close(output.double(), reference(x, wd, wu), **TOLERANCES[x.dtype])


def test_prefetch_up_mtp_regression():
    path = Path("/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e08_t10_diagnosis/98_flydsl.pt")
    if not path.exists():
        pytest.skip("recorded checkpoint regression sample unavailable")
    case = torch.load(path, weights_only=True, map_location="cuda")
    reader = GRRead(10, case["wd"], case["wu"])
    torch.testing.assert_close(
        reader(case["x"]).double(), reference(case["x"], case["wd"], case["wu"]), **TOLERANCES[torch.bfloat16]
    )


@pytest.mark.parametrize("rows", [1, 16, 24, 32])
@pytest.mark.parametrize("padding", [4, 8, 16, 32])
def test_hidden_lds_padding(rows, padding):
    x, wd, wu = synthetic(rows, seed=109)
    config = replace(default_config(rows), hidden_pad=padding, prefetch_low=False)
    reader = GRRead(rows, wd, wu, config)
    torch.testing.assert_close(reader(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])


def test_low_load_and_stride_cache_keys():
    configs = [replace(default_config(7), hidden_pad=p, prefetch_low=low) for p in (0, 4, 8) for low in (False, True)]
    x, wd, wu = synthetic(7, seed=113)
    ref = reference(x, wd, wu)
    keys = set()
    for config in configs:
        reader = GRRead(7, wd, wu, config)
        torch.testing.assert_close(reader(x).double(), ref, **TOLERANCES[x.dtype])
        key = _launchers(7, x.dtype, config)[1].manager_key
        assert key is not None
        keys.add(key)
    assert len(keys) == len(configs)
