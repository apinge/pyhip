from pathlib import Path

import pytest
import torch

from .support import TOLERANCES, reference, synthetic
from .three_stage import ThreeStageGRRead, _reduce_launcher


@pytest.mark.parametrize("rows", range(1, 25))
def test_three_stage_fp64(rows):
    x, wd, wu = synthetic(rows, seed=53)
    reader = ThreeStageGRRead(rows, wd, wu)
    torch.testing.assert_close(reader(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
    activation = torch.nn.functional.silu(x.double() @ wd.double().T / 4)
    torch.testing.assert_close(reader.activation.reshape(-1, 320)[:rows].double(), activation, rtol=2e-5, atol=2e-5)
    assert torch.count_nonzero(reader.activation.reshape(-1, 320)[rows:]).item() == 0


@pytest.mark.parametrize("threads", [64, 128, 256])
@pytest.mark.parametrize("vec", [1, 2, 4])
def test_reducer_configurations(threads, vec):
    x, wd, wu = synthetic(7, seed=59)
    reader = ThreeStageGRRead(7, wd, wu, reduce_threads=threads, reduce_vec=vec)
    torch.testing.assert_close(reader(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])


@pytest.mark.parametrize("rows", [1, 17, 24])
def test_three_stage_changed_graph_inputs(rows):
    x, wd, wu = synthetic(rows, seed=61)
    reader = ThreeStageGRRead(rows, wd, wu)
    for _ in range(3):
        reader(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = reader(x)
    for _ in range(5):
        x.mul_(0.95).add_(0.03125)
        reader.activation.fill_(float("nan"))
        for _ in range(30):
            graph.replay()
        torch.testing.assert_close(actual.double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
        assert torch.count_nonzero(reader.activation.reshape(-1, 320)[rows:]).item() == 0


def test_reducer_cache_keys():
    keys = {_reduce_launcher(7, 16, 16, threads, vec).manager_key for threads in (64, 128) for vec in (1, 2, 4)}
    assert len(keys) == 6


def test_three_stage_contract():
    x, wd, wu = synthetic(1)
    assert ThreeStageGRRead(0, wd, wu)(x[:0]).shape == (0, 2560)
    with pytest.raises(ValueError, match="0..24"):
        ThreeStageGRRead(25, wd, wu)
    with pytest.raises(ValueError, match="reducer"):
        ThreeStageGRRead(1, wd, wu, reduce_threads=32)
    with pytest.raises(ValueError, match="BF16"):
        ThreeStageGRRead(1, wd.half(), wu.half())


def test_three_stage_mtp_regression():
    path = Path("/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e08_t10_diagnosis/98_flydsl.pt")
    if not path.exists():
        pytest.skip("recorded checkpoint regression sample unavailable")
    case = torch.load(path, weights_only=True, map_location="cuda")
    reader = ThreeStageGRRead(10, case["wd"], case["wu"])
    torch.testing.assert_close(
        reader(case["x"]).double(), reference(case["x"], case["wd"], case["wu"]), **TOLERANCES[torch.bfloat16]
    )
