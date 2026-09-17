import pytest
import torch

from .kernel import Config
from .register_gate import RegisterGateGRRead
from .support import TOLERANCES, reference, synthetic


@pytest.mark.parametrize("rows", range(1, 33))
def test_register_gate_fp64(rows):
    x, wd, wu = synthetic(rows, seed=67)
    reader = RegisterGateGRRead(rows, wd, wu)
    torch.testing.assert_close(reader(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])


@pytest.mark.parametrize("rows", [1, 7, 16, 17, 24, 25, 32])
def test_register_gate_stream_mapping(rows):
    x, wd, wu = synthetic(rows, seed=71)
    pattern = torch.arange(rows * 10240, device=x.device).reshape(rows, 4, 2560)
    pattern = (pattern % 97 - 48).float() / 32 + torch.arange(4, device=x.device)[None, :, None] * 0.125
    x.copy_(pattern.reshape_as(x))
    wd.zero_()
    wu.zero_()
    reader = RegisterGateGRRead(rows, wd, wu)
    expected = 0.5 * x.double().reshape(rows, 4, 2560).mean(1)
    torch.testing.assert_close(reader(x).double(), expected, **TOLERANCES[x.dtype])


@pytest.mark.parametrize(
    "config",
    [
        Config(block_m=32, compensate_hidden=True),
        Config(down_n=32, up_n=64, waves=2, compensate_hidden=True),
        Config(up_n=256, compensate_hidden=True),
    ],
)
def test_register_gate_alternate_layouts(config):
    x, wd, wu = synthetic(24, seed=73)
    reader = RegisterGateGRRead(24, wd, wu, config)
    torch.testing.assert_close(reader(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])


@pytest.mark.parametrize("rows", [1, 16, 24, 25, 32])
@pytest.mark.parametrize("ordered", [False, True])
def test_register_gate_changed_graph_inputs(rows, ordered):
    x, wd, wu = synthetic(rows, seed=79)
    reader = RegisterGateGRRead(rows, wd, wu, ordered=ordered)
    for _ in range(3):
        reader(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = reader(x)
    for _ in range(4):
        x.mul_(0.95).add_(0.03125)
        for _ in range(30):
            graph.replay()
        torch.testing.assert_close(actual.double(), reference(x, wd, wu), **TOLERANCES[x.dtype])
