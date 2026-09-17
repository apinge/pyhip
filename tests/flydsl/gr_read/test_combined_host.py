import pytest
import torch

from .combined_host import (
    CombinedHostGRRead,
    CombinedPaddedGRRead,
    CombinedThreeStageGRRead,
)
from .kernel import MAX_ROWS
from .support import TOLERANCES, reference, synthetic


@pytest.mark.parametrize("reader_type", [CombinedHostGRRead, CombinedThreeStageGRRead, CombinedPaddedGRRead])
@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1))
def test_combined_host_fp64(reader_type, rows):
    x, wd, wu = synthetic(rows, seed=83)
    reader = reader_type(rows, wd, wu)
    torch.testing.assert_close(reader(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])


@pytest.mark.parametrize("reader_type", [CombinedHostGRRead, CombinedThreeStageGRRead, CombinedPaddedGRRead])
@pytest.mark.parametrize("rows", [1, 17, 24, 25, 31, 32])
def test_combined_host_changed_graph_inputs(reader_type, rows):
    x, wd, wu = synthetic(rows, seed=89)
    reader = reader_type(rows, wd, wu)
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


@pytest.mark.parametrize("reader_type", [CombinedHostGRRead, CombinedThreeStageGRRead, CombinedPaddedGRRead])
def test_combined_host_contract(reader_type):
    x, wd, wu = synthetic(1)
    assert reader_type(0, wd, wu)(x[:0]).shape == (0, 2560)
    reader = reader_type(1, wd, wu)
    with pytest.raises(ValueError, match="contiguous"):
        reader(torch.empty(1, 20480, dtype=x.dtype, device=x.device)[:, ::2])
