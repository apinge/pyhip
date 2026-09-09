from dataclasses import replace

import pytest
import torch

from .kernel import Config, GRRead, _launchers
from .support import TOLERANCES, reference, synthetic, torch_mix


@pytest.mark.parametrize("rows", range(1, 25))
def test_matches_fp64(rows):
    dtype = torch.bfloat16
    x, wd, wu = synthetic(rows, dtype)
    kernel = GRRead(rows, wd, wu)
    actual = kernel(x)
    torch.testing.assert_close(actual.double(), reference(x, wd, wu), **TOLERANCES[dtype])
    down = kernel.partial.reshape(kernel.config.split_k, -1, 320).sum(0)[:rows]
    expected_down = x.double() @ wd.double().T
    torch.testing.assert_close(down.double(), expected_down, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("rows", [1, 4, 7, 16, 17, 24])
def test_graph_observes_changed_inputs(rows):
    x, wd, wu = synthetic(rows, seed=7)
    kernel = GRRead(rows, wd, wu)
    for _ in range(3):
        kernel(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = kernel(x)
    for i in range(6):
        x.mul_(0.97).add_(0.03125 * (i + 1))
        graph.replay()
        torch.testing.assert_close(actual.double(), reference(x, wd, wu), **TOLERANCES[x.dtype])


@pytest.mark.parametrize("rows", [1, 24])
def test_zero_weights_give_stream_mean(rows):
    x, wd, wu = synthetic(rows, seed=11)
    wd.zero_()
    wu.zero_()
    kernel = GRRead(rows, wd, wu)
    expected = 0.5 * x.double().reshape(rows, 4, 2560).mean(1)
    torch.testing.assert_close(kernel(x).double(), expected, **TOLERANCES[x.dtype])


def test_range_and_layout_contract():
    x, wd, wu = synthetic(1)
    for rows in (-1, 25, 32):
        with pytest.raises(ValueError, match="0..24"):
            GRRead(rows, wd, wu)
    with pytest.raises(ValueError, match="BF16"):
        GRRead(1, wd.half(), wu.half())
    empty = GRRead(0, wd, wu)
    assert empty(x[:0]).shape == (0, 2560)
    kernel = GRRead(1, wd, wu)
    noncontiguous = torch.empty((1, 20480), dtype=x.dtype, device=x.device)[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        kernel(noncontiguous)


@pytest.mark.parametrize("config", [Config(split_k=4), Config(split_k=16, block_k=16)])
def test_alternate_split_k(config):
    x, wd, wu = synthetic(7, seed=13)
    kernel = GRRead(7, wd, wu, config=config)
    torch.testing.assert_close(kernel(x).double(), reference(x, wd, wu), **TOLERANCES[x.dtype])


@pytest.mark.parametrize("field", ["fast_math", "preshuffle", "skip_padding", "compensate_hidden"])
def test_compile_cache_distinguishes_boolean_options(field):
    x, wd, wu = synthetic(1, seed=29)
    first = Config(compensate_hidden=True)
    second = replace(first, **{field: False})
    a = GRRead(1, wd, wu, first)
    b = GRRead(1, wd, wu, second)
    ref = reference(x, wd, wu)
    torch.testing.assert_close(a(x).double(), ref, **TOLERANCES[x.dtype])
    torch.testing.assert_close(b(x).double(), ref, **TOLERANCES[x.dtype])
    assert _launchers(1, x.dtype, first)[1].manager_key != _launchers(1, x.dtype, second)[1].manager_key


def test_accuracy_relative_to_bf16_eager():
    x, wd, wu = synthetic(8)
    ref = reference(x, wd, wu)
    fused = GRRead(8, wd, wu)(x).double()
    eager = torch_mix(x, wd, wu).double()
    assert (fused - ref).abs().max() <= (eager - ref).abs().max() * 1.5 + 1e-6


def test_checkpoint_mtp_activation_rounding_regression():
    from pathlib import Path

    path = Path("/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e08_t10_diagnosis/98_flydsl.pt")
    if not path.exists():
        pytest.skip("run diagnose.py to obtain the recorded checkpoint regression case")
    case = torch.load(path, weights_only=True, map_location="cuda")
    ref = reference(case["x"], case["wd"], case["wu"])
    accurate = GRRead(10, case["wd"], case["wu"])(case["x"])
    torch.testing.assert_close(accurate.double(), ref, **TOLERANCES[torch.bfloat16])
    legacy = GRRead(10, case["wd"], case["wu"], Config(compensate_hidden=False))(case["x"])
    tol = TOLERANCES[torch.bfloat16]
    assert ((legacy.double() - ref).abs() / (tol["atol"] + tol["rtol"] * ref.abs())).max() > 1
