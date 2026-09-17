"""CPU-only host-guard tests; mocked devices do not validate gfx950 GPU code."""

import warnings
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

if __package__:
    from . import kernel, large_down, prefetch_up
    from .combined_host import CombinedPaddedGRRead
    from .experimental_triton import ExtendedTritonGRRead
else:
    import kernel
    import large_down
    import prefetch_up
    from combined_host import CombinedPaddedGRRead
    from experimental_triton import ExtendedTritonGRRead


def weights():
    device = torch.device("cuda:0")
    wd = Mock(shape=(320, 10240), dtype=torch.bfloat16, device=device, is_cuda=True)
    wu = Mock(shape=(10240, 320), dtype=torch.bfloat16, device=device, is_cuda=True)
    wd.is_contiguous.return_value = wu.is_contiguous.return_value = True
    return wd, wu


def fake_baseline():
    compiled = Mock(n_regs=219, n_spills=0)
    compiled.metadata = SimpleNamespace(shared=16384, num_warps=2)
    kernel_mock = Mock()
    kernel_mock.warmup.return_value = compiled
    return SimpleNamespace(
        _GFX942_MIX_CONFIG={"num_warps": 2, "kpack": 2},
        _hc_mix_persistent_kernel=kernel_mock,
        _get_counters=Mock(),
    )


def expect_value_error(call, message):
    try:
        call()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message}")


def main():
    assert not torch.cuda.is_initialized()
    with ExitStack() as stack:
        props = stack.enter_context(patch.object(torch.cuda, "get_device_properties"))
        stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=Mock()))
        stack.enter_context(patch.object(torch.version, "hip", "mock-rocm"))
        stack.enter_context(patch.object(torch, "empty", side_effect=lambda *a, **k: Mock()))
        stack.enter_context(patch.object(torch, "empty_like", side_effect=lambda *a, **k: Mock()))
        stack.enter_context(patch.object(kernel.flyc, "compile", side_effect=lambda *a, **k: Mock()))
        for module in (kernel, prefetch_up):
            stack.enter_context(patch.object(module, "preshuffle_weight", side_effect=lambda value: value))
            stack.enter_context(patch.object(module, "_launchers", return_value=(Mock(), Mock())))
        stack.enter_context(patch.object(large_down, "down_launcher", return_value=Mock()))
        stack.enter_context(patch.object(large_down, "pair_launcher", return_value=Mock()))
        readers = (
            kernel.GRRead,
            prefetch_up.GRRead,
            CombinedPaddedGRRead,
            lambda rows, wd, wu: large_down.LargeDownGRRead(CombinedPaddedGRRead(rows, wd, wu)),
        )
        for arch, cu_count in (("gfx942:sramecc+:xnack-", 80), ("gfx950:sramecc+:xnack-", 256), ("gfx942", 304)):
            props.return_value = SimpleNamespace(gcnArchName=arch, multi_processor_count=cu_count)
            for factory in readers:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reader = factory(17, *weights())
                assert reader.rows == 17
                assert len(caught) == int(not arch.startswith("gfx942"))
                if caught:
                    assert caught[0].category is RuntimeWarning
                    assert arch in str(caught[0].message)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reader = ExtendedTritonGRRead(17, *weights(), fake_baseline())
            assert len(caught) == int(not arch.startswith("gfx942") or cu_count != 80)
            assert reader.resources["grid"] == 80
            assert reader.arguments[-2] == 80
            if caught:
                assert caught[0].category is RuntimeWarning
                assert arch in str(caught[0].message)
                assert str(cu_count) in str(caught[0].message)

        # These remain hard contracts even when the architecture check is advisory.
        props.return_value = SimpleNamespace(gcnArchName="gfx950", multi_processor_count=256)
        for factory in readers:
            wd, wu = weights()
            wd.is_cuda = False
            expect_value_error(lambda: factory(17, wd, wu), "ROCm")
            wd, wu = weights()
            wd.dtype = torch.float32
            expect_value_error(lambda: factory(17, wd, wu), "BF16")
        for module in (kernel, prefetch_up):
            config = module.Config(block_m=32, up_n=256, compensate_hidden=True)
            expect_value_error(config.validate, "LDS capacity")
        baseline = fake_baseline()
        baseline._hc_mix_persistent_kernel.warmup.return_value.metadata.shared = 65537
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            expect_value_error(lambda: ExtendedTritonGRRead(17, *weights(), baseline), "LDS capacity")
    assert not torch.cuda.is_initialized()
    print("PASS: gfx950 warns and proceeds; gfx942 stays quiet; ROCm/BF16/LDS guards remain hard")


if __name__ == "__main__":
    main()
