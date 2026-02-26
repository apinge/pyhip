"""
Call HIP Gemma norm from Python via pyhip.module() (same pattern as gemm-simple).
No pre-build: run from this directory and pyhip compiles gemma_norm_hip.cpp on the fly.

  cd /root/workspace/pyhip/archive/norm && python test_gemma_norm_hip.py
  pytest test_gemma_norm_hip.py -v
"""
import os
import sys

import pytest
import torch
import pyhip

# Load module: triggers compile of gemma_norm_hip.cpp when needed (like gemm-simple)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
hip = pyhip.module("gemma_norm_hip.cpp")


def _grid_block_smem_rmsnorm(batch_size, hidden_size):
    """Match GemmaRMSNorm: vec_size=8, block = (32, num_warps), smem = num_warps * 4."""
    vec_size = 8
    block_size = min(1024, hidden_size // vec_size)
    num_warps = (block_size + 31) // 32
    grid = [batch_size, 1, 1]
    block = [32, num_warps, 1]
    smem = num_warps * 4  # float
    return grid, block, smem


def _grid_block_smem_fused(batch_size, hidden_size):
    """Match GemmaFusedAddRMSNorm: smem = (padded_warps + hidden_size) * 4."""
    vec_size = 8
    block_size = min(1024, hidden_size // vec_size)
    num_warps = (block_size + 31) // 32
    padded_warps = ((num_warps + 3) // 4) * 4
    smem = (padded_warps + hidden_size) * 4
    grid = [batch_size, 1, 1]
    block = [32, num_warps, 1]
    return grid, block, smem


def gemma_rmsnorm(output, input_tensor, weight, eps=1e-6, enable_pdl=False, stream=None):
    """Gemma-style RMSNorm: output = (input / RMS(input)) * (1 + weight). fp16 or bf16, contiguous."""
    (enable_pdl,)  # unused on HIP
    assert output.is_contiguous() and input_tensor.is_contiguous() and weight.is_contiguous()
    assert output.dtype == input_tensor.dtype == weight.dtype
    assert output.dtype in (torch.float16, torch.bfloat16)
    batch_size, hidden_size = input_tensor.shape
    grid, block, smem = _grid_block_smem_rmsnorm(batch_size, hidden_size)
    eps_f = float(eps)
    if output.dtype == torch.float16:
        hip.gemma_rmsnorm_fp16(
            grid, block,
            output.data_ptr(), input_tensor.data_ptr(), weight.data_ptr(),
            hidden_size, eps_f, sharedMemBytes=smem,
        )
    else:
        hip.gemma_rmsnorm_bf16(
            grid, block,
            output.data_ptr(), input_tensor.data_ptr(), weight.data_ptr(),
            hidden_size, eps_f, sharedMemBytes=smem,
        )
    return output


def gemma_fused_add_rmsnorm(input_tensor, residual, weight, eps=1e-6, enable_pdl=False, stream=None):
    """residual += input; input = (residual / RMS(residual)) * (1 + weight). In-place. fp16 or bf16."""
    (enable_pdl,)  # unused on HIP
    assert input_tensor.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    assert input_tensor.dtype == residual.dtype == weight.dtype
    assert input_tensor.dtype in (torch.float16, torch.bfloat16)
    batch_size, hidden_size = input_tensor.shape
    grid, block, smem = _grid_block_smem_fused(batch_size, hidden_size)
    eps_f = float(eps)
    if input_tensor.dtype == torch.float16:
        hip.gemma_fused_add_rmsnorm_fp16(
            grid, block,
            input_tensor.data_ptr(), residual.data_ptr(), weight.data_ptr(),
            hidden_size, eps_f, sharedMemBytes=smem,
        )
    else:
        hip.gemma_fused_add_rmsnorm_bf16(
            grid, block,
            input_tensor.data_ptr(), residual.data_ptr(), weight.data_ptr(),
            hidden_size, eps_f, sharedMemBytes=smem,
        )


# -----------------------------------------------------------------------------
# Reference (Python) for testing
# -----------------------------------------------------------------------------
def _gemma_rms_norm_ref(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    return x.to(orig_dtype)


def _gemma_fused_add_rms_norm_ref(x, residual, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x + residual
    residual_out = x.clone()
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    return x.to(orig_dtype), residual_out


# -----------------------------------------------------------------------------
# Pytest (parametrized like flashinfer tests/utils/test_norm.py)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_gemma_norm(batch_size, hidden_size, dtype, specify_out, enable_pdl, contiguous):
    eps = 1e-6
    if contiguous:
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda", dtype=dtype)[:, :hidden_size]

    w = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # Ref from same contiguous view (kernel requires contiguous)
    x_in = x.contiguous()
    y_ref = _gemma_rms_norm_ref(x_in, w, eps)

    if specify_out:
        y = torch.empty_like(x_in)
        gemma_rmsnorm(y, x_in, w, eps=eps, enable_pdl=enable_pdl)
    else:
        y = torch.empty_like(x_in)
        gemma_rmsnorm(y, x_in, w, eps=eps, enable_pdl=enable_pdl)

    torch.cuda.synchronize()
    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(y_ref, y, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_gemma_fused_add_rmsnorm(batch_size, hidden_size, dtype, enable_pdl, contiguous):
    eps = 1e-6
    if contiguous:
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda", dtype=dtype)[:, :hidden_size]

    residual = torch.randn_like(x.contiguous())
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_in = x.contiguous()
    x_native, residual_native = _gemma_fused_add_rms_norm_ref(
        x_in.clone(), residual.clone(), weight, eps
    )

    x_fused = x_in.clone()
    residual_fused = residual.clone()
    gemma_fused_add_rmsnorm(x_fused, residual_fused, weight, eps=eps, enable_pdl=enable_pdl)

    torch.cuda.synchronize()
    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(x_fused, x_native, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual_fused, residual_native, rtol=rtol, atol=atol)


# -----------------------------------------------------------------------------
# Run as script (single quick check)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.set_default_device("cuda")
    torch.manual_seed(42)

    # Quick smoke (fp16 + bf16)
    test_gemma_norm(1, 1024, torch.float16, False, False, True)
    test_gemma_fused_add_rmsnorm(1, 1024, torch.float16, False, True)
    test_gemma_norm(1, 1024, torch.bfloat16, False, False, True)
    test_gemma_fused_add_rmsnorm(1, 1024, torch.bfloat16, False, True)
    print("All tests passed.")
