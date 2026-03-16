# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Gemma RMSNorm test: aligned with ROCm/aiter PR#2148 (op_tests/test_gemma_rms_norm.py).
# Run from this directory; pyhip compiles gemma_norm.cpp on the fly.
#
#   cd /root/workspace/pyhip/archive/gemma_norm && python test_gemma_norm.py
#   pytest test_gemma_norm.py -v
#
# Optional: compare with aiter (install: cd aiter_20260302 && python3 setup.py develop).
# Gemma uses output = (x/RMS)* (1+weight); aiter rms_norm uses (x/RMS)*weight, so pass weight_aiter=1+weight.
import os
import sys
import argparse
import time

import pytest
import torch
import pyhip

# Optional aiter: same math as Gemma by passing weight_aiter = 1 + weight to aiter.rms_norm / rmsnorm2d_fwd_with_add
try:
    import aiter
    from aiter.ops.rmsnorm import rms_norm as aiter_rms_norm
    from aiter.ops.rmsnorm import rmsnorm2d_fwd_with_add as aiter_rmsnorm2d_fwd_with_add
    AITER_AVAILABLE = True
except ImportError:
    aiter = None
    AITER_AVAILABLE = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
hip = pyhip.module("gemma_norm.cpp")

WARP_SIZE = 64
NUM_ITERS = 101
NUM_WARMUP = 2


def _grid_block_smem_rmsnorm(batch_size, hidden_size):
    vec_size = 8
    block_size = min(1024, hidden_size // vec_size)
    num_warps = (block_size + WARP_SIZE - 1) // WARP_SIZE
    grid = [batch_size, 1, 1]
    block = [WARP_SIZE, num_warps, 1]
    smem = num_warps * 4
    return grid, block, smem


def _grid_block_smem_fused(batch_size, hidden_size):
    vec_size = 8
    block_size = min(1024, hidden_size // vec_size)
    num_warps = (block_size + WARP_SIZE - 1) // WARP_SIZE
    padded_warps = ((num_warps + 3) // 4) * 4
    smem = (padded_warps + hidden_size) * 4
    grid = [batch_size, 1, 1]
    block = [WARP_SIZE, num_warps, 1]
    return grid, block, smem


def gemma_rmsnorm(output, input_tensor, weight, eps=1e-6, stream=None):
    """Dispatch by dtype: call hip.gemma_rmsnorm_fp16 or hip.gemma_rmsnorm_bf16."""
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


def gemma_fused_add_rmsnorm(input_tensor, residual, weight, eps=1e-6, stream=None):
    """Dispatch by dtype: call hip.gemma_fused_add_rmsnorm_fp16 or _bf16."""
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
# Reference (same as aiter PR#2148 / sglang GemmaRMSNorm.forward_native)
# -----------------------------------------------------------------------------
def _gemma_rms_norm_ref_native(x, w, eps=1e-6, residual=None):
    """
    Reference from sglang GemmaRMSNorm.forward_native
    https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/layernorm.py
    """
    orig_dtype = x.dtype
    if residual is not None:
        x = x + residual
        residual_out = x.clone()
    else:
        residual_out = None
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x, residual_out


def _perftest_run(func, *args, **kwargs):
    """Run func with warmup + iters, return (last_result, avg_us)."""
    for _ in range(NUM_WARMUP):
        result = func(*args, **kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(NUM_ITERS):
        result = func(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / NUM_ITERS
    avg_us = avg_ms * 1000.0
    return result, avg_us


def run_torch(input, weight, eps, residual=None):
    output, residual_out = _gemma_rms_norm_ref_native(
        input.clone(),
        weight,
        eps,
        residual=residual.clone() if residual is not None else None,
    )
    return output, residual_out


def run_gemma(input, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = torch.empty_like(input, dtype=input.dtype, device=input.device)
        gemma_rmsnorm(output, input, weight, eps=eps)
    else:
        output = input.clone()
        residual_out = residual.clone()
        gemma_fused_add_rmsnorm(output, residual_out, weight, eps=eps)
    return output, residual_out


def _run_gemma_rmsnorm(dtype, m, n):
    dim = (m, n)
    input_t = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    (a, _), avg_a = _perftest_run(run_torch, input_t, weight, 1e-6)
    (b, _), avg_b = _perftest_run(run_gemma, input_t, weight, 1e-6)
    speedup = avg_a / avg_b if avg_b > 0 else 0.0
    msg = (
        f"[perf] dim: {str(dim):<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, "
        f"gemma avg: {avg_b:<8.2f} us, speedup: {speedup:.2f}x"
    )
    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=msg)


def _run_gemma_fused_add_rmsnorm(dtype, m, n):
    dim = (m, n)
    input_t = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    (a, res_a), avg_a = _perftest_run(run_torch, input_t, weight, 1e-6, res)
    (b, res_b), avg_b = _run_gemma_fused_with_timing(input_t, weight, res, 1e-6)
    speedup = avg_a / avg_b if avg_b > 0 else 0.0
    msg = (
        f"[perf] dim: {str(dim):<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, "
        f"gemma avg: {avg_b:<8.2f} us, speedup: {speedup:.2f}x"
    )
    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=msg)
    torch.testing.assert_close(res_a, res_b, rtol=rtol, atol=atol, msg="gemma res check")


def _run_gemma_fused_with_timing(input_t, weight, res, eps):
    """Run fused add rmsnorm with timing; returns (output, residual_out), avg_us."""
    output = input_t.clone()
    residual_out = res.clone()

    for _ in range(NUM_WARMUP):
        gemma_fused_add_rmsnorm(output, residual_out, weight, eps=eps)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(NUM_ITERS):
        output = input_t.clone()
        residual_out = res.clone()
        gemma_fused_add_rmsnorm(output, residual_out, weight, eps=eps)
    end.record()
    torch.cuda.synchronize()
    avg_us = start.elapsed_time(end) / NUM_ITERS * 1000.0
    return (output, residual_out), avg_us


# Same as aiter PR#2148 test_gemma_rms_norm.py
L_DTYPE_STR = ["fp16", "bf16"]
D_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
L_M = [1, 2, 128, 256, 8000, 8294, 33176]
L_N = [256, 4096]

# Perf test: buffer rotation like test-pa.py
BUF_COPY = 8
# Warmup iters before timed perf (reduces single-shape vs full-sweep variance: GPU freq, cache, driver state)
PERF_WARMUP_ITERS = 20


def _rw_bytes_rmsnorm(m, n, elem_bytes=2):
    """Read input + weight, write output. (B*H + H + B*H) * elem_bytes."""
    return (2 * m * n + n) * elem_bytes


def _rw_bytes_fused(m, n, elem_bytes=2):
    """Read input, residual, weight; write input, residual. (2*B*H*2 + H) * elem_bytes."""
    return (4 * m * n + n) * elem_bytes


def _flops_rmsnorm(m, n):
    """Nominal FLOPs for gemma rmsnorm: ~4 per element (square, rsqrt, 2 muls for scale)."""
    return 4 * m * n


def _flops_fused_add_rmsnorm(m, n):
    """Nominal FLOPs for fused add rmsnorm: add + rmsnorm ~ 5 per element."""
    return 5 * m * n


def run_aiter_rmsnorm(input, weight, eps):
    """Gemma-style via aiter: pass weight_ck = 1 + weight to standard rms_norm."""
    weight_ck = (1.0 + weight).to(weight.dtype)
    return aiter_rms_norm(input, weight_ck, eps), None


def run_aiter_fused_add_rmsnorm(input, residual, weight, eps):
    """Gemma-style via aiter: pass weight_ck = 1 + weight to rmsnorm2d_fwd_with_add."""
    weight_ck = (1.0 + weight).to(weight.dtype)
    out = torch.empty_like(input)
    res_out = torch.empty_like(residual)
    aiter_rmsnorm2d_fwd_with_add(out, input, residual, res_out, weight_ck, eps)
    return out, res_out


def run_perf_rmsnorm(dtype, m, n, num_iters=10):
    """Perf test with pyhip.cudaPerf; print torch / gemma / aiter (if available) and speedups."""
    dim = (m, n)
    elem_bytes = 2  # fp16/bf16
    rw_bytes = _rw_bytes_rmsnorm(m, n, elem_bytes)
    inputs = [torch.randn(dim, dtype=dtype, device="cuda") for _ in range(BUF_COPY)]
    weights = [torch.randn(n, dtype=dtype, device="cuda") for _ in range(BUF_COPY)]
    outputs = [torch.empty(dim, dtype=dtype, device="cuda") for _ in range(BUF_COPY)]
    # Warmup so single-shape run matches full-sweep (GPU freq/cache steady)
    for _ in range(PERF_WARMUP_ITERS):
        run_torch(inputs[0], weights[0], 1e-6)
        gemma_rmsnorm(outputs[0], inputs[0], weights[0], eps=1e-6)
        if AITER_AVAILABLE:
            run_aiter_rmsnorm(inputs[0], weights[0], 1e-6)
    torch.cuda.synchronize()
    i = 0
    latencies_torch = []
    for _ in range(num_iters):
        with pyhip.cudaPerf(0, rw_bytes, name="torch", verbose=0) as p:
            run_torch(inputs[i], weights[i], 1e-6)
        latencies_torch.append(p.dt_ms * 1e3)  # us
        i = (i + 1) % BUF_COPY
    i = 0
    latencies_gemma = []
    for _ in range(num_iters):
        with pyhip.cudaPerf(0, rw_bytes, name="gemma", verbose=0) as p:
            gemma_rmsnorm(outputs[i], inputs[i], weights[i], eps=1e-6)
        latencies_gemma.append(p.dt_ms * 1e3)  # us
        i = (i + 1) % BUF_COPY
    avg_torch_us = sum(latencies_torch) / num_iters
    avg_gemma_us = sum(latencies_gemma) / num_iters
    speedup_gemma = avg_torch_us / avg_gemma_us if avg_gemma_us > 0 else 0.0
    flops = _flops_rmsnorm(m, n)
    # TFLOPS = FLOPs / (time_sec * 1e12), time_sec = avg_gemma_us * 1e-6
    time_sec = avg_gemma_us * 1e-6
    gemma_tflops = (flops / (time_sec * 1e12)) if avg_gemma_us > 0 else 0.0
    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    line = (
        f"[perf rmsnorm] dim={dim} dtype={dtype_str}: "
        f"torch {avg_torch_us:.2f} us, gemma(ours) {avg_gemma_us:.2f} us {gemma_tflops:.2f} TFLOPS"
    )
    if AITER_AVAILABLE:
        i = 0
        latencies_aiter = []
        for _ in range(num_iters):
            with pyhip.cudaPerf(0, rw_bytes, name="aiter", verbose=0) as p:
                run_aiter_rmsnorm(inputs[i], weights[i], 1e-6)
            latencies_aiter.append(p.dt_ms * 1e3)
            i = (i + 1) % BUF_COPY
        avg_aiter_us = sum(latencies_aiter) / num_iters
        speedup_aiter = avg_torch_us / avg_aiter_us if avg_aiter_us > 0 else 0.0
        gemma_vs_aiter = avg_aiter_us / avg_gemma_us if avg_gemma_us > 0 else 0.0
        line += f", aiter {avg_aiter_us:.2f} us, gemma/aiter {gemma_vs_aiter:.2f}x"
    print(line)


def run_perf_fused_add_rmsnorm(dtype, m, n, num_iters=10):
    """Perf test fused add rmsnorm with pyhip.cudaPerf; print torch / gemma / aiter (if available) and speedups."""
    dim = (m, n)
    elem_bytes = 2
    rw_bytes = _rw_bytes_fused(m, n, elem_bytes)
    inputs = [torch.randn(dim, dtype=dtype, device="cuda") for _ in range(BUF_COPY)]
    residuals = [torch.randn(dim, dtype=dtype, device="cuda") for _ in range(BUF_COPY)]
    weights = [torch.randn(n, dtype=dtype, device="cuda") for _ in range(BUF_COPY)]
    # Warmup so single-shape run matches full-sweep (GPU freq/cache steady)
    for _ in range(PERF_WARMUP_ITERS):
        run_torch(inputs[0], weights[0], 1e-6, residual=residuals[0])
        out = inputs[0].clone()
        res_out = residuals[0].clone()
        gemma_fused_add_rmsnorm(out, res_out, weights[0], eps=1e-6)
        if AITER_AVAILABLE:
            run_aiter_fused_add_rmsnorm(inputs[0].clone(), residuals[0].clone(), weights[0], 1e-6)
    torch.cuda.synchronize()
    i = 0
    latencies_torch = []
    for _ in range(num_iters):
        with pyhip.cudaPerf(0, rw_bytes, name="torch", verbose=0) as p:
            run_torch(inputs[i], weights[i], 1e-6, residual=residuals[i])
        latencies_torch.append(p.dt_ms * 1e3)
        i = (i + 1) % BUF_COPY
    i = 0
    latencies_gemma = []
    for _ in range(num_iters):
        with pyhip.cudaPerf(0, rw_bytes, name="gemma", verbose=0) as p:
            out = inputs[i].clone()
            res_out = residuals[i].clone()
            gemma_fused_add_rmsnorm(out, res_out, weights[i], eps=1e-6)
        latencies_gemma.append(p.dt_ms * 1e3)
        i = (i + 1) % BUF_COPY
    avg_torch_us = sum(latencies_torch) / num_iters
    avg_gemma_us = sum(latencies_gemma) / num_iters
    speedup_gemma = avg_torch_us / avg_gemma_us if avg_gemma_us > 0 else 0.0
    flops = _flops_fused_add_rmsnorm(m, n)
    # TFLOPS = FLOPs / (time_sec * 1e12), time_sec = avg_gemma_us * 1e-6
    time_sec = avg_gemma_us * 1e-6
    gemma_tflops = (flops / (time_sec * 1e12)) if avg_gemma_us > 0 else 0.0
    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    line = (
        f"[perf fused_add_rmsnorm] dim={dim} dtype={dtype_str}: "
        f"torch {avg_torch_us:.2f} us, gemma(ours) {avg_gemma_us:.2f} us {gemma_tflops:.2f} TFLOPS"
    )
    if AITER_AVAILABLE:
        i = 0
        latencies_aiter = []
        for _ in range(num_iters):
            with pyhip.cudaPerf(0, rw_bytes, name="aiter", verbose=0) as p:
                run_aiter_fused_add_rmsnorm(inputs[i].clone(), residuals[i].clone(), weights[i], 1e-6)
            latencies_aiter.append(p.dt_ms * 1e3)
            i = (i + 1) % BUF_COPY
        avg_aiter_us = sum(latencies_aiter) / num_iters
        speedup_aiter = avg_torch_us / avg_aiter_us if avg_aiter_us > 0 else 0.0
        gemma_vs_aiter = avg_aiter_us / avg_gemma_us if avg_gemma_us > 0 else 0.0
        line += f", aiter {avg_aiter_us:.2f} us, gemma/aiter {gemma_vs_aiter:.2f}x"
    print(line)


@pytest.mark.parametrize("dtype", [D_DTYPES[k] for k in L_DTYPE_STR])
@pytest.mark.parametrize("m", L_M)
@pytest.mark.parametrize("n", L_N)
def test_gemma_rmsnorm_pytest(dtype, m, n):
    _run_gemma_rmsnorm(dtype, m, n)


@pytest.mark.parametrize("dtype", [D_DTYPES[k] for k in L_DTYPE_STR])
@pytest.mark.parametrize("m", L_M)
@pytest.mark.parametrize("n", L_N)
def test_gemma_fused_add_rmsnorm_pytest(dtype, m, n):
    _run_gemma_fused_add_rmsnorm(dtype, m, n)


if __name__ == "__main__":
    l_dtype = L_DTYPE_STR
    l_m = L_M
    l_n = L_N
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Gemma RMSNorm test (fp16/bf16 only)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=l_dtype,
        nargs="?",
        const=None,
        default=None,
        help="Data type: fp16 or bf16",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="?",
        default=None,
        help="Token number (M)",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        nargs="?",
        default=None,
        help="Hidden size (N)",
    )
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype_list = [D_DTYPES[key] for key in l_dtype]
    else:
        l_dtype_list = [D_DTYPES[args.dtype]]
    if args.m is not None:
        l_m = [args.m]
    if args.n is not None:
        l_n = [args.n]

    print("\nstart gemma rmsnorm test(no residual) ---")
    for dtype in l_dtype_list:
        for m in l_m:
            for n in l_n:
                _run_gemma_rmsnorm(dtype, m, n)

    torch.cuda.synchronize()
    print("\nstart gemma rmsnorm fuse add test")
    for dtype in l_dtype_list:
        for m in l_m:
            for n in l_n:
                _run_gemma_fused_add_rmsnorm(dtype, m, n)

    print("All tests passed.")

    # Perf test: torch vs gemma(ours) vs aiter (if installed). Gemma = aiter with weight_ck=1+weight.
    print("\n--- perf (torch / gemma(ours) / aiter) ---")
    if not AITER_AVAILABLE:
        print("(install aiter for 3-way compare: cd aiter_20260302 && python3 setup.py develop)")
    for dtype in l_dtype_list:
        for m in l_m:
            for n in l_n:
                run_perf_rmsnorm(dtype, m, n)
    torch.cuda.synchronize()
    for dtype in l_dtype_list:
        for m in l_m:
            for n in l_n:
                run_perf_fused_add_rmsnorm(dtype, m, n)
    print("perf done.")
