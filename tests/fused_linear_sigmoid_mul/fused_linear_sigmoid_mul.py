"""y = sigmoid(x @ W^T + b) * m — Qwen2 shared expert gate fuse.

x,m: (N, H) bf16; W: (1, H); b: (1,) or None. Kernel: fused_linear_sigmoid_mul.cpp
"""

import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import pyhip

hip = pyhip.module("fused_linear_sigmoid_mul.cpp")


def fused_linear_sigmoid_mul_ref(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    shared_output: torch.Tensor,
) -> torch.Tensor:
    return torch.sigmoid(F.linear(hidden_states, weight, bias)) * shared_output




@triton.jit
def fused_linear_sigmoid_mul_triton(
    hidden_states_ptr,
    weight_ptr,
    shared_output_ptr,
    output_ptr,
    N,
    H,
    shared_output_stride_row,
    output_stride_row,
    BLOCK_H: tl.constexpr,
):
    """每行: sum_q = sum_h x[n,h]*w[0,h]（分块向量逐元乘再规约）；out = sigmoid(sum_q) * m[n,:]."""
    pid_n = tl.program_id(0)
    row_start_xm = pid_n * shared_output_stride_row
    row_start_out = pid_n * output_stride_row
    row_ok = pid_n < N

    sum_q = 0.0
    for i in range(0, H, BLOCK_H):
        col_offsets = i + tl.arange(0, BLOCK_H)
        col_mask = col_offsets < H
        mask_1d = row_ok & col_mask
        x = tl.load(hidden_states_ptr + row_start_xm + col_offsets, mask=mask_1d, other=0.0)
        w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0)
        prod = x.to(tl.float32) * w.to(tl.float32)
        sum_q = sum_q + tl.sum(prod)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-sum_q))

    for i in range(0, H, BLOCK_H):
        col_offsets = i + tl.arange(0, BLOCK_H)
        col_mask = col_offsets < H
        mask_2d = row_ok & col_mask[None, :]
        y_ptrs = shared_output_ptr + row_start_xm + col_offsets[None, :]
        y = tl.load(y_ptrs, mask=mask_2d, other=0.0)
        out = sigmoid_x * y.to(tl.float32)
        out_ptrs = output_ptr + row_start_out + col_offsets[None, :]
        tl.store(out_ptrs, out.to(output_ptr.dtype.element_ty), mask=mask_2d)


def fused_linear_sigmoid_mul_triton_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    shared_output: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        output = out
    assert hidden_states.dim() == 2 and weight.dim() == 2 and shared_output.dim() == 2
    n, h = shared_output.shape
    assert hidden_states.shape == (n, h)
    assert weight.shape == (1, h)
    if output is None:
        output = torch.empty_like(shared_output)
    block_h = 2048
    grid = (triton.cdiv(n, 1),)
    fused_linear_sigmoid_mul_triton[grid](
        hidden_states,
        weight,
        shared_output,
        output,
        n,
        h,
        shared_output.stride(0),
        output.stride(0),
        BLOCK_H=block_h,
        num_warps=4,
        num_stages=2,
    )
    return output


def triton_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    shared_output: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """整段 fused kernel；bias 未在 Triton 里实现时须为 None。"""
    assert bias is None, "fused triton kernel 当前未加 bias"
    # if out is None:
    #     out = torch.empty_like(shared_output)
    fused_linear_sigmoid_mul_triton_impl(hidden_states, weight, shared_output, out=out)
    return out


WARP_SIZE = 64


def fused_linear_sigmoid_mul_hip(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    shared_output: torch.Tensor,
    output: torch.Tensor,
):
    N, H = hidden_states.shape
    assert bias is None
    threads_per_block = [64, 4, 1]
    blocks_per_grid = [N, 1, 1]
    hip.fused_linear_sigmoid_mul_kernel(
        blocks_per_grid,
        threads_per_block,
        hidden_states.data_ptr(),
        weight.data_ptr(),
        shared_output.data_ptr(),
        output.data_ptr(),
        N,
        H,
        sharedMemBytes=4 * 4,
    )


def main():
    if not torch.cuda.is_available():
        print("skip: CUDA not available")
        return
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    N, H = 8000, 4096
    x = torch.randn(N, H, device=device, dtype=torch.bfloat16)
    W = torch.randn(1, H, device=device, dtype=torch.bfloat16)
    b = None
    m = torch.randn(N, H, device=device, dtype=torch.bfloat16)

    ref = fused_linear_sigmoid_mul_ref(x, W, b, m)
    out = torch.empty_like(m)
    fused_linear_sigmoid_mul_hip(x, W, b, m, out)
    ok = torch.allclose(out, ref, rtol=2e-2, atol=2e-2)
    print("allclose(ref, hip) bias", ok)
    if not ok:
        print("max_abs_diff", (out.float() - ref.float()).abs().max().item())
        exit(1)

    out_triton = torch.empty_like(m)
    triton_impl(x, W, b, m, out=out_triton)
    ok_triton = torch.allclose(out_triton, ref, rtol=2e-2, atol=2e-2)
    print("allclose(ref, triton_impl fused triton)", ok_triton)
    if not ok_triton:
        print(
            "max_abs_diff triton",
            (out_triton.float() - ref.float()).abs().max().item(),
        )
        exit(1)

    rw_bytes = (N * H + H + N * H + N * H) * x.element_size()
    times_ref = []
    for _ in range(10):
        with pyhip.cudaPerf(rw_bytes=rw_bytes, name="fused_linear_sigmoid_mul_ref", verbose=1) as p:
            _ = fused_linear_sigmoid_mul_ref(x, W, b, m)
        times_ref.append(p.dt_ms)
    times_hip = []
    for _ in range(10):
        with pyhip.cudaPerf(rw_bytes=rw_bytes, name="fused_linear_sigmoid_mul_hip", verbose=1) as p:
            fused_linear_sigmoid_mul_hip(x, W, b, m, out)
        times_hip.append(p.dt_ms)
    mean_ref = sum(times_ref) / len(times_ref)
    mean_hip = sum(times_hip) / len(times_hip)
    print(
        f"fused_linear_sigmoid_mul_ref (torch) mean {mean_ref * 1e3:.3f} us  (10 runs)",
    )
    print(
        f"fused_linear_sigmoid_mul_hip (HIP)   mean {mean_hip * 1e3:.3f} us  (10 runs)",
    )
    out_bench = torch.empty_like(m)
    times_triton = []
    for _ in range(10):
        with pyhip.cudaPerf(
            rw_bytes=rw_bytes, name="fused_linear_sigmoid_mul_triton", verbose=1
        ) as p:
            triton_impl(x, W, b, m, out=out_bench)
        times_triton.append(p.dt_ms)
    mean_triton = sum(times_triton) / len(times_triton)
    print(
        f"triton_impl (fused Triton) mean {mean_triton * 1e3:.3f} us  (10 runs)",
    )


if __name__ == "__main__":
    main()
