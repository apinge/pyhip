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
    """
    hidden_states: (8000, 4096) bf16
    weight: (1, 4096) bf16
    bias: None
    shared_output: (8000, 4096) bf16
    output: (8000, 4096) bf16
    """
    return torch.sigmoid(F.linear(hidden_states, weight, bias)) * shared_output


# --- Triton: fused_sigmoid_mul_broadcast（与 sglang 同源，便于日后对照）---
# Commit: https://github.com/zejunchen-zejun/sglang/commit/2d1b284120456e69e0c03363219bdd19bb9b2f32
# 源文件: https://github.com/zejunchen-zejun/sglang/blob/2d1b284120456e69e0c03363219bdd19bb9b2f32/python/sglang/srt/layers/elementwise.py
# Diff 中 qwen2_moe 用法: gate_output = shared_expert_gate(hidden_states); fused_sigmoid_mul_broadcast(gate_output, shared_output, out=shared_output)


@triton.jit
def _fused_sigmoid_mul_broadcast_kernel(
    X,
    Y,
    OUT,
    N,
    H,
    stride_y,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """sigmoid(x[n,0]) * y[n,h]，x 为 [N,1] 与 y [N,H] broadcast。"""
    # 上游: https://github.com/zejunchen-zejun/sglang/blob/2d1b284120456e69e0c03363219bdd19bb9b2f32/python/sglang/srt/layers/elementwise.py
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    row_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    row_mask = row_offsets < N
    col_mask = col_offsets < H

    x = tl.load(X + row_offsets, mask=row_mask, other=0.0)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))

    y_ptrs = Y + row_offsets[:, None] * stride_y + col_offsets[None, :]
    mask_2d = row_mask[:, None] & col_mask[None, :]
    y = tl.load(y_ptrs, mask=mask_2d, other=0.0)

    out = sigmoid_x[:, None] * y.to(tl.float32)

    out_ptrs = OUT + row_offsets[:, None] * stride_y + col_offsets[None, :]
    tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=mask_2d)


def fused_sigmoid_mul_broadcast(
    x: torch.Tensor,
    y: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert x.dim() == 2 and y.dim() == 2
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 1
    n, h = y.shape
    if out is None:
        out = torch.empty_like(y)
    block_n = 32
    block_h = 1024
    grid = (triton.cdiv(n, block_n), triton.cdiv(h, block_h))
    _fused_sigmoid_mul_broadcast_kernel[grid](
        x,
        y,
        out,
        n,
        h,
        y.stride(0),
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        num_warps=8,
        num_stages=2,
    )
    return out


def ref_triton(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    shared_output: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """对齐 qwen2_moe: gate_output = linear(hidden); fused_sigmoid_mul_broadcast(gate, m, out=m)."""
    gate_output = F.linear(hidden_states, weight, bias)
    if out is None:
        out = torch.empty_like(shared_output)
    fused_sigmoid_mul_broadcast(gate_output, shared_output, out=out)
    return out


WARP_SIZE = 64

def _grid_block_smem_fused(batch_size, hidden_size):
    vec_size = 8
    block_size = min(1024, hidden_size // vec_size)

    #row_tile = 1
    #padded_warps = 0 # ((num_warps + 3) // 4) * 4
    #smem = (padded_warps + hidden_size*row_tile) * 2
    smem = 4*4;
    grid = [batch_size, 1, 1]
    block = [WARP_SIZE, 1, 1]
    return grid, block, smem
    
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
        sharedMemBytes=4*4,
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
    b = None # 没bias 就是None
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
    ref_triton(x, W, b, m, out=out_triton)
    ok_triton = torch.allclose(out_triton, ref, rtol=2e-2, atol=2e-2)
    print("allclose(ref, ref_triton linear+fused_sigmoid_mul_broadcast)", ok_triton)
    if not ok_triton:
        print(
            "max_abs_diff triton",
            (out_triton.float() - ref.float()).abs().max().item(),
        )
        exit(1)

    # x, W, m 读 + out 写（bf16）；Triton 路径另含 gate [N,1]，此处用同一量级近似
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
            rw_bytes=rw_bytes, name="ref_triton_linear+fused_sigmoid_mul_broadcast", verbose=1
        ) as p:
            ref_triton(x, W, b, m, out=out_bench)
        times_triton.append(p.dt_ms)
    mean_triton = sum(times_triton) / len(times_triton)
    print(
        f"ref_triton (F.linear + fused_sigmoid_mul_broadcast) mean {mean_triton * 1e3:.3f} us  (10 runs)",
    )


if __name__ == "__main__":
    main()
