"""y = sigmoid(x @ W^T + b) * m — Qwen2 shared expert gate fuse.

x,m: (N, H) bf16; W: (1, H); b: (1,) or None. Kernel: fused_linear_sigmoid_mul.cpp
"""

import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F

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


    #x, W, m 读 + out 写（bf16）
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


if __name__ == "__main__":
    main()
