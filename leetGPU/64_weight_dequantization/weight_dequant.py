import pytest
import pyhip


@pyhip.module("weight_dequant.cpp")
def weight_dequant_hip_vec4(X, S, Y, M, N, TILE_SIZE, s_cols): ...


import torch

torch.cuda.set_device(6)
torch.set_default_device("cuda")
torch.manual_seed(0)
cur_gpu_device = torch.cuda.get_device_name()
print(f"{torch.get_default_device()=}")


def check_all_close(out, out_ref, rtol=0.01, atol=0.01, verbose=False):
    if not torch.allclose(out, out_ref, rtol=0.01, atol=0.01):
        if verbose:
            print(out)
            print(out_ref)
        torch.testing.assert_close(out, out_ref, rtol=0.01, atol=0.01)
        # torch.testing.assert_close(out, out_ref)
    else:
        print("PASS")


import triton
import triton.language as tl

def get_configs():
    return [
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8),
    ]

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'TILE_SIZE'],
)
@triton.jit
def weight_dequant_triton(
    X,
    S,
    Y,
    M,
    N,
    TILE_SIZE,
    s_rows,
    s_cols,
    stride_xr,
    stride_xc,
    stride_yr,
    stride_yc,
    stride_sr,
    stride_sc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    y_r = row * BLOCK_M + tl.arange(0, BLOCK_M)
    y_c = col * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_xy = (y_r[:, None] < M) & (y_c[None, :] < N)


    x_ptrs = X + y_r[:, None] * stride_xr + y_c[None,:]*stride_xc
    y_ptrs = Y + y_r[:, None] * stride_yr + y_c[None,:]*stride_yc

    s_r = y_r // TILE_SIZE
    s_c = y_c // TILE_SIZE


    mask_s = (s_r[:, None] < s_rows) & (s_c[None, :] < s_cols)
    s_ptrs = S + (s_r[:, None] * stride_sr + s_c[None, :] * stride_sc)
    s_val  = tl.load(s_ptrs,mask=mask_s,other=0.0)
    
    x_val = tl.load(x_ptrs,mask=mask_xy,other=0.0)
    #注意这里是元素乘
    y_val = x_val*s_val
    tl.store(y_ptrs,y_val,mask=mask_xy)

    pass


def triton_impl(
    X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int
):
    Y.zero_()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    s_rows = (M + TILE_SIZE - 1) // TILE_SIZE
    s_cols = (N + TILE_SIZE - 1) // TILE_SIZE

    stride_xr, stride_xc = X.stride()
    stride_yr, stride_yc = Y.stride()
    stride_sr, stride_sc = S.stride()
    weight_dequant_triton[grid](
        X,
        S,
        Y,
        M,
        N,
        TILE_SIZE,
        s_rows,
        s_cols,
        stride_xr,
        stride_xc,
        stride_yr,
        stride_yc,
        stride_sr,
        stride_sc,
        # BLOCK_M=16,
        # BLOCK_N=16,
    )
    pass


def reference_impl(
    X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int
):

    Y.zero_()
    # S shape: (ceil(M/TILE_SIZE), ceil(N/TILE_SIZE))
    # We expand S to match X's shape (M, N)

    # Expand rows
    S_expanded = S.repeat_interleave(TILE_SIZE, dim=0)
    # Crop if M is not a multiple of TILE_SIZE
    if S_expanded.shape[0] > M:
        S_expanded = S_expanded[:M, :]

    # Expand cols
    S_expanded = S_expanded.repeat_interleave(TILE_SIZE, dim=1)
    # Crop if N is not a multiple of TILE_SIZE
    if S_expanded.shape[1] > N:
        S_expanded = S_expanded[:, :N]

    # Perform element-wise multiplication
    # Ensure Y is updated in-place
    Y.copy_(X.to(Y.dtype) * S_expanded.to(Y.dtype))


def hip_impl(
    X: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, M: int, N: int, TILE_SIZE: int
):
    Y.zero_()

    threadsPerBlock = [16, 16]
    blocksPerGrid = [
        (N + threadsPerBlock[0] - 1) // threadsPerBlock[0],
        (M + threadsPerBlock[1] - 1) // threadsPerBlock[1],
    ]
    s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
    weight_dequant_hip_vec4(
        blocksPerGrid,
        threadsPerBlock,
        X.data_ptr(),
        S.data_ptr(),
        Y.data_ptr(),
        M,
        N,
        TILE_SIZE,
        s_cols,
        sharedMemBytes=4 * 4,
    )
    pass


def test_weight_dequant():

    dtype = torch.float32
    device = "cuda"
    """
    1 ≤ M, N ≤ 8192
    TILE_SIZE ∈ {16, 32, 64, 128}
    """
    # case 1
    # M, N = 256, 256
    # TILE_SIZE = 128
    # X = torch.randn(M, N, device="cuda", dtype=torch.float32)
    # # S shape
    # s_rows = (M + TILE_SIZE - 1) // TILE_SIZE
    # s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
    # S = torch.randn(s_rows, s_cols, device="cuda", dtype=torch.float32)
    # Y = torch.empty_like(X)

    # case 2 这个case是有问题的只是官方的一种示例 TILE_SIZE ∈ {16, 32, 64, 128}
    # M, N = 4, 4
    # TILE_SIZE = 2
    # X = torch.tensor(
    #     [[10, 10, 5, 5], [10, 10, 5, 5], [2, 2, 8, 8], [2, 2, 8, 8]],
    #     dtype=dtype,
    #     device=device,
    # )
    # S = torch.tensor([[0.5, 2.0], [4.0, 0.25]], dtype=dtype, device=device)

    # s_rows = (M + TILE_SIZE - 1) // TILE_SIZE
    # s_cols = (N + TILE_SIZE - 1) // TILE_SIZE

    # Y = torch.empty_like(X)

    # case3 performance test
    M, N = 8192, 8192
    TILE_SIZE = 128
    X = torch.randn(M, N, device="cuda", dtype=torch.float32)
    s_rows = (M + TILE_SIZE - 1) // TILE_SIZE
    s_cols = (N + TILE_SIZE - 1) // TILE_SIZE
    S = torch.randn(s_rows, s_cols, device="cuda", dtype=torch.float32)
    Y = torch.empty_like(X)

    ref_output = torch.empty(X.shape, device="cuda", dtype=torch.float32)
    triton_output = torch.empty(X.shape, device="cuda", dtype=torch.float32)
    hip_output = torch.empty(X.shape, device="cuda", dtype=torch.float32)
    reference_impl(
        X,
        S,
        ref_output,
        M,
        N,
        TILE_SIZE,
    )
    torch.cuda.synchronize()
    hip_impl(
        X,
        S,
        hip_output,
        M,
        N,
        TILE_SIZE,
    )
    torch.cuda.synchronize()
    triton_impl(
        X,
        S,
        triton_output,
        M,
        N,
        TILE_SIZE,
    )
    torch.cuda.synchronize()

    print("check hip")
    check_all_close(ref_output, hip_output, verbose=True)
    print("check triton")
    check_all_close(ref_output, triton_output, verbose=True)

    # # Performance
    implementations = {
        "Reference (Torch)": lambda: reference_impl(
            X,
            S,
            ref_output,
            M,
            N,
            TILE_SIZE,
        ),
        "Triton": lambda: triton_impl(
            X,
            S,
            hip_output,
            M,
            N,
            TILE_SIZE,
        ),
        "HIP/CUDA": lambda: hip_impl(
            X,
            S,
            hip_output,
            M,
            N,
            TILE_SIZE,
        ),
    }
    warmup_iters = 10
    test_iters = 100

    print(
        f"\n{'Implementation':<20} | {'Avg Time (ms)':<15} | {'Throughput (GB/s)':<15}"
    )
    print("-" * 55)

    # * 4 是因为 float32 数据类型
    # 假设是单通道 (C=1) 和单批次 (N=1)
    total_bytes = (S.numel() + X.numel() + Y.numel()) * 4

    for name, func in implementations.items():
        # 1. Warm up

        for _ in range(warmup_iters):
            func()
        torch.cuda.synchronize()

        # 2. Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(test_iters):
            func()
        end_event.record()

        torch.cuda.synchronize()

        avg_ms = start_event.elapsed_time(end_event) / test_iters

        # 计算带宽 GB/s (Bytes / 1e9 / (ms / 1000)) = Bytes / ms / 1e6
        bandwidth = total_bytes / (avg_ms * 1e6)

        print(f"{name:<20} | {avg_ms:>13.4f} | {bandwidth:>13.2f}")
    """
        gfx942
        triton 3.6.0+git20251a31
        torch  2.9.1+rocm7.1.0.lw.git351ff442
        hip 7.1.0
        Implementation       | Avg Time (ms)   | Throughput (GB/s)
        -------------------------------------------------------
        Reference (Torch)    |        0.7832 |        685.50
        HIP/CUDA (LDS 16X26) |        0.7844 |        684.50
        
        
        1 ≤ M, N ≤ 8192
        TILE_SIZE ∈ {16, 32, 64, 128}
        本题有个比较苛刻的tile size的条件 
        如果以[16,16]做threadperblock LDS相对简单 
        因为每个threadBlock的scale一定是一个值
        LDS 只在threadIdx.x ==0 && threadIdx.y ==0 时把scale准备好就行 （别忘了__syncthreads()）

        反而是在完全不用LDS的情况下[256,1]的性能更高
        Implementation       | Avg Time (ms)   | Throughput (GB/s)
        -------------------------------------------------------
        Reference (Torch)    |        0.7850 |        683.91
        HIP/CUDA   (native)  |        0.6891 |        779.07

        加入triton triton的实现在 case1 性能不如CUDA 但是在大输入case3下
        Implementation       | Avg Time (ms)   | Throughput (GB/s)
        -------------------------------------------------------
        Reference (Torch)    |        0.7863 |        682.78
        Triton               |        0.3187 |       1684.73
        HIP/CUDA             |        0.6891 |        779.06

        # 增加了向量化
        在[16,16]上比[256,1]更好
        Implementation       | Avg Time (ms)   | Throughput (GB/s)
        -------------------------------------------------------
        Reference (Torch)    |        0.7935 |        676.64
        Triton               |        0.3074 |       1746.41
        HIP/CUDA  (vec)      |        0.3693 |       1453.69

    """


if __name__ == "__main__":
    test_weight_dequant()
