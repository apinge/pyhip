import pytest
import pyhip

import triton
import triton.language as tl


@pyhip.module("max_pooling.cpp")
def max_pooling(
    input, output, N, C, H, W, kernel_size, stride, padding, H_out, W_out
): ...


def get_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    # 当 H_out 或 W_out 改变时，Autotune 会重新寻找最优 BLOCK_SIZE
    key=["H_out", "W_out"],
)
@triton.jit
def max_pooling_2d(
    input_ptr,
    output_ptr,
    N,
    C,
    H,
    W,
    kernel_size,
    stride,
    padding,
    H_out,
    W_out,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    #    KERNEL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = tl.program_id(1)
    channel_idx = tl.program_id(2)

    # output_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # out_h = output_offsets // W_out
    # out_w = output_offsets % W_out

    offsets = tl.arange(0, BLOCK_SIZE)
    out_h = (pid * BLOCK_SIZE + offsets) // W_out
    out_w = (pid * BLOCK_SIZE + offsets) % W_out
    output_offsets = out_h * W_out + out_w

    max_val = tl.full([BLOCK_SIZE], value=-float("inf"), dtype=tl.float32)

    h_start = out_h * stride - padding
    w_start = out_w * stride - padding

    out_idx = pid * BLOCK_SIZE + offsets
    out_idx_mask = out_idx < (H_out * W_out)

    for i in range(kernel_size):
        for j in range(kernel_size):
            curr_h = h_start + i
            curr_w = w_start + j
            mask = (
                out_idx_mask
                & (curr_h >= 0)
                & (curr_h < H)
                & (curr_w >= 0)
                & (curr_w < W)
            )
            input_offsets = (
                (batch_idx * C * H * W) + (channel_idx * H * W) + (curr_h * W + curr_w)
            )
            # other usage
            curr_val = tl.load(
                input_ptr + input_offsets, mask=mask, other=-float("inf")
            )
            max_val = tl.maximum(max_val, curr_val)

    output_base = batch_idx * C * H_out * W_out + channel_idx * H_out * W_out

    tl.store(output_ptr + output_base + output_offsets, max_val, mask=out_idx_mask)

    pass


import torch

torch.cuda.set_device(6)
torch.set_default_device("cuda")
torch.manual_seed(0)
cur_gpu_device = torch.cuda.get_device_name()
print(f"{torch.get_default_device()=}")


def check_all_close(out, out_ref, rtol=0.01, atol=0.01, verbose=False):
    if not torch.allclose(out, out_ref, rtol=0.01, atol=0.01):
        if verbose:
            # torch.set_printoptions(threshold=float('inf'))
            print(out)
            print(out_ref)
        torch.testing.assert_close(out, out_ref, rtol=0.01, atol=0.01)
        # torch.testing.assert_close(out, out_ref)
    else:
        print("PASS")


def reference_impl(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    H: int,
    W: int,
    kernel_size: int,
    stride: int,
    padding: int,
):
    output.zero_()
    input_tensor = input.view(N, C, H, W)

    # Apply max pooling
    result = torch.nn.functional.max_pool2d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    output.copy_(result.flatten())


def triton_impl(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    N: int,
    C: int,
    H: int,
    W: int,
    kernel_size: int,
    stride: int,
    padding: int,
):
    output_tensor.zero_()
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    grid = lambda META: (
        triton.cdiv(H_out * W_out, META["BLOCK_SIZE"]),
        N,
        C,
    )
    max_pooling_2d[grid](
        input_tensor,
        output_tensor,
        N,
        C,
        H,
        W,
        kernel_size,
        stride,
        padding,
        H_out,
        W_out,
        #        BLOCK_SIZE=64,
        #        KERNEL_SIZE=4,
    )
    pass


def hip_impl(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    N: int,
    C: int,
    H: int,
    W: int,
    kernel_size: int,
    stride: int,
    padding: int,
):
    output_tensor.zero_()
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    threadsPerBlock = [4, 4, 64]  # not larger than 1024
    blocksPerGrid = [
        (N + threadsPerBlock[0] - 1) // threadsPerBlock[0],
        (C + threadsPerBlock[1] - 1) // threadsPerBlock[1],
        (H_out * W_out + threadsPerBlock[2] - 1) // threadsPerBlock[2],
    ]
    # print(threadsPerBlock)
    # print(blocksPerGrid)
    max_pooling(
        blocksPerGrid,
        threadsPerBlock,
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        N,
        C,
        H,
        W,
        kernel_size,
        stride,
        padding,
        H_out,
        W_out,
    )
    pass


def test_max_pooling():

    dtype = torch.float32

    # N, C, H, W = 1, 1, 5, 5
    # kernel_size, stride, padding = 2, 1, 1
    # H_out = (H + 2 * padding - kernel_size) // stride + 1
    # W_out = (W + 2 * padding - kernel_size) // stride + 1

    # # Create input with extreme values
    # input_tensor = torch.tensor(
    #     [
    #         [
    #             [
    #                 [1e6, -1e6, 0.0, 1e-6, -1e-6],
    #                 [float("inf"), float("-inf"), 1.0, 2.0, 3.0],
    #                 [4.0, 5.0, 6.0, 7.0, 8.0],
    #                 [9.0, 10.0, 11.0, 12.0, 13.0],
    #                 [14.0, 15.0, 16.0, 17.0, 18.0],
    #             ]
    #         ]
    #     ],
    #     device="cuda",
    #     dtype=dtype,
    # )
    # output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)

    N, C, H, W = 4, 64, 256, 256  # 4 batches, 64 channels, 256x256 spatial
    kernel_size, stride, padding = 3, 2, 1

    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    # Use seeded random for reproducible performance tests
    torch.manual_seed(123)
    input_tensor = torch.randn(N, C, H, W, device="cuda", dtype=dtype)
    output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)

    # N, C, H, W = 1, 1, 3, 3
    # kernel_size, stride, padding = 2, 1, 0

    # # Create input tensor: [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    # input_tensor = torch.tensor(
    #     [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], device="cuda", dtype=dtype
    # )
    

    # Calculate output dimensions
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output_tensor = torch.empty(N * C * H_out * W_out, device="cuda", dtype=dtype)

    print(f"H_out{H_out}, W_out{W_out}")

    ref_output = torch.empty(output_tensor.shape, device="cuda", dtype=dtype)
    reference_impl(input_tensor, ref_output, N, C, H, W, kernel_size, stride, padding)
    torch.cuda.synchronize() # 强制同步，保险起见
    # define triton output
    triton_output = torch.empty(output_tensor.shape, device="cuda", dtype=dtype)
    triton_impl(input_tensor, triton_output, N, C, H, W, kernel_size, stride, padding)
    torch.cuda.synchronize() # 强制同步，保险起见

    hip_impl(input_tensor, output_tensor, N, C, H, W, kernel_size, stride, padding)
    torch.cuda.synchronize() # 强制同步，保险起见

    print("check cuda")
    check_all_close(ref_output, output_tensor, verbose=True)
    print("check triton")
    check_all_close(ref_output, triton_output, verbose=True)

    # Performance
    implementations = {
        "Reference (Torch)": lambda out: reference_impl(input_tensor, out, N, C, H, W, kernel_size, stride, padding),
        "Triton": lambda out: triton_impl(input_tensor, out, N, C, H, W, kernel_size, stride, padding),
        "HIP/CUDA": lambda out: hip_impl(input_tensor, out, N, C, H, W, kernel_size, stride, padding)
    }
    warmup_iters = 10
    test_iters = 100
    
    print(f"\n{'Implementation':<20} | {'Avg Time (ms)':<15} | {'Throughput (GB/s)':<15}")
    print("-" * 55)

    # * 4 是因为 float32 数据类型
    total_bytes = (input_tensor.numel() + (N * C * H_out * W_out)) * 4

    for name, func in implementations.items():
        # 1. Warm up
        target_output = output_tensor
        for _ in range(warmup_iters):
            func(target_output)
        torch.cuda.synchronize()

        # 2. Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(test_iters):
            func(target_output)
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
    Reference (Torch)    |        0.1727 |        485.73
    Triton               |        0.1311 |        639.69
    HIP/CUDA             |        0.3925 |        213.70
    """


if __name__ == "__main__":
    test_max_pooling()
