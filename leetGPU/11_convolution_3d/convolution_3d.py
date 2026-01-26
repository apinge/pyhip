import pytest
import pyhip


@pyhip.module("convolution_3d.cpp")
def convolution_3d(
    input,
    kernel,
    output,
    input_depth,
    input_rows,
    input_cols,
    kernel_depth,
    kernel_rows,
    kernel_cols,
): ...


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


import triton
import triton.language as tl


def get_configs():
    configs = []
    for block_r in [16, 32, 64]:
        for block_c in [16, 32, 64]:

            for num_warps in [4, 8]:
                configs.append(
                    triton.Config(
                        {'BLOCK_R': block_r, 'BLOCK_C': block_c}, 
                        num_warps=num_warps
                    )
                )
    return configs
@triton.autotune(
    configs=get_configs(),
    key=['output_rows', 'output_cols', 'kernel_depth', 'kernel_rows', 'kernel_cols'],
)
@triton.jit
def convolution_3d_triton(
    input,
    kernel,
    output,
    input_depth,
    input_rows,
    input_cols,
    kernel_depth,
    kernel_rows,
    kernel_cols,
    stride_id,
    stride_ir,
    stride_ic,
    stride_kd,
    stride_kr,
    stride_kc,
    stride_od,
    stride_or,
    stride_oc,
    output_depth,
    output_rows,
    output_cols,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    # DEPTH_BOCK: tl.constexpr,
):
    pos = tl.program_id(0)
    depth = tl.program_id(1)  # output_depth
    num_pid_c = tl.cdiv(output_cols, BLOCK_C)
    row = pos // num_pid_c
    col = pos % num_pid_c

    curr_output_depth_ptr = depth * stride_od

    out_r = row * BLOCK_R + tl.arange(0, BLOCK_R)
    out_c = col * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_out = (out_r[:, None] < output_rows) & (out_c[None, :] < output_cols)
    accumulator = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)

    # input_depth_tmp = input_cols*input_rows
    # kernel_depth_tmp = kernel_cols*kernel_rows
    for k in range(kernel_depth):
        input_depth_offset = (k + depth) * stride_id
        kernel_depth_offset = k * stride_kd
        for i in range(kernel_rows):
            in_r = out_r + i
            for j in range(kernel_cols):
                in_c = out_c + j
                mask_in = (in_r[:, None] < input_rows) & (in_c[None, :] < input_cols)
                input_ptrs = (
                    input
                    + input_depth_offset
                    + in_r[:, None] * stride_ir
                    + in_c[None, :] * stride_ic
                )
                val = tl.load(input_ptrs, mask=mask_in, other=0.0)
                k_ptr = kernel_depth_offset + kernel + (i * stride_kr) + j * stride_kc
                weight = tl.load(k_ptr)
                accumulator += val * weight
    # tl.device_print(" depth", depth)
    out_ptrs = (
        output
        + curr_output_depth_ptr
        + out_r[:, None] * stride_or
        + out_c[None, :] * stride_oc
    )
    tl.store(out_ptrs, accumulator, mask=mask_out)

    pass


def triton_impl(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    output.zero_()

    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    # stride_id, stride_ir, stride_ic = input_rows*input_cols,input_cols, 1
    # stride_kd, stride_kr, stride_kc = kernel_cols*kernel_rows, kernel_cols,1
    # stride_od, stride_or, stride_oc = output_rows*output_cols, output_cols,1
    stride_id, stride_ir, stride_ic = input.stride()
    stride_kd, stride_kr, stride_kc = kernel.stride()
    stride_od, stride_or, stride_oc = output.stride()

    grid = lambda META: (
        triton.cdiv(output_rows, META["BLOCK_R"])
        * triton.cdiv(output_cols, META["BLOCK_C"]),
        output_depth,
    )
    convolution_3d_triton[grid](
        input,
        kernel,
        output,
        input_depth,
        input_rows,
        input_cols,
        kernel_depth,
        kernel_rows,
        kernel_cols,
        stride_id,
        stride_ir,
        stride_ic,
        stride_kd,
        stride_kr,
        stride_kc,
        stride_od,
        stride_or,
        stride_oc,
        output_depth,
        output_rows,
        output_cols,
        # BLOCK_R=16,
        # BLOCK_C=16,
    )
    pass


def reference_impl(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    output.zero_()
    assert input.shape == (input_depth, input_rows, input_cols)
    assert kernel.shape == (kernel_depth, kernel_rows, kernel_cols)
    assert output.shape == (
        input_depth - kernel_depth + 1,
        input_rows - kernel_rows + 1,
        input_cols - kernel_cols + 1,
    )
    assert input.dtype == kernel.dtype == output.dtype
    assert input.device == kernel.device == output.device

    input_expanded = input.unsqueeze(0).unsqueeze(0)
    kernel_expanded = kernel.unsqueeze(0).unsqueeze(0)

    result = torch.nn.functional.conv3d(
        input_expanded, kernel_expanded, bias=None, stride=1, padding=0
    )

    output.copy_(result.squeeze(0).squeeze(0))


def hip_impl(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    output.zero_()

    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    threadsPerBlock = [256, 1]
    blocksPerGrid = [
        (output_cols * output_rows + threadsPerBlock[0] - 1) // threadsPerBlock[0],
        (output_depth + threadsPerBlock[1] - 1) // threadsPerBlock[1],
    ]
    convolution_3d(
        blocksPerGrid,
        threadsPerBlock,
        input.data_ptr(),
        kernel.data_ptr(),
        output.data_ptr(),
        input_depth,
        input_rows,
        input_cols,
        kernel_depth,
        kernel_rows,
        kernel_cols,
    )
    pass


def test_2d_convolution():

    dtype = torch.float32
    device = "cuda"
    # case 1
    # input = torch.tensor(
    #     [
    #         [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    #         [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    #         [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
    #     ],
    #     dtype=dtype,
    #     device="cuda",
    # )
    # kernel = torch.tensor(
    #     [[[1, 0, 0], [1, 1, 1], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 1]]],
    #     dtype=dtype,
    #     device="cuda",
    # )
    # output_tensor = torch.empty((2, 1, 1), device="cuda", dtype=dtype)

    # input_depth = 3
    # input_rows = 3
    # input_cols = 3
    # kernel_depth = 2
    # kernel_rows = 3
    # kernel_cols = 3

    # case 2
    # input = torch.tensor(
    #     [
    #         [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    #         [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
    #     ],
    #     dtype=dtype,
    #     device=device,
    # )
    # kernel = torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=dtype, device=device)
    # output_tensor = torch.zeros((2, 2, 2), dtype=dtype, device=device)
    # input_depth = 2
    # input_rows = 3
    # input_cols = 4
    # kernel_depth = 1
    # kernel_rows = 2
    # kernel_cols = 3

    # case 3 performance test
    input_depth, input_rows, input_cols = 256, 128, 128
    kernel_depth, kernel_rows, kernel_cols = 5, 5, 5
    input = torch.empty(
        input_depth, input_rows, input_cols, device="cuda", dtype=dtype
    ).uniform_(-1.0, 1.0)
    kernel = torch.empty(
        kernel_depth, kernel_rows, kernel_cols, device="cuda", dtype=dtype
    ).uniform_(-1.0, 1.0)
    output_tensor = torch.zeros(
        input_depth - kernel_depth + 1,
        input_rows - kernel_rows + 1,
        input_cols - kernel_cols + 1,
        device="cuda",
        dtype=dtype,
    )

    ref_output = torch.empty(output_tensor.shape, device="cuda", dtype=torch.float32)
    triton_output = torch.empty(output_tensor.shape, device="cuda", dtype=torch.float32)
    hip_output = torch.empty(output_tensor.shape, device="cuda", dtype=torch.float32)
    reference_impl(
        input,
        kernel,
        ref_output,
        input_depth,
        input_rows,
        input_cols,
        kernel_depth,
        kernel_rows,
        kernel_cols,
    )
    torch.cuda.synchronize()
    hip_impl(
        input,
        kernel,
        hip_output,
        input_depth,
        input_rows,
        input_cols,
        kernel_depth,
        kernel_rows,
        kernel_cols,
    )
    torch.cuda.synchronize()
    triton_impl(
        input,
        kernel,
        triton_output,
        input_depth,
        input_rows,
        input_cols,
        kernel_depth,
        kernel_rows,
        kernel_cols,
    )
    torch.cuda.synchronize()

    print("check hip")
    check_all_close(ref_output, hip_output, verbose=True)
    print("check triton")
    check_all_close(ref_output, triton_output, verbose=True)

    # # Performance
    implementations = {
        "Reference (Torch)": lambda: reference_impl(
            input,
            kernel,
            ref_output,
            input_depth,
            input_rows,
            input_cols,
            kernel_depth,
            kernel_rows,
            kernel_cols,
        ),
        "Triton": lambda: triton_impl(
            input,
            kernel,
            triton_output,
            input_depth,
            input_rows,
            input_cols,
            kernel_depth,
            kernel_rows,
            kernel_cols,
        ),
        "HIP/CUDA": lambda: hip_impl(
            input,
            kernel,
            hip_output,
            input_depth,
            input_rows,
            input_cols,
            kernel_depth,
            kernel_rows,
            kernel_cols,
        ),
    }
    warmup_iters = 10
    test_iters = 100

    print(
        f"\n{'Implementation':<20} | {'Avg Time (ms)':<15} | {'Throughput (GB/s)':<15}"
    )
    print("-" * 55)

    # * 4 是因为 float32 数据类型
    total_bytes = (input.numel() + kernel.numel() + output_tensor.numel()) * 4

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
    Reference (Torch)    |        5.6020 |          5.76
    Triton               |        0.4697 |         68.72
    HIP/CUDA             |        0.7979 |         40.45

    """
if __name__ == "__main__":
    test_2d_convolution()
