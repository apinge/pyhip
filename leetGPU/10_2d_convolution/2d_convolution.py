import pytest
import pyhip


import triton
import triton.language as tl


# @pyhip.module("2d_convolution_kernel.cpp")
# def convolution_2d_kernel(
#     input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols
# ): ...


@pyhip.module("2d_convolution_kernel.cpp")
def convolution_2d_optimized(
    input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols
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


def get_autotune_config():
    return [

        triton.Config({'ROW_BLOCK': 8, 'COL_BLOCK': 8}, num_warps=2),
        triton.Config({'ROW_BLOCK': 16, 'COL_BLOCK': 16}, num_warps=4),
        triton.Config({'ROW_BLOCK': 32, 'COL_BLOCK': 8}, num_warps=4),
        triton.Config({'ROW_BLOCK': 8, 'COL_BLOCK': 32}, num_warps=4),
        triton.Config({'ROW_BLOCK': 32, 'COL_BLOCK': 16}, num_warps=8),
        triton.Config({'ROW_BLOCK': 16, 'COL_BLOCK': 32}, num_warps=8),
        triton.Config({'ROW_BLOCK': 32, 'COL_BLOCK': 32}, num_warps=8),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['output_rows', 'output_cols', 'kernel_rows', 'kernel_cols'],
)
@triton.jit
def convolution_2d(
    input,
    kernel,
    output,
    input_rows,
    intput_cols,
    kernel_rows,
    kernel_cols,
    stride_ir,
    stride_ic,
    stride_kr,
    stride_kc,
    stride_or,
    stride_oc,
    output_rows,
    output_cols,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    out_r = row* ROW_BLOCK+tl.arange(0,ROW_BLOCK)
    out_c = col* COL_BLOCK+tl.arange(0,COL_BLOCK)
    mask_out = (out_r[:,None]<output_rows) & (out_c[None,:]<output_cols)
    accumulator = tl.zeros([ROW_BLOCK, COL_BLOCK], dtype=tl.float32)

   
    for i in range(kernel_rows):
        for j in range(kernel_cols):
            """
            如果这里卷积本身有stride
            应该是 in_r = out_r *conv_r +j
            in_c = out_c*cov_c +j 
            """
            in_r = out_r+i
            in_c = out_c+j
            mask_in =  (in_r[:,None] < input_rows ) &(in_c[None,:] < intput_cols)

            input_ptrs = input +  in_r[:,None]*stride_ir + in_c[None,:]*stride_ic
            val = tl.load(input_ptrs, mask=mask_in, other=0.0)
            k_ptr = kernel + (i*stride_kr) + j*stride_kc
            weight = tl.load(k_ptr)
            accumulator += val*weight
    
    out_ptrs = output + out_r[:,None]*stride_or + out_c[None,:]*stride_oc
    tl.store(out_ptrs,accumulator, mask=mask_out)

    pass


def triton_impl(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    output.zero_()

    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    grid = lambda META: (
        triton.cdiv(output_rows, META["ROW_BLOCK"]),
        triton.cdiv(output_cols, META["COL_BLOCK"]),
    )
    stride_ir, stride_ic = input_cols, 1
    stride_kr, stride_kc = kernel_cols,1
    stride_or, stride_oc = output_cols,1
 
    convolution_2d[grid](
        input,
        kernel,
        output,
        input_rows,
        input_cols,
        kernel_rows,
        kernel_cols,
        stride_ir,
        stride_ic,
        stride_kr,
        stride_kc,
        stride_or,
        stride_oc,
        output_rows,
        output_cols,
        # ROW_BLOCK=16,
        # COL_BLOCK=16,
    )
    pass


def reference_impl(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    output.zero_()
    # Reshape flattened arrays to 2D matrices
    input_2d = input.view(input_rows, input_cols)
    kernel_2d = kernel.view(kernel_rows, kernel_cols)
    # Prepare tensors for conv2d (add batch and channel dimensions)
    kernel_prepared = kernel_2d.unsqueeze(0).unsqueeze(0)
    input_prepared = input_2d.unsqueeze(0).unsqueeze(0)
    # Perform cross-correlation using PyTorch's F.conv2d
    # (which does cross-correlation by default)
    result = torch.nn.functional.conv2d(input_prepared, kernel_prepared, padding=0)
    # Copy result to output tensor (removing the extra dimensions and flattening)
    output.copy_(result.view(-1))


def hip_impl(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    output.zero_()

    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    threadsPerBlock = [256, 1]
    blocksPerGrid = [
        (output_cols + threadsPerBlock[0] - 1) // threadsPerBlock[0],
        (output_rows + threadsPerBlock[1] - 1) // threadsPerBlock[1],
    ]
    convolution_2d_optimized(
        blocksPerGrid,
        threadsPerBlock,
        input.data_ptr(),
        kernel.data_ptr(),
        output.data_ptr(),
        input_rows,
        input_cols,
        kernel_rows,
        kernel_cols,
    )
    pass


def test_2d_convolution():

    dtype = torch.float32
    # input = torch.empty(128 * 64, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    # kernel = torch.empty(7 * 7, device="cuda", dtype=dtype).uniform_(-0.2, 0.2)
    # input_rows, input_cols = 128, 64
    # kernel_rows, kernel_cols = 7, 7
    input_rows = 3072
    input_cols = 3072
    kernel_rows = 15
    kernel_cols = 15
    input = torch.empty(input_rows * input_cols, device="cuda", dtype=dtype).uniform_(-1.0, 1.0)
    kernel = torch.empty(kernel_rows * kernel_cols, device="cuda", dtype=dtype).uniform_(
            -1.0, 1.0
        )
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    output = torch.empty(output_rows * output_cols, device="cuda", dtype=dtype)

    ref_output = torch.empty(
        output_rows * output_cols, device="cuda", dtype=torch.float32
    )
    triton_output = torch.empty(
        output_rows * output_cols, device="cuda", dtype=torch.float32
    )
    reference_impl(
        input, kernel, ref_output, input_rows, input_cols, kernel_rows, kernel_cols
    )
    torch.cuda.synchronize()
    hip_impl(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols)
    torch.cuda.synchronize()
    triton_impl(input, kernel, triton_output, input_rows, input_cols, kernel_rows, kernel_cols)
    torch.cuda.synchronize()

    print("check hip")
    check_all_close(ref_output, output, verbose=True)
    print("check triton")
    check_all_close(ref_output, triton_output, verbose=True)

    # Performance
    implementations = {
        "Reference (Torch)": lambda out: reference_impl(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols),
        "Triton": lambda out: triton_impl(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols),
        "HIP/CUDA": lambda out: hip_impl(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols)
    }
    warmup_iters = 10
    test_iters = 100
    
    print(f"\n{'Implementation':<20} | {'Avg Time (ms)':<15} | {'Throughput (GB/s)':<15}")
    print("-" * 55)

    # * 4 是因为 float32 数据类型
    # 假设是单通道 (C=1) 和单批次 (N=1)
    total_bytes = (input_rows * input_cols + kernel_rows * kernel_cols + output_rows * output_cols) * 4

    for name, func in implementations.items():
        # 1. Warm up
        target_output = output
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
    Reference (Torch)    |        9.5677 |          7.86
    Triton               |        1.8562 |         40.49
    HIP/CUDA             |        1.5456 |         48.63

    ## 模板展开
    如果针对 kernel_col = 15的测 定义模板展开
    
    Implementation       | Avg Time (ms)   | Throughput (GB/s)
    -------------------------------------------------------
    Reference (Torch)    |        9.5312 |          7.89
    Triton               |        1.9255 |         39.03
    HIP/CUDA             |        1.3007 |         57.78

    ## Recursive Template Instantiation + Static Dispatch
    这个本身没提升性能 只是突破了pyhip 用不了常数模板的限制
    Implementation       | Avg Time (ms)   | Throughput (GB/s)
    -------------------------------------------------------
    Reference (Torch)    |        9.5371 |          7.88
    Triton               |        1.9264 |         39.01
    HIP/CUDA             |        1.2984 |         57.88

    """

if __name__ == "__main__":
    test_2d_convolution()
