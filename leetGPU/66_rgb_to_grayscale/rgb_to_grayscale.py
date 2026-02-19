import pyhip

import triton
import triton.language as tl

import torch

# HIP Kernel 壳子：具体实现不在此处，由 .cpp 提供
@pyhip.module("rgb_to_grayscale.cpp")
def rgb_to_grayscale_kernel(input, output, width, height): ...


def check_all_close(out, out_ref, rtol=0.01, atol=0.01, verbose=False):
    if not torch.allclose(out, out_ref, rtol=rtol, atol=atol):
        if verbose:
            print(out)
            print(out_ref)
        torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
    else:
        print("PASS")


def reference_impl(input: torch.Tensor, output: torch.Tensor, width: int, height: int):
    assert input.shape == (height * width * 3,)
    assert output.shape == (height * width,)
    assert input.dtype == output.dtype == torch.float32
    assert input.device == output.device

    # Reshape input to (height, width, 3) for easier processing
    rgb_image = input.view(height, width, 3)

    # Apply RGB to grayscale conversion: gray = 0.299*R + 0.587*G + 0.114*B
    grayscale = (
        0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]
    )

    # Flatten and store in output
    output.copy_(grayscale.flatten())


def generate_example_test() -> dict:
    width, height = 2, 2
    # RGB values for a 2x2 image
    # Pixel (0,0): R=255, G=0, B=0 (red)
    # Pixel (0,1): R=0, G=255, B=0 (green)
    # Pixel (1,0): R=0, G=0, B=255 (blue)
    # Pixel (1,1): R=128, G=128, B=128 (gray)
    input_data = torch.tensor(
        [
            255.0,
            0.0,
            0.0,  # red
            0.0,
            255.0,
            0.0,  # green
            0.0,
            0.0,
            255.0,  # blue
            128.0,
            128.0,
            128.0,  # gray
        ],
        device="cuda",
        dtype=torch.float32,
    )
    output = torch.zeros(width * height, device="cuda", dtype=torch.float32)
    return {
        "input": input_data,
        "output": output,
        "width": width,
        "height": height,
    }

def generate_performance_test() -> dict:
    width, height = 2048, 2048
    input_size = width * height * 3
    output_size = width * height
    return {
        "input": torch.randint(0, 256, (input_size,), device="cuda", dtype=torch.float32),
        "output": torch.zeros(output_size, device="cuda", dtype=torch.float32),
        "width": width,
        "height": height,
    }

@triton.jit
def rgb_to_grayscale_triton(input_ptr, output_ptr, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < width*height
    

    r_ptr = input_ptr + offsets*3
    g_ptr = input_ptr + offsets*3 +1
    b_ptr = input_ptr + offsets*3+2

    r = tl.load(r_ptr,mask=mask)
    g = tl.load(g_ptr,mask = mask)
    b = tl.load(b_ptr,mask  = mask)

  
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    tl.store(output_ptr + offsets, gray, mask=mask)
    pass

def triton_impl(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    width: int,
    height: int,
):
    output_tensor.zero_()
    num_pixels = width * height
    grid = lambda META: (triton.cdiv(num_pixels, META["BLOCK_SIZE"]),)
    rgb_to_grayscale_triton[grid](
        input_tensor,
        output_tensor,
        width=width,
        height=height,
        BLOCK_SIZE=256,
    )


def hip_impl(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    width: int,
    height: int,
):
    output_tensor.zero_()
    num_pixels = width * height  # 每个像素一个灰度输出
    threads_per_block = [256, 1, 1]
    blocks_per_grid = [
        (num_pixels + threads_per_block[0] - 1) // threads_per_block[0],
        1,
        1,
    ]
    rgb_to_grayscale_kernel(
        blocks_per_grid,
        threads_per_block,
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        width,
        height,
        #sharedMemBytes=256 * 3 * 4,
    )


def test_rgb_to_grayscale():
    # 正确性：小例子
    example = generate_example_test()
    input_data = example["input"]
    output_tensor = example["output"]
    width = example["width"]
    height = example["height"]

    ref_output = torch.empty_like(output_tensor, device="cuda", dtype=torch.float32)
    reference_impl(input_data, ref_output, width, height)
    torch.cuda.synchronize()

    hip_output = torch.empty(output_tensor.shape, device="cuda", dtype=torch.float32)
    hip_impl(input_data, hip_output, width, height)
    torch.cuda.synchronize()
    print("check HIP (correctness)")
    check_all_close(ref_output, hip_output, verbose=True)

    triton_output = torch.empty(output_tensor.shape, device="cuda", dtype=torch.float32)
    triton_impl(input_data, triton_output, width, height)
    torch.cuda.synchronize()
    print("check Triton (correctness)")
    check_all_close(ref_output, triton_output, verbose=True)

    # 性能：大图
    perf = generate_performance_test()
    input_tensor = perf["input"]
    output_tensor = perf["output"]
    width, height = perf["width"], perf["height"]

    implementations = {
        "Reference (Torch)": lambda out: reference_impl(input_tensor, out, width, height),
        "Triton": lambda out: triton_impl(input_tensor, out, width, height),
        "HIP/CUDA": lambda out: hip_impl(input_tensor, out, width, height),
    }
    warmup_iters = 10
    test_iters = 100

    print(f"\n{'Implementation':<20} | {'Avg Time (ms)':<15} | {'Throughput (GB/s)':<15}")
    print("-" * 55)

    # float32 = 4 bytes，读写总量
    total_bytes = (input_tensor.numel() + output_tensor.numel()) * 4

    for name, func in implementations.items():
        target_output = output_tensor
        for _ in range(warmup_iters):
            func(target_output)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(test_iters):
            func(target_output)
        end_event.record()
        torch.cuda.synchronize()

        avg_ms = start_event.elapsed_time(end_event) / test_iters
        bandwidth = total_bytes / (avg_ms * 1e6)
        print(f"{name:<20} | {avg_ms:>13.4f} | {bandwidth:>13.2f}")


if __name__ == "__main__":
    test_rgb_to_grayscale()
