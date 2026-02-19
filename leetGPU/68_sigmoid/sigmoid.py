import pyhip

import triton
import triton.language as tl

import torch

# HIP Kernel 壳子：具体实现不在此处，由 .cpp 提供
@pyhip.module("sigmoid.cpp")
def sigmoid_kernel(X, Y, N): ...



@triton.jit
def sigmoid_triton(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    # tl.exp 是 Triton 内置的
    y = 1.0/(1.0+tl.exp(-x))
    tl.store(Y_ptr + offsets, y, mask=mask)


def check_all_close(out, out_ref, rtol=1e-5, atol=1e-5, verbose=False):
    if not torch.allclose(out, out_ref, rtol=rtol, atol=atol):
        if verbose:
            print(out)
            print(out_ref)
        torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
    else:
        print("PASS")


def reference_impl(X: torch.Tensor, Y: torch.Tensor, N: int):
    assert X.shape == Y.shape
    assert X.dtype == torch.float32
    assert Y.dtype == torch.float32
    assert X.device.type == "cuda"
    assert Y.device.type == "cuda"
    torch.sigmoid(X, out=Y)


def generate_example_test() -> dict:
    dtype = torch.float32
    N = 4
    X = torch.tensor([0.0, 1.0, -1.0, 2.0], device="cuda", dtype=dtype)
    Y = torch.empty(N, device="cuda", dtype=dtype)
    return {"X": X, "Y": Y, "N": N}


def generate_performance_test() -> dict:
    dtype = torch.float32
    N = 50000000
    return {
        "X": torch.empty(N, device="cuda", dtype=dtype).uniform_(-10.0, 10.0),
        "Y": torch.empty(N, device="cuda", dtype=dtype),
        "N": N,
    }


def triton_impl(X: torch.Tensor, Y: torch.Tensor, N: int):
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    sigmoid_triton[grid](X, Y, N=N, BLOCK_SIZE=1024)


def hip_impl(X: torch.Tensor, Y: torch.Tensor, N: int):
    threads_per_block = [256, 1, 1]
    blocks_per_grid = [(N + threads_per_block[0] - 1) // threads_per_block[0], 1, 1]
    sigmoid_kernel(
        blocks_per_grid,
        threads_per_block,
        X.data_ptr(),
        Y.data_ptr(),
        N,
    )


def test_sigmoid():
    # 正确性：小例子
    example = generate_example_test()
    X, Y, N = example["X"], example["Y"], example["N"]

    ref_output = torch.empty_like(Y, device="cuda", dtype=torch.float32)
    reference_impl(X, ref_output, N)
    torch.cuda.synchronize()

    hip_output = torch.empty_like(Y, device="cuda", dtype=torch.float32)
    hip_impl(X, hip_output, N)
    torch.cuda.synchronize()
    print("check HIP (correctness)")
    check_all_close(ref_output, hip_output, verbose=True)

    triton_output = torch.empty_like(Y, device="cuda", dtype=torch.float32)
    triton_impl(X, triton_output, N)
    torch.cuda.synchronize()
    print("check Triton (correctness)")
    check_all_close(ref_output, triton_output, verbose=True)

    # 性能
    perf = generate_performance_test()
    X = perf["X"]
    Y = perf["Y"]
    N = perf["N"]

    implementations = {
        "Reference (Torch)": lambda out: reference_impl(X, out, N),
        "Triton": lambda out: triton_impl(X, out, N),
        "HIP/CUDA": lambda out: hip_impl(X, out, N),
    }
    warmup_iters = 10
    test_iters = 100

    print(f"\n{'Implementation':<20} | {'Avg Time (ms)':<15} | {'Throughput (GB/s)':<15}")
    print("-" * 55)
    total_bytes = (X.numel() + Y.numel()) * 4

    for name, func in implementations.items():
        target = Y
        for _ in range(warmup_iters):
            func(target)
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(test_iters):
            func(target)
        end_ev.record()
        torch.cuda.synchronize()
        avg_ms = start_ev.elapsed_time(end_ev) / test_iters
        bandwidth = total_bytes / (avg_ms * 1e6)
        print(f"{name:<20} | {avg_ms:>13.4f} | {bandwidth:>13.2f}")


if __name__ == "__main__":
    test_sigmoid()
