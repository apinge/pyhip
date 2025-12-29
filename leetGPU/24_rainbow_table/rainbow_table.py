
import pytest
import pyhip
@pyhip.module("rainbow_table_kernel.cpp")
def fnv1a_hash_kernel(input,output,N,R): ...

import torch
torch.cuda.set_device(6)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
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
def fnv1a_hash(x: torch.Tensor) -> torch.Tensor:
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261
    x_int = x.to(torch.int64)
    hash_val = torch.full_like(x_int, OFFSET_BASIS, dtype=torch.int64)
    for byte_pos in range(4):
        byte = (x_int >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME
        hash_val = hash_val & 0xFFFFFFFF
    return hash_val
def reference_impl(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    assert input.shape == (N,)
    assert output.shape == (N,)
    assert input.dtype == torch.int32
    assert output.dtype == torch.uint32

    current = input

    # Apply hash R times
    for _ in range(R):
        current = fnv1a_hash(current)

    # Reinterpret the lower 32 bits as uint32
    output.copy_(current.to(torch.int32).view(torch.uint32))

def test_rainbow_table():

    kernel = fnv1a_hash_kernel

    input_tensor = torch.tensor( [0, 1, 2147483647], device="cuda", dtype=torch.int32)
    N = input_tensor.shape[0]
    output_tensor = torch.empty(N, device="cuda", dtype=torch.uint32)
    ref_output_tensor = torch.empty(N, device="cuda", dtype=torch.uint32)

    R = 3
    reference_impl(input_tensor,ref_output_tensor,N,R)

    threadsPerBlock = 256
    blocksPerGrid = (N + threadsPerBlock - 1) // threadsPerBlock
   # breakpoint()
    kernel([blocksPerGrid],[threadsPerBlock],input_tensor.data_ptr(), output_tensor.data_ptr(), N, R)


    # ev_start = torch.cuda.Event(enable_timing=True)
    # ev_end = torch.cuda.Event(enable_timing=True)

    # for i in range(3):
    #     torch.cuda._sleep(1_000_000_000)
    #     ev_start.record()
    #     kernel([(width*height+256-1)//256],[256],image.data_ptr(), width, height)
    #     ev_end.record()
    #     torch.cuda.synchronize()
    #     dt_ms = ev_start.elapsed_time(ev_end)/1
    #     flops = width*height
    #     bytes_per_elem = 1 # char
    #     rd_bytes = (width*height* 2) * bytes_per_elem
    #     print(f"dt = {dt_ms*1e3:.5f} us {flops*1e-9/dt_ms:.5f} TFLOPS  {rd_bytes*1e-6/dt_ms:.5f} GB/s per-layer  {width=} {height=} ")
    
   

    check_all_close(ref_output_tensor, output_tensor)



if __name__ == "__main__":
    test_rainbow_table()