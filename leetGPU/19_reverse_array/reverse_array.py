
import pytest
import pyhip
@pyhip.module("reverse_array_kernel.cpp")
def reverse_array_kernel(input, N): ...

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

def reference_impl(input: torch.Tensor, N: int):
    assert input.shape == (N,)
    assert input.dtype == torch.float32

    # Reverse the array in-place
    input[:] = torch.flip(input, [0])
        
def test_reverse_array():

    kernel = reverse_array_kernel
    input = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.float32)
    N = input.shape[0]

    ref_output = input.clone()
    reference_impl(ref_output,N)


    threadsPerBlock = 256
    blocksPerGrid = (N + threadsPerBlock - 1) // threadsPerBlock
    kernel([blocksPerGrid],[threadsPerBlock],input.data_ptr(), N)


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
    
   

    check_all_close(ref_output, input)



if __name__ == "__main__":
    test_reverse_array()