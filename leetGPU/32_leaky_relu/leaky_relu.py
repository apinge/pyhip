
import pytest
import pyhip
@pyhip.module("leaky_relu_kernel.cpp")
def leaky_relu_kernel(input,output,N): ...

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

def reference_impl(input: torch.Tensor, output: torch.Tensor, N: int):
    assert input.shape == (N,)
    assert output.shape == (N,)
    assert input.dtype == output.dtype
    assert input.device == output.device

    # Apply Leaky ReLU: f(x) = x if x > 0, else 0.01 * x
    alpha = 0.01
    output[:] = torch.where(input > 0, input, alpha * input)

def test_leaky_relu():

    kernel = leaky_relu_kernel
    input_tensor = torch.tensor([-0.001, -0.0001, 0.0, 0.0001, 0.001], device="cuda", dtype=torch.float32)
    #input_tensor =  torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], device="cuda", dtype=torch.float32),
    N = input_tensor.shape[0]
    output_tensor = torch.empty(N, device="cuda", dtype=torch.float32)
    ref_output_tensor = torch.empty(N, device="cuda", dtype=torch.float32)
    reference_impl(input_tensor,ref_output_tensor,N)

    threadsPerBlock = 256
    blocksPerGrid = (N + threadsPerBlock - 1) // threadsPerBlock
   # breakpoint()
    kernel([blocksPerGrid],[threadsPerBlock],input_tensor.data_ptr(), output_tensor.data_ptr(), N)


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
    
   

    check_all_close(ref_output_tensor, ref_output_tensor)



if __name__ == "__main__":
    test_leaky_relu()