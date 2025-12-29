
import pytest
import pyhip
@pyhip.module("1d_convolution_kernel.cpp")
def convolution_1d_kernel(input,kernel,output,input_size,kernel_size): ...

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

def reference_impl(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    assert input.shape == (input_size,)
    assert kernel.shape == (kernel_size,)
    assert output.shape == (input_size - kernel_size + 1,)
    assert input.dtype == kernel.dtype == output.dtype
    assert input.device == kernel.device == output.device
    
    # Create strided view of input for all windows
    windows = input.unfold(0, kernel_size, 1)
    
    # Use einsum for explicit cross-correlation
    # 'ij,j->i' means: for each window i, multiply with kernel j and sum over j
    output.copy_(torch.einsum('ij,j->i', windows, kernel))
        
def test_1d_convolution():

    kernel = convolution_1d_kernel
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=torch.float32)
    kernel_tensor = torch.tensor([0.25], device="cuda", dtype=torch.float32)
    input_size = input_tensor.shape[0]
    kernel_size = kernel_tensor.shape[0]
    output_size = input_size - kernel_size + 1
    output_tensor = torch.empty(output_size, device="cuda", dtype=torch.float32)

   
    ref_output_tensor = torch.empty(output_size, device="cuda", dtype=torch.float32)
    reference_impl(input_tensor,kernel_tensor, ref_output_tensor,input_size,kernel_size)
 

   
    threadsPerBlock = 256
    blocksPerGrid = (output_size + threadsPerBlock - 1) // threadsPerBlock
    kernel([blocksPerGrid],[threadsPerBlock],input_tensor.data_ptr(), kernel_tensor.data_ptr(), output_tensor.data_ptr(), input_size, kernel_size)
  
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
    test_1d_convolution()