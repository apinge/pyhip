
import pytest
import pyhip
@pyhip.module("monte_carlo_integration_kernel.cpp")
def monte_carlo_integration(y_samples,result,n_samples): ...

import torch
import torch.nn.functional as F

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

def reference_impl(y_samples: torch.Tensor, result: torch.Tensor, a: float, b: float, n_samples: int):
    assert y_samples.shape == (n_samples,)
    assert result.shape == (1,)
    assert y_samples.dtype == result.dtype
    assert y_samples.device == result.device
    assert b > a
    
    # Monte Carlo integration: integral ≈ (b - a) * mean(y_samples)
    mean_y = torch.mean(y_samples)
    integral = (b - a) * mean_y
    
    result[0] = integral

def test_monte_carlo_intergration():

    kernel = monte_carlo_integration
    dtype = torch.float32
    y_samples = torch.tensor([0.0625, 0.25, 0.5625, 1.0, 1.5625, 2.25, 3.0625, 4.0], device="cuda", dtype=dtype)
    result = torch.empty(1, device="cuda", dtype=dtype)
    ref_result = torch.empty(1, device="cuda", dtype=dtype)
    n_samples=8
    a = 0.0
    b = 2.0
    reference_impl(y_samples,ref_result,a,b,n_samples)



    threadsPerBlock = 256
    blocksPerGrid = (n_samples + threadsPerBlock - 1) // threadsPerBlock
   # breakpoint()
    kernel([blocksPerGrid],[threadsPerBlock],y_samples.data_ptr(), result.data_ptr(), a,b,n_samples)


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
    
   
    print(ref_result)
    check_all_close(ref_result,result)



if __name__ == "__main__":
    test_monte_carlo_intergration()