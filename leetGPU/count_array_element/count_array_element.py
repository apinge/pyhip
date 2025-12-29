
import pytest
import pyhip
@pyhip.module("count_equal_kernel.cpp")
def count_equal_kernel(input,output,N,K): ...

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

def reference_impl(input: torch.Tensor, N: int,K:int):
    assert input.shape == (N,)
    

    return (input==K).sum().item()

def test_count_array_element():

    kernel = count_equal_kernel

    dtype = torch.int32
    A = torch.tensor([1, 2, 3, 4, 1], device="cuda", dtype=dtype)
    K = 1
    N = A.shape[0]
    output = torch.empty(1, device="cuda", dtype=dtype)
    output_ref = reference_impl(A,N, K)


    threadsPerBlock = 256
    blocksPerGrid = (N + threadsPerBlock - 1) // threadsPerBlock
   # breakpoint()
    kernel([blocksPerGrid],[threadsPerBlock],A.data_ptr(), output.data_ptr(), N,K)

    print(output)
    assert output[0].item() == output_ref



if __name__ == "__main__":
    test_count_array_element()