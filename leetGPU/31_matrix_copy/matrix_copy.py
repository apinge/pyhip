
import pytest
import pyhip
@pyhip.module("matrix_copy_kernel.cpp")
def matrix_copy_kernel(A,B,N): ...

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

def reference_impl(A: torch.Tensor, B: torch.Tensor, N: int):
    assert A.shape == (N, N)
    assert B.shape == (N, N)
    assert A.dtype == B.dtype
    assert A.device == B.device

    # Copy matrix A to B
    B[:] = A

def test_matrix_copy():

    kernel = matrix_copy_kernel

    dtype = torch.float32
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype)
    N = A.shape[0]
    B = torch.empty(N, N, device="cuda", dtype=dtype)



    threadsPerBlock = 256
    blocksPerGrid = (N*N + threadsPerBlock - 1) // threadsPerBlock
   # breakpoint()
    kernel([blocksPerGrid],[threadsPerBlock],A.data_ptr(), B.data_ptr(), N)


    
    check_all_close(A,B)


if __name__ == "__main__":
    test_matrix_copy()