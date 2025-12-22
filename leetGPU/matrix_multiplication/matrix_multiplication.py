
import pytest
import pyhip
@pyhip.module("matrix_multiplication_kernel.cpp")
def matrix_multiplication_kernel(A, B, C, M, N, K): ...

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
@pytest.mark.parametrize("M, N, K",[
    (16,16,32),
    (8,1,32),
     (5,6,7),
    (1,20,20),
])
def test_matrix_multiplication(M,N,K):

    #kernel = matrix_multiplication
    kernel = matrix_multiplication_kernel
    '''
    (M,N) (N,K) => (M,K)
    '''
    # M = 16
    # N = 16
    # K = 32

    A = torch.randn(M,N, dtype=torch.float16)
    B = torch.randn(N,K, dtype=torch.float16)
    out = torch.zeros((M, K), dtype=torch.float16)

    kernel([(K+16-1)//16, (M+16-1)//16],[16,16],A.data_ptr(), out.data_ptr(), out.data_ptr(), M,N,K)

    out_ref = torch.matmul(A,B)

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    for i in range(3):
        torch.cuda._sleep(1_000_000_000)
        ev_start.record()
        kernel([(K+16-1)//16, (M+16-1)//16],[16,16],A.data_ptr(), B.data_ptr(), out.data_ptr(), M,N,K)
        ev_end.record()
        torch.cuda.synchronize()
        dt_ms = ev_start.elapsed_time(ev_end)/1
        flops = M*N*K*2
        bytes_per_elem = 2
        rd_bytes = (M * N * K * 2) * bytes_per_elem
        print(f"dt = {dt_ms*1e3:.5f} us {flops*1e-9/dt_ms:.5f} TFLOPS  {rd_bytes*1e-6/dt_ms:.5f} GB/s per-layer  {M=} {N=} {K=} ")
    check_all_close(out, out_ref)



if __name__ == "__main__":
    test_matrix_multiplication()

