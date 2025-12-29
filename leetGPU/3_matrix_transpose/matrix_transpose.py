
import pytest
import pyhip
@pyhip.module("matrix_transpose_kernel.cpp")
def matrix_transpose_kernel(input,output,rows,cols): ...

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

def reference_impl(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    assert input.shape == (rows, cols)
    assert output.shape == (cols, rows)
    assert input.dtype == output.dtype
    assert input.device == output.device

    output.copy_(input.transpose(0, 1))
        
def test_matrix_transpose():

    #kernel = matrix_transpose
    kernel = matrix_transpose_kernel
    '''
    (rows,cols) => (cols,rows)
    '''
    rows, cols = 2, 3
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda", dtype=torch.float32)
    ref_output_tensor = torch.empty(cols, rows, device="cuda", dtype=torch.float32) 

    reference_impl(input_tensor,ref_output_tensor,rows,cols)

   
    output_tensor = torch.empty(cols, rows, device="cuda", dtype=torch.float32)

    kernel([(cols+16-1)//16, (rows+16-1)//16],[16,16],input_tensor.data_ptr(), output_tensor.data_ptr(), rows,cols)


    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    for i in range(3):
        torch.cuda._sleep(1_000_000_000)
        ev_start.record()
        kernel([(cols+16-1)//16, (rows+16-1)//16],[16,16],input_tensor.data_ptr(), output_tensor.data_ptr(), rows,cols)
        ev_end.record()
        torch.cuda.synchronize()
        dt_ms = ev_start.elapsed_time(ev_end)/1
        flops = rows*cols
        bytes_per_elem = 4
        rd_bytes = (rows*cols* 2) * bytes_per_elem
        print(f"dt = {dt_ms*1e3:.5f} us {flops*1e-9/dt_ms:.5f} TFLOPS  {rd_bytes*1e-6/dt_ms:.5f} GB/s per-layer  {rows=} {cols=} ")
    check_all_close(output_tensor, ref_output_tensor)



if __name__ == "__main__":
    test_matrix_transpose()