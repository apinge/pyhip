
import pytest
import pyhip
@pyhip.module("color_inversion_kernel.cpp")
def color_inversion_kernel(image,width,height): ...

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

def reference_impl(image: torch.Tensor, width: int, height: int):
    assert image.shape == (height * width * 4,)
    assert image.dtype == torch.uint8

    # Reshape to (height, width, 4) for easier processing
    image_reshaped = image.view(height, width, 4)

    # Invert RGB channels (first 3 channels), keep alpha unchanged
    image_reshaped[:, :, :3] = 255 - image_reshaped[:, :, :3]
        
def test_color_inversion():

    kernel = color_inversion_kernel
    width, height = 2, 2
    image = torch.tensor([
                    [[255, 0, 0, 255], [0, 255, 0, 255]],
                    [[0, 0, 255, 255], [128, 128, 128, 255]]
                ], dtype=torch.uint8, device="cuda").flatten()
    ref_image = image.clone()
    reference_impl(ref_image,width,height)

    kernel([(width*height+256-1)//256],[256],image.data_ptr(), width, height)


    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

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
    
   

    check_all_close(ref_image, image)



if __name__ == "__main__":
    test_color_inversion()