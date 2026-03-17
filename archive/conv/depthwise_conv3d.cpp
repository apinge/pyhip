// Extracted from PyTorch DepthwiseConv3d.cu (lines 25-95), no torch dependency.
// Uses raw pointers and grid-stride loop; invokable via pyhip.module.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Grid-stride loop equivalent to CUDA_KERNEL_LOOP
#define HIP_KERNEL_LOOP(i, n)                                                       \
  for (int64_t i = (int64_t)(blockIdx.x) * blockDim.x + threadIdx.x; i < (n);        \
       i += (int64_t)(blockDim.x) * gridDim.x)

__global__ void conv_depthwise3d_cuda_kernel(
    const void* input_void,
    void* output_void,
    const void* kernel_void,
    const void* bias_void,
    int batch,
    int iC,
    int oC,
    int iT,
    int iH,
    int iW,
    int oT,
    int oH,
    int oW,
    int kT,
    int kH,
    int kW,
    int strideT,
    int strideH,
    int strideW,
    int paddingT,
    int paddingH,
    int paddingW,
    int dilationT,
    int dilationH,
    int dilationW)
{
  const __fp16* input = static_cast<const __fp16*>(input_void);
  __fp16* output = static_cast<__fp16*>(output_void);
  const __fp16* kernel = static_cast<const __fp16*>(kernel_void);
  const __fp16* bias = static_cast<const __fp16*>(bias_void);

  const int channel_multiplier = oC / iC;
  const int num_output = batch * oC * oT * oH * oW;

  HIP_KERNEL_LOOP(index, num_output) {
    const int out_col = index % oW;
    const int out_row = (index / oW) % oH;
    const int out_frame = (index / oW / oH) % oT;
    const int out_channel = (index / oW / oH / oT) % oC;
    const int b = index / oW / oH / oT / oC;

    const int in_channel = out_channel / channel_multiplier;

    const int in_col_start = out_col * strideW - paddingW;
    const int in_row_start = out_row * strideH - paddingH;
    const int in_frame_start = out_frame * strideT - paddingT;

    float sum = 0.0f;
    const __fp16* kernel_ptr = kernel + out_channel * kT * kH * kW;
    const int input_stride_c = iT * iH * iW;
    const int input_stride_t = iH * iW;
    const __fp16* input_ptr = input + b * iC * input_stride_c
                              + in_channel * input_stride_c
                              + in_frame_start * input_stride_t
                              + in_row_start * iW
                              + in_col_start;

    for (int k_frame = 0; k_frame < kT; ++k_frame) {
      const int in_frame = in_frame_start + k_frame * dilationT;
      for (int k_row = 0; k_row < kH; ++k_row) {
        const int in_row = in_row_start + k_row * dilationH;
        for (int k_col = 0; k_col < kW; ++k_col) {
          const float op1 = (float)*(kernel_ptr++);
          const int in_col = in_col_start + k_col * dilationW;
          if (in_frame >= 0 && in_row >= 0 && in_col >= 0 &&
              in_frame < iT && in_row < iH && in_col < iW) {
            sum += op1 * (float)*input_ptr;
          }
          input_ptr += dilationW;
        }
        input_ptr += iW * dilationH - kW * dilationW;
      }
      input_ptr += iW * (iH * dilationT - kH * dilationH);
    }
    if (bias != nullptr) {
      sum += (float)bias[out_channel];
    }

    const int output_stride_c = oT * oH * oW;
    const int output_stride_t = oH * oW;
    output[b * oC * output_stride_c + out_channel * output_stride_c
           + out_frame * output_stride_t + out_row * oW + out_col] = (__fp16)sum;
  }
}
