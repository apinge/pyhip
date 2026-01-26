#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>




__global__ void __launch_bounds__(256, 1) convolution_3d(
    const float* input, const float* kernel, float* output, int input_depth,
    int input_rows, int input_cols, int kernel_depth, int kernel_rows,int kernel_cols
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x; 
    int depth = blockIdx.y * blockDim.y + threadIdx.y;
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;
    int output_row_idx =  pos/output_cols;
    int output_col_idx = pos%output_cols;
    if(output_row_idx<output_rows && output_col_idx<output_cols){
        float sum = 0.0f;
        int output_depth = depth*output_rows*output_cols;
        int input_depth_tmp = input_cols*input_rows;
        int kernel_depth_tmp = kernel_cols*kernel_rows;
        for(int k = 0;k<kernel_depth;++k){
            int input_depth_offset = (k+depth)*input_depth_tmp;
            int kernel_depth_offset = k*kernel_depth_tmp;
         for (int i = 0; i < kernel_rows; ++i) {
            int input_row_offset =  input_depth_offset+(i + output_row_idx) * input_cols + output_col_idx; 
            int kernel_row_offset = kernel_depth_offset+ i*kernel_cols;

             for (int j = 0; j < kernel_cols; ++j) {
                sum += kernel[kernel_row_offset + j] * input[input_row_offset + j];
            }
        }
    }
        output[pos+output_depth] = sum;
    }

}

// // input, kernel, output are device pointers
// extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
//                       int input_rows, int input_cols, int kernel_depth, int kernel_rows,
//                       int kernel_cols) {}
