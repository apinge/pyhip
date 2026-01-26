#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>


// __global__ void convolution_2d_kernel(const float* input, const float* kernel, float* output, int input_rows,
//                       int input_cols, int kernel_rows, int kernel_cols) {
//     int col = blockIdx.x * blockDim.x + threadIdx.x; 
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int output_cols = input_cols - kernel_cols+1;
//     int output_rows = input_rows - kernel_rows+1;
//     if(row < output_rows && col < output_cols){
//         float sum = 0.0f;
// #pragma unroll
//         for(int j =0;j<kernel_cols;++j)
// #pragma unroll
//             for(int i =0;i<kernel_rows;++i){
//                 sum += kernel[i*kernel_cols+j] * input[(i+row)*input_cols+col+j];
//             }

//         output[row*output_cols+col] = sum;
//     }

//  }
// #define KERNEL_ROWS (15)
//#define KERNEL_COLS (15) //只展开一个效果好
template <int K_COLS>
__device__ __forceinline__ void convolution_core(
    const float* __restrict__ input, 
    const float* __restrict__ kernel, 
    float* __restrict__ output, 
    const int input_rows, const int input_cols, 
    const int kernel_rows, const int kernel_cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    int output_cols = input_cols - kernel_cols + 1;
    int output_rows = input_rows - kernel_rows + 1;

    if (row < output_rows && col < output_cols) {
        float sum = 0.0f;

//#pragma unroll
        for (int i = 0; i < kernel_rows; ++i) {
//      for (int i = 0; i < KERNEL_ROWS; ++i) {
            // 预计算行偏移，减少乘法次数
            int input_row_offset = (i + row) * input_cols + col;
 
            int kernel_row_offset = i*K_COLS;
            
//#pragma unroll 4 // 只进行适度展开，防止寄存器溢出
#pragma unroll
       for (int j = 0; j < K_COLS; ++j) {
//            for (int j = 0; j < kernel_cols; ++j) {
                sum += kernel[kernel_row_offset + j] * input[input_row_offset + j];
            }
        }
        output[row * output_cols + col] = sum;
    }
}

#if __cplusplus >= 201703L // support cpp 17
    template <int CURRENT_K>
__device__ void dispatch(
        int target_k, const float* input, const float* kernel, float* output,
        int rows, int cols, int k_rows, int k_cols)  {
    if constexpr (CURRENT_K > 0) {
        if (target_k == CURRENT_K) {
            convolution_core<CURRENT_K>(input, kernel, output, rows, cols, k_rows, k_cols);
        } else {
            dispatch<CURRENT_K - 1>(target_k, input, kernel, output, rows, cols, k_rows, k_cols);
        }
    }
}
#else
template <int CURRENT_K>
struct ConvDispatcher {
    static __device__ __forceinline__ void dispatch(
        int target_k, const float* input, const float* kernel, float* output,
        int rows, int cols, int k_rows, int k_cols) 
    {
        if (target_k == CURRENT_K) {
            convolution_core<CURRENT_K>(input, kernel, output, rows, cols, k_rows, k_cols);
        } else {
            ConvDispatcher<CURRENT_K - 1>::dispatch(target_k, input, kernel, output, rows, cols, k_rows, k_cols);
        }
    }
};

template <>
struct ConvDispatcher<0> {
    static __device__ __forceinline__ void dispatch(...) {
        // 可以留空或者报错
    }
};
#endif

__global__ void convolution_2d_optimized(
    const float* input, const float* kernel, float* output,
    int input_rows, int input_cols,
    int kernel_rows, int kernel_cols
) {

    if (kernel_cols >= 1 && kernel_cols <= 31) {
#if __cplusplus >= 201703L // support cpp 17
        dispatch<31>(
                kernel_cols, input, kernel, output, 
                input_rows, input_cols, kernel_rows, kernel_cols
            );
#else
            ConvDispatcher<31>::dispatch(
                kernel_cols, input, kernel, output, 
                input_rows, input_cols, kernel_rows, kernel_cols
            );
#endif
        }

}

// input, kernel, output are device pointers
// extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
//                       int input_cols, int kernel_rows, int kernel_cols) {
//     int output_rows = input_rows-kernel_rows+1;
//     int output_cols = input_cols-output_cols+1;
//     dim3 threadsPerBlock(32, 32);
//     dim3 blocksPerGrid((output_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                        (output_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     convolution_2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
//     cudaDeviceSynchronize();

//                       }