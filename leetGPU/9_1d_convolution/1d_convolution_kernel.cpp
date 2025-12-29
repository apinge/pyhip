#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>


__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id<input_size-kernel_size+1){
       // printf("id:%d input_size%d, kernel_size %d\n",id,input_size,kernel_size);
        float sum = 0.0f;
        for(int k = 0;k<kernel_size; ++k){
            sum += input[id+k]*kernel[k];
           // printf("%d\tid+k%d\tk%d\n",id,id+k,k);
        }
         output[id] = sum;
    }

                                      }

// // input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
// extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
//                       int kernel_size) {
//     int output_size = input_size - kernel_size + 1;
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

//     convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
//                                                               kernel_size);
//     cudaDeviceSynchronize();
// }
