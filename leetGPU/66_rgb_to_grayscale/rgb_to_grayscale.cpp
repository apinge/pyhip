
#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    // 我们需要加载 256 * 3 个 float 到 LDS
    // __shared__ float tile[256 * 3];
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int total = width * height;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if(idx<total){
        int offset = idx*3;
        float r = input[offset];
        float g = input[offset+1];
        float b = input[offset+2];
        float gray = 0.299*r + 0.587*g + 0.114*b;
        output[idx] = gray;
    }
    
}

// input, output are device pointers
// extern "C" void solve(const float* input, float* output, int width, int height) {
//     int total_pixels = width * height;
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

//     rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
//     cudaDeviceSynchronize();
// }
