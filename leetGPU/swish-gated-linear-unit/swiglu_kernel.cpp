#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>


__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid<halfN){
        float x = input[tid];
        float g = input[tid + halfN];
        // 显式使用 1.0f，并将计算逻辑展开以利于编译器优化
        float res = g * (x / (1.0f + __expf(-x)));
        output[tid] = res;
    }
}

// input, output are device pointers
// extern "C" void solve(const float* input, float* output, int N) {
//     int halfN = N / 2;
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

//     swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
//     cudaDeviceSynchronize();
// }

// input, output are device pointers
// extern "C" void solve(const float* input, float* output, int N) {
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

//     silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
//     cudaDeviceSynchronize();
// }
