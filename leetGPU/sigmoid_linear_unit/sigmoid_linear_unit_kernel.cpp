#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>


__global__ void silu_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid<N){
        output[tid] = input[tid]/(1.0+__expf(-input[tid]));
    }
}

// input, output are device pointers
// extern "C" void solve(const float* input, float* output, int N) {
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

//     silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
//     cudaDeviceSynchronize();
// }
