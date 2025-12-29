#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id<N){
        if(input[id]==K)
            atomicAdd(output,1);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
// extern "C" void solve(const int* input, int* output, int N, int K) {
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

//     count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
//     cudaDeviceSynchronize();
// }