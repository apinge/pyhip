#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>


__global__ void relu_kernel(const float* input, float* output, int N) {

    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id<N){
        if(input[id]<=0) output[id] = 0.0f;
        else output[id] = input[id]; 
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
// extern "C" void solve(const float* input, float* output, int N) {
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

//     relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
//     cudaDeviceSynchronize();
// }
