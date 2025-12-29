#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>


__global__ void reverse_array_kernel(float* input, int N) {

    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id<N/2){
        float tmp = input[id];
        input[id] = input[N-1-id];
        input[N-1-id] = tmp;
    }
}

// input is device pointer
// extern "C" void solve(float* input, int N) {
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

//     reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
//     cudaDeviceSynchronize();
// }
