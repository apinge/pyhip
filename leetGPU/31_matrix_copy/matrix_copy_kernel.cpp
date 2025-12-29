#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>


__global__ void matrix_copy_kernel(const float* A, float* B, int N) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id<N*N){
        B[id] = A[id];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
// extern "C" void solve(const float* A, float* B, int N) {
//     int total = N * N;
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
//     copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
//     cudaDeviceSynchronize();
// }
