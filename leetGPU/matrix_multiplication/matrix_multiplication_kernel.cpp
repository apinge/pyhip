#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

__global__ void matrix_multiplication_kernel(const __fp16* A, const __fp16* B, __fp16* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float sum = 0.0f; 
        for (int z = 0; z < N; ++z) {
            sum += A[row * N + z] * B[z * K + col];
        }
        C[row * K + col] = sum; 
    }
}