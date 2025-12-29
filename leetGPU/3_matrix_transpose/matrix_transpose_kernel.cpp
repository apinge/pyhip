#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>


__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < cols && row < rows){
        //printf("row %d col %d row*cols+col %d col*rows+row%d\n",row,col,row*cols+col,col*rows+row);
        output[col*rows+row] = input[row*cols+col];
    }
}

// // input, output are device pointers (i.e. pointers to memory on the GPU)
// extern "C" void solve(const float* input, float* output, int rows, int cols) {
//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
//     cudaDeviceSynchronize();
// }