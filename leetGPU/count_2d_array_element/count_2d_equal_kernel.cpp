#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

// __global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
//     int col = threadIdx.x + blockDim.x*blockIdx.x;
//     int row = threadIdx.y + blockDim.y*blockIdx.y;
//     if(row<N && col<M){
//         if(input[row*M+col]==K){
//             atomicAdd(output,1);
//         }
//     }


// }


// kernel has problem!
// what is 
__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int col = threadIdx.x + blockDim.x*blockIdx.x;
    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int val = (row < N && col < M && input[row*M + col] == K);

    unsigned long long mask = __activemask();  // 必须是 64-bit
    int cnt = __reduce_add_sync(mask, val);
    /* __reduce_add_sync
    对 mask 中为 1 的所有 lane，
    把它们各自的 val 加在一起，
    然后 把这个和广播给 mask 中的每一个 lane
    */

    if ((threadIdx.x & 63) == 0 && (threadIdx.y &63)==0) {
        atomicAdd(output, cnt);
    }


}

// input, output are device pointers (i.e. pointers to memory on the GPU)
// extern "C" void solve(const int* input, int* output, int N, int M, int K) {
//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
//     cudaDeviceSynchronize();
// }