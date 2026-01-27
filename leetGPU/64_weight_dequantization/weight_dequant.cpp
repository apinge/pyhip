#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>


__global__ void __launch_bounds__(256, 1) weight_dequant_hip_vec4(
    const float* __restrict__ X, const float* __restrict__ S, float* __restrict__ Y, const int M, const int N, const int TILE_SIZE,const int s_cols)
{
   // 每个线程处理 4 个连续列
    int col_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col_base < N) {
        // 只有当这一组 4 个元素都属于同一个 TILE 时，性能最高
        // 由于 TILE_SIZE 是 16 的倍数，且 col_base 是 4 的倍数，
        // 只要 col_base 不是 TILE_SIZE 的边缘，它们就一定属于同一个 S 元素。
        
        // 这里的处理逻辑：
        // 如果剩余宽度 >= 4，直接向量化加载
        //printf("row %d col_base %d\n",row,col_base);
        if (col_base + 3 < N) {
            float scale = S[(row / TILE_SIZE) * s_cols + (col_base / TILE_SIZE)];
            float4 x_vec = reinterpret_cast<const float4*>(&X[row * N + col_base])[0];
            float4 y_vec;
            y_vec.x = x_vec.x * scale;
            y_vec.y = x_vec.y * scale;
            y_vec.z = x_vec.z * scale;
            y_vec.w = x_vec.w * scale;
            reinterpret_cast<float4*>(&Y[row * N + col_base])[0] = y_vec;
        } 
        // 否则，处理剩余的 1~3 个元素（尾巴）
        else {
            for (int i = 0; i < (N - col_base); ++i) {
                int curr_col = col_base + i;
                float scale = S[(row / TILE_SIZE) * s_cols + (curr_col / TILE_SIZE)];
                Y[row * N + curr_col] = X[row * N + curr_col] * scale;
            }
        }
    }


}

//通用版本 不容易错 朴素实现
__global__ void __launch_bounds__(256, 1) weight_dequant_hip_native(
    const float* __restrict__ X, const float* __restrict__ S, float* __restrict__ Y, const int M, const int N, const int TILE_SIZE,const int s_cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // __shared__ float s_tile[4];

   
    if (row < M && col < N) {
        int pos = row * N + col;
        int s_row = row / TILE_SIZE;
        int s_col = col / TILE_SIZE;
        float scale  =S[s_row * s_cols + s_col];
       // Y[pos] = X[pos] * scale;
        Y[pos] = X[pos] * scale;

        // reinterpret_cast<float4*>(&Y[row * N + col])[0] = y_vec;
    }


}


// only for thread block 16X16
__global__ void __launch_bounds__(256, 1) weight_dequant_hip_16x16(
    const float* __restrict__ X, const float* __restrict__ S, float* __restrict__ Y, const int M, const int N, const int TILE_SIZE,const int s_cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float s_tile[4];

    if( threadIdx.x ==0 && threadIdx.y ==0){
        int s_row = row / TILE_SIZE;
        int s_col = col / TILE_SIZE;
        s_tile[0] =S[s_row * s_cols + s_col];
    }
    __syncthreads(); // 一定要加 确保LDS同步完成 这是threadblock内的同步
    if (row < M && col < N) {
        int pos = row * N + col;
       // Y[pos] = X[pos] * scale;
        Y[pos] = X[pos] * s_tile[0];

        // reinterpret_cast<float4*>(&Y[row * N + col])[0] = y_vec;
    }


}

__global__ void __launch_bounds__(256, 1) weight_dequant_hip_lds(
    const float* X, const float* S, float* Y, 
    const int M, const int N, const int TILE_SIZE, const int s_cols)
{
    // 每个 Block 只需要缓存它覆盖到的那一小块 S
    // blockDim 是 (16, 16)，TILE_SIZE >= 16
    __shared__ float s_cache[4]; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;


    if (ty == 0 && tx < 4) {
   
        int r_off = (tx / 2) * 15; 
        int c_off = (tx % 2) * 15;
        int s_r = (row + r_off) / TILE_SIZE;
        int s_c = (col + c_off) / TILE_SIZE;
        if (s_r < (M+TILE_SIZE-1)/TILE_SIZE && s_c < s_cols)
            s_cache[tx] = S[s_r * s_cols + s_c];
    }
    __syncthreads();

    if (row < M && col < N) {

        int s_idx = ((row % TILE_SIZE + ty >= TILE_SIZE) ? 2 : 0) + 
                    ((col % TILE_SIZE + tx >= TILE_SIZE) ? 1 : 0);
        float scale = S[(row / TILE_SIZE) * s_cols + (col / TILE_SIZE)];
        Y[row * N + col] = X[row * N + col] * scale;
    }
}

// input, kernel, output are device pointers
// extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE)  {

//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
//     int  s_cols = (N + TILE_SIZE - 1) / TILE_SIZE;
//     weight_dequant_hip<<<blocksPerGrid, threadsPerBlock>>>(X,S,Y,M,N,TILE_SIZE,s_cols);
//     cudaDeviceSynchronize();

//                       }