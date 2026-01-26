#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

__global__ void max_pooling(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding, int H_out, int W_out) {
  
// 1. 严格对应 Python 传入的维度
    int batch = blockIdx.x * blockDim.x + threadIdx.x; 
    int channel = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = blockIdx.z * blockDim.z + threadIdx.z;

    // 2. 边界检查
    if (batch < N && channel < C && pos < H_out * W_out) {
        
        // 3. 计算在当前平面内的输出坐标
        int pos_h = pos / W_out;
        int pos_w = pos % W_out;

        // 4. 计算输入窗口在原图的起始点 (stride 作用于此处)
        /*
        Stride 作用于： 输出坐标到输入坐标的映射映射逻辑（决定窗口在哪开始）。
        Kernel 内部循环： 步长永远是 1（决定窗口内如何采样）。
        */
        int h_start = pos_h * stride - padding;
        int w_start = pos_w * stride - padding;

       // float max_val = -3.40282e38f; // -FLT_MAX
        float max_val = -__FLT_MAX__;//这里有个坑点 __FLX_MIN__是最小正值 如果最小负值要用-__FLT_MAX__
        // 5. 计算该 Batch 和 Channel 的起始基础偏移
        // 输入是 (N, C, H, W)
        int in_base = batch * (C * H * W) + channel * (H * W);

        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int cur_h = h_start + i;
                int cur_w = w_start + j;

                // 检查是否在原图合法范围内 (考虑 padding)
                if (cur_h >= 0 && cur_h < H && cur_w >= 0 && cur_w < W) {
                    float val = input[in_base + cur_h * W + cur_w];
                    if (val > max_val) max_val = val;
         
                }
            }
        }

        // 6. 写入输出
        // 你的 output_tensor 对应 N*C*H_out*W_out 的线性空间
        int out_idx = batch * (C * H_out * W_out) + channel * (H_out * W_out) + pos;
        output[out_idx] = max_val;
    }

}


// input, output are device pointers (i.e. pointers to memory on the GPU)
// extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
//                       int kernel_size, int stride, int padding) {
       
//             int H_out = (H + 2 * padding - kernel_size) / stride + 1;
//             int W_out = (W + 2 * padding - kernel_size) / stride + 1;

//             dim3 threadsPerBlock(4, 16, 16);

       
//             dim3 blocksPerGrid(
//                 (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                 (C + threadsPerBlock.y - 1) / threadsPerBlock.y,
//                 (H_out * W_out + threadsPerBlock.z - 1) / threadsPerBlock.z
//             );

//             max_pooling<<<blocksPerGrid, threadsPerBlock>>>(
//                 d_input, d_output, 
//                 N, C, H, W, 
//                 kernel_size, stride, padding, 
//                 H_out, W_out
//             );
//                       }