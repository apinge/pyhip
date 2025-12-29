#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>



__global__ void color_inversion_kernel(unsigned char* image, int width, int height) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; 
    if(id<width*height){
        int offset = id*4;
        image[0+offset]= 255-image[0+offset];
        image[1+offset]= 255-image[1+offset];
        image[2+offset]= 255-image[2+offset];
    }
}
// // image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
// extern "C" void solve(unsigned char* image, int width, int height) {
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

//     invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
//     cudaDeviceSynchronize();
// }