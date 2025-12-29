#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

__global__ void post_process_kernel(float* result, float a, float b, int n_samples) {
    // Only one thread do this
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float val = *result;
        val /= n_samples;
        val *= (b - a);
        *result = val;
    }
}

__global__ void monte_carlo_integration(const float* y_samples, float* result, float a, float b, int n_samples) {
    int tid = threadIdx.x + blockDim.x*blockIdx;
    if(tid<n_samples){
        result[0] +=  (b-a)*(y_samples[tid]/n_samples);
    }
}

// y_samples, result are device pointers
// extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (n_samples + threadsPerBlock - 1) / threadsPerBlock;

//     monte_carlo_integration<<<blocksPerGrid, threadsPerBlock>>>(y_samples, result, a,b,n_samples);
 
//     cudaDeviceSynchronize();
// }