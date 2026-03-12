/*
 * Gemma norm kernels for pyhip: dispatch layer.
 * Template kernels live in gemma_norm_kernels.cuh (__device__ template).
 * Python/pytest cannot call templates; we expose non-template __global__ names
 * that pyhip discovers: gemma_rmsnorm_fp16, gemma_rmsnorm_bf16,
 * gemma_fused_add_rmsnorm_fp16, gemma_fused_add_rmsnorm_bf16.
 */
#include "gemma_norm_kernels.cuh"

// --------------- dispatch: __global__ wrappers that call template __device__ ---------------

__global__ void gemma_rmsnorm_fp16(__half* __restrict__ output,
                                   const __half* __restrict__ input,
                                   const __half* __restrict__ weight,
                                   uint32_t hidden_size,
                                   float eps) {
  gemma_rmsnorm_device<__half>(output, input, weight, hidden_size, eps);
}

__global__ void gemma_rmsnorm_bf16(__bf16* __restrict__ output,
                                   const __bf16* __restrict__ input,
                                   const __bf16* __restrict__ weight,
                                   uint32_t hidden_size,
                                   float eps) {
  gemma_rmsnorm_device<__bf16>(output, input, weight, hidden_size, eps);
}

__global__ void gemma_fused_add_rmsnorm_fp16(__half* __restrict__ input,
                                             __half* __restrict__ residual,
                                             const __half* __restrict__ weight,
                                             uint32_t hidden_size,
                                             float eps) {
  gemma_fused_add_rmsnorm_device<__half>(input, residual, weight, hidden_size, eps);
}

__global__ void gemma_fused_add_rmsnorm_bf16(__bf16* __restrict__ input,
                                             __bf16* __restrict__ residual,
                                             const __bf16* __restrict__ weight,
                                             uint32_t hidden_size,
                                             float eps) {
  gemma_fused_add_rmsnorm_device<__bf16>(input, residual, weight, hidden_size, eps);
}
