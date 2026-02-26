/*
 * Gemma norm __global__ kernels for pyhip.module().
 * Run Python from this directory; pyhip compiles this .cpp on the fly (no pre-build).
 * Uses gemma_norm_hip.cuh for vec types and helpers; exposes fp16 and bf16 kernels.
 */
#include "gemma_norm_hip.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

constexpr uint32_t VEC_SIZE = 8;

// Gemma RMSNorm: one block per row. grid = (batch_size), block = (32, num_warps).
// output = (input / RMS(input)) * (1 + weight). Strides = hidden_size.
__global__ void gemma_rmsnorm_fp16(__half* __restrict__ output,
                                    const __half* __restrict__ input,
                                    const __half* __restrict__ weight,
                                    uint32_t hidden_size, float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride_input = hidden_size;
  const uint32_t stride_output = hidden_size;
  const float weight_bias = 1.f;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = gemma_norm::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    gemma_norm::vec_t_half<VEC_SIZE> input_vec;
    input_vec.fill(__float2half(0.f));
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float v = __half2float(input_vec[j]);
      sum_sq += v * v;
    }
  }

#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = gemma_norm::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    gemma_norm::vec_t_half<VEC_SIZE> input_vec, weight_vec, output_vec;
    input_vec.fill(__float2half(0.f));
    weight_vec.fill(__float2half(0.f));
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = __float2half(__half2float(input_vec[j]) * rms_rcp *
                                   (weight_bias + __half2float(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                       thread_id * VEC_SIZE);
    }
  }
}

// Gemma FusedAdd RMSNorm: residual += input; input = (residual / RMS(residual)) * (1 + weight).
__global__ void gemma_fused_add_rmsnorm_fp16(__half* __restrict__ input,
                                             __half* __restrict__ residual,
                                             const __half* __restrict__ weight,
                                             uint32_t hidden_size, float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride_input = hidden_size;
  const uint32_t stride_residual = hidden_size;
  const float weight_bias = 1.f;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = gemma_norm::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + gemma_norm::ceil_div(num_warps, 4u) * 4;

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    gemma_norm::vec_t_half<VEC_SIZE> input_vec, residual_vec;
    input_vec.fill(__float2half(0.f));
    residual_vec.fill(__float2half(0.f));
    gemma_norm::vec_t_float<VEC_SIZE> x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = __half2float(input_vec[j]) + __half2float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = __float2half(x);
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                         thread_id * VEC_SIZE);
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = gemma_norm::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    gemma_norm::vec_t_half<VEC_SIZE> input_vec, weight_vec;
    gemma_norm::vec_t_float<VEC_SIZE> x_vec;
    input_vec.fill(__float2half(0.f));
    weight_vec.fill(__float2half(0.f));
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = __float2half(x_vec[j] * rms_rcp *
                                  (weight_bias + __half2float(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE +
                     thread_id * VEC_SIZE);
    }
  }
}

// -----------------------------------------------------------------------------
// BF16 versions (same logic as fp16, __bf16 + __bfloat162float / __float2bfloat16)
// -----------------------------------------------------------------------------
struct vec_bf16_8 {
  uint4 data;
  __device__ __forceinline__ __bf16& operator[](uint32_t i) { return ((__bf16*)&data)[i]; }
  __device__ __forceinline__ void load(const __bf16* ptr) { data = *reinterpret_cast<const uint4*>(ptr); }
  __device__ __forceinline__ void store(__bf16* ptr) const { *reinterpret_cast<uint4*>(ptr) = data; }
};

__global__ void gemma_rmsnorm_bf16(__bf16* __restrict__ output,
                                   const __bf16* __restrict__ input,
                                   const __bf16* __restrict__ weight,
                                   uint32_t hidden_size, float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride_input = hidden_size;
  const uint32_t stride_output = hidden_size;
  const float weight_bias = 1.f;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = gemma_norm::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_bf16_8 input_vec;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    } else {
      *reinterpret_cast<uint4*>(&input_vec.data) = uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float v = __bfloat162float(input_vec[j]);
      sum_sq += v * v;
    }
  }

#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = gemma_norm::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_bf16_8 input_vec, weight_vec, output_vec;
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    } else {
      *reinterpret_cast<uint4*>(&input_vec.data) = uint4{0, 0, 0, 0};
      *reinterpret_cast<uint4*>(&weight_vec.data) = uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = __float2bfloat16(__bfloat162float(input_vec[j]) * rms_rcp *
                                       (weight_bias + __bfloat162float(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                       thread_id * VEC_SIZE);
    }
  }
}

__global__ void gemma_fused_add_rmsnorm_bf16(__bf16* __restrict__ input,
                                             __bf16* __restrict__ residual,
                                             const __bf16* __restrict__ weight,
                                             uint32_t hidden_size, float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride_input = hidden_size;
  const uint32_t stride_residual = hidden_size;
  const float weight_bias = 1.f;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = gemma_norm::ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + gemma_norm::ceil_div(num_warps, 4u) * 4;

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_bf16_8 input_vec, residual_vec;
    float x_buf[VEC_SIZE];
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    } else {
      *reinterpret_cast<uint4*>(&input_vec.data) = uint4{0, 0, 0, 0};
      *reinterpret_cast<uint4*>(&residual_vec.data) = uint4{0, 0, 0, 0};
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = __bfloat162float(input_vec[j]) + __bfloat162float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = __float2bfloat16(x);
      x_buf[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                         thread_id * VEC_SIZE);
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        smem_x[i * num_threads * VEC_SIZE + thread_id * VEC_SIZE + j] = x_buf[j];
      }
    }
  }

#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = gemma_norm::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_bf16_8 weight_vec, output_vec;
    float x_buf[VEC_SIZE];
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        x_buf[j] = smem_x[i * num_threads * VEC_SIZE + thread_id * VEC_SIZE + j];
      }
    } else {
      *reinterpret_cast<uint4*>(&weight_vec.data) = uint4{0, 0, 0, 0};
      for (uint32_t j = 0; j < VEC_SIZE; j++) x_buf[j] = 0.f;
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = __float2bfloat16(x_buf[j] * rms_rcp *
                                      (weight_bias + __bfloat162float(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE +
                      thread_id * VEC_SIZE);
    }
  }
}
