/*
 * Gemma RMSNorm template kernels for pyhip (standalone, no torch).
 * Same logic as aiter csrc/kernels/gemma_norm_kernels.cu; T = __half or __bf16.
 * Use from gemma_norm.cpp via non-template __global__ dispatch (gemma_rmsnorm_fp16/bf16, etc.).
 */
#ifndef GEMMA_NORM_KERNELS_CUH_
#define GEMMA_NORM_KERNELS_CUH_

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <type_traits>

namespace {

template <typename T>
__forceinline__ __device__ float dtype2acctype(T x) { return x; }
template <>
__forceinline__ __device__ float dtype2acctype<__half>(__half x) { return __half2float(x); }
template <>
__forceinline__ __device__ float dtype2acctype<__hip_bfloat16>(__hip_bfloat16 x) { return __bfloat162float(x); }
template <>
__forceinline__ __device__ float dtype2acctype<__bf16>(__bf16 x) { return __bfloat162float(x); }

template <typename T>
__forceinline__ __device__ T acctype2dtype(float x) { return x; }
template <>
__forceinline__ __device__ __half acctype2dtype<__half>(float x) { return __float2half(x); }
template <>
__forceinline__ __device__ __hip_bfloat16 acctype2dtype<__hip_bfloat16>(float x) { return __float2bfloat16(x); }
template <>
__forceinline__ __device__ __bf16 acctype2dtype<__bf16>(float x) { return __float2bfloat16(x); }

constexpr uint32_t WARP_SIZE = 64;
constexpr uint32_t VEC_SIZE = 8;

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(T1 x, T2 y) {
  return (x + y - 1) / y;
}

namespace gemma_norm {

__forceinline__ __device__ float rsqrt(float x) { return __frsqrt_rn(x); }

__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  return __shfl_xor(x, lane_mask, WARP_SIZE);
}

template <size_t N>
struct vec_t_float {
  float d[N];
  __device__ __forceinline__ float& operator[](size_t i) { return d[i]; }
  __device__ __forceinline__ void fill(float val) {
    for (size_t i = 0; i < N; ++i) d[i] = val;
  }
  __device__ __forceinline__ void load(const float* ptr) {
    for (size_t i = 0; i < N; ++i) d[i] = ptr[i];
  }
  __device__ __forceinline__ void store(float* ptr) const {
    for (size_t i = 0; i < N; ++i) ptr[i] = d[i];
  }
};

template <typename T>
struct alignas(16) VecT8 {
  T d[VEC_SIZE];

  __device__ __forceinline__ void load(const T* ptr) {
    using VecType = uint4;
    *reinterpret_cast<VecType*>(d) = *reinterpret_cast<const VecType*>(ptr);
  }

  __device__ __forceinline__ void store(T* ptr) const {
    using VecType = uint4;
    *reinterpret_cast<VecType*>(ptr) = *reinterpret_cast<const VecType*>(d);
  }

  __device__ __forceinline__ void fill(T val) {
#pragma unroll
    for (int i = 0; i < static_cast<int>(VEC_SIZE); ++i) {
      d[i] = val;
    }
  }

  __device__ __forceinline__ T& operator[](size_t i) { return d[i]; }
  __device__ __forceinline__ const T& operator[](size_t i) const { return d[i]; }
};

}  // namespace gemma_norm

// --------------- template device implementation (single place for kernel logic) ---------------

template <typename T>
__device__ void gemma_rmsnorm_device(T* __restrict__ output,
                                     const T* __restrict__ input,
                                     const T* __restrict__ weight,
                                     uint32_t hidden_size,
                                     float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride_input = hidden_size;
  const uint32_t stride_output = hidden_size;
  const float weight_bias = 1.f;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * WARP_SIZE;
  const uint32_t num_threads = num_warps * WARP_SIZE;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    gemma_norm::VecT8<T> input_vec;
    input_vec.fill(acctype2dtype<T>(0.f));
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float v = dtype2acctype<T>(input_vec[j]);
      sum_sq += v * v;
    }
  }

#pragma unroll
  for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = gemma_norm::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    gemma_norm::VecT8<T> input_vec, weight_vec, output_vec;
    input_vec.fill(acctype2dtype<T>(0.f));
    weight_vec.fill(acctype2dtype<T>(0.f));
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = acctype2dtype<T>(dtype2acctype<T>(input_vec[j]) * rms_rcp *
                                       (weight_bias + dtype2acctype<T>(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride_output + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }
}

template <typename T>
__device__ void gemma_fused_add_rmsnorm_device(T* __restrict__ input,
                                               T* __restrict__ residual,
                                               const T* __restrict__ weight,
                                               uint32_t hidden_size,
                                               float eps) {
  const uint32_t d = hidden_size;
  const uint32_t stride_input = hidden_size;
  const uint32_t stride_residual = hidden_size;
  const float weight_bias = 1.f;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * WARP_SIZE;
  const uint32_t num_threads = num_warps * WARP_SIZE;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4u) * 4;

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    gemma_norm::VecT8<T> input_vec, residual_vec;
    gemma_norm::vec_t_float<VEC_SIZE> x_vec;
    input_vec.fill(acctype2dtype<T>(0.f));
    residual_vec.fill(acctype2dtype<T>(0.f));
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = dtype2acctype<T>(input_vec[j]) + dtype2acctype<T>(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = acctype2dtype<T>(x);
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

#pragma unroll
  for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sum_sq += gemma_norm::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = gemma_norm::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    gemma_norm::VecT8<T> input_vec, weight_vec;
    gemma_norm::vec_t_float<VEC_SIZE> x_vec;
    weight_vec.fill(acctype2dtype<T>(0.f));
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = acctype2dtype<T>(x_vec[j] * rms_rcp *
                                      (weight_bias + dtype2acctype<T>(weight_vec[j])));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }
}

}  // namespace

#endif /* GEMMA_NORM_KERNELS_CUH_ */
