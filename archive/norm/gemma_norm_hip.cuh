/*
 * Standalone Gemma norm kernels for HIP (ROCm).
 * Extracted from FlashInfer include/flashinfer/attention/generic/norm.cuh
 * and gpu_iface (HIP backend). Only fp16 (__half), GemmaRMSNorm and GemmaFusedAddRMSNorm.
 */
#ifndef GEMMA_NORM_HIP_CUH_
#define GEMMA_NORM_HIP_CUH_

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h> // for bfloat16
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>

#define GEMMA_NORM_HIP_CALL(func) \
  do {                            \
    hipError_t e = (func);        \
    if (e != hipSuccess) return e; \
  } while (0)

namespace gemma_norm {

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(T1 x, T2 y) {
  return (x + y - 1) / y;
}

__forceinline__ __device__ float rsqrt(float x) {
  return __frsqrt_rn(x);
}

__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  return __shfl_xor(x, lane_mask, 32);
}

/* ----- vec_t half 1,2,4,8 (HIP __half) ----- */
template <size_t N>
struct vec_t_half;
template <>
struct vec_t_half<1> {
  __half data;
  __device__ __forceinline__ __half& operator[](size_t i) { return ((__half*)(&data))[i]; }
  __device__ __forceinline__ void fill(__half val) { data = val; }
  __device__ __forceinline__ void load(const __half* ptr) { data = *ptr; }
  __device__ __forceinline__ void store(__half* ptr) const { *ptr = data; }
};
template <>
struct vec_t_half<2> {
  __half2 data;
  __device__ __forceinline__ __half& operator[](size_t i) { return ((__half*)(&data))[i]; }
  __device__ __forceinline__ void fill(__half val) { data = __half2(val, val); }
  __device__ __forceinline__ void load(const __half* ptr) { data = *((__half2*)ptr); }
  __device__ __forceinline__ void store(__half* ptr) const { *((__half2*)ptr) = data; }
};
template <>
struct vec_t_half<4> {
  uint2 data;
  __device__ __forceinline__ __half& operator[](size_t i) { return ((__half*)(&data))[i]; }
  __device__ __forceinline__ void fill(__half val) {
    __half2 v = __half2(val, val);
    *(__half2*)(&data.x) = v;
    *(__half2*)(&data.y) = v;
  }
  __device__ __forceinline__ void load(const __half* ptr) { data = *((uint2*)ptr); }
  __device__ __forceinline__ void store(__half* ptr) const { *((uint2*)ptr) = data; }
};
template <>
struct vec_t_half<8> {
  int4 data;
  __device__ __forceinline__ __half& operator[](size_t i) { return ((__half*)&data)[i]; }
  __device__ __forceinline__ void fill(__half val) {
    __half2 v = __half2(val, val);
    *(__half2*)(&data.x) = v;
    *(__half2*)(&data.y) = v;
    *(__half2*)(&data.z) = v;
    *(__half2*)(&data.w) = v;
  }
  __device__ __forceinline__ void load(const __half* ptr) { data = *((int4*)ptr); }
  __device__ __forceinline__ void store(__half* ptr) const { *((int4*)ptr) = data; }
};

/* ----- vec_t float 1,2,4,8 ----- */
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

template <uint32_t VEC_SIZE>
__global__ void RMSNormKernel(__half* __restrict__ input, __half* __restrict__ weight,
                              __half* __restrict__ output, const uint32_t d,
                              const uint32_t stride_input, const uint32_t stride_output,
                              float weight_bias, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_half<VEC_SIZE> input_vec;
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
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_half<VEC_SIZE> input_vec, weight_vec, output_vec;
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

template <uint32_t VEC_SIZE>
__global__ void FusedAddRMSNormKernel(__half* __restrict__ input, __half* __restrict__ residual,
                                      __half* __restrict__ weight, const uint32_t d,
                                      const uint32_t stride_input, const uint32_t stride_residual,
                                      float weight_bias, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4) * 4;

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_half<VEC_SIZE> input_vec, residual_vec;
    input_vec.fill(__float2half(0.f));
    residual_vec.fill(__float2half(0.f));
    vec_t_float<VEC_SIZE> x_vec;
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
    sum_sq += shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t_half<VEC_SIZE> input_vec, weight_vec;
    vec_t_float<VEC_SIZE> x_vec;
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

#define DISPATCH_VEC_SIZE(vec_size, VEC_SIZE, ...) \
  switch (vec_size) {                               \
    case 8: {                                        \
      constexpr uint32_t VEC_SIZE = 8;                \
      __VA_ARGS__                                    \
      break;                                         \
    }                                                \
    case 4: {                                        \
      constexpr uint32_t VEC_SIZE = 4;               \
      __VA_ARGS__                                    \
      break;                                         \
    }                                                \
    case 2: {                                        \
      constexpr uint32_t VEC_SIZE = 2;                \
      __VA_ARGS__                                    \
      break;                                         \
    }                                                \
    case 1: {                                        \
      constexpr uint32_t VEC_SIZE = 1;                \
      __VA_ARGS__                                    \
      break;                                         \
    }                                                \
    default:                                         \
      return hipErrorInvalidValue;                   \
  }

inline hipError_t GemmaRMSNorm(__half* input, __half* weight, __half* output,
                               uint32_t batch_size, uint32_t d, uint32_t stride_input,
                               uint32_t stride_output, float eps = 1e-5f, bool enable_pdl = false,
                               hipStream_t stream = 0) {
  (void)enable_pdl;  // HIP has no PDL equivalent
  const uint32_t vec_size = std::gcd(16u / sizeof(__half), d);
  const uint32_t block_size = std::min(1024u, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32u);
  const uint32_t smem_size = num_warps * sizeof(float);
  const float weight_bias = 1.f;

  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);

  DISPATCH_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = RMSNormKernel<VEC_SIZE>;
    void* args[] = {&input, &weight, &output, &d, &stride_input, &stride_output,
                    const_cast<float*>(&weight_bias), const_cast<float*>(&eps)};
    hipError_t e = hipFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                       hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (e != hipSuccess) return e;
    return hipLaunchKernel(reinterpret_cast<void*>(kernel), nblks, nthrs, args, smem_size, stream);
  });
  return hipSuccess;
}

inline hipError_t GemmaFusedAddRMSNorm(__half* input, __half* residual, __half* weight,
                                       uint32_t batch_size, uint32_t d, uint32_t stride_input,
                                       uint32_t stride_residual, float eps = 1e-5f,
                                       bool enable_pdl = false, hipStream_t stream = 0) {
  (void)enable_pdl;
  const uint32_t vec_size = std::gcd(16u / sizeof(__half), d);
  const uint32_t block_size = std::min(1024u, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32u);
  const uint32_t smem_size = (ceil_div(num_warps, 4u) * 4 + d) * sizeof(float);
  const float weight_bias = 1.f;

  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);

  DISPATCH_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormKernel<VEC_SIZE>;
    void* args[] = {&input, &residual, &weight, &d, &stride_input, &stride_residual,
                    const_cast<float*>(&weight_bias), const_cast<float*>(&eps)};
    hipError_t e = hipFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                       hipFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (e != hipSuccess) return e;
    return hipLaunchKernel(reinterpret_cast<void*>(kernel), nblks, nthrs, args, smem_size, stream);
  });
  return hipSuccess;
}

}  // namespace gemma_norm

#endif  // GEMMA_NORM_HIP_CUH_
