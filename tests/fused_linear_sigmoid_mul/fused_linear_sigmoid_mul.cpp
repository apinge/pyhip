
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include "hip/hip_runtime.h"
#include <cstdlib>

constexpr uint32_t WARP_SIZE = 64;
constexpr uint32_t VEC_SIZE = 8;

template <typename scalar_t>
struct __align__(16) vec8_t
{
  scalar_t x, y, z, w, u, v, s, t;

  __device__ vec8_t() : x(0), y(0), z(0), w(0), u(0), v(0), s(0), t(0) {}
  __device__ vec8_t(scalar_t x, scalar_t y, scalar_t z, scalar_t w, scalar_t u,
                      scalar_t v, scalar_t s, scalar_t t)
      : x(x), y(y), z(z), w(w), u(u), v(v), s(s), t(t) {}

  __device__ vec8_t operator*(const vec8_t& other) const
  {
    return vec8_t(x * other.x, y * other.y, z * other.z, w * other.w, u * other.u,
                  v * other.v, s * other.s, t * other.t);
  }

  __device__ vec8_t operator*(const float& scale) const
  {
    return vec8_t(x * scale, y * scale, z * scale, w * scale, u * scale, v * scale,
                  s * scale, t * scale);
  }

  __device__ vec8_t operator+(const vec8_t& other) const
  {
    return vec8_t(x + other.x, y + other.y, z + other.z, w + other.w, u + other.u,
                  v + other.v, s + other.s, t + other.t);
  }

  __device__ void operator+=(const vec8_t& other)
  {
    x += other.x;
    y += other.y;
    z += other.z;
    w += other.w;
    u += other.u;
    v += other.v;
    s += other.s;
    t += other.t;
  }

  __device__ scalar_t sum() const { return x + y + z + w + u + v + s + t; }
};

// 以下为逐元素 SiLU 占位；与 torch.sigmoid(linear)*m 对齐需在 vectorize 路径上改成 gate 标量等
__device__ __forceinline__ float silu_kernel(const __hip_bfloat16& x)
{
  const float xf = __bfloat162float(x);
  return xf * __builtin_amdgcn_rcpf(1.0f + expf(-xf));
}

__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
    return __shfl_xor(x, lane_mask, WARP_SIZE);
  }

  


__global__ void __launch_bounds__(256, 1) fused_linear_sigmoid_mul_kernel(
    const __hip_bfloat16* __restrict__ hidden_states,
    const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ shared_output,
    __hip_bfloat16* __restrict__ output,
    const int N,
    const int H)
{
  const vec8_t<__hip_bfloat16>* vectorized_in =
      reinterpret_cast<const vec8_t<__hip_bfloat16>*>(hidden_states);
  const vec8_t<__hip_bfloat16>* vectorized_weight =
      reinterpret_cast<const vec8_t<__hip_bfloat16>*>(weight);
  const vec8_t<__hip_bfloat16>* vectorized_shared_output =
      reinterpret_cast<const vec8_t<__hip_bfloat16>*>(shared_output);
  vec8_t<__hip_bfloat16>* vectorized_output =
      reinterpret_cast<vec8_t<__hip_bfloat16>*>(output);

  const int vec_hidden_size = 4096>> 3;
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  const uint32_t tid = tx + ty * 64;
  const uint32_t row_offset = vec_hidden_size * blockIdx.x;
  constexpr uint32_t rounds = 2;
  extern __shared__ float smem[];
  float dot_partial = 0.f;
#pragma unroll
  for (int i = 0; i < rounds; ++i)
  {
    const int32_t local_idx = i * (int32_t)(256) + (int32_t)tid;
    const int32_t vec_idx = (int32_t)row_offset + local_idx;
    vec8_t<__hip_bfloat16> curr_in = vectorized_in[vec_idx];
    vec8_t<__hip_bfloat16> curr_w = vectorized_weight[local_idx];

    const __hip_bfloat16* xi = reinterpret_cast<const __hip_bfloat16*>(&curr_in);
    const __hip_bfloat16* wi = reinterpret_cast<const __hip_bfloat16*>(&curr_w);
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j)
    {
      const float a = __bfloat162float(xi[j]);
      const float b = __bfloat162float(wi[j]);
      dot_partial += a * b;
    }
  }
#pragma unroll
  for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2)
  {
    dot_partial += shfl_xor_sync(dot_partial, offset);
  }

  smem[ty] = dot_partial;
  __syncthreads();
  if (tx == 0 && ty == 0)
  {
    float s = smem[0];
    for (uint32_t w = 1; w < blockDim.y; ++w)
      s += smem[w];
    smem[0] = s;
  }
  __syncthreads();
  const float dot = smem[0];
  const float sig = 1.0f / (1.0f + expf(-dot));
  #pragma unroll
  for (int i = 0; i < rounds; ++i)
  {
    const int32_t local_idx = i * (int32_t)(256) + (int32_t)tid;
    const int32_t vec_idx = (int32_t)row_offset + local_idx;
    vec8_t<__hip_bfloat16> curr_share_output = vectorized_shared_output[vec_idx];
    __hip_bfloat16* po = reinterpret_cast<__hip_bfloat16*>(&vectorized_output[vec_idx]);
    const __hip_bfloat16* pm = reinterpret_cast<const __hip_bfloat16*>(&curr_share_output);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j)
    {
      const float x = __bfloat162float(pm[j]) * sig;
      po[j] = __float2bfloat16(x);
    }
  }


}
