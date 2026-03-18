
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h> // for bfloat16

using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using float16x8 = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

template <typename scalar_t>
struct __align__(16) vec8_t
{
  scalar_t x, y, z, w, u, v, s, t;

  __device__ vec8_t() : x(0), y(0), z(0), w(0), u(0), v(0), s(0), t(0) {}
  __device__ vec8_t(scalar_t x, scalar_t y, scalar_t z, scalar_t w, scalar_t u,
                    scalar_t v, scalar_t s, scalar_t t)
      : x(x), y(y), z(z), w(w), u(u), v(v), s(s), t(t) {}

  __device__ vec8_t operator*(const vec8_t &other) const
  {
    return vec8_t(x * other.x, y * other.y, z * other.z, w * other.w,
                  u * other.u, v * other.v, s * other.s, t * other.t);
  }

  __device__ vec8_t operator*(const float &scale) const
  {
    return vec8_t(x * scale, y * scale, z * scale, w * scale, u * scale,
                  v * scale, s * scale, t * scale);
  }

  __device__ vec8_t operator+(const vec8_t &other) const
  {
    return vec8_t(x + other.x, y + other.y, z + other.z, w + other.w,
                  u + other.u, v + other.v, s + other.s, t + other.t);
  }

  __device__ void operator+=(const vec8_t &other)
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

//   __device__ vec8_t& operator=(const vec8_t &other)
//   {
//     x = other.x;
//     y = other.y;
//     z = other.z;
//     w = other.w;
//     u = other.u;
//     v = other.v;
//     s = other.s;
//     t = other.t;
//     return *this;
//   }

  __device__ scalar_t sum() const { return x + y + z + w + u + v + s + t; }
};

constexpr uint32_t WARP_SIZE = 64;
constexpr uint32_t VEC_SIZE = 8;

template <>
struct __align__(16) vec8_t<__fp16>
{
  half2 c0, c1, c2, c3;

  __device__ vec8_t() {
    // 使用 make_half2 明确指定
    c0 = make_half2(0.0f, 0.0f);
    c1 = make_half2(0.0f, 0.0f);
    c2 = make_half2(0.0f, 0.0f);
    c3 = make_half2(0.0f, 0.0f);
}

  __device__ vec8_t(__fp16 x, __fp16 y, __fp16 z, __fp16 w,
                    __fp16 u, __fp16 v, __fp16 s, __fp16 t) {
                        c0 = make_half2(x, y);
                        c1 = make_half2(z, w);
                        c2 = make_half2(u, v);
                        c3 = make_half2(s, t);
                    }
   

  __device__ vec8_t operator*(const vec8_t &other) const
  {
    vec8_t res;
    res.c0 = c0 * other.c0;
    res.c1 = c1 * other.c1;
    res.c2 = c2 * other.c2;
    res.c3 = c3 * other.c3;
    return res;
  }

  __device__ vec8_t operator*(const float &scale) const
  {
    half2 s2 = __float2half2_rn(scale);
    vec8_t res;
    res.c0 = c0 * s2;
    res.c1 = c1 * s2;
    res.c2 = c2 * s2;
    res.c3 = c3 * s2;
    return res;
  }

  __device__ vec8_t operator+(const vec8_t &other) const
  {
    vec8_t res;
    res.c0 = c0 + other.c0;
    res.c1 = c1 + other.c1;
    res.c2 = c2 + other.c2;
    res.c3 = c3 + other.c3;
    return res;
  }

  __device__ void operator+=(const vec8_t &other)
  {
    c0 = c0 + other.c0;
    c1 = c1 + other.c1;
    c2 = c2 + other.c2;
    c3 = c3 + other.c3;
  }

  __device__ vec8_t& operator=(const vec8_t &other)
  {
    c0 = other.c0;
    c1 = other.c1;
    c2 = other.c2;
    c3 = other.c3;
    return *this;
  }

  __device__ float sum() const {
    half2 res = c0 + c1 + c2 + c3;
    return __half2float(res.x) + __half2float(res.y);
  }

  __device__ __fp16 get(int i) const {
      const __fp16* ptr = reinterpret_cast<const __fp16*>(&c0);
      return ptr[i];
  }
};
__forceinline__ __device__ float gemma_shfl_xor_sync(float x, int lane_mask) {
    return __shfl_xor(x, lane_mask, WARP_SIZE);
  }

__global__ void  gemma_fused_add_rmsnorm_fp16(
    __half* __restrict__ input,
    __half* __restrict__ residual,
    const __half* __restrict__ weight,
    const uint32_t hidden_size,
    float eps)
{
    // (void)input;
    // (void)residual;
    // (void)weight;
    // (void)hidden_size;
    // (void)eps;

    vec8_t<__fp16> *vectorized_in =
        reinterpret_cast<vec8_t<__fp16>  *>(input);
    vec8_t<__fp16> const *vectorized_weight =
        reinterpret_cast<vec8_t<__fp16> const *>(weight);
    vec8_t<__fp16>  *vectorized_residule =
        reinterpret_cast<vec8_t<__fp16>  *>(residual);
    
    const int vec_hidden_size = hidden_size >> 3; //除以8
    const int row_tile = 1; //先写死blockDim.y;
    extern __shared__ __half smem[];
    vec8_t<__fp16> * smem_vec = reinterpret_cast<vec8_t<__fp16>*>(smem);
    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t row_offset = (4096>>3)*blockIdx.x; //blockIdx.x*row_tile*vec_hidden_size;
    const uint32_t rounds = hidden_size>>9;//除以64个thread X 8 vec
    const uint32_t vec_per_thread = (WARP_SIZE>>3); // 8, 每线程负责 8 个 vec

    vec8_t<__fp16> v8_variance = {0, 0, 0, 0, 0, 0, 0, 0};
    vec8_t<__fp16> one = {1, 1, 1, 1, 1, 1, 1, 1};

    // 每线程从自己负责的段起始加载，而非都从 row_offset 加载
    const uint32_t thread_start = row_offset +  tx;
    vec8_t<__fp16> curr_vec = vectorized_in[thread_start];
    vec8_t<__fp16> curr_residule = vectorized_residule[thread_start];

    //vec8_t<__fp16> curr_weight = vectorized_weight[0];
    for(uint32_t i = 1; i < rounds; i++){
        // load next: 第 i 段对应全局 [row_offset + i*64, row_offset + i*64+63]
        int32_t local_idx = i * 64 + tx;
        int32_t vec_idx = row_offset + i * 64 + tx;  // 原 thread_start+i 错误：i=1 时 thread0 会读到 row_offset+1 而非 row_offset+64
        vec8_t<__fp16> next_vec = vectorized_in[vec_idx];
        vec8_t<__fp16> next_residule = vectorized_residule[vec_idx];
        
        curr_vec += curr_residule;
       
        // store residule

        *(float32x4*)&(vectorized_residule[vec_idx-64])= *(float32x4*)&curr_vec;
        smem_vec[local_idx-64] = curr_vec;
        v8_variance += curr_vec * curr_vec;

        curr_vec = next_vec;
        curr_residule = next_residule;
     
    }
    curr_vec += curr_residule;
    int32_t local_idx =  (rounds-1) * 64 + tx;//vec_per_thread *tx +rounds-1;
    int32_t vec_idx = local_idx+ row_offset;
  
     *(float32x4*)&vectorized_residule[vec_idx] = *(float32x4*)&curr_vec;
      smem_vec[local_idx] = curr_vec;
    v8_variance += curr_vec * curr_vec;
    float sum_sq = v8_variance.sum();

    // printf("v8_variance_sum: %f\n", sum_sq); //64
    
    
    #pragma unroll
    for (uint32_t offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1){
        sum_sq += __shfl_xor(sum_sq, offset, WARP_SIZE);
    }
    // if(tx == 0 && ty == 0)
    // {
        // printf("sum_sq: %f\n", sum_sq); //4096
    // }
    float s_variance = rsqrtf(sum_sq /hidden_size + eps);
      //  if(tx == 0 && ty == 0)
    // {
    //     printf("sum_sq: %f, s_variance: %f\n", sum_sq, s_variance); //384
    // }
    // 与 torch 一致：在 float 下做 normalized * (1+weight)，再一次性转 fp16，避免两次舍入导致 weight!=0 时精度差
    for(uint32_t i = 0; i < rounds; i++){
        int32_t row_local =i * 64 + tx;// tx *  vec_per_thread  + i;
        int32_t vec_idx = row_local + row_offset;
        vec8_t<__fp16> tmp = smem_vec[row_local];
        vec8_t<__fp16> v8_w = vectorized_weight[row_local];
        vec8_t<__fp16> one_w = v8_w + one;
        float scale = s_variance;
        vec8_t<__fp16> out;
        __half* out_ptr = reinterpret_cast<__half*>(&out);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float v = __half2float(tmp.get(j)) * scale * __half2float(one_w.get(j));
            out_ptr[j] = __float2half_rn(v);
        }
        vectorized_in[vec_idx] = out;
    }

}

