// A: 16x32, B: 32x16, D: 16x16, Out: 16x16 — row-major fp16。语义: Out = (A @ B) @ D
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using fp16_t = _Float16;
using fp16x4_t = __attribute__((vector_size(4 * sizeof(fp16_t)))) fp16_t;
using fp32x4_t = __attribute__((vector_size(4 * sizeof(float)))) float;
using fp16x8_t = __attribute__((vector_size(8 * sizeof(fp16_t)))) fp16_t;

__global__ void ab_swap(const fp16_t* A, const fp16_t* B, const fp16_t* D, fp16_t* Out) {
    fp16x8_t a_reg;
    fp16x8_t b_reg;
    fp32x4_t c_reg {};
    fp32x4_t out_reg {};
    fp16x4_t d_reg {};
    fp16x4_t c_half_reg {};
 
    a_reg = *reinterpret_cast<const fp16x8_t*>(A + 8 * (threadIdx.x / 16) + 32 * (threadIdx.x % 16));
#pragma unroll
    for (int i = 0; i < 8; i++) {
        b_reg[i] = *(B + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 128);
    }

    c_reg = __builtin_amdgcn_mfma_f32_16x16x32f16(b_reg, a_reg, c_reg, 0, 0, 0);

#pragma unroll  
    for (int i = 0; i < 4; i++) {
        d_reg[i] = *(D + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64);
    }

#pragma unroll  
    for (int i = 0; i < 4; i++) {
        c_half_reg[i] = __float2half(c_reg[i]);
    }

    out_reg = __builtin_amdgcn_mfma_f32_16x16x16f16(c_half_reg, d_reg, out_reg, 0, 0, 0);
    
    
#pragma unroll
    for (int i = 0; i < 4; i++) {
        *(Out + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64) =  __float2half(out_reg[i]);
    }
}

