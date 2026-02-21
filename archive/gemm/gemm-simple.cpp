/*
 * MFMA kernels from ROCm blog (row-major), plus bf16 16x16x16.
 * https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// -------- FP32 32x32x2 (blog: __builtin_amdgcn_mfma_f32_32x32x2f32) --------
using fp32x16_t = __attribute__((vector_size(16 * sizeof(float)))) float;

__global__ void mfma_fp32_32x32x2_fp32(const float* A, const float* B, float* C) {
    float a_reg;
    float b_reg;
    fp32x16_t c_reg {};

    const float* ldg_a_ptr = A + threadIdx.x / 32 + 2 * (threadIdx.x % 32);
    const float* ldg_b_ptr = B + threadIdx.x % 32 + (threadIdx.x / 32) * 32;

    a_reg = *ldg_a_ptr;
    b_reg = *ldg_b_ptr;

    c_reg = __builtin_amdgcn_mfma_f32_32x32x2f32(a_reg, b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++) {
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + i * 32 * 8]          = c_reg[i * 4];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 1 + i * 32 * 8] = c_reg[i * 4 + 1];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 2 + i * 32 * 8] = c_reg[i * 4 + 2];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 3 + i * 32 * 8] = c_reg[i * 4 + 3];
    }
}

// -------- FP16 16x16x16 (blog: __builtin_amdgcn_mfma_f32_16x16x16f16) --------
using fp16_t = _Float16;
using fp16x4_t = __attribute__((vector_size(4 * sizeof(fp16_t)))) fp16_t;
using fp32x4_t = __attribute__((vector_size(4 * sizeof(float)))) float;

__global__ void mfma_fp32_16x16x16_fp16(const fp16_t* A, const fp16_t* B, float* C) {
    fp16x4_t a_reg;
    fp16x4_t b_reg;
    fp32x4_t c_reg {};

    a_reg = *reinterpret_cast<const fp16x4_t*>(A + 4 * (threadIdx.x / 16) + 16 * (threadIdx.x % 16));

    for (int i = 0; i < 4; i++) {
        b_reg[i] = *(B + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64);
    }

    c_reg = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++) {
        *(C + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64) = c_reg[i];
    }
}
// // ----------------------------------------
// // Step 1: Load scalar addresses
// // ----------------------------------------
// // s[6:7] = pA base address
// // s[8:9] = pB base address
// // s[10:11] = pC base address
// // wait 2 cycles for address setup
// s_load_dwordx2 s[6:7], s[0:1], 0x0
// s_load_dwordx2 s[8:9], s[0:1], 0x8
// s_load_dwordx2 s[10:11], s[0:1], 0x10
// s_nop 0x1

// // ----------------------------------------
// // Step 2: Compute lane / cblock / r
// // ----------------------------------------
// // lane_id = threadIdx.x % 64
// v_and_b32 v1, 0x3f, v0
// v_and_b32 v2, 0xf, v1      // r = lane_id & 15
// v_lshrrev_b32 v1, 0x4, v1  // cblock = lane_id >> 4

// // ----------------------------------------
// // Step 3: Create buffer descriptors for load
// // ----------------------------------------
// // self.desc.pA = {base=s6,s7, size=0x20000, stride=0x200}
// // self.desc.pB = {base=s8,s9, size=0x20000, stride=0x200}
// // self.desc.pC = {base=s10,s11, size=0x20000, stride=0x400}
// s_mov_b32 s15, 0x20000
// s_mov_b32 s12, s6
// s_mov_b32 s13, s7
// s_mov_b32 s14, 0x200
// s_mov_b32 s19, 0x20000
// s_mov_b32 s16, s8
// s_mov_b32 s17, s9
// s_mov_b32 s18, 0x200
// s_mov_b32 s23, 0x20000
// s_mov_b32 s20, s10
// s_mov_b32 s21, s11
// s_mov_b32 s22, 0x400

// // ----------------------------------------
// // Step 4: Compute voffset for each lane
// // ----------------------------------------
// // voffset = (r << 5) + (cblock << 3)
// v_lshlrev_b32 v3, 0x5, v2    // v3 = r << 5
// v_lshlrev_b32 v4, 0x3, v1    // v4 = cblock << 3
// v_add_u32_e32 v3, v3, v4     // v3 = voffset

// // ----------------------------------------
// // Step 5: Load A and B fragments
// // ----------------------------------------
// // load_dwordx2 = 4 bf16 elements (2 per dword)
// // note: offen = offset from base
// buffer_load_dwordx2 v[4:5], v3, s[12:15], 0x0 offen // a_frag
// buffer_load_dwordx2 v[6:7], v3, s[16:19], 0x0 offen // b_frag
// s_waitcnt vmcnt(0)

// // ----------------------------------------
// // Step 6: MFMA compute
// // ----------------------------------------
// // c_frag = a_frag @ b_frag
// v_mfma_f32_16x16x16_bf16 v[4:7], v[4:5], v[6:7], 0

// // ----------------------------------------
// // Step 7: Compute C base offset
// // ----------------------------------------
// // base_offset = (r << 2) + (cblock << 6)
// v_lshlrev_b32 v2, 0x2, v2   // r << 2
// v_lshlrev_b32 v1, 0x6, v1   // cblock << 6
// v_add_u32_e32 v1, v2, v1    // base_offset

// // ----------------------------------------
// // Step 8: Store C fragment
// // ----------------------------------------
// buffer_store_dword v4, v1, s[20:23], 0x0 offen       // c_frag[0]
// buffer_store_dword v5, v1, s[20:23], 0x0 offen offset:64
// buffer_store_dword v6, v1, s[20:23], 0x0 offen offset:128
// buffer_store_dword v7, v1, s[20:23], 0x0 offen offset:192
__global__ void mfma_fp32_16x16x16_fp16_A_BT(const fp16_t* A, const fp16_t* B, float* C) {
    fp16x4_t a_reg;
    fp16x4_t b_reg;
    fp32x4_t c_reg {};

    a_reg = *reinterpret_cast<const fp16x4_t*>(A + 4 * (threadIdx.x / 16) + 16 * (threadIdx.x % 16));
    b_reg = *reinterpret_cast<const fp16x4_t*>(B + 4 * (threadIdx.x / 16) + 16 * (threadIdx.x % 16));
    // for (int i = 0; i < 4; i++) {
    //     b_reg[i] = *(B + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64);
    // }

    c_reg = __builtin_amdgcn_mfma_f32_16x16x16f16(a_reg, b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++) {
        *(C + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64) = c_reg[i];
    }
}

__global__ void mfma_fp32_32x32x8_fp16_A_BT(const fp16_t* A, const fp16_t* B, float* C) {
    fp16x4_t a_reg;
    fp16x4_t b_reg;
    fp32x16_t c_reg {};

    a_reg = *reinterpret_cast<const fp16x4_t*>(A + 4 * (threadIdx.x / 32) + 8 * (threadIdx.x % 32));
    b_reg = *reinterpret_cast<const fp16x4_t*>(B + 4 * (threadIdx.x / 32) + 8 * (threadIdx.x % 32));
 

    c_reg = __builtin_amdgcn_mfma_f32_32x32x8f16(a_reg, b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++) {
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + i * 32 * 8]          = c_reg[i * 4];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 1 + i * 32 * 8] = c_reg[i * 4 + 1];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 2 + i * 32 * 8] = c_reg[i * 4 + 2];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 3 + i * 32 * 8] = c_reg[i * 4 + 3];
    }
}


__global__ void mfma_fp32_32x32x8_fp16(const fp16_t* A, const fp16_t* B, float* C) {
    fp16x4_t a_reg;
    fp16x4_t b_reg;
    fp32x16_t c_reg {};

    a_reg = *reinterpret_cast<const fp16x4_t*>(A + 4 * (threadIdx.x / 32) + 8 * (threadIdx.x % 32));
    //b_reg = *reinterpret_cast<const fp16x4_t*>(B + 4 * (threadIdx.x / 32) + 8 * (threadIdx.x % 32));
    for (int i = 0; i < 4; i++) {
        b_reg[i] = *(B + i * 8 + 4 * (threadIdx.x / 32) + 8 * (threadIdx.x % 32));
    }

    c_reg = __builtin_amdgcn_mfma_f32_32x32x8f16(a_reg, b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++) {
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + i * 32 * 8]          = c_reg[i * 4];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 1 + i * 32 * 8] = c_reg[i * 4 + 1];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 2 + i * 32 * 8] = c_reg[i * 4 + 2];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 3 + i * 32 * 8] = c_reg[i * 4 + 3];
    }
}


// -------- BF16 16x16x16 (已有，不改) --------
using bf16_t = __bf16;
using bf16x4_t = __attribute__((vector_size(4 * sizeof(bf16_t)))) bf16_t;

__global__ void sgemm_bf16_simple(const bf16_t* A, const bf16_t* B, float* C) {
    bf16x4_t a_reg;
    bf16x4_t b_reg;
    fp32x4_t c_reg {};

    a_reg = *reinterpret_cast<const bf16x4_t*>(A + 4 * (threadIdx.x / 16) + 16 * (threadIdx.x % 16));

    for (int i = 0; i < 4; i++) {
        b_reg[i] = *(B + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64);
    }

    c_reg = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_reg, b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++) {
        *(C + i * 16 + threadIdx.x % 16 + (threadIdx.x / 16) * 64) = c_reg[i];
    }
}

// -------- FP8 32x32x16 (blog: __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8) --------
using fp8_t = __hip_fp8_storage_t;
//using fp8_storage_t = uint8_t;
using fp8x8_t = __attribute__((vector_size(8 * sizeof(fp8_t)))) fp8_t;
using fp32x16_t_fp8 = __attribute__((vector_size(16 * sizeof(float)))) float;

__global__ void mfma_fp32_32x32x16_fp8_fp8(const fp8_t* A, const fp8_t* B, float* C) {
    fp8x8_t a_reg;
    fp8x8_t b_reg;
    fp32x16_t_fp8 c_reg {};

    a_reg = *reinterpret_cast<const fp8x8_t*>(A + (threadIdx.x / 32) * 8 + (threadIdx.x % 32) * 16);

    for (int i = 0; i < 8; i++) {
        b_reg[i] = *(B + i * 32 + threadIdx.x % 32 + (threadIdx.x / 32) * 8 * 32);
    }

    c_reg = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8((long)a_reg, (long)b_reg, c_reg, 0, 0, 0);

    for (int i = 0; i < 4; i++) {
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + i * 32 * 8]          = c_reg[i * 4];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 1 + i * 32 * 8] = c_reg[i * 4 + 1];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 2 + i * 32 * 8] = c_reg[i * 4 + 2];
        C[threadIdx.x % 32 + (threadIdx.x / 32) * 4 * 32 + 32 * 3 + i * 32 * 8] = c_reg[i * 4 + 3];
    }
}
