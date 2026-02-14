/*******************************************************************************
 * MFMA 16x16x16 BF16 GEMM — naive/unoptimized (no LDS, no tiling).
 * pyhip entry (kernel only).
 * Originated from MLSE.LIB.Git.Training/011_mfma_exercise/01_simple_hgemm_bf16.cpp
 * Block/wave 划分与 gemm_test 流程参考自 rocWMMA 示例：
 * https://github.com/ROCm/rocWMMA/blob/develop_deprecated/samples/simple_hgemm.cpp
 ******************************************************************************/

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

///////////////
/// Helpers ///
///////////////

using bf16x4   = __attribute__((__vector_size__(4 * sizeof(__bf16)))) __bf16;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

__device__ void fill_frag(float32x4& frag, float value)
{
#pragma unroll
    for(int i = 0; i < 4; i++)
        frag[i] = value;
}

/////////////////
/// Constants ///
/////////////////

const int WAVE_SIZE   = 64;
// Thread block
// : T_BLOCK_X must be multiple of WAVE_SIZE.
// Note: Each wave will compute one BLOCK_M x BLOCK_N output block
// Note: Workgroup will compute
//  T_BLOCK_X / WAVE_SIZE x T_BLOCK_Y output blocks
const int T_BLOCK_X   = 4 * WAVE_SIZE;
const int T_BLOCK_Y   = 4;
const int BLOCK_M     = 16;
const int BLOCK_N     = 16;
const int BLOCK_K     = 16;

using float32_t = float;

// BF16 fragment: 4 elements per thread for 16x16x16 (same layout as f16 MFMA)
// MFMA 16x16x16 bf16 1k: same I/O shape as f16, uses __builtin_amdgcn_mfma_f32_16x16x16bf16_1k
__device__ float32x4 mfma_f32_16x16x16bf16_1k(bf16x4 aFrag, bf16x4 bFrag, float32x4 accumFrag)
{
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(aFrag, bFrag, accumFrag, 0, 0, 0);
}

// patten可以参考 https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
// A 为列优先：本 lane 在 16x16 块内取 (row, col) 起沿列方向连续 4 个元素
__device__ bf16x4 load_A_16x16_col_major(__bf16 const* input, int ld)
{
    uint32_t lane  = threadIdx.x & 63;
    uint32_t row   = lane % BLOCK_M;           // 行 [0, 16)
    uint32_t col0  = (lane / BLOCK_M) * 4;      // 起始列 0/4/8/12，共取 4 列
    int      base   = (int)(row + col0 * ld);   // 列优先: row + col*ld
    int      stride = ld;                       // 下一列同一行

    return bf16x4{
        input[base],
        input[base + stride],
        input[base + 2 * stride],
        input[base + 3 * stride],
    };
}

// B 为行优先：本 lane 在 16x16 块内取 (row, col) 起沿行方向连续 4 个元素
__device__ bf16x4 load_B_16x16_row_major(__bf16 const* input, int ld)
{
    uint32_t lane = threadIdx.x & 63;
    uint32_t row0 = (lane / BLOCK_N) * 4;       // 起始行 0/4/8/12，共取 4 行
    uint32_t col  = lane % BLOCK_N;             // 列 [0, 16)
    int      base   = (int)(row0 * ld + col);   // 行优先: row*ld + col
    int      stride = ld;                       // 下一行同一列

    return bf16x4{
        input[base],
        input[base + stride],
        input[base + 2 * stride],
        input[base + 3 * stride],
    };
}

// C/D 的 16x16 块在列优先下的 lane 划分：
// - 64 个 lane：lane = 0..63 → row_group = lane/16 ∈ {0,1,2,3}，col = lane%16 ∈ {0..15}。
// - row = row_group*4，即本 lane 负责的行带为 [row, row+4)（共 4 行），列号为 col。
// - 列优先下一列内连续 4 个 float 即 C(row:row+4, col)，线性下标 base = row + col*ld。
// - 因此：本 lane 读/写的是 16x16 块里的「第 col 列、第 row_group 段」的 4 个元素，即 C[row:row+4, col]。
// C的patten和B其实是一样的 见https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
// C能这样连续读 因为C是 col major 连续读同col上四个float ，而B呢 是row major 连续读同col上的四个float，所以B是代码层面是离散地load
__device__ float32x4 load_C_16x16_col_major(float32_t const* input, int ld)
{
    uint32_t lane  = threadIdx.x & 63;
    uint32_t row   = (lane / BLOCK_N) * 4;   // 0, 4, 8, 12
    uint32_t col   = lane % BLOCK_N;         // 0..15
    int      offset = (int)(row + col * ld); // 列优先: 第 col 列、从 row 行起的 4 个 float
    return *((float32x4*)(input + offset));
}

__device__ void store_C_16x16_col_major(float32_t* output, float32x4 cFrag, int ld)
{
    uint32_t lane  = threadIdx.x & 63;
    uint32_t row   = (lane / BLOCK_N) * 4;
    uint32_t col   = lane % BLOCK_N;
    int      offset = (int)(row + col * ld);
    *((float32x4*)(output + offset)) = cFrag;
}

// 索引关系说明（workgroup 64x64 vs MFMA 16x16）：
// - 一个 workgroup 负责输出 D 上的一块 64x64 = 4*BLOCK_M x 4*BLOCK_N。
// - 这块 64x64 由 4x4 个 16x16 的 tile 组成，每个 tile 由一个 wave 用一次 MFMA 计算。
// - (waveGridX, waveGridY) = 该 wave 对应的 16x16 tile 在"全局 tile 网格"里的格子下标。
// - 本 wave 负责的 16x16 在 D 上的左上角：(cRow, cCol) = (waveGridX*BLOCK_M, waveGridY*BLOCK_N)，
//   即该 tile 覆盖 D 的行 [cRow, cRow+16)、列 [cCol, cCol+16)。
// - 在某个 workgroup 内：waveGridX = blockIdx.x*4 + (threadIdx.x/64)，waveGridY = blockIdx.y*4 + threadIdx.y，
//   即 block 内 4x4 个 wave 对应 4x4 个 16x16 tile，拼成该 workgroup 的 64x64。
// D = alpha * (A*B) + beta * C; A MxK col-major lda=m, B KxN row-major ldb=n, C/D MxN col-major ldc=ldd=m
__global__ void sgemm_bf16_d(uint32_t     m,
                             uint32_t     n,
                             uint32_t     k,
                             __bf16 const* a,
                             __bf16 const* b,
                             float32_t const* c,
                             float32_t*       d,
                             uint32_t     lda,
                             uint32_t     ldb,
                             uint32_t     ldc,
                             uint32_t     ldd,
                             float32_t    alpha,
                             float32_t    beta)
{
    auto fragA   = bf16x4{};
    auto fragB   = bf16x4{};
    auto fragAcc = float32x4{};

    fill_frag(fragAcc, 0.0f);

    // 本 thread 所在 wave 对应的 16x16 tile 的格子下标 -> 该 tile 在 D 上的左上角 (cRow, cCol)
    auto waveGridX = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
    auto waveGridY = (blockIdx.y * blockDim.y + threadIdx.y);
    auto cRow      = waveGridX * BLOCK_M;
    auto cCol      = waveGridY * BLOCK_N;

    if(cRow < m && cCol < n)
    {
        // A/B 布局与本循环取块（以 waveGridX==0, waveGridY==0 为例）：
        // - A: M×K 列优先，lda=m，A(row,col) 在 a[col*m+row]。指针 a + (cRow + i*lda) = a + (0 + i*m)
        //   表示从第 i 列开始；load_A_16x16_col_major 读 16 行×16 列 => A 的行 [cRow, cRow+16)、列 [i, i+16]。
        //   即每次取 A 的一块 16×16：行 [cRow:cRow+16]，列 [i:i+16]。
        // - B: K×N 行优先，ldb=n，B(row,col) 在 b[row*n+col]。指针 b + (i*ldb + cCol) = b + (i*n + 0)
        //   表示从第 i 行开始；load_B_16x16_row_major 读 16 行×16 列 => B 的行 [i, i+16)、列 [cCol, cCol+16]。
        //   即每次取 B 的一块 16×16：行 [i:i+16]，列 [cCol:cCol+16]。
        // 本 wave 写 D[cRow:cRow+16, cCol:cCol+16] = sum_i A[cRow:cRow+16, i:i+16] * B[i:i+16, cCol:cCol+16]。
        for(int i = 0; i < k; i += BLOCK_K)
        {
            fragA   = load_A_16x16_col_major(a + (cRow + i * lda), lda);
            fragB   = load_B_16x16_row_major(b + (i * ldb + cCol), ldb);
            fragAcc = mfma_f32_16x16x16bf16_1k(fragA, fragB, fragAcc);
        }

        auto fragC = load_C_16x16_col_major(c + (cRow + cCol * ldc), ldc);
        for(int i = 0; i < 4; ++i)
            fragC[i] = alpha * fragAcc[i] + beta * fragC[i];
        store_C_16x16_col_major(d + (cRow + cCol * ldd), fragC, ldd);
    }
}

/* ---------- 原始 C++ 主机逻辑（仅保留作参考，由 Python 侧负责分配与校验）----------
// CPU reference: D = alpha * (A_bf16 * B_bf16) + beta * C (A,B in bf16 bits, C/D float)
// A: M×K 列优先 lda=m；B: K×N 行优先 ldb=n；C/D: M×N 列优先 ldc=ldd=m

#include <iostream>
#include <vector>
#include <cstdlib>
#include <limits>

#define CHECK_HIP_ERROR(status) \
    do { if((status) != hipSuccess) { fprintf(stderr, "hip error: %s at %s:%d\n", hipGetErrorString(status), __FILE__, __LINE__); exit(EXIT_FAILURE); } } while(0)
template<typename IntT1, typename IntT2>
__host__ static constexpr IntT1 ceilDiv(IntT1 num, IntT2 div) { return (num + div - 1) / div; }

static void gemm_cpu_bf16_ref(uint32_t m, uint32_t n, uint32_t k,
    uint16_t const* a, uint16_t const* b, float const* c, float* d,
    uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t ldd, float alpha, float beta)
{
    auto rowMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return row * ld + col; };
    auto colMjr = [](uint32_t row, uint32_t col, uint32_t ld) { return col * ld + row; };
    for(uint32_t i = 0; i < m; ++i)
        for(uint32_t j = 0; j < n; ++j) {
            float accum = 0.f;
            for(uint32_t h = 0; h < k; ++h) {
                float ah = __bfloat162float(*reinterpret_cast<__bf16 const*>(&a[colMjr(i,h,lda)]));
                float bh = __bfloat162float(*reinterpret_cast<__bf16 const*>(&b[rowMjr(h,j,ldb)]));
                accum += ah * bh;
            }
            d[colMjr(i,j,ldd)] = alpha * accum + beta * c[colMjr(i,j,ldc)];
        }
}

static void fillRandBf16(uint16_t* mat, uint32_t m, uint32_t n)
{
    srand((unsigned)time(nullptr));
    for(uint32_t i = 0; i < m; ++i)
        for(uint32_t j = 0; j < n; ++j) {
            int value = (rand() % 5);
            if(value % 3 == 0) value = -value;
            __bf16 b = __float2bfloat16(static_cast<float>(value));
            mat[i * n + j] = *reinterpret_cast<uint16_t*>(&b);
        }
}

__host__ void gemm_test(uint32_t m, uint32_t n, uint32_t k, float alpha, float beta)
{
    if((m < (BLOCK_M * T_BLOCK_X / WAVE_SIZE) || n < (BLOCK_N * T_BLOCK_Y) || k < BLOCK_K)
       || (m % BLOCK_M || n % BLOCK_N || k % BLOCK_K)) { std::cout << "Unsupported size!\n"; return; }
    int lda = m, ldb = n, ldc = m, ldd = ldc;
    std::vector<uint16_t> matrixA(m * k), matrixB(k * n);
    std::vector<float32_t> matrixC(m * n), matrixD(m * n, std::numeric_limits<float>::signaling_NaN());
    fillRandBf16(matrixA.data(), m, k);
    fillRandBf16(matrixB.data(), k, n);
    // fillRand(matrixC.data(), m, n);  // need fillRand from common.hpp
    __bf16 *d_a, *d_b; float32_t *d_c, *d_d;
    size_t bytesA = matrixA.size()*sizeof(uint16_t), bytesB = matrixB.size()*sizeof(uint16_t);
    size_t bytesC = matrixC.size()*sizeof(float32_t), bytesD = matrixD.size()*sizeof(float32_t);
    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA)); CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC)); CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));
    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));
    auto blockDim = dim3(T_BLOCK_X, T_BLOCK_Y);
    auto gridDim  = dim3(ceilDiv(m, BLOCK_M * T_BLOCK_X / WAVE_SIZE), ceilDiv(n, BLOCK_N * T_BLOCK_Y));
    sgemm_bf16_d<<<gridDim, blockDim>>>(m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));
    gemm_cpu_bf16_ref(m, n, k, matrixA.data(), matrixB.data(), matrixC.data(), matrixD.data(), lda, ldb, ldc, ldd, alpha, beta);
    CHECK_HIP_ERROR(hipFree(d_a)); CHECK_HIP_ERROR(hipFree(d_b)); CHECK_HIP_ERROR(hipFree(d_c)); CHECK_HIP_ERROR(hipFree(d_d));
}

int main() { gemm_test(256, 256, 256, 2.1f, 2.1f); return 0; }
---------- */
