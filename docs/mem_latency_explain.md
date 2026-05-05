# `mem_latency.md` 第 38–53 段说明（GEMM pipeline 示例）

对应原文：`docs/mem_latency.md` 中 “Example Usage in gemm” 小节。

这段是在用 **`mem_latency.md` 里测到的延迟/吞吐**，给一个 **具体 GEMM kernel 的算力与访存怎么对齐** 的推导示例（偏 pipeline / 调度设计说明）。

---

## Kernel 设定（原文约 38–43 行）

- **每个 CU 上跑 4 个 wave**（一个 block 里 4 wave 协作）。
- 用 **`v_mfma_f32_32x32x8_f16`**，文档里按 **latency 和吞吐都是约 32 cycle** 来估。
- **每个 wave** 负责一块 **128×128** 的 C 累加（由 16 个 32×32 小块拼出来），寄存器占用刚好压在 AccGPR 限制里。
- **四个 wave 合起来**算一块更大的 **C（256×256 量级）**；**A、B** 在 LDS 里是 **`256×32` 的 half 块**，先 **4 wave 合作从 HBM 装进 LDS**，再各自 **从 LDS 读到 VGPR** 喂给 MFMA。

---

## 内层循环算力 vs HBM 预取（原文约 45–49 行）

- 每个 wave 内层做的是 **`128×32 @ 32×128 → 128×128`**。单次 MFMA 做的是更小的 **`32×8 @ 8×32 → 32×32`**，所以要 **`4×4×4 = 64` 条 MFMA**，按 32 cycle/条粗算 **`64×32 = 2048` cycle**。
- 与此同时要从 HBM **预取 A+B**：**`256×32×2×2 bytes = 32768 B`**。表里 HBM 吞吐按 **`32 B/cycle`**，在 **2048 cycle** 里理论能拉 **`2048×32 ≈ 64KB`**，大于 **32KB**，所以结论是：**这一段更像算力瓶颈（compute-bound）**，HBM 带宽够盖住这轮计算时间（粗粒度上）。
- **延迟**：HBM load **500–800 cycle**，所以 **`wait vmcnt` / 写 LDS 之前**，load 要提前 **至少约 800 cycle** 发出去。
- **条数**：若用 **`buffer_load dwordx4`**（一次 16B×…见主文档表里口径），**4 个 wave 一共约 32 条这类 load**，**每个 wave 8 条**。

---

## 为什么要“摊开”发 global load（issue 不堵）（原文约 49 行）

- 这些 load **共享 CU 上通往内存子系统的通路**，发射太快会 **堵 issue**（与主文档里 issue-cycle 的解释一致）。
- 文档按测到的 **`1 条请求 / 32 cycle`** 的全局节奏来估；**4 个 wave 平分**，每个 wave 理想大约是 **`1 条 / (4×32) = 128 cycle`**。
- 所以 **每个 wave 的 8 条 load** 不要挤在一起发，最好 **均匀散落在约 32 条 MFMA 的时间跨度里**（文中写成 **`8×4=32` MFMA** 这种对应关系），避免 **issue-blocking**。

---

## 从 LDS 读 A/B（`ds_read_b128`）（原文约 51–53 行）

- 每个 wave 每轮内层要从 LDS 读 **两块 `128×32` 的 half**，共 **`16384 B`**，按 **`ds_read_b128`** 粗算 **16 条读指令**。
- **延迟 ~64 cycle**：所以要 **比真正用到数据提前若干条 MFMA**（文中写 **至少提前约 2 条 MFMA**，因为 MFMA 按 32 cycle 估）。
- **吞吐 ~8 cycle**（并且提到 **至少两个 SIMD 一起发** 时的口径）：可以和 MFMA **交错得比较密**（文中 **1:1** 是示意性的调度密度）。
- **4 个 wave**：合在一起大约 **每 32 cycle 发 4 条 `ds_read_b128`**，对应 **`1 条 / 8 cycle`** 的饱和节奏（与主文档表里 LDS 读相关的 issue/吞吐讨论一致）。

---

**一句话：** 先用 **MFMA 条数 × 32 cycle** 定“计算骨架有多长”，再用 **HBM 32 B/cycle** 判断 **预取是否算力主导**，接着用 **issue-cycle / 共享带宽** 说明 **global load 要摊开**，最后用 **LDS 延迟 + LDS 读吞吐** 说明 **`ds_read_b128` 要提前发、并且多 wave 一起怎样才能贴满读口**。

# 接口用法

## m0 到底是什么
为什么要置为m0 呢
和白皮书上说的也不太一致
```C++
asm volatile("s_mov_b32 m0, %0"::"s"(0));
```
- CDNA4 白皮书

> LDS
> ◦ If the LDS-ADDRESS is out-of-range (addr < 0 or >= (MIN(lds_size, m0)):
> ▪ Writes out-of-range are discarded; it is undefined if SIZE is not a multiple of write-data-size.
> ▪ Reads return the value zero.
>◦ If any source-VGPR is out-of-range, use the VGPR0 value is used.
> ◦ If the dest-VGPR is out of range, nullify the instruction (issue with exec=0)

> 3.6.5. LDS Allocation and Clamping
> LDS is allocated per work-group or per-wavefront when work-groups are not in use. LDS space is allocated to a
>  work-group or wavefront in contiguous blocks of 1280 bytes on 1280-byte alignment. LDS allocations do not
> wrap around the LDS storage. All accesses to LDS are restricted to the space allocated to that wavefront/workgroup.
> Clamping of LDS reads and writes is controlled by two size registers, which contain values for the size of the
> LDS space allocated by SPI to this wavefront or work-group, and a possibly smaller value specified in the LDS
> instruction (size is held in M0). The LDS operations use the smaller of these two sizes to determine how to
> clamp the read/write addresses.
> 3.7. M0 Memory Descriptor
> There is one 32-bit M0 register per wavefront, which can be used for:
> • Local Data Share (LDS)
> ◦ LDS addressing for Memory/Vfetch → LDS: {14’h0, lds_offset[17:0]} // in bytes
> ◦ { base[5:0], 16’h0}
> • Indirect GPR addressing for both vector and scalar instructions. M0 is an unsigned index.

HipKittens

```C++
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>, int N_THREADS = WARP_THREADS>
__attribute__((always_inline)) 
__device__ __forceinline__ void load(ST& dst, const GL& src, const COORD& idx,
                                const uint32_t* __restrict__ swizzled_offsets,
                                i32x4 SRD,
                                const void* base_ptr, const uint32_t lds_base)
{
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "only supporting 16 and 8-bit dtypes");

    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * N_THREADS;
    constexpr int memcpy_per_tile  = (ST::rows * ST::cols * sizeof(T)) / bytes_per_memcpy;
    static_assert(bytes_per_memcpy % 16 == 0, "LDS bump must be 16-aligned");

    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);
    constexpr int elem_per_warp   = elem_per_thread * kittens::WARP_THREADS;

    // ---- compute per-tile base pointer and scalar offset (SOFF) ----
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* __restrict__ gptr = (T*)&src[unit_coord];

    uint32_t SOFF = to_sgpr_u32(static_cast<uint32_t>(
    reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)
    ));

    // // ---- LDS base (byte address) as SGPR (wave-uniform) ----
    // const int num_warps = N_THREADS / kittens::WARP_THREADS;
    // const int wid = warpid() % num_warps;
    // uint32_t lds_base = to_sgpr_u32(static_cast<uint32_t>(
    // reinterpret_cast<uintptr_t>(&dst.data[0]) + wid * elem_per_warp * sizeof(T)
    // ));

    // ---- SGPR cursor we bump each iteration (no new readfirstlane) ----
    uint32_t lds_cur = lds_base;
    asm volatile("" : "+s"(lds_cur)); 

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        int32_t lds_byte = lds_cur;                 // still SGPR
        asm volatile("" : "+s"(lds_byte));           // keep it SGPR at the use

        asm volatile("s_mov_b32 m0, %0" :: "s"(lds_byte));
        llvm_amdgcn_raw_buffer_load_lds(
            SRD, 
            (as3_uint32_ptr)0, 
            16, 
            swizzled_offsets[i], 
            SOFF, 
            0,
            static_cast<int>(coherency::cache_all)
        );

        // SGPR bump (compiler emits s_add_u32)
        lds_cur += bytes_per_memcpy;
    }
}
```
soffset写道lds的地址 为啥这样做
指令不应该叫 GLOBAL_LOAD_LDS_DWORD吗
```C++
   asm volatile("s_mov_b32 m0, %0"::"s"(0));
    for(int i = 0; i < 64*1024; i += 64) {
        dc += get_cycles([&](){
            buffer_load_dword_lds<0>(buff, soffset, voffset);
            s_waitcnt_vmcnt<0>();
        });
        count ++;
        soffset =  (__builtin_amdgcn_readfirstlane(lds[0][0]) * sizeof(int)/sizeof(int32x4_t))*sizeof(int32x4_t);
    }
```
## llvm_amdgcn_raw_buffer_load_lds

```C++ 
extern "C" __device__ void 
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                as3_uint32_ptr lds_ptr,
                                int size,
                                int voffset, 
                                int soffset, 
                                int offset,  // does not change (0); instruction offset
                                int aux) __asm("llvm.amdgcn.raw.buffer.load.lds"); // cache coherency
```