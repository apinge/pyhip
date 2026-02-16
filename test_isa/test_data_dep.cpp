#include <hip/hip_runtime.h>

// workgroup and data dependency: out[tid] = s_data[255-tid] = base + (255 - tid)
__global__ void test_data_dep(int* out, int base) {
    __shared__ int s_data[256];
    int scalar_val = __builtin_amdgcn_readfirstlane(base);
    int val  = scalar_val + threadIdx.x;
    s_data[threadIdx.x] = val;
    __syncthreads();
    int idx = 255 - threadIdx.x;
    out[threadIdx.x] = s_data[idx];
}

// ---------------------------------------------------------------------------
// 内联汇编版本，与上面汇编逐条对应。LDS 256 dwords，s_barrier 后读 s_data[255-tid]。
//
// 1. threadIdx.x 是 v0 吗？是的。
//    在 AMD GPU 的硬件 ABI 中，当一个 Kernel 启动时，硬件会自动初始化每个线程的寄存器。
//    对于 workitem（线程）在当前 Workgroup 里的索引，规则如下：
//    v0: 存放 threadIdx.x
//    v1: 存放 threadIdx.y（如果维度 > 1）
//    v2: 存放 threadIdx.z（如果维度 > 1）
//    所以在这段代码里，所有对 v0 的初始操作，本质上都是在处理 threadIdx.x。
//
// 2. 为什么要有 << 2 (v_lshlrev_b32_e32 v1, 2, v0)？
//    这叫「从索引到字节偏移」的转换。
//    逻辑背景：你的 out 或者 LDS 存储的是 int 类型（b32）。
//    硬件机制：GPU 的内存寻址（无论是 LDS 还是 Global Memory）都是按字节（Byte）计算的。
//    计算逻辑：1 个 int 占用 4 个字节。第 i 个元素的起始地址偏移量是 i * 4。
//    汇编实现：在二进制中，乘以 4 等同于左移 2 位（i << 2）。
//    v1 = v0 << 2 实际上就是计算 threadIdx.x * sizeof(int)，为后面的内存读写（ds_write / global_store）准备好精确的字节位置。
//
// 3. 为什么要 v_sub_u32_e32 v0, 0, v1？
//    这行代码初看很诡异，但结合后面的 ds_read_b32 v0, v0 offset:1020 就能看出这是一个「倒序读取」或「固定偏移定位」的骚操作。
//    数学意义：v0 = 0 - v1。由于 v1 是 threadIdx.x * 4，那么 v0 现在存的是一个负的字节偏移量。
//    组合技解析：v0 = -(threadIdx.x * 4)；ds_read_b32 v0, v0 offset:1020；
//    最终地址 = v0 (寄存器值) + 1020 (指令立即数)，即：地址 = 1020 - (threadIdx.x * 4)。
//    为什么要这么做？这通常用于反向访问数据。假设 Workgroup 大小是 256，int 数组总长度是 256*4 = 1024 字节。
//    当 threadIdx.x = 0 时，读取偏移 1020（最后一个元素）。
//    当 threadIdx.x = 255 时，读取偏移 1020 - 255*4 = 0（第一个元素）。
//    这段代码实现了一个「数据左右镜像翻转」的逻辑。
//
// 4. 整体代码逻辑串联
//    加载参数：从内核参数加载 base (s4) 和 out 指针 (s[2:3])。
//    计算偏移：v1 = threadIdx.x * 4 (字节地址)。
//    写入 LDS：每个线程把 base + threadIdx.x 的值写进自己对应的 LDS 位置。
//    同步：s_barrier 确保所有线程都写完了。
//    反向读取：利用 1020 - v1 的技巧，从 LDS 里把数据倒着读出来存入 v0。
//    全局存储：把倒序拿到的值 v0 写回到全局内存 out 中。
//
// 5. s_waitcnt 这里为什么写了两次？
//    第一个 s_waitcnt lgkmcnt(0)：等待 s_load (内核参数) 完成。
//    第二个 s_waitcnt lgkmcnt(0)：非常关键！它是在等 ds_write 结束。
//    如果没有这个等，后面的 s_barrier 可能起不到作用，导致数据还没写完别的线程就去读了。
//    第三个 s_waitcnt lgkmcnt(0)：等待 ds_read (从 LDS 读取) 的数据返回，这样才能执行最后的 global_store。
// ---------------------------------------------------------------------------
__global__ void test_data_dep_asm(int* out, int base) {
    (void)out;
    (void)base;
    __asm__ volatile(
        "s_load_dword s4, s[0:1], 0x8                     // s4 = base (kernarg+8)\n"
        "s_load_dwordx2 s[2:3], s[0:1], 0x0               // s[2:3] = out (kernarg+0)\n"
        "v_lshlrev_b32_e32 v1, 2, v0                      // v1 = v0 << 2 (byte offset)\n"
        "s_waitcnt lgkmcnt(0)\n"
        "v_add_u32_e32 v0, s4, v0                         // v0 = base + tid\n"
        "ds_write_b32 v1, v0                              // LDS[tid] = v0\n"
        "v_sub_u32_e32 v0, 0, v1                          // v0 = -v1 (byte offset for 255-tid)\n"
        "s_waitcnt lgkmcnt(0)\n"
        "s_barrier\n"
        "ds_read_b32 v0, v0 offset:1020                   // v0 = LDS[255-tid] (1020/4=255 dwords)\n"
        "s_waitcnt lgkmcnt(0)\n"
        "global_store_dword v1, v0, s[2:3]                // out[tid] = v0\n"
        :
        :
        : "s2", "s3", "s4", "v0", "v1", "memory"
    );
}
/*
s_waitcnt lgkmcnt(0)确认异步内存操作完成仅针对当前 Wavefront，是 “我” 的数据写完没？（确保本 Wave 的寄存器到内存路径清空）。
s_barrier协调所有线程的进度针对整个 Workgroup 是 “大家” 都写完没？（确保所有 整个workgroup的Wave 都到达同一个进度点）
*/