#include <hip/hip_runtime.h>

__global__ void test_readfirstlane(int* out, int base) {
    // 假设 base 每个线程传入的值是 threadIdx.x
    // 但 readfirstlane 会强行只取第一个线程的值 （每个wave的第一个线程 不是workgoup）
    // 这个first lane很有意思 如果workgroup是64 当然是第0个thread
    // 如果是 256 每个wave的第0 个thread
    // readfirstlane 的"First"是指 "当前 Wave 内部活跃的第一个 Lane"。它看不见别的 Wave，更管不了整个 Workgroup
    int val = base + (threadIdx.x&63);
    int scalar_val = __builtin_amdgcn_readfirstlane(val);
    // 即使是 threadIdx.x 为 50 的线程，这里拿到的 scalar_val 也是 wave 内 first lane 的值
    out[threadIdx.x] = scalar_val;
}

// ---------------------------------------------------------------------------
// 内联汇编版本：与上面 test_readfirstlane 等价的逻辑，用于对照 ISA 分析。
// 语义：out[threadIdx.x] = base + (threadIdx.x & 63) 的 wave-first-lane 值。
// 即每个 wave 内所有线程写入的是该 wave 第一个 lane 的 (base + (tid&63))。
//
// ISA 分析（s_waitcnt lgkmcnt(0) 与标量内存）：
//
// 这段汇编里的 s_waitcnt lgkmcnt(0) 等待的是 Scalar Memory Reads（标量内存读取）的完成。
// 具体到这段代码，它在等前两行 s_load_dword 指令把数据真正从内存（或 L2 Cache）
// 搬运到寄存器 s4 和 s[2:3] 中。
//
// 1. 为什么需要等？
//    AMD GPU 的内存指令是异步的。当执行 s_load_dword s4, ... 时，指令发射（Issue）后，
//    标量执行单元（Scalar Unit）不会原地踏步等数据回来，而是直接往下走。
//    如果没有 s_waitcnt：程序会立刻执行后面的 s_add_i32 s0, s0, s4。但此时 s4 里的数据
//    可能还在内存总线上跑着呢，寄存器里还是旧的脏数据。
//    有了 s_waitcnt lgkmcnt(0)：处理器会在这里"阻塞"，直到所有处于在途状态（In-flight）
//    的 Local/Global/Kernel Memory (LGKM) 请求全部返回。
//
// 2. 指令解析：什么是 LGKM？
//    waitcnt 后面跟的参数决定了它在等哪种计数器：
//    - lgkmcnt: LDS, Global (Scalar), Kernel (Scalar), Message。这里主要负责等待 s_load 系列指令。
//    - 0: 计数器必须减到 0。意思是"等之前所有发出的标量加载指令全部完成"。
//
// 3. 数据流依赖图：
//    加载阶段：
//      s_load_dword s4, s[0:1], 0x8           // 发起请求，lgkmcnt += 1（base）
//      s_load_dwordx2 s[2:3], s[0:1], 0x0     // 发起请求，lgkmcnt += 1（out 指针）
//    不相关计算（在此期间利用延迟，Latency Hiding）：
//      v_readfirstlane_b32 s0, v0             // 不依赖 s_load，不需要等
//      s_and_b32 s0, s0, 63                   // lane id in wave
//      v_lshlrev_b32_e32 v0, 2, v0            // 字节偏移 out[tid]
//    同步点：
//      s_waitcnt lgkmcnt(0)                   // 确保 s4 和 s[2:3] 已就绪
//    使用阶段：
//      s_add_i32 s0, s0, s4                   // 必须使用 s4
//      global_store_dword v0, v1, s[2:3]      // 必须使用 s[2:3] 作为基址
//
// 4. 观察：v_readfirstlane 和 v_lshlrev 被插在 s_load 和 s_waitcnt 之间，是编译器
//    指令调度（Instruction Scheduling）的优化，用不依赖内存的指令填补空档，隐藏内存延迟。
//
// 若不写 s_waitcnt，极可能因使用未就绪的地址/数据而导致错误结果（全 0 或随机数）或 Memory Fault。
// ---------------------------------------------------------------------------
__global__ void test_readfirstlane_asm(int* out, int base) {
    (void)out; // 这里避免compiler优化掉它认为没用的输入变量
    (void)base;
    // s[0:1] = kernarg 段指针（由 ABI 在 kernel 入口提供），v0 = workitem id
    __asm__ volatile(
        "s_load_dword s4, s[0:1], 0x8                     // s4 = load_dword_from(s[0:1] + 0x8, glc=0); 8.2.1.1. Scalar Memory Addressing\n"
        "s_load_dwordx2 s[2:3], s[0:1], 0x0               // s[2:3] = load_dwordx2_from(s[0:1] + 0x0, glc=0); 8.2.1.1. Scalar Memory Addressing\n"
        "v_readfirstlane_b32 s0, v0\n"
        "s_and_b32 s0, s0, 63                             // s0 = s0 & 63\n"
        "v_lshlrev_b32_e32 v0, 2, v0                      // v0.b32 = v0 << 2;\n"
        "s_waitcnt lgkmcnt(0)\n"
        "s_add_i32 s0, s0, s4                             // s0.i32 = s0 + s4; scc=overflow_or_carry\n"
        "v_mov_b32_e32 v1, s0                             // v1 = s0;\n"
        "global_store_dword v0, v1, s[2:3]                // save_dword_to_addr(v1, addr=v0 + s[2:3])\n"
        :
        : //"s"(out), "s"(base) // 将变量绑定到 SGPR  
        : "s0", "s2", "s3", "s4", "v0", "v1", "memory"
    );
}

// ---------------------------------------------------------------------------
// 内联汇编语法与破坏列表 / memory 说明：
//
// __asm__ ( "代码" : 输出 : 输入 : 破坏列表 (Clobber List) );
//
// 破坏列表里你看到的 "s0", "s2", "s3", "s4", "v0", "v1" 都在告诉编译器：
// “我在汇编里乱动了这些寄存器，你别在里面存重要东西！”
//
// 保护机制：编译器在安排 C++ 变量时也会用到这些寄存器。如果你不在这里声明，
// 编译器可能刚好把你的 out 指针存在 s2 里，结果你的汇编一运行就把 s2 改成了别的值，程序直接崩溃。
// 声明之后：编译器会避开这些寄存器，或者在执行汇编前把这些寄存器里的重要数据备份到栈里。
//
// "memory" 是什么意思？
// 这被称为 Memory Barrier（内存屏障/内存栅栏）。含义：
// - 禁止重排序：不能为了优化把汇编块之后的内存操作挪到汇编块之前执行，反之亦然。
// - 强制刷新缓存：这段汇编可能以编译器“看不见”的方式读写内存。因此执行前必须把寄存器里
//   的内存变量写回内存；执行完后必须认为之前的缓存都失效，需要重新从内存读取。
// 在本例中：因为执行了 global_store_dword，直接修改了 out 指向的内容。若不加 "memory"，
// 编译器可能还认为内存没变，继续使用之前缓存的结果。
// ---------------------------------------------------------------------------
__global__ void test_readfirstlane_asm_2(int* out, int base) {
    __asm__ volatile(
        "v_readfirstlane_b32 s0, v0\n"
        "s_and_b32 s0, s0, 63                             // s0 = s0 & 63\n"
        "v_lshlrev_b32_e32 v0, 2, v0                      // v0.b32 = v0 << 2;\n"
        "s_add_i32 s0, s0, %1                             // s0.i32 = s0 + s4; scc=overflow_or_carry (%1=base)\n"
        "v_mov_b32_e32 v1, s0                             // v1 = s0;\n"
        "global_store_dword v0, v1, %0                    // save_dword_to_addr(v1, addr=v0 + s[2:3]) (%0=out)\n"
        :
        : "s"(out), "s"(base)          // %0 是 out, %1 是 base
        : "s0", "v0", "v1", "memory"   // 注意：不要把 %0, %1 用的寄存器也写进破坏列表
    );
}