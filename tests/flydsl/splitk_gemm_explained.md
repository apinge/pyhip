# 当前 FlyDSL Split-K GEMM：从算法、preshuffle 到 gfx942 IR

本文只解释当前目录里的这一条具体路径：

- 源码：[test_gemm.py](./test_gemm.py)
- 初始 FlyDSL IR：[00_origin.mlir](./test_gemm_splitk/gemm_splitk_0/00_origin.mlir)
- 降到 ROCDL 后的 IR：[08_convert_fly_to_rocdl.mlir](./test_gemm_splitk/gemm_splitk_0/08_convert_fly_to_rocdl.mlir)
- LLVM IR：[21_llvm_ir.ll](./test_gemm_splitk/gemm_splitk_0/21_llvm_ir.ll)
- 最终 gfx942 ISA：[22_final_isa.s](./test_gemm_splitk/gemm_splitk_0/22_final_isa.s)
- preshuffle 背景材料：`/opt/ai-framework-labs/flydsl_samples/preshuffle_explained.md`

目标是先建立一张完整地图，后续再逐段细扣 lane layout、地址公式、LDS bank 和指令调度。

## 1. 一句话概括

这是一条 BF16 GEMM：

```text
C[M, N] = A[M, K] @ W[N, K]^T
```

本文中的 `W` 就是 GEMM 的 B operand：host 代码叫它 weight，kernel 参数叫 `arg_b`。后面写成 `B/W` 时表示同一个矩阵。

它让一个 256-thread workgroup 中的 4 个 wave 共同计算同一个 `64 x 64` 输出 tile：

- 4 个 wave 在 K 维分工；
- 每个 wave 都持有一份完整的 `64 x 64` FP32 部分和；
- K 全部算完后，4 份部分和写入 LDS；
- workgroup 内做四路规约；
- 最后转成 BF16，写回 C。

这里的 Split-K 是“单 workgroup 内的 4-wave Split-K”，不是多个 workgroup 分别计算 K 后再用 global atomic 或第二个 kernel 合并。

## 2. 当前 dump 对应的具体问题

当前 `test_gemm.py` 配置为：

| 项目 | 数值 |
|---|---:|
| GPU | AMD Instinct MI308X / gfx942 |
| FlyDSL | 0.3.1 |
| A | BF16 `[32, 10240]` |
| W | BF16 `[320, 10240]`，传入 kernel 前 preshuffle |
| C | BF16 `[32, 320]` |
| `TILE_M` | 64 |
| `TILE_N` | 64 |
| `TILE_K` | 64，每个 wave 的 K 工作量 |
| split waves | 4 |
| block | 256 threads = 4 waves |
| grid | `(1, 5, 1)` |
| K-group 数量 | `10240 / (64 * 4) = 40` |

`M` 在生成的 kernel 中仍是运行时参数，但这次运行传入的是 32。`N=320`、`K=10240` 和 tile 配置已经作为编译期常量进入 IR。

初始 IR 中可以直接看到：

```mlir
gpu.func @gemm_splitk_0(...) kernel attributes {
  known_block_size = array<i32: 256, 1, 1>
}
```

launcher 中可以看到：

```mlir
blocks in (%ceil_div_M_64, %c5, %c1)
threads in (%c256, %c1, %c1)
```

所以本次 `M=32` 时：

```text
grid.x = ceil(32 / 64) = 1
grid.y = 320 / 64 = 5
```

一共启动 5 个 workgroup，每个 workgroup 负责一个名义上的 `64 x 64` C tile。

## 3. 先看数学上的 Split-K

普通 GEMM 的一个输出元素为：

```text
C[m,n] = sum(k=0..10239) A[m,k] * W[n,k]
```

当前 kernel 把 K 写成：

```text
K = 40 groups * 4 waves * 64 K/wave
  = 40 * 4 * 64
  = 10240
```

### 3.1 `40 groups` 具体是什么意思

`40 groups` 指完整的 `K=10240` 被外层 K 循环切成了 40 个宽度为 256 的逻辑分块：

```text
10240 / 256 = 40
```

这里一个 K-group 的 256 又由 workgroup 内的 4 个 wave 分担：

```text
一个 K-group = 4 waves * 64 K/wave
             = 4 * 64
             = 256 K
```

所以完整 K 的层次关系是：

```text
MFMA_K       = 16
每个 wave K = TILE_K = 64 = 4 * MFMA_K
每个 group K= 4 waves * 64 = 256
完整 K      = 40 groups * 256 = 10240
```

把完整 K 轴画开：

```text
K=10240
|
+-- group 0  : K[   0 :  256]
+-- group 1  : K[ 256 :  512]
+-- group 2  : K[ 512 :  768]
|      ...
+-- group 39 : K[9984 : 10240]
```

每个 group 内部再分给 4 个 wave：

```text
                 一个 K-group，共 256 个 K

A[:, k0:k0+256]                    W[:, k0:k0+256]
        |                                  |
        +-- S0：64 个 K -- Wave 0 ---------+ -> P0 += A[:,S0] @ W[:,S0]^T
        +-- S1：64 个 K -- Wave 1 ---------+ -> P1 += A[:,S1] @ W[:,S1]^T
        +-- S2：64 个 K -- Wave 2 ---------+ -> P2 += A[:,S2] @ W[:,S2]^T
        +-- S3：64 个 K -- Wave 3 ---------+ -> P3 += A[:,S3] @ W[:,S3]^T
```

图中的 `S0..S3` 表示 K 坐标集合。为了方便理解，图上把它们画成四份；真实 lane 访问顺序还会被 tiled-MMA K layout 交错。

源码中的循环次数正是：

```python
loop_end = K // TILE_K // 4
         = 10240 // 64 // 4
         = 40
```

dump 出来的初始 IR 也直接固化为：

```mlir
%c40 = arith.constant 40 : index
scf.for %k = %c0 to %c40 step %c1
```

这 40 个 group 是顺序执行的。每个 wave 在 40 次循环中持续更新自己的寄存器 partial；只有全部 40 个 group 结束以后，4 个 wave 才把最终 partial 写入 LDS 做四路规约。

对每个外层 K-group，4 个 wave 各自处理 64 个 K 位置。精确的 K 顺序经过 tiled-MMA layout 交错，不应简单理解为最终寄存器里始终是四段朴素连续区间；但从数学上，每个 wave 负责当前 256-wide K-group 的四分之一。

可以抽象为：

```text
P0[m,n] = sum(k in S0) A[m,k] * W[n,k]
P1[m,n] = sum(k in S1) A[m,k] * W[n,k]
P2[m,n] = sum(k in S2) A[m,k] * W[n,k]
P3[m,n] = sum(k in S3) A[m,k] * W[n,k]

C[m,n] = P0[m,n] + P1[m,n] + P2[m,n] + P3[m,n]
```

其中 `S0..S3` 不重叠，并集覆盖完整的 K。

这条路径的核心收益是：4 个 wave 不在 M/N 上切不同输出块，而是一起推进同一个输出 tile 的 K reduction，适合 M 较小、K 很长的场景。

## 4. 一个 workgroup 到底算什么

源码中：

```python
a_tile = flat_divide(A, make_tile(64, 64 * 4))[..., blk_x, ...]
b_tile = flat_divide(B, make_tile(64, 64 * 4))[..., blk_y, ...]
c_tile = flat_divide(C, make_tile(64, 64))[..., blk_x, blk_y]
```

因此一个 workgroup 的名义工作集是：

```text
A tile: 64 x 256
B tile: 64 x 256
C tile: 64 x 64
```

K=10240 时，这样的 K-group 一共有 40 个。

初始 IR 对应为：

```text
A: (64, 256, ?, 40), stride (..., 256)
B: ((16,4), (8,32), 5, 40), K-group stride 4096 elements
C: (64, 64, ?, 5)
```

这里 B 看起来不像普通二维矩阵，是因为它已经按照 preshuffled 物理布局解释。

### 4.1 全局矩阵如何分给 5 个 workgroup

先忽略 preshuffle，只从 GEMM 的逻辑坐标看：

```text
A: [M=32, K=10240]

       K = 10240 ------------------------------------------------------>
     +------------------------------------------------------------------+
M=32 |                         A[0:32, 0:10240]                          |
     +------------------------------------------------------------------+
       所有 block_y 都读取同一个 A 行块；名义 tile 是 64 行，后 32 行越界


B/W: [N=320, K=10240]，数学上在 GEMM 中使用 W^T

       K = 10240 ------------------------------------------------------>
     +------------------------------------------------------------------+  n=0
 64  | W tile 0: W[  0: 64, :]  -> block(0,0)                           |
     +------------------------------------------------------------------+  n=64
 64  | W tile 1: W[ 64:128, :]  -> block(0,1)                           |
     +------------------------------------------------------------------+  n=128
 64  | W tile 2: W[128:192, :]  -> block(0,2)                           |
     +------------------------------------------------------------------+  n=192
 64  | W tile 3: W[192:256, :]  -> block(0,3)                           |
     +------------------------------------------------------------------+  n=256
 64  | W tile 4: W[256:320, :]  -> block(0,4)                           |
     +------------------------------------------------------------------+  n=320


C: [M=32, N=320]

          N columns --------------------------------------------------->
         0          64         128        192        256        320
       +----------+----------+----------+----------+----------+
M=32   | block0   | block1   | block2   | block3   | block4   |
       | C[:,0:64]|C[:,64:128]| ...     | ...      |C[:,256:320]|
       +----------+----------+----------+----------+----------+
```

用表格表示 block 的输入和输出关系：

| workgroup | A 的逻辑区域 | W 的逻辑区域 | 产生的 C 区域 |
|---|---|---|---|
| `(0,0)` | `A[0:64, 0:10240]` | `W[0:64, 0:10240]` | `C[0:64, 0:64]` |
| `(0,1)` | `A[0:64, 0:10240]` | `W[64:128, 0:10240]` | `C[0:64, 64:128]` |
| `(0,2)` | `A[0:64, 0:10240]` | `W[128:192, 0:10240]` | `C[0:64, 128:192]` |
| `(0,3)` | `A[0:64, 0:10240]` | `W[192:256, 0:10240]` | `C[0:64, 192:256]` |
| `(0,4)` | `A[0:64, 0:10240]` | `W[256:320, 0:10240]` | `C[0:64, 256:320]` |

这里写的是名义 tile 坐标。实际 `M=32`，所以只有 `A[0:32,:]` 和 `C[0:32,:]` 有效；第 32 到 63 行由 buffer bounds 保护。

### 4.2 放大一个 workgroup：A 拿哪块、W 拿哪块

以 `block(0,2)` 为例，它最终负责：

```text
C[0:64, 128:192]
```

完整 K 被切成 40 个 256-wide group。第 `g` 个 group 的逻辑输入为：

```text
k0 = g * 256

A_g = A[0:64,     k0:k0+256]     shape = [64, 256]
W_g = W[128:192,  k0:k0+256]     shape = [64, 256]
```

图示：

```text
                         当前 K-group: [k0, k0+256)

A_g，固定 M rows                         W_g，固定 N rows

       256 个 K                                      256 个 K
  +-------------------------+                  +-------------------------+
  | A[0:64, k0:k0+256]      |                  | W[128:192, k0:k0+256]  |
  |                         |                  |                         |
  |  S0  S1  S2  S3         |                  |  S0  S1  S2  S3         |
  +-------------------------+                  +-------------------------+
     |   |   |   |                                |   |   |   |
     |   |   |   +----------- Wave 3 -------------+   |   |   |
     |   |   +--------------- Wave 2 -----------------+   |   |
     |   +------------------- Wave 1 ---------------------+   |
     +----------------------- Wave 0 -------------------------+

Wave 0: A[0:64, S0] x W[128:192, S0]^T -> P0[64,64]
Wave 1: A[0:64, S1] x W[128:192, S1]^T -> P1[64,64]
Wave 2: A[0:64, S2] x W[128:192, S2]^T -> P2[64,64]
Wave 3: A[0:64, S3] x W[128:192, S3]^T -> P3[64,64]

最终：P0 + P1 + P2 + P3 -> C[0:64, 128:192]
```

每个 `S_w` 有 64 个 K 位置：

```text
|S0| = |S1| = |S2| = |S3| = 64
S0 union S1 union S2 union S3 = [k0, k0+256)
```

图中把 `S0..S3` 横向画开只是为了表达“所有权”。真实 lane load 顺序由：

```text
wave layout: (1,1,4):(0,0,1)
K layout:    (4,16,2):(1,8,4)
```

共同决定，其中存在交错。A 和 W 必须对同一个 `S_w` 取值，才能形成合法的 K reduction。

40 个 group 依次累加：

```text
g=0:  K[   0: 256]
g=1:  K[ 256: 512]
...
g=39: K[9984:10240]
```

每个 wave 的 `P_w` 一直留在寄存器中，直到 40 个 group 全部算完。

### 4.3 一个 K-group 内的计算图

```text
                         一个 workgroup / 一个 C 64x64 tile

              A_g [64x256]                    W_g [64x256]
                    |                               |
                    +---------------+---------------+
                                    |
                     按同一组 K 集合 S0/S1/S2/S3 分给 4 waves
                                    |
            +-----------------------+-----------------------+
            |                       |                       |                       |
            v                       v                       v                       v
         Wave 0                  Wave 1                  Wave 2                  Wave 3
       A[:,S0],W[:,S0]         A[:,S1],W[:,S1]         A[:,S2],W[:,S2]         A[:,S3],W[:,S3]
            |                       |                       |                       |
            v                       v                       v                       v
       partial P0[64x64]       partial P1[64x64]       partial P2[64x64]       partial P3[64x64]

  注意：上面的 P0..P3 会跨 40 个 K-group 持续累加，不是每个 group 都进 LDS。
```

#### 4.3.1 从一条 MFMA 推导每线程的 C accumulator 数量

gfx942 的一个 wave 有 64 个 lane，也就是 64 个 thread。`MFMA 16x16x16` 中的三个维度分别表示输出行数 M、输出列数 N，以及本次累加的 K 深度。一条 MFMA 由整个 wave 协作，更新一个 `16x16` 输出块：

```text
每个 wave：16 * 16 = 256 个 FP32 accumulator
每个 thread：256 / 64 = 4 个 FP32 accumulator
```

覆盖一个 `64x64` 输出 tile 时，M、N 两个方向都要展开 4 次，因此共有 `4*4=16` 个输出块：

```text
                         N = 64
              +-------+-------+-------+-------+
              | 16x16 | 16x16 | 16x16 | 16x16 |
              +-------+-------+-------+-------+
              | 16x16 | 16x16 | 16x16 | 16x16 |
     M = 64   +-------+-------+-------+-------+
              | 16x16 | 16x16 | 16x16 | 16x16 |
              +-------+-------+-------+-------+
              | 16x16 | 16x16 | 16x16 | 16x16 |
              +-------+-------+-------+-------+
```

固定一轮 K 深度为 16 的计算：

```text
每 wave 的 MFMA 条数 = (64 / 16) * (64 / 16) = 16 条
每 thread 的 C accumulator = 16 块 * 每块 4 个 = 64 个 FP32
```

“4 条 MFMA，每条每线程 4 个，所以共 16 个 FP32”只适用于这 4 条指令更新 4 个不同输出块的情况；4 个 `16x16` 块只能覆盖整个 `64x64` tile 的四分之一。

源码中的初始化直接对应每线程的 64 个 accumulator：

```python
c_frag.store(Vec.filled(TILE_M * TILE_N // 64, 0, fx.Float32))
#                      64 * 64 / 64 = 64
```

初始 IR 中，外层 K 循环携带的状态也是每线程的 `vector<64xf32>`。

#### 4.3.2 沿 K 重复 MFMA 时复用 accumulator

对一个 wave、一个 K-group：

```text
64x64 输出 tile = 4x4 个 16x16 MFMA 输出块
wave 的 K 深度 64 = 4 个 MFMA_K=16

所以：4（M 方向）* 4（N 方向）* 4（K 方向）= 64 条 MFMA
```

沿 K 的 4 轮更新同一批输出位置，始终复用每线程的 64 个 FP32 accumulator；全部 40 个 K-group 也持续累加到这批寄存器中。

| MFMA 的重复方向 | 更新哪些 C 元素 | accumulator 数量如何变化 |
|---|---|---|
| 沿 M/N 展开 | 不同输出位置 | 为不同输出块分配更多 accumulator |
| 沿 K 累加 | 相同输出位置 | 复用已有 accumulator |

### 4.4 40 个 K-group 算完后如何规约

每个 wave 最终有一份完整的 FP32 partial：

```text
P0[64,64]  P1[64,64]  P2[64,64]  P3[64,64]
```

LDS 被逻辑解释为 `[256,64]`，容纳 4 份 `64x64` partial，共 65536 bytes。当前写入 layout 按每份 partial 的 16 行交错放置。令 `m = 16 * m_block + m_in`，则 swizzle 前的逻辑坐标为：

```text
P_w[m,n] -> LDS[64 * m_block + 16 * w + m_in, n]

LDS 逻辑行       存放内容（每行均有 64 列）
  0..15          P0[ 0:16, :]
 16..31          P1[ 0:16, :]
 32..47          P2[ 0:16, :]
 48..63          P3[ 0:16, :]
 64..79          P0[16:32, :]
 80..95          P1[16:32, :]
 96..111         P2[16:32, :]
112..127         P3[16:32, :]
   ...
192..207         P0[48:64, :]
208..223         P1[48:64, :]
224..239         P2[48:64, :]
240..255         P3[48:64, :]
```

这个映射来自源码中写 LDS 的 thread-value layout `((16,4,4),4):((1,256,16),64)` 和 tile `(64,16)`；最终物理地址还要应用 `S<3,3,3>` swizzle。

然后：

```text
         P0[m,n] ----+
         P1[m,n] ----+
                      +--> FP32 add --> C_fp32[m,n]
         P2[m,n] ----+
         P3[m,n] ----+

 C_fp32[m,n] -> fast FP32-to-BF16 -> global C[m,n]
```

#### 4.4.1 为什么每线程最终只有 16 个输出

计算阶段，每个 wave 都持有一份 `64x64` partial；规约完成后，整个 workgroup 共同持有一份 `64x64` 最终 C。源码分别构造写 LDS、读 LDS 和写 global 的 copy layout，在规约时重新分配输出位置，由 256 个线程共同完成：

```text
计算阶段，每线程的 partial 数量 = 64 * 64 / 64  = 64
写回阶段，每线程的最终输出数量 = 64 * 64 / 256 = 16
```

对于分配给某个线程的 16 个 `(m,n)` 坐标，它从 LDS 中读取 `P0..P3` 在这些相同坐标上的值，共 `4*16=64` 个 FP32，逐坐标相加后得到 16 个最终值。这里的四路规约发生在 wave 的部分和之间。

| 阶段 | 每个 thread | 每个 wave | 整个 workgroup |
|---|---|---|---|
| 计算结束，尚未规约 | 64 个 FP32 partial | 一份 `64x64` partial | 四份 `64x64` partial |
| 规约完成，准备写回 | 16 个最终值 | 1024 个最终值 | 一份 `64x64` 最终 C |

因此，“每个 wave 的 C 是 `64x64`”描述的是计算阶段的 partial tile。写回阶段，每个 wave 负责最终 C 的四分之一。以上数量按 `64x64` 逻辑 tile 计算，当前 `M=32` 时的有效输出边界另见第 11 节。

```text
计算阶段：每线程持有 64 个 FP32 partial accumulator
规约阶段：每线程从四个 wave 的 partial 各取 16 个值
           -> 4 x vector<16xf32>
           -> 三次相加
           -> 16 个最终 BF16 输出
```

#### 4.4.2 为什么是 16 条 LDS 写指令和 16 条 LDS 读指令

`ds_write_b128` 和 `ds_read_b128` 的 `b128` 表示每个参与执行的 lane 传输 128 bit，即 `128/32=4` 个 FP32。这些指令由整个 wave 执行；一条指令在 64 个 lane 上合计传输 `64*4=256` 个 FP32。

```text
写 LDS：每线程 64 个 FP32 / 每条 4 个 = 16 条 ds_write_b128
读 LDS：每线程 (4 * 16) 个 FP32 / 每条 4 个 = 16 条 ds_read_b128
```

因此，`16 x ds_write_b128` 中的 16 是指令条数，每线程实际写入 64 个 FP32。例如最终 ISA 的 `ds_write_b128 v66, v[30:33]` 就将每个 lane 的 4 个 32-bit 寄存器写入 LDS。

ISA 正好对应：

```text
16 x ds_write_b128      把每线程 64 个 FP32 partial 写入 LDS
1  x s_barrier          等所有 wave 写完
16 x ds_read_b128       读取四份 partial，共 64 个 FP32
packed FP32 add         四路规约成 16 个 FP32
4  x buffer_store_dwordx2
                         写回 16 个 BF16
```

## 5. 这条 kernel 为什么需要 preshuffle weight

host 侧先创建普通 BF16 权重：

```python
w_ = torch.randn([N, K], dtype=torch.bfloat16)
```

再调用 AITER：

```python
w_shuffled = shuffle_weight(w_).reshape(N // 16, -1)
```

默认 `shuffle_weight(..., layout=(16,16))` 对 BF16 有：

```text
BN = 16
BK = 32
每个 128-bit load = 16 bytes = 8 个 BF16
```

逻辑坐标拆分为：

```text
n = n_blk * 16 + n_in
k = k_blk * 32 + k_sub * 8 + k_lane

n_in   in [0, 15]
n_blk  in [0, 19]      因为 N/16 = 320/16 = 20
k_lane in [0, 7]
k_sub  in [0, 3]
k_blk  in [0, 319]     因为 K/32 = 10240/32 = 320
```

原始 row-major 物理顺序近似是：

```text
[n_blk, n_in, k_blk, k_sub, k_lane]
```

对当前 `W.shape = [320, 10240]`，AITER 实际先执行：

```python
x.view(
    -1,
    N // 16,
    16,
    K // 32,
    4,
    8,
)
```

代入具体数字：

```text
[batch, n_blk, n_in, k_blk, k_sub, k_lane]
= [1,     20,    16,   320,   4,     8]
```

忽略最前面的 batch 维，就是上面写的：

```text
[n_blk, n_in, k_blk, k_sub, k_lane]
= [20,    16,   320,   4,     8]
```

各维乘积仍然是原始权重元素总数：

```text
1 * 20 * 16 * 320 * 4 * 8
= 3,276,800
= 320 * 10240
```

preshuffle 后变为：

```text
[n_blk, k_blk, k_sub, n_in, k_lane]
```

AITER 对上面的 6D view 执行：

```python
x.permute(0, 1, 3, 4, 2, 5)
```

因此完整 shape 和维度顺序变成：

```text
[batch, n_blk, k_blk, k_sub, n_in, k_lane]
= [1,     20,    320,   4,     16,   8]
```

忽略 batch 维：

```text
[n_blk, k_blk, k_sub, n_in, k_lane]
= [20,    320,   4,     16,   8]
```

`contiguous()` 按这个新顺序真正重排物理内存，随后又 reshape 回：

```text
[N, K] = [320, 10240]
```

测试代码再把它看成：

```text
[N/16, 16*K] = [20, 163840]
```

最后两次 reshape 都不会撤销 preshuffle；它们只改变 tensor 的表面 shape，底层元素顺序仍是 `[1,20,320,4,16,8]` 对应的连续顺序。

也就是把 `k_blk/k_sub` 提到 `n_in` 前面，让 16 个 lane 对应的 16 行小片段在内存里紧邻。

kernel 用下面的 layout 重新解释这个物理 buffer：

```python
shape  = ((16, N // 16), (8, 4, K // 32))
stride = ((8, 16 * K),  (1, 128, 512))
```

代入当前数字，dump 中就是：

```mlir
shape  = ((16,20),(8,4,320))
stride = ((8,163840),(1,128,512))
```

这里 B 的 K 维被拆成：

```text
(8, 4, 320)
 |  |   |
 |  |   +-- k_blk：一共有 320 个 K32 block
 |  +------ k_sub：每个 K32 block 内有 4 个 K8 小段
 +--------- k_lane：每个 K8 小段内有 8 个 BF16
```

也就是：

```text
K = 8 * 4 * 320
  = 10240
```

这里中间的 `4` 是 `k_sub=0,1,2,3`，不是四个 wave。四个 wave 会在后面的线程 partition 中共同选择 `k_blk=320` 这一维。

### 5.1 图：从整个 preshuffled B 放大到一个 wave 的 load

先不看 offset 公式。下面从整个 B buffer 开始逐级放大。

#### 图一：preshuffle 后整个 B 的最外层仍按 16 行一组

B/W 的逻辑 shape 是 `[320,10240]`。N 方向被切成 20 个 `n_blk`，每个 `n_blk` 包含 16 行：

```text
preshuffled B 的最外层

 n_blk     对应逻辑 W rows                  被哪个 workgroup 使用

   0       W[  0: 16, :]  ┐
   1       W[ 16: 32, :]  │
   2       W[ 32: 48, :]  ├── block_y = 0，W[0:64, :]
   3       W[ 48: 64, :]  ┘
          ------------------------------------------------
   4       W[ 64: 80, :]  ┐
   5       W[ 80: 96, :]  │
   6       W[ 96:112, :]  ├── block_y = 1，W[64:128, :]
   7       W[112:128, :]  ┘
          ------------------------------------------------
   8       W[128:144, :]  ┐
   9       W[144:160, :]  │
  10       W[160:176, :]  ├── block_y = 2，W[128:192, :]
  11       W[176:192, :]  ┘
          ------------------------------------------------
  12..15                  ─── block_y = 3，W[192:256, :]
  16..19                  ─── block_y = 4，W[256:320, :]
```

所以 `block_y=2` 直接选择最外层连续的四个 `n_blk=8,9,10,11`。

每个 `n_blk` 的大小为：

```text
16 rows * K=10240 = 163840 BF16 elements
```

这就是 layout 中 `n_blk stride = 163840` 的来源。

#### 图二：放大 block_y=2 的一个 K-group

一个 K-group 包含 256 个 K，也就是 8 个 `K32 block`。每个格子代表逻辑上的 `16 rows x 32 K`：

```text
K-group g

k_blk:       8g+0       8g+1       8g+2       8g+3       8g+4       8g+5       8g+6       8g+7
wave:        Wave 0     Wave 1     Wave 2     Wave 3     Wave 0     Wave 1     Wave 2     Wave 3
k_repeat:       0          0          0          0          1          1          1          1
             +----------+----------+----------+----------+----------+----------+----------+----------+
n_blk=8     | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    |
W[128:144]  | microtile| microtile| microtile| microtile| microtile| microtile| microtile| microtile|
             +----------+----------+----------+----------+----------+----------+----------+----------+
n_blk=9     | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    |
W[144:160]  | microtile| microtile| microtile| microtile| microtile| microtile| microtile| microtile|
             +----------+----------+----------+----------+----------+----------+----------+----------+
n_blk=10    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    |
W[160:176]  | microtile| microtile| microtile| microtile| microtile| microtile| microtile| microtile|
             +----------+----------+----------+----------+----------+----------+----------+----------+
n_blk=11    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    | 16x32    |
W[176:192]  | microtile| microtile| microtile| microtile| microtile| microtile| microtile| microtile|
             +----------+----------+----------+----------+----------+----------+----------+----------+
```

Wave 0 选择第 0 列和第 4 列，并且四个 `n_blk` row 都要：

```text
Wave 0 从 B 取的 microtiles

                         第一个 K32                  第二个 K32
                         k_blk=8g+0                  k_blk=8g+4

n_blk=8  W[128:144]      [16x32]                     [16x32]
n_blk=9  W[144:160]      [16x32]                     [16x32]
n_blk=10 W[160:176]      [16x32]                     [16x32]
n_blk=11 W[176:192]      [16x32]                     [16x32]

合起来：
4 * 16 N rows = 64 N rows
2 * 32 K      = 64 K

所以 Wave 0 逻辑上拿到 W[128:192, S_wave0]，shape 为 [64,64]。
```

其他 wave 只是换列：

```text
Wave 1 -> 第 1、5 列
Wave 2 -> 第 2、6 列
Wave 3 -> 第 3、7 列
```

#### 图三：A 和 B 选择的逻辑坐标

仍以 Wave 0 为例。下面两侧都画逻辑矩阵坐标，每个格子表示需要哪些元素；物理内存顺序在图四中展开：

```text
A 的逻辑坐标                            W/B 的逻辑坐标

A rows 0:64                            W rows 128:192

   k_blk=8g+0   k_blk=8g+4                k_blk=8g+0   k_blk=8g+4
   +----------+ +----------+              +----------+ +----------+
A0 | 16x32    | | 16x32    |          B0  | 16x32    | | 16x32    |
   +----------+ +----------+              +----------+ +----------+
A1 | 16x32    | | 16x32    |          B1  | 16x32    | | 16x32    |
   +----------+ +----------+              +----------+ +----------+
A2 | 16x32    | | 16x32    |          B2  | 16x32    | | 16x32    |
   +----------+ +----------+              +----------+ +----------+
A3 | 16x32    | | 16x32    |          B3  | 16x32    | | 16x32    |
   +----------+ +----------+              +----------+ +----------+

   A0..A3 是四个 M16 row blocks           B0..B3 是四个 N16 row blocks

                          使用完全相同的两个 k_blk
                                        |
                                        v
                     A[:, S_wave0] @ B[:, S_wave0]^T
                                        |
                                        v
                              P0[64,64]
```

A、B 在逻辑上选择同一组 K，才能计算对应的 `A[m,k] * W[n,k]`。preshuffle 改变 W 元素的存放位置，kernel 的 B layout 负责把逻辑坐标映射到新地址。

#### 图四：把一个 B 微块展开成真实的内存顺序

固定 `block_y=2`、`g=0`、Wave 0，放大它的第一个 B 微块 `W[128:144, 0:32]`。先给每行的四段数据起名字，方便追踪它们移动到哪里：

```text
x_r = W[128+r,  0: 8]     y_r = W[128+r,  8:16]
z_r = W[128+r, 16:24]     t_r = W[128+r, 24:32]

r = 0..15
每个 [x_r] / [y_r] / [z_r] / [t_r] 都是 8 个 BF16 = 16 bytes
```

shuffle 前，W 的物理内存按完整的 K 行排列。图中每行读到最右端，再接下一行；两行起点相隔 `10240*2=20480 bytes`：

```text
                         K 方向 ->
                 0:8    8:16   16:24  24:32       32:10240
W row 128      [ x0 ][ y0 ][ z0 ][ t0 ][ 该行剩余的 K 数据 ... ]
W row 129      [ x1 ][ y1 ][ z1 ][ t1 ][ 该行剩余的 K 数据 ... ]
W row 130      [ x2 ][ y2 ][ z2 ][ t2 ][ 该行剩余的 K 数据 ... ]
   ...
W row 143      [x15 ][y15 ][z15 ][t15 ][ 该行剩余的 K 数据 ... ]

相邻 lane 0、1、2 需要的 x0、x1、x2，原来相距一整行。
```

preshuffle 把同一个 K8 小段的 16 行数据放到一起。下面是该微块重排后的连续 1024 bytes，按每行 256 bytes 换行显示；这里的换行表示内存地址继续递增：

```text
微块内 byte offset                  物理内存顺序 ->

   0             [ x0 ][ x1 ][ x2 ] ... [x15 ]
                    |     |     |          |
                  lane0 lane1 lane2      lane15

 256             [ y0 ][ y1 ][ y2 ] ... [y15 ]
                    |     |     |          |
                 lane16 lane17 lane18    lane31

 512             [ z0 ][ z1 ][ z2 ] ... [z15 ]
                    |     |     |          |
                 lane32 lane33 lane34    lane47

 768             [ t0 ][ t1 ][ t2 ] ... [t15 ]
                    |     |     |          |
                 lane48 lane49 lane50    lane63

1024             下一个 K32 微块的起点
```

例如，`x1` 始终是逻辑元素 `W[129,0:8]`；shuffle 后它紧挨着 `x0=W[128,0:8]` 存放。数据值和逻辑坐标保持对应，物理地址发生变化。

在这一条 B 的 `buffer_load_dwordx4` 中：

```text
lane  0：微块 byte [   0:  16] -> W[128,  0: 8]
lane  1：微块 byte [  16:  32] -> W[129,  0: 8]
lane 15：微块 byte [ 240: 256] -> W[143,  0: 8]
lane 16：微块 byte [ 256: 272] -> W[128,  8:16]
lane 63：微块 byte [1008:1024] -> W[143, 24:32]

每个 lane：8 个 BF16 = 16 bytes
整个 wave：64 * 16 bytes = 连续 1024 bytes = 一个 16x32 BF16 微块
```

对照对应的 A load，A 仍按 row-major 地址读取：lane 0、1 分别从 `A[0,0:8]` 和 `A[1,0:8]` 取数，起点相差 20480 bytes。B 的 lane 0、1 起点只相差 16 bytes。这正是 preshuffle 改善当前 B 访问的地方。

#### 图五：这些连续微块在整个 B buffer 中如何排列

把图四中的连续 1024 bytes 记作一个 `Tj`，其中 `j=k_blk`。固定 `n_blk` 时，`Tj` 包含该组 16 行的 `K[32j:32j+32]`：

```text
地址递增 ->

n_blk=8，W rows 128:144：
[ T0 ][ T1 ][ T2 ][ T3 ][ T4 ][ T5 ][ T6 ][ T7 ] ... [ T319 ]
   ^                       ^
 Wave 0                  Wave 0       <- g=0 时 Wave 0 选这两个微块

紧接着是 n_blk=9，W rows 144:160：
[ T0 ][ T1 ][ T2 ][ T3 ][ T4 ][ T5 ][ T6 ][ T7 ] ... [ T319 ]
   ^                       ^
 Wave 0                  Wave 0

再接 n_blk=10，W rows 160:176；随后 n_blk=11，W rows 176:192。

每个 Tj：1024 bytes
每个 n_blk：320 * 1024 = 327680 bytes
```

Wave 0 在这个 K-group 中对 `n_blk=8,9,10,11` 各读 `T0` 和 `T4`，所以执行 8 条 B load。每条 load 的 64 个 lane 覆盖一个连续微块；不同 load 之间会跳到另一个微块。因此，preshuffle 保证这里每条 B load 的 lane 地址连续，整个 wave 的全部输入仍分布在 8 个微块中。

ROCm 上很多 MFMA GEMM 都采用类似思想：把 weight 提前变成贴近 lane/MFMA 消费顺序的物理布局。但具体 permutation 不是所有 kernel 完全相同，它仍取决于 dtype、MFMA atom、load 宽度和 tiled-MMA layout。

#### 图看懂以后，再把它压缩成 offset 公式

上面的图对应：

```text
offset =
    n_blk  * 163840
  + k_blk  * 512
  + k_sub  * 128
  + n_in   * 8
  + k_lane * 1
```

各 stride 的图形含义是：

```text
163840 -> 跳到下一个 16-row n_blk
512    -> 跳到右边下一个 16x32 microtile
128    -> 在 microtile 内跳到下一个 K8 横条
8      -> 在 K8 横条内跳到下一行
1      -> 这一行内移动一个 BF16
```

几个具体例子：

```text
W[0, 0:8]   -> physical element offset [0, 8)
W[1, 0:8]   -> physical element offset [8, 16)
W[15, 0:8]  -> physical element offset [120, 128)
W[0, 8:16]  -> physical element offset [128, 136)
W[0, 32:40] -> physical element offset [512, 520)
```

### 5.2 `block(0,2)` 如何从 preshuffled B 中拿到 `W[128:192,S_w]`

这里把 workgroup、wave、lane 和 preshuffled B 的坐标完整对上。

先把线程号拆开：

```text
tid  = 64 * wave_id + lane
lane = 16 * q + r

wave_id in [0,3]
lane    in [0,63]
q       in [0,3]
r       in [0,15]
```

对于 `block(0,2)`：

```text
block_x = 0
block_y = 2

M tile 起点 = block_x * 64 = 0
N tile 起点 = block_y * 64 = 128
```

在第 `g` 个 K-group 中，每个线程访问的逻辑 K 可以写成：

```text
k = 256*g
  + 32*wave_id
  + 8*q
  + 128*k_repeat
  + k_lane

k_repeat in [0,1]
k_lane   in [0,7]
```

这里需要区分三个不同维度：

```text
wave_id  in [0,1,2,3]   -> 4 个 wave
k_repeat in [0,1]       -> 每个 wave 取两个不同的 K32 block
q/k_sub  in [0,1,2,3]   -> 一个 K32 block 内的四个连续 K8 小段
```

所以 `k_repeat` 确实只有两个值 0、1，不是 0、1、2、3。四个值来自独立的 `wave_id`。

它们共同把 preshuffle layout 中的 `k_blk=320` 维拆开：

```text
k_blk = 8*g + wave_id + 4*k_repeat
```

为什么是 8？因为一个 K-group 包含：

```text
4 waves * 2 k_repeat = 8 个 K32 block
```

每个 K32 block 有 32 个 K，因此：

```text
8 K32 blocks * 32 K/block = 256 K/group
```

对 `g=0` 展开：

| `k_repeat` | Wave 0 | Wave 1 | Wave 2 | Wave 3 |
|---:|---|---|---|---|
| 0 | `k_blk=0`，`K[0:32]` | `k_blk=1`，`K[32:64]` | `k_blk=2`，`K[64:96]` | `k_blk=3`，`K[96:128]` |
| 1 | `k_blk=4`，`K[128:160]` | `k_blk=5`，`K[160:192]` | `k_blk=6`，`K[192:224]` | `k_blk=7`，`K[224:256]` |

可以直观看成：

```text
                         wave_id
                  0         1         2         3
              +---------+---------+---------+---------+
k_repeat=0    | k_blk 0 | k_blk 1 | k_blk 2 | k_blk 3 |
              | K 0:32  | K32:64  | K64:96  | K96:128 |
              +---------+---------+---------+---------+
k_repeat=1    | k_blk 4 | k_blk 5 | k_blk 6 | k_blk 7 |
              |K128:160 |K160:192 |K192:224 |K224:256 |
              +---------+---------+---------+---------+
```

然后每一个 `k_blk` 内部再由 `q`，也就是 preshuffle shape 中的 `k_sub`，拆成四段：

```text
一个 K32 block

q/k_sub=0 -> 相对 K[ 0: 8]
q/k_sub=1 -> 相对 K[ 8:16]
q/k_sub=2 -> 相对 K[16:24]
q/k_sub=3 -> 相对 K[24:32]
```

最后 `k_lane=0..7` 选择这个 K8 小段里的具体元素。

因此完整层次是：

```text
g          -> 选择第几个 256-K group，共 40 个
k_repeat   -> 选择 group 的前/后四个 K32 block
wave_id    -> 选择这四个 K32 block 中的哪一个
q/k_sub    -> 选择该 K32 block 中的哪一个 K8 小段
k_lane     -> 选择 K8 小段中的具体 BF16
```

维度数量也完全对上 preshuffle 的 `k_blk=320`：

```text
40 groups * 2 k_repeat * 4 waves
= 320 K32 blocks
```

因此 4 个 wave 在一个 256-wide K-group 中实际拿到的 K 集合是：

```text
Wave 0: [  0: 32] union [128:160]，再加 256*g
Wave 1: [ 32: 64] union [160:192]，再加 256*g
Wave 2: [ 64: 96] union [192:224]，再加 256*g
Wave 3: [ 96:128] union [224:256]，再加 256*g
```

所以前文中的 `S0..S3` 不是四个简单连续的 64-wide 区间，而是每个 wave 各拿两个 32-wide 区间：

```text
一个 K-group: 0 ---------------------------------------------------- 255

第一半  K[0:128]
         +---------+---------+---------+---------+
         | Wave 0  | Wave 1  | Wave 2  | Wave 3  |
         |  0:32   | 32:64   | 64:96   | 96:128  |
         +---------+---------+---------+---------+

第二半  K[128:256]
         +---------+---------+---------+---------+
         | Wave 0  | Wave 1  | Wave 2  | Wave 3  |
         |128:160  |160:192  |192:224  |224:256  |
         +---------+---------+---------+---------+
```

对 A，线程还会用 `m_repeat in [0,3]` 选择 4 组相隔 16 行的 M：

```text
m = block_x*64 + r + 16*m_repeat
  = r, r+16, r+32, r+48
```

对 B/W，线程用 `n_repeat in [0,3]` 选择当前 N tile 中 4 组相隔 16 行的 N：

```text
n = block_y*64 + r + 16*n_repeat
  = 128+r, 144+r, 160+r, 176+r
```

于是同一个线程在同一个 K 坐标上，同时取得：

```text
A[m, k]
W[n, k]
```

这就是 A fragment 和 B fragment 能在 MFMA 中正确配对的关键。

以 `g=0` 的几个线程为例：

| tid | wave | `q,r` | A 的 M rows | W 的 N rows | 两段 K |
|---:|---:|---:|---|---|---|
| 0 | 0 | `0,0` | `0,16,32,48` | `128,144,160,176` | `0:8`、`128:136` |
| 1 | 0 | `0,1` | `1,17,33,49` | `129,145,161,177` | `0:8`、`128:136` |
| 16 | 0 | `1,0` | `0,16,32,48` | `128,144,160,176` | `8:16`、`136:144` |
| 64 | 1 | `0,0` | `0,16,32,48` | `128,144,160,176` | `32:40`、`160:168` |

当前 `M=32`，所以上表里 A 的 row 32、48 等访问会 OOB-zero；但对应 fragment 槽位和 MFMA 仍然存在。

现在看 preshuffled B 的物理坐标。对 `block_y=2`：

```text
n_blk_base = block_y * (64/16)
           = 2 * 4
           = 8
```

每个线程的 B 坐标为：

```text
n_blk  = 8 + n_repeat
n_in   = r
k_blk  = 8*g + wave_id + 4*k_repeat
k_sub  = q
k_lane = 0..7
```

代入 preshuffle offset：

```text
B_physical_offset =
    (8 + n_repeat)                    * 163840
  + (8*g + wave_id + 4*k_repeat)      * 512
  + q                                 * 128
  + r                                 * 8
  + k_lane
```

这里每一项都能在 IR partition 中找到：

```mlir
B thread partition
shape  = ((8,1),4,2,40)
stride = ((1,0),163840,2048,4096)
```

对应关系：

| partition 维度 | shape | stride | 含义 |
|---|---:|---:|---|
| `k_lane` | 8 | 1 | 一次读取连续 8 个 BF16 |
| broadcast | 1 | 0 | layout 中的广播维 |
| `n_repeat` | 4 | 163840 | `n_blk=8,9,10,11`，即 N rows `128:192` |
| `k_repeat` | 2 | 2048 | 跳到当前 wave 的第二个 32-wide K 段 |
| `g` | 40 | 4096 | 跳到下一个 256-wide K-group |

`tid` 本身提供了剩余坐标：

```text
wave_id -> k_blk 在 group 内的 wave 偏移
q       -> k_sub
r       -> n_in
```

因此代码里虽然没有直接写：

```python
W[128:192, S_wave]
```

但下面三层组合已经表达了完全相同的选择：

```text
block_y
  -> 先选 W 的 64-row N tile

make_tiled_copy_B(...).get_slice(tid)
  -> 再按 wave/lane 选择 N row 和 K 子集

b_tensor_thr[..., g]
  -> 最后选择当前第 g 个 256-wide K-group
```

### 5.3 `64 x 64` C tile 内不是 4 个 MFMA，而是 `4 x 4` 个 MFMA 输出块

一个 `MFMA(16,16,16)` 一次产生一个 `16 x 16` 的 C 子块。`64 x 64` 输出 tile 在 M/N 上分别切成 4 份：

```text
A 的 M row groups：

A0 = A[ 0:16, S_w]
A1 = A[16:32, S_w]
A2 = A[32:48, S_w]
A3 = A[48:64, S_w]

block(0,2) 的 W/B row groups：

B0 = W[128:144, S_w]
B1 = W[144:160, S_w]
B2 = W[160:176, S_w]
B3 = W[176:192, S_w]
```

固定一个 wave 和一个 `K=16` MFMA slice，A/B 的组合形成 16 个输出子块：

```text
                         C 的 N 方向
                  128:144   144:160   160:176   176:192
                    B0         B1         B2         B3
               +----------+----------+----------+----------+
M   0:16  A0   | A0*B0^T  | A0*B1^T  | A0*B2^T  | A0*B3^T  |
               +----------+----------+----------+----------+
   16:32  A1   | A1*B0^T  | A1*B1^T  | A1*B2^T  | A1*B3^T  |
               +----------+----------+----------+----------+
   32:48  A2   | A2*B0^T  | A2*B1^T  | A2*B2^T  | A2*B3^T  |
               +----------+----------+----------+----------+
   48:64  A3   | A3*B0^T  | A3*B1^T  | A3*B2^T  | A3*B3^T  |
               +----------+----------+----------+----------+
```

每一个格子是一个 `16 x 16` C subtile，需要一条 wave-level：

```asm
v_mfma_f32_16x16x16_bf16
```

所以不是“一个 `64 x 64` 里只有 4 个 MFMA”，而是：

```text
一个 K=16 slice:
    4 个 M blocks * 4 个 N blocks
  = 16 条 MFMA

一个 wave 在一个 K-group 中负责 K=64:
    64 / 16 = 4 个 K slices

合计:
    16 MFMA/K-slice * 4 K-slices
  = 64 MFMA/wave/K-group
```

这 16 个 C subtile 在整个 K 循环中持续累加：

```text
C00 += A0 @ B0^T
C01 += A0 @ B1^T
...
C32 += A3 @ B2^T
C33 += A3 @ B3^T
```

实现中 `fx.gemm()` 的 operand 顺序写成了 `b_frag, a_frag`，这是为了匹配当前 MFMA/register layout；逻辑数学仍然是：

```text
C[M,N] += A[M,K] @ W[N,K]^T
```

对当前 `M=32`：

```text
A0、A1 对应有效输出行
A2、A3 的 A load 越界返回 0

所以 C 的前两行 16-row subtile 有效，后两行 subtile 虽然仍执行 MFMA，结果不会形成有效 global C store。
```

### 5.4 先只看一个具体 wave：Wave 0 到底拿 A/B 的哪里

先不看 `q`、`r` 和 offset 公式。固定下面这个场景：

```text
workgroup = block(0,2)
K-group   = group 0，也就是完整 K 的 [0:256]
wave      = Wave 0
```

这个 workgroup 要计算：

```text
C[0:64, 128:192]
```

因此它需要：

```text
A 的 M rows: 0:64
B 的 N rows: 128:192
```

#### 第一步：4 个 wave 怎样瓜分当前 K[0:256]

“32-wide”只表示“在 K 方向连续 32 列”，没有额外含义。

当前 K-group 有 256 列。实际分配如下：

```text
K[0:256]

          第一半 K[0:128]                         第二半 K[128:256]

     0       32      64      96      128      160      192      224      256
     +-------+-------+-------+-------+--------+--------+--------+--------+
     |Wave 0 |Wave 1 |Wave 2 |Wave 3 | Wave 0 | Wave 1 | Wave 2 | Wave 3 |
     +-------+-------+-------+-------+--------+--------+--------+--------+

Wave 0 拿：K[ 0: 32] 和 K[128:160]
Wave 1 拿：K[32: 64] 和 K[160:192]
Wave 2 拿：K[64: 96] 和 K[192:224]
Wave 3 拿：K[96:128] 和 K[224:256]
```

因此 Wave 0 总共拿 64 个 K：

```text
Wave 0 K = K[0:32] union K[128:160]
```

#### 第二步：Wave 0 从 A 拿什么

先把 A 的 64 行分为四组，每组 16 行：

```text
A0 = A[ 0:16, :]
A1 = A[16:32, :]
A2 = A[32:48, :]
A3 = A[48:64, :]
```

Wave 0 从四组 A row 中都取自己的 64 个 K：

```text
                 K[0:32]              K[128:160]

A0 rows  0:16   A[ 0:16,  0:32]      A[ 0:16,128:160]
A1 rows 16:32   A[16:32,  0:32]      A[16:32,128:160]
A2 rows 32:48   A[32:48,  0:32]      A[32:48,128:160]
A3 rows 48:64   A[48:64,  0:32]      A[48:64,128:160]
```

画成矩阵块：

```text
                         Wave 0 读取的 A

                      K[0:32]          K[128:160]
                  +---------------+   +---------------+
A rows  0:16      |      A00      |   |      A04      |
                  +---------------+   +---------------+
A rows 16:32      |      A10      |   |      A14      |
                  +---------------+   +---------------+
A rows 32:48      |      A20      |   |      A24      |  当前 M=32，OOB -> 0
                  +---------------+   +---------------+
A rows 48:64      |      A30      |   |      A34      |  当前 M=32，OOB -> 0
                  +---------------+   +---------------+
```

#### 第三步：Wave 0 从 preshuffled B 拿什么

`block_y=2` 选择 B/W 的 N rows `128:192`。同样分为四组，每组 16 行：

```text
B0 = W[128:144, :]
B1 = W[144:160, :]
B2 = W[160:176, :]
B3 = W[176:192, :]
```

Wave 0 必须从 B 取和 A 完全相同的 K：

```text
                         Wave 0 读取的 B/W

                      K[0:32]          K[128:160]
                  +---------------+   +---------------+
W rows 128:144    |      B00      |   |      B04      |
                  +---------------+   +---------------+
W rows 144:160    |      B10      |   |      B14      |
                  +---------------+   +---------------+
W rows 160:176    |      B20      |   |      B24      |
                  +---------------+   +---------------+
W rows 176:192    |      B30      |   |      B34      |
                  +---------------+   +---------------+
```

上图是逻辑矩阵坐标。B 已经 preshuffle，所以物理内存并不是四条普通 row-major 矩形；但是 kernel 的 B layout 会把 preshuffled 地址重新解释回这些逻辑坐标。

可以先记住：

```text
A 和 B 的 M/N 行不同，但 K 必须完全相同。

A[某些 M rows, Wave 0 的 K]
                x
B[某些 N rows, Wave 0 的同一组 K]^T
                =
Wave 0 的 C partial
```

#### 第四步：Wave 0 的 64 个 lane 怎样合作读取一个 K[0:32]

现在只放大 Wave 0 的第一块 `K[0:32]`。

Wave 里有 64 个 lane，分成四组，每组 16 个 lane：

```text
Wave 0 的 64 lanes

lanes  0:15  -> 读取 K[ 0: 8]
lanes 16:31  -> 读取 K[ 8:16]
lanes 32:47  -> 读取 K[16:24]
lanes 48:63  -> 读取 K[24:32]
```

每组中的 16 个 lane 对应一个 16-row A/B block 的 16 个 row：

```text
lanes 0:15 读取 K[0:8] 时：

lane 0  对应 row-in-block 0
lane 1  对应 row-in-block 1
...
lane 15 对应 row-in-block 15
```

以 lane 0 为例，它同时服务四个 A row group 和四个 B row group：

```text
lane 0 从 A 读取：

A[ 0, 0:8]
A[16, 0:8]
A[32, 0:8]   当前 M=32，返回 0
A[48, 0:8]   当前 M=32，返回 0

lane 0 从 preshuffled B 读取：

W[128, 0:8]
W[144, 0:8]
W[160, 0:8]
W[176, 0:8]
```

lane 1 做同样的事，但 row-in-block 变成 1：

```text
A rows: 1, 17, 33, 49
B rows: 129, 145, 161, 177
K:      0:8
```

lane 16 回到 row-in-block 0，但 K 换成下一段：

```text
A rows: 0, 16, 32, 48
B rows: 128, 144, 160, 176
K:      8:16
```

这就是原来公式里的两个变量：

```text
q = lane // 16
    表示当前 lane 属于四个 16-lane 小组中的哪一组
    也决定它读取 K[0:8]、K[8:16]、K[16:24] 或 K[24:32]

r = lane % 16
    表示当前 lane 在小组内是第几号
    也决定它对应 16-row block 中的哪一行
```

所以不是一上来凭空定义 `q/r`；它们只是把 64 个 lane 画成这个 `4 x 16` 表：

```text
                    r = row-in-16-row-block
             0       1       2                  15
          +-------+-------+-------+----- ... -----+
q = 0     |lane 0 |lane 1 |lane 2 | ... |lane 15 | -> K[ 0: 8]
          +-------+-------+-------+----- ... -----+
q = 1     |lane16 |lane17 |lane18 | ... |lane 31 | -> K[ 8:16]
          +-------+-------+-------+----- ... -----+
q = 2     |lane32 |lane33 |lane34 | ... |lane 47 | -> K[16:24]
          +-------+-------+-------+----- ... -----+
q = 3     |lane48 |lane49 |lane50 | ... |lane 63 | -> K[24:32]
          +-------+-------+-------+----- ... -----+
```

#### 第五步：这 32 个 K 怎样变成两次 K=16 MFMA

每个 lane 一次读取连续 8 个 BF16：

```text
[x0 x1 x2 x3 | x4 x5 x6 x7]
```

前 4 个交给第一轮 MFMA，后 4 个交给第二轮 MFMA：

```text
第一轮 MFMA 的 K=16：

lanes  0:15 提供 K[ 0: 4]
lanes 16:31 提供 K[ 8:12]
lanes 32:47 提供 K[16:20]
lanes 48:63 提供 K[24:28]

合起来共 4+4+4+4 = 16 个 K


第二轮 MFMA 的 K=16：

lanes  0:15 提供 K[ 4: 8]
lanes 16:31 提供 K[12:16]
lanes 32:47 提供 K[20:24]
lanes 48:63 提供 K[28:32]

同样合起来是 16 个 K
```

可以把它画成：

```text
Wave 0 的 K[0:32]

K:   0:4   4:8   8:12  12:16  16:20  20:24  24:28  28:32
     +-----+-----+------+------+------+------+------+------+
     |step0|step1|step0 |step1 |step0 |step1 |step0 |step1 |
     +-----+-----+------+------+------+------+------+------+
       q0    q0     q1     q1     q2     q2     q3     q3

step0 = 第一轮 K=16 MFMA 使用
step1 = 第二轮 K=16 MFMA 使用
```

Wave 0 的第二块 `K[128:160]` 完全重复这个 pattern，于是再产生两轮 K=16 MFMA。

最终 Wave 0 一共有四轮 K=16：

```text
第 0 轮：来自 K[  0: 32] 中标记为 step0 的位置
第 1 轮：来自 K[  0: 32] 中标记为 step1 的位置
第 2 轮：来自 K[128:160] 中标记为 step0 的位置
第 3 轮：来自 K[128:160] 中标记为 step1 的位置
```

#### 第六步：一轮 K=16 如何产生 `64x64` C partial

固定上面的某一轮 K=16，A 有四个 16-row block，B 有四个 16-row block：

```text
                         B/N 方向
                  B0         B1         B2         B3
               128:144    144:160    160:176    176:192
              +----------+----------+----------+----------+
A0   0:16     | C00 MFMA | C01 MFMA | C02 MFMA | C03 MFMA |
              +----------+----------+----------+----------+
A1  16:32     | C10 MFMA | C11 MFMA | C12 MFMA | C13 MFMA |
              +----------+----------+----------+----------+
A2  32:48     | C20 MFMA | C21 MFMA | C22 MFMA | C23 MFMA |
              +----------+----------+----------+----------+
A3  48:64     | C30 MFMA | C31 MFMA | C32 MFMA | C33 MFMA |
              +----------+----------+----------+----------+
```

每个格子都是一个 `16x16x16` MFMA：

```text
一轮 K=16       -> 4 * 4 = 16 条 MFMA
Wave 0 共四轮   -> 16 * 4 = 64 条 MFMA
```

因此 Wave 0 的完整工作可以画成：

```text
A rows 0:64，取 Wave 0 的 64 个 K
                         x
B rows 128:192，取完全相同的 64 个 K
                         |
                         v
      4 轮 K=16，每轮计算 4x4 个 C subtile
                         |
                         v
              P0[64,64]，保存在 Wave 0 寄存器
```

Wave 1、2、3 做完全相同的事情，只是各自使用不同的 64 个 K，分别得到 `P1`、`P2`、`P3`。

#### 最后再看通用公式

理解上面的图以后，才需要下面这组简写：

```text
w    = wave_id
lane = wave 内 lane_id
q    = lane // 16       # 选择哪一个连续 8-K 小段
r    = lane % 16        # 选择 16-row block 中的哪一行

base0 = 256*g + 32*w
base1 = base0 + 128

m = block_x*64 + r + 16*m_repeat
n = block_y*64 + r + 16*n_repeat

k = base0/base1 + 8*q + k_lane
```

其中：

```text
m_repeat, n_repeat in [0,3]
k_lane              in [0,7]
```

#### 最终 ISA 中的一个真实配对

```asm
buffer_load_dwordx4 v[78:81], ...   ; B 的连续 8 个 BF16
buffer_load_dwordx4 v[94:97], ...   ; A 的连续 8 个 BF16
```

前四个 BF16 进入第一轮 MFMA：

```asm
v_mfma_f32_16x16x16_bf16 \
    v[30:33], v[78:79], v[94:95], v[30:33]
```

后四个 BF16 进入第二轮 MFMA，并继续累加到同一 C subtile：

```asm
v_mfma_f32_16x16x16_bf16 \
    v[30:33], v[80:81], v[96:97], v[30:33]
```

第二个 `K[128:160]` 段继续贡献第三、第四轮：

```asm
v_mfma_f32_16x16x16_bf16 \
    v[30:33], v[110:111], v[126:127], v[30:33]

v_mfma_f32_16x16x16_bf16 \
    v[30:33], v[112:113], v[128:129], v[30:33]
```

四条指令都累加到 `v[30:33]`，也就是同一个 `16x16` C subtile 的 lane-local FP32 accumulator。

## 6. preshuffle、K layout 和 128-bit load 如何配合

该 kernel 使用：

```python
BufferCopy128b()
MFMA(16, 16, 16, BFloat16)
K layout = (4, 16, 2):(1, 8, 4)
```

一个 `buffer_load_dwordx4` 正好读取：

```text
128 bits = 16 bytes = 8 BF16
```

初始 IR 中，每个线程的 A/B source partition 都是：

```text
((8,1),4,2,40)
```

忽略 broadcast 的 size-1 维度，一个线程在每个 K-group 中为每个 operand 读取：

```text
8 * 4 * 2 = 64 BF16
```

因此每线程、每个 K-group：

```text
A: 8 次 128-bit load = 128 bytes
B: 8 次 128-bit load = 128 bytes
```

最终 ISA 的循环体中确实有 16 条静态的：

```asm
buffer_load_dwordx4
```

即 8 条用于 A、8 条用于 B。preshuffle 的作用，就是让 B 的每次 16-byte load 直接得到 tiled-MMA 随后要消费的 lane-local 数据。

## 7. MFMA 计算量如何对上

使用的原子指令是：

```text
v_mfma_f32_16x16x16_bf16
```

一次 MFMA 覆盖：

```text
M = 16
N = 16
K = 16
```

按乘加算 2 FLOPs，一条 wave-level MFMA 对应：

```text
2 * 16 * 16 * 16 = 8192 FLOPs
```

一个 wave 的部分和覆盖完整 `64 x 64` C tile，但只覆盖 K 的四分之一：

```text
每个 K-group，每个 wave 的 K = 64
```

所以每个 wave、每个 K-group 需要：

```text
(64/16) * (64/16) * (64/16)
= 4 * 4 * 4
= 64 条 MFMA
```

这与最终 ISA 循环体中静态出现的 64 条：

```asm
v_mfma_f32_16x16x16_bf16
```

完全一致。

K 循环有 40 次，所以每个 wave 动态执行：

```text
64 MFMA/loop * 40 loops = 2560 MFMA
```

4 个 wave 合起来正好完成一个 `64 x 64 x 10240` GEMM tile：

```text
2560 * 4 * 8192 FLOPs
= 83,886,080 FLOPs
= 2 * 64 * 64 * 10240
```

初始 IR 中，Python 的两个 `sub_k` 逻辑调用已经表现为：

```mlir
scf.for ... to %c40 ...
  fly.gemm ...
  fly.gemm ...
```

layout lowering 再把每个高层 `fly.gemm` 展开为覆盖多个 `16 x 16` C subtile 的 MFMA 指令。

## 8. 每个 wave 为什么要有完整的 C partial

tiled-MMA 的 wave layout 是：

```python
fx.make_layout((1, 1, 4), (0, 0, 1))
```

M、N 方向的 stride 都为 0，说明 4 个 wave 不在 M/N 方向切不同输出区域；它们在 K 方向分工。

所以每个 wave 都需要一个完整 `64 x 64` partial C：

```text
64 * 64 = 4096 FP32 values/wave
```

MFMA 的结果分散在 wave 的 64 个 lane 上，因此每个 lane 持有：

```text
4096 / 64 = 64 FP32 accumulators
```

初始 IR 正好显示：

```mlir
vector<64xf32>
```

这也是该 kernel VGPR 压力很高的根本原因之一。

最终 code object 元数据报告：

```text
vgpr_count: 138
sgpr_count: 21
vgpr_spill_count: 0
private_segment_fixed_size: 0
```

## 9. LDS 中如何完成四路规约

4 个 wave 的部分和总大小为：

```text
4 waves * 64 * 64 * 4 bytes
= 65536 bytes
= 64 KiB
```

源码把 LDS 看成：

```text
(TILE_M * 4, TILE_N) = (256, 64)
```

并叠加：

```text
Swizzle S<3,3,3>
```

初始 IR 中对应：

```mlir
allocBytes = 65536
memref<f32, shared, S<3,3,3> ... (256,64):(64,1)>
```

最终 ISA 元数据再次确认：

```text
.amdhsa_group_segment_fixed_size 65536
```

规约分三步：

1. 4 个 wave 把各自的 `64 x 64` FP32 partial 写入 LDS。
2. 执行一次 workgroup barrier。
3. 256 个线程重新分工，每线程读取 4 份 partial 中对应的 16 个输出值，然后做三次向量加法。

对一个线程而言：

```text
写 LDS: 64 FP32 = 256 bytes
读 LDS: 4 * 16 FP32 = 256 bytes
最终负责: 16 个 BF16 输出 = 32 bytes
```

ISA 中可以看到对应的静态指令数量：

```text
16 x ds_write_b128
1  x s_barrier
16 x ds_read_b128
4  x buffer_store_dwordx2
```

这条 Split-K 没有 global atomic，也没有额外 reduction kernel。所有 partial reduction 都在一个 workgroup 的 LDS 内完成。

## 10. FP32 partial 如何变成 BF16

四路相加之后，源码用位操作快速完成 FP32 到 BF16：

```python
round_bit = 0x8000 bitcast to f32
acc = ((acc + round_bit).bitcast(u32) >> 16).to(u16).bitcast(bf16)
```

初始 IR 中是：

```text
vector<16xf32>
  -> add 0x8000 对应的 bit pattern
  -> bitcast vector<16xi32>
  -> shift right 16
  -> truncate vector<16xi16>
  -> bitcast vector<16xbf16>
```

最终 ISA 使用 packed FP32 add 和 `v_perm_b32` 整理 BF16 数据，再通过 64-bit buffer store 写回。

## 11. 当前 M=32 为什么性能不高

当前输出只有：

```text
C = [32, 320]
```

但 tile 固定为 `64 x 64`：

- N=320 正好是 5 个完整 N tile；
- M=32 只有一个 M tile，而且只使用名义 `64` 行中的一半；
- kernel 仍然执行完整 `64 x 64` MFMA 和 LDS reduction；
- 越界的 A load/C store 由 buffer bounds 处理，但计算资源已经消耗。

有用 FLOPs 为：

```text
2 * 32 * 320 * 10240 = 209,715,200 FLOPs
```

5 个 workgroup 的名义 tile 计算量为：

```text
5 * 2 * 64 * 64 * 10240 = 419,430,400 FLOPs
```

也就是说，仅从 M tail 看，约一半 MFMA 工作没有对应有效输出。

本次 dump 运行结果约为：

```text
splitk: 67 us, 3.2 TFLOPS
torch:  13 us, 16.4 TFLOPS
```

这个结果主要用于观察 IR，不适合直接代表 Split-K kernel 在饱和大问题上的性能。

## 12. 从源码一路看到 ISA

推荐按下面顺序阅读：

| 阶段 | 文件 | 重点 |
|---|---|---|
| Python DSL | `test_gemm.py` | tile、wave split、preshuffle view、LDS reduction |
| 初始 IR | `00_origin.mlir` | shape/stride、`scf.for to 40`、`vector<64xf32>`、64 KiB LDS |
| layout lowering | `03_fly_layout_lowering.mlir` | tiled copy/MMA 如何展开成线程坐标和向量操作 |
| ROCDL | `08_convert_fly_to_rocdl.mlir` | 128-bit buffer load、MFMA、barrier 和 buffer store |
| LLVM IR | `21_llvm_ir.ll` | `llvm.amdgcn.*` intrinsic 和 buffer resource |
| ISA | `22_final_isa.s` | 最终 load/MFMA/LDS/reduction/store 指令 |

几个适合直接搜索的命令：

```bash
cd /opt/pyhip/tests/flydsl/test_gemm_splitk/gemm_splitk_0

rg -n "scf.for|fly.gemm|vector<64xf32>|allocBytes" 00_origin.mlir
rg -n "raw.ptr.buffer.load|rocdl.mfma|barrier" 08_convert_fly_to_rocdl.mlir
rg -n "llvm.amdgcn" 21_llvm_ir.ll
rg -n "buffer_load_dwordx4|v_mfma|ds_write|ds_read|s_barrier|buffer_store" 22_final_isa.s
```

## 13. 当前先记住的完整数据流

```text
普通 row-major W[N,K]
        |
        | AITER shuffle_weight(layout=(16,16))
        v
preshuffled W physical buffer
        |
        | FlyDSL 用 ((16,20),(8,4,320)) / ((8,163840),(1,128,512)) 解释
        v
每线程 128-bit coalesced buffer loads
        |
        | tiled-MMA retile
        v
4 waves 分别计算同一 64x64 C tile 的 K partial
        |
        | 每 wave 64 个 FP32 accumulators/lane
        v
4 份 64x64 FP32 partial 写入 64 KiB LDS
        |
        | barrier + 四路向量加法
        v
每线程得到 16 个最终 FP32 C 元素
        |
        | fast FP32 -> BF16
        v
BufferCopy64b 写回 C[M,N]
```

## 14. 后续值得逐项细扣的问题

后续可以沿这几个问题继续展开：

1. `(1,1,4):(0,0,1)` 如何把 `wave_id` 映射到当前 256-wide K-group。
2. `(4,16,2):(1,8,4)` 如何决定两次高层 `fly.gemm` 的 K 交错顺序。
3. A/B 的 `partition_S` 为什么得到 `((8,1),4,2,40)`，每个维度分别对应什么。
4. 为什么两条高层 `fly.gemm` 最终展开为循环体中的 64 条 MFMA。
5. `S<3,3,3>` 如何改变 LDS 地址，以及是否完全消除了当前读写模式的 bank conflict。
6. 16 条 `ds_write_b128` 和 16 条 `ds_read_b128` 分别对应哪些 C 坐标。
7. `M=32` 时，哪些 lane 的 global load/store 被 bounds checking 屏蔽。
8. 64 KiB LDS 和 138 VGPR 如何共同限制 occupancy。
9. 当前 BF16 rounding 写法与严格 round-to-nearest-even 的差异。

本文先固定整体框架。后续分析某一个 layout 时，应同时对照 Python、`00_origin.mlir`、ROCDL IR 和最终 ISA，避免只根据某一层的表面 shape 猜测真实的数据移动。

## 15. `M=32 < TILE_M=64` 时到底如何 padding

launcher 是：

```python
gemm_splitk(...).launch(
    grid=(div_up(M, TILE_M), div_up(N, TILE_N), 1),
    block=(256, 1, 1),
)
```

当前参数代入后：

```text
grid.x = ceil(32 / 64) = 1
grid.y = ceil(320 / 64) = 5
```

这里的 `div_up()` 只负责“尾块也要启动”，并不会创建一个新的 padded tensor。

### 15.1 没有真实分配 `A_pad[64,K]`

PyTorch 实际分配的仍然是：

```text
A storage = 32 * 10240 * 2 bytes = 655360 bytes
C storage = 32 *   320 * 2 bytes =  20480 bytes
```

kernel 则按名义 `64 x 64` tile 做地址计算：

```text
名义 A tile rows: m = 0..63
真实 A rows:      m = 0..31

名义 C tile rows: m = 0..63
真实 C rows:      m = 0..31
```

所以它更接近“虚拟 zero padding”，不是额外申请并填充一个 `64 x K` 的 A。

```text
A 的名义 64-row tile

m=0  +------------------------------------------+
     |                                          |
     | A[0:32, 0:10240]：真实数据               | 32 rows
     |                                          |
m=32 +------------------------------------------+ <-- A buffer 的真实末尾
     |                                          |
     | A[32:64, :]：没有物理存储                | 32 rows
     | buffer load 越界，计算时视为 0           |
     |                                          |
m=64 +------------------------------------------+
```

### 15.2 `make_buffer_tensor(max_size=False)` 提供硬件边界

源码对 A、B、C 都调用了：

```python
a_tensor = fx.rocdl.make_buffer_tensor(arg_a, max_size=False)
b_tensor = fx.rocdl.make_buffer_tensor(arg_b, max_size=False)
c_tensor = fx.rocdl.make_buffer_tensor(arg_c, max_size=False)
```

`max_size=False` 会根据 tensor layout 的 `cosize` 和元素宽度生成真实的 buffer byte count，而不是把 descriptor 长度设成最大值。

当前 `00_origin.mlir` 中，A 的 descriptor 长度按运行时 M 计算：

```mlir
%21 = fly.cosize(%A_layout)
%22 = fly.get_scalar(%21)
%23 = arith.muli %22, %c2_i32
%24 = arith.extsi %23 : i32 to i64
%25 = fly.make_ptr(..., %24, ...)
```

因为 BF16 是 2 bytes，所以实际就是：

```text
A_num_records_bytes = M * K * 2
                    = 32 * 10240 * 2
                    = 655360
```

C 同理：

```text
C_num_records_bytes = M * N * 2
                    = 32 * 320 * 2
                    = 20480
```

B/W 没有 M tail，长度是编译期常量：

```text
B_num_records_bytes = N * K * 2
                    = 320 * 10240 * 2
                    = 6553600
```

在 `21_llvm_ir.ll` 中可以直接看到三个 buffer resource：

```llvm
; A：长度由运行时 M 算出
%22 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc...(i64 %21, ...)

; preshuffled B：固定 6553600 bytes
%23 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc...(i64 6553600, ...)

; C：长度由运行时 M 算出
%28 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc...(i64 %27, ...)
```

### 15.3 第一个越界地址正好落在哪里

A 是 row-major `[M,K]`，一行大小为：

```text
10240 BF16 * 2 bytes = 20480 bytes
```

第 32 行的首地址相对 A base 为：

```text
32 * 20480 = 655360 bytes
```

这正好等于 A descriptor 的 byte count，因此 `m=32` 已经是第一个 OOB row。

C 的一行大小为：

```text
320 BF16 * 2 bytes = 640 bytes
```

第 32 行的首地址为：

```text
32 * 640 = 20480 bytes
```

也正好等于 C descriptor 的 byte count。

### 15.4 硬件如何处理越界 load/store

对当前 AMD buffer load/store 路径：

```text
A 的 OOB buffer load  -> 返回 0
C 的 OOB buffer store -> 丢弃，不写显存
```

因此名义 tile 的上下两半可以画成：

```text
                   A tile [64, K]

valid rows 0:32    A_real --------------------+
                                              |
OOB rows 32:64     zero from buffer load -----+----> MFMA
                                                   |
                                                   v
                                      nominal C partial [64,64]
                                                   |
                         +-------------------------+------------------+
                         |                                            |
                         v                                            v
                  rows 0:32 store                              rows 32:64 store
                  地址在 C 范围内                              地址超过 C bound
                  写入真实 C                                   被硬件丢弃
```

因为越界 A 行读到的是 0，所以名义 C tile 的第 32–63 行累加结果也应为 0；即使这些行仍经过 MFMA 和 LDS reduction，最终 global store 也不会越过真实 C allocation。

### 15.5 为什么 IR 中仍然显示 A tile 是 64 行

初始 IR 同时保留了两个信息：

```mlir
; 原始逻辑 A 的 M 是动态值
!fly.memref<bf16, ..., (?,10240):(10240,1)>

; tiled view 的内部 tile 固定为 64 x 256
!fly.memref<bf16, ..., (64,256,?,40):(10240,1,655360,256)>
```

含义是：

- tile 形状固定，方便生成无分支的 tiled-copy 和 MFMA；
- 原始 tensor 的真实 M 仍保留在 buffer descriptor bound 中；
- 边界安全交给 buffer load/store，而不是改变 tile shape。

因此最终 ISA 中看不到围绕每个 load/store 的显式：

```text
if m < M
```

仍然是直接发出 `buffer_load_dwordx4` 和 `buffer_store_dwordx2`，是否越界由 buffer resource descriptor 和硬件判断。

### 15.6 这种做法的正确性和代价

正确性上：

```text
OOB A = 0
0 * B = 0
OOB C store = drop
```

所以不需要单独的 padding kernel，也不需要在 hot loop 中加入逐元素条件分支。

性能代价是：

- `M=32` 仍按 `TILE_M=64` 执行完整 MFMA；
- 仍为 64 行 partial 使用完整的寄存器和 LDS；
- 仍执行完整的 LDS 写入、读取和规约；
- 只有最终 32 行是有效输出。

所以当前 case 在 M 方向只有约 50% 的有效 tile 利用率。这解释了为什么该配置适合研究 IR 和边界行为，但不是展示峰值性能的理想 shape。

如果 `M=65`，逻辑也是一样：

```text
grid.x = ceil(65/64) = 2

block_x=0 -> rows 0:64，全部有效
block_x=1 -> rows 64:128，只有 row 64 有效，其余 buffer access OOB
```

这种“固定 tile + descriptor bounds”的方式，就是当前 kernel 对任意 M tail 的处理方式。
