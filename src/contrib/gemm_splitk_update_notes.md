# gemm_splitk_update fp8 适配修改记录

## 背景

`gemm_splitk_update.py` 从老 branch 拿过来，kernel 和 test 接口只支持 fp4 (`float4_e2m1fn_x2`)。当前机器是 gfx942 (MI300)，不支持 fp4（仅 gfx950 支持），导致 `get_fp4type_if_valid()` 返回 `None`，所有测试被跳过，输出为空。

目标：让脚本在 gfx942 上用 `torch.float8_e4m3fnuz` 跑起来。

## 问题分析

跑脚本后发现三层问题：

1. **数据类型不支持** — `get_fp4type_if_valid()` 在 gfx942 上返回 `None`，test 全跳过
2. **Kernel 代码过旧** — 文件里内嵌的 `gemm_splitk_wd` kernel 只有 fp4 路径，缺少 fp8 的 offset 计算、scale 处理、quant_type 传递
3. **指令不兼容** — `v_cvt_pk_bf16_f32` 是 gfx950 (CDNA4) 专用指令，gfx942 (CDNA3) 不支持

## 修改思路

### 第一步：尝试在老 kernel 上加 fp8 支持

参照已有的 `jit_gemm_splitk.py`，逐项补齐 fp8 路径：

- kernel assert 放宽为支持 fp4/fp8/bf16
- `voffset_b`：fp8 用 `J.threadIdx.x * 16`（而非 fp4 的 `J.lane_id + J.warp_id` 拆分）
- `voffset_a`：fp8 用 `J.threadIdx.x // 16`（而非 fp4 的 `lane_div_16 + J.warp_id`）
- `voffset_scale`：fp8 blockwise 128x128 的 scale layout `[N//128, K//128]`
- `gemm_splitk()` 调用：fp8 需传 `quant_type_str='per_1x128'`
- output 转换：`v_cvt_pk_bf16_f32` → `uni_cvt_pk_bf16_f32`（CDNA3 兼容）

结果：kernel 能编译，但输出全零。老 kernel 和新 `gemm_splitk.py` 底层的 LDS swizzle pattern 不一致，逐项修复成本高且容易出错。

### 第二步：直接复用 jit_gemm_splitk 模块

放弃维护文件内嵌的 kernel 副本，改为 import 已验证的 `pyhip.contrib.jit_gemm_splitk.gemm_splitk_wd`。这个模块已经正确支持 fp4/fp8/bf16 的所有 offset、scale、LDS swizzle 逻辑。

### 第三步：修复 jit_gemm_splitk.py 自身的两个 bug

在实际运行中发现 `jit_gemm_splitk.py` 也有从老代码残留的问题：

| bug | 原因 | 修复 |
|-----|------|------|
| `gemm_splitk() got unexpected keyword argument 'fp8_ptpc'` | 老接口参数名，`gemm_splitk.py` 已改为 `quant_type_str` | `fp8_ptpc=False` → `quant_type_str='per_1x128'` |
| `v_cvt_pk_bf16_f32: instruction not supported on this GPU` | gfx950 专用指令 | → `uni_cvt_pk_bf16_f32`（自动选 CDNA3/4 路径） |

### 第四步：适配测试数据准备

fp8 和 fp4 的量化方式完全不同：

- **fp4**: `per_1x32` 量化 + `e8m0_shuffle` scale + `shuffle_weight(w_qt)`
- **fp8**: blockwise `per_128x128` 量化 + `float32` scale + `shuffle_weight(w_qt, layout=(16, 16))`

fp8 要求 N 和 K 都是 128 的倍数，TILE_N 也必须整除 128。

精度验证也从 `torch.allclose(rtol=0.1, atol=0.03)` 改为 `calc_diff` 阈值 0.02，与 `test_jit_gemm_splitk.py` 一致。

### 第五步：调整 tile 搜索范围

fp8 在大 tile 下寄存器压力大。TILE_M=64 + TILE_N=128 会导致 VGPR 溢出（`cannot allocate vGPRs`）。参照 `test_jit_gemm_splitk.py`，将 tile_m 搜索范围从 `[64, 32, 16]` 收缩为 `[16, 32]`。

## 修改文件清单

| 文件 | 改动 |
|------|------|
| `src/contrib/gemm_splitk_update.py` | 删除内嵌 kernel，import `jit_gemm_splitk.gemm_splitk_wd`；加 fp8 数据准备；精度验证改 `calc_diff`；tile 搜索改 `[16,32]` |
| `src/contrib/jit_gemm_splitk.py` | `fp8_ptpc` → `quant_type_str`；`v_cvt_pk_bf16_f32` → `uni_cvt_pk_bf16_f32` |

## 验证结果

gfx942 上 fp8 (`float8_e4m3fnuz`)，N=9216, K=4096：

- accuracy test: M=2~8192+ 全部通过
- perf test: M=2 ~5 tflops → M=256 ~90 tflops
