# `tests/contrib/gemm` 目录内 GEMM 一览

本文档概括本目录下各测试所覆盖的 **GEMM 变体**、**参与运算的数据类型**、**量化形态**、**与 CDNA3 / CDNA4（以 `gcnArchName` 中是否含 `gfx950` 等为线索）的关系**，以及 **主要实现文件路径**。

> **说明**：仓库内常以 `J.arch` / `torch.cuda.get_device_properties().gcnArchName` 判断架构。`common/gemm_splitk.py` 等处用 `"gfx950" in J.arch` 区分 **CDNA4（gfx950 系）** 与 **CDNA3（如 gfx942 等）** 的指令与寄存器布局差异。未在代码中写死架构的 kernel 仍可能仅在部分 ISA 上通过编译或验证，以你本机 ROCm / pyhip 版本为准。

---

## 总览表

| 目录内入口 | 数据类型（典型） | 量化 / 缩放 | CDNA3 | CDNA4 | 主要实现文件 |
|------------|------------------|-------------|-------|--------|----------------|
| `test_cdna4.py` | A/B/C：`bf16` | 无量化 | ✓（`v_mfma_f32_16x16x32_bf16`） | ✓ | `pyhip/src/contrib/gemm_cdna4.py` |
| `test_4wave_cdna4_slicing.py` | 同上 | 无量化 | ✓ | ✓（含 XCD / slice 调度等实验逻辑，见源码注释） | `pyhip/src/contrib/gemm_4wave_slicing.py` + `gemm_cdna4.py` |
| `test_fp8_8wave.py` | A/B：`fp8` / `bf16` / `fp16`；C：`bf16` | `fp8` 路径可选 **128×128 block 的 f32 scale**（`use_f32_blockscales_128`）；权重可 **preshuffle** | `fp8`：多为 **16×16×16 bf16 MFMA** 分解路径 | `fp8`：**`v_mfma_f32_16x16x128_f8f6f4`** 等 | `pyhip/src/contrib/gemm_fp8.py` |
| `test_a8w8_f32_blockscale.py` | A/W：`fp8`；scale：`fp32`；输出：`bf16` 等 | **块缩放** `128×128`，另有 CK / ASM / preshuffle 对比 | 依赖 **aiter** 与 pyhip 路径是否在目标 ISA 上可用 | 同上 | 测试内：`aiter.gemm_a8w8_*`、`pyhip.contrib.gemm_fp8`；JIT：`gemm_fp8.py` |
| `test_w8a8_block_fp8_linear.py` | 输入：`bf16`（在线量化为 `fp8`）；W：`fp8`；scale：`fp32`；C：`bf16` | **per-1×128** 激活量化 + **128×128** 权重块缩放 | **jit**：经 `gemm_splitk` 走 CDNA3/4 分支；**gluon**：Triton Gluon `cdna3`/`cdna4` 混合 API | 同上 | `pyhip/src/contrib/w8a8_block_fp8_linear.py`（组合 `gemm_fp8`、`gluon/gemm_splitk`、aiter） |
| `test_a4w4_mxfp4.py` | A：经 `fp4`+per-1×32 scale；B：`fp4×2`+e8m0；C：`bf16` | **MXFP4 / block scale**（e8m0 + shuffle） | **JIT 路径**：权重为 `float4_e2m1fn_x2` 时依赖 **gfx950**（测试内 `get_fp4type_if_valid`） | ✓（**scaled MFMA**：`v_mfma_scale_f32_16x16x128_f8f6f4`） | `pyhip/src/contrib/gemm_a4w4.py`；对比 **aiter** `gemm_a4w4` |
| `test_jit_gemm_splitk.py` | A：`bf16`；W：`bf16` / `fp8` / `fp4`；C：`bf16` | `fp4`：解压 + scale；`fp8`：per-token 或块缩放等（见 `common/gemm_splitk.py`） | ✓（显式 `is_cdna4 == False` 分支：`16×16×16` bf16 MFMA 等） | ✓（`gfx950`：`cvt_scalef32_pk_bf16_fp4`、`16×16×32` MFMA 等） | `pyhip/src/contrib/jit_gemm_splitk.py` → `common/gemm_splitk.py`、`common/gemm.py` |
| `ck_kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle.s` | CK 实例化中含 **`f8_fnuz`** 等 | **Composable Kernel** 多 D/blockscale + B preshuffle 的 **汇编片段**（调试用 / 对照） | 与 CK 构建目标 gfx 相关（常见 MI300 族） | 视 CK 目标架构而定 | 上游 ROCm **CK** 生成；非 pyhip 手写维护 |

---

## 各文件简要说明

### 1. `test_cdna4.py` → `gemm_kernel`

- **类型**：BF16 GEMM，累加寄存器为 `f32`，写回 `bf16`。
- **平台**：4-wave、`v_mfma_f32_16x16x32_bf16`；文件名强调 CDNA4 场景，但指令形态与 CDNA3 MI 系列常用 bf16 MFMA 一致，**是否“仅 CDNA4”以你环境验证为准**。
- **实现**：`pyhip/src/contrib/gemm_cdna4.py`。

### 2. `test_4wave_cdna4_slicing.py` → `gemm_kernel_slicing`

- **类型**：与上相同（BF16）。
- **平台**：在 `gemm_cdna4` 基础上增加 **slice / XCD / 时间戳** 等实验逻辑；注释引用 `gfx950-gluon-tutorials` 的 v8 slice 内核思路。
- **实现**：`pyhip/src/contrib/gemm_4wave_slicing.py`。

### 3. `test_fp8_8wave.py` → `gemm_8wave_fp8bf16fp16`

- **类型**：`AB_dtype` 可为 `fp8`、`bf16`、`fp16`；输出固定 **`bf16`**。
- **量化**：纯精度对比时可无量化；`gemm_fp8.py` 内 **`use_f32_blockscales_128`** 开启时为 **FP8 矩阵乘 + f32 块缩放**（与 block 128 调度配套）。
- **平台**：`fp8` 主路径使用 **`v_mfma_f32_16x16x128_f8f6f4`**（偏 CDNA4 扩展 K 块 MFMA）；`bf16`/`fp16` 分支为 **`v_mfma_f32_16x16x32_bf16` / `_f16`**。
- **实现**：`pyhip/src/contrib/gemm_fp8.py`。

### 4. `test_a8w8_f32_blockscale.py`

- **类型**：激活与权重 **`fp8`**，**`fp32` 块缩放张量**，输出多为 **`bf16`**（与 aiter dtypes 一致）。
- **量化**：标准 **A8W8 + F32 blockwise scale（128×128）**；对比 **CK / preshuffle / ASM** 等 aiter 入口与 pyhip。
- **平台**：核心算子来自 **aiter**；本地 JIT 部分复用 `gemm_fp8`（见测试 `from pyhip.contrib.gemm_fp8 import *`）。
- **实现**：`pyhip/src/contrib/gemm_fp8.py` + 外部 **`aiter`** 包。

### 5. `test_w8a8_block_fp8_linear.py` → `w8a8_block_fp8_linear`

- **类型**：逻辑上 **W8A8**：权重 `fp8` + `fp32` block scale；激活 **`bf16`** 在线压到 **`fp8`**（per-1×128）。
- **量化**：**激活 per-1×128** + **权重 128×128 block**。
- **平台**：`method="jit"` 走 **`gemm_8wave_fp8bf16fp16`**；小 M 或 `gluon` 走 **`pyhip/contrib/gluon/gemm_splitk.py`**（内部 `gl.amd.cdna3` 与 `gl.amd.cdna4.mfma` 混用）；`aiter` 走 Triton/Python CK 封装。
- **实现**：`pyhip/src/contrib/w8a8_block_fp8_linear.py`。

### 6. `test_a4w4_mxfp4.py` → `gemm_a4w4_kernel` / aiter

- **类型**：**MXFP4**：权重 **`float4_e2m1fn_x2`** + **e8m0** scale；激活经 **`per_1x32_f4_quant_hip`** 得到 `fp4` + scale；输出 **`bf16`**。
- **量化**：**A4W4** + micro-scaling（与 ROCm/aiter ck_gemm_a4w4_blockscale 一致）。
- **平台**：JIT 使用 **`v_mfma_scale_f32_16x16x128_f8f6f4`**，属 **CDNA4 类 scaled MFMA**；测试里 **`float4_e2m1fn_x2` 仅在 `gcnArchName` 含 `950` 时启用**。
- **实现**：`pyhip/src/contrib/gemm_a4w4.py`；参考实现 **`aiter.gemm_a4w4`**（见测试内注释链接）。

### 7. `test_jit_gemm_splitk.py` → `gemm_splitk_wd`

- **类型**：**Split-K** 小批量 GEMM；权重可为 **`bf16` / `fp8` / `fp4`**。
- **量化**：`fp8` 分 **per-token（ptpc）** 与 **块缩放** 等分支；`fp4` 用 **`v_cvt_scalef32_pk_bf16_fp4`**（注释标明 **gfx950**）。
- **平台**：`pyhip/src/contrib/common/gemm_splitk.py` 内 **`is_cdna4 = "gfx950" in J.arch`**，对 **MFMA 条数、fp8→bf16 转换指令、fp4 解压** 等做 **CDNA3 / CDNA4 分支**。
- **实现**：`pyhip/src/contrib/jit_gemm_splitk.py`，核心循环 **`pyhip/src/contrib/common/gemm_splitk.py`**。

### 8. `ck_kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle.s`

- **类型**：Composable Kernel 生成的 **AMDGCN 汇编**，符号中含 **`f8_fnuz`**、blockscale、preshuffle 等模板参数。
- **用途**：与本目录 ROCm profiling（如 `run_trace.sh`）对照，**非** pyhip 运行时直接加载的主路径。

### 9. 辅助脚本

- **`run_trace.sh` / `trace.yaml`**：`rocprofv3` 采集 `test_4wave_cdna4_slicing.py` 等性能轨迹用。

---

## 相关公共模块（不在本目录但在调用链上）

| 路径 | 作用 |
|------|------|
| `pyhip/src/contrib/common/loaders.py` | MFMA global→LDS loader、preshuffle 等 |
| `pyhip/src/contrib/common/gemm.py` | `UGEMM` 等通用封装 |
| `pyhip/src/contrib/gluon/gemm_splitk.py` | Triton Gluon 版 split-K / blockscale GEMM |

---

## 维护提示

- 新增测试时，建议在本文件 **总览表** 中补充一行，并注明 **是否依赖 aiter**、**`gfx950` 专用 dtype（如 fp4_e2m1）** 以及 **主 MFMA / 缩放指令** 名称，便于后续做架构矩阵（CDNA3 vs CDNA4）核对。
