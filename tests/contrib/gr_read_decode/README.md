# GR read decode，T=1..32

从 `pyhip` 的 `be1555f` 提取最终选中的 H64 decode：
`h64_layout.py` 的 plane up + `large_down.py` 的 split-K down。
代码都在本目录，目标为 **FlyDSL 0.3.2**，配合 ROCm PyTorch；无需模型、SGLang 或旧仓库。
本目录是独立测试示例，没有接入 `src/` 的 prefill 接口。

## 算法

固定 BF16 输入 `X[T,10240]`、`W_down[320,10240]`、`W_up[10240,320]`，
输出 `Y[T,2560]`，四路 stream。计算：

```text
hidden = SiLU(X @ W_down.T / 4)
gate   = sigmoid(hidden @ W_up.T).reshape(T, 4, 2560)
Y      = mean(gate * X.reshape(T, 4, 2560), dim=1)
```

完整调用固定两个 kernel：

1. **Down**：global split-K=4，每个 CTA 的 4 个 wave 再分工 K，
   128-bit X/W 读取和双缓冲预取；LDS 归约后写 FP32 `P[4,T_pad,320]`。
   BM 在 T<=16 时为16，否则为32；BK=128，T<=25 使用两拍预取。
2. **Up**：先归约四份 P，再做 SiLU，把激活拆为 BF16 high/low 两份参与 MFMA，
   保留精度；随后做 sigmoid、四路加权平均，写 BF16 Y。
   BM16/BN128、4 waves、A-first、128-bit W load、hidden LDS stride324。

| T | Up BK | 跳过 P 内部 padding 读取 | 提前加载 W_up tile |
| --- | ---: | --- | --- |
| 1..8 | 160 | 是 | 是 |
| 9..16 | 32 | 否，Down 已清零 | 是 |
| 17..28 | 160 | 是 | 否 |
| 29..31 | 32 | 是 | 否 |
| 32 | 160 | 无尾行 | 否 |

`prepare_weights()` 在测试准备阶段执行一次。W_down 使用 N16/K32 preshuffle；
W_up 先做 `[4,40,2,4,2,4,320] → permute(1,0,2,4,3,5,6)`，再 preshuffle，
与 prefill 的 H64 布局相同。运行时持有一份 packed W_down/W_up，不重新排列权重。
P/Y 由每个测试实例独立持有，重复调用会覆盖输出；需要保存结果时自行 clone。

## 运行

在仓库根目录运行；也可进入本目录直接执行脚本。GPU 编号按机器调整：

```bash
# 已有 ROCm PyTorch 环境中，安装目标 FlyDSL 版本
python3 -m pip install -r tests/contrib/gr_read_decode/requirements.txt

# 全部 T=1..32，两对随机权重；普通 assert，无需 pytest
HIP_VISIBLE_DEVICES=2 python3 tests/contrib/gr_read_decode/test_gr_read.py

# 只检查指定的 T
HIP_VISIBLE_DEVICES=2 python3 tests/contrib/gr_read_decode/test_gr_read.py --rows 1 16 17 32
```

正确性使用 FP64 reference，逐元素 `rtol=1e-2, atol=5e-3`。
每档 capture 一次，让 live rows 从 T 降到0再回到T，检查 zero/stale/NaN 尾行、
P/Y 预污染、内部 padding、前后 guard、输入与权重不变。默认共 **6528 次 replay**。
不要使用 `python -O`，它会关闭 assert。

## 最终版 T=1..32 benchmark

在仓库根目录执行下面这一条命令，跑完全部32档，只测最终 H64 实现，无需模型：

```bash
HIP_VISIBLE_DEVICES=2 python3 tests/contrib/gr_read_decode/bench_gr_read.py \
  --rows {1..32} --weights 100 --rounds 3 --samples 7 --seed 707 \
  --output tests/contrib/gr_read_decode/results/final_t1_32.jsonl
```

`--output` 必须是新文件名，重跑时换名字；也可以省略该参数，自动生成带时间戳的文件。
进入本目录后可直接运行 `HIP_VISIBLE_DEVICES=2 python3 bench_gr_read.py`，默认参数同上。
每档输出 `H64 final Graph: ... us`，单位是**一次完整 GR read 的微秒数**。

性能计时沿用原 H64 报告：每图顺序执行100个实例两轮，样本 replay 3次，
CUDA Event 时间除以600；报告完整两段调用，不计编译、packing、参考与准备。
21个样本全部保留在 `results/graph_*.jsonl`，`--output` 可指定新的结果文件。
`--weights`、`--rounds`、`--samples` 可做快速检查，改变它们后的时延不能直接套用原表。
性能脚本要求 `rocm-smi`、支持 PTL 查询的 `amd-smi`，检查空闲GPU及
PTL Enabled/VECTOR,F8；准备后和结束后固定静置2秒再读取门禁，不修改设备设置。

算法选择针对 MI308X/gfx942；其他 ROCm 架构发出 warning，
仍需在目标机器运行正确性与性能检查。

## 本次验证（2026-09-21）

MI308X/GPU2，Torch 2.12.0/ROCm 7.2.4，**FlyDSL 0.3.2**：
全部32档、6528次随机权重 replay 通过；最终 benchmark 的100对随机权重
在初始输入和改变输入后均通过FP64检查，保留全部672个计时样本。
全部32档编译出的 GPU `.text` 与原 H64 实现一致。

| T | 38号 H64 报告 µs | 最终版 FlyDSL 0.3.2 µs |
| ---: | ---: | ---: |
| 1 | 10.991 | 10.991 |
| 16 | 13.046 | 13.039 |
| 32 | 19.418 | 19.429 |

最终版全部32档与38号表的最大绝对差为 **0.0293 µs**。
38号表使用真实HC权重，本次最终入口使用随机权重，计时工作量相同；
同一最终入口从0.3.1切到0.3.2，32档跨运行最大绝对偏差为0.145%。
