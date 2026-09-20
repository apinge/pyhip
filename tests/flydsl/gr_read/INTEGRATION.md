# 给集成端 Codex：Qwen3.8 GR read 接口与 Graph Bucket 契约

**布局范围：本文仅适用于旧 `h*C+c` W_up 和 `SmallBatchGRRead` / `LargeDownGRRead`。** 与同事 prefill `3f472e33` 共用 H64/stream/lane 权重的新入口是 `h64_layout.H64GRRead`，请按 [H64_LAYOUT.md](H64_LAYOUT.md) 准备权重和调用。两种 packed W_up 的 shape 相同但排列不同，禁止混用；下文“既定格式”指本文记录的旧版契约。

日期：2026-09-17。现有 pyhip 基线为 `d3510c6`，最终 kernel 算法来自 `a716010`。本次新增接口测试和本文，没有修改 kernel、weight 格式或本机 SGLang 生产分发。

请集成最终 `down_wave_splitk_pipeline + up_gate` 两段方案，不移植历史 control。不要在另一个项目中 import 测试脚本作为生产 API；测试中的 `pack_original_weights`、`selected_configs`、`compile_bucket` 是已执行验证的参考实现，生产适配层应放进目标仓库正常的 kernel/模型模块。

## 1. 先明确三个行数

| 名称 | 含义 | capture=4，实际=3 |
| --- | --- | ---: |
| B | 传到 GR 的物理 tensor 行数，也是该次 capture/编译的行数 | 4 |
| M | 本轮有效 token 行数，普通 decode 的有效行在前缀 | 3 |
| T_pad | kernel 内部 P 的 M tile 对齐行数 | 16 |

**编译、选择配置和分配 P/Y 全部按 B，不按 M。当前 kernel 没有运行时 M 参数。**

普通 decode 下 0 <= M <= B <= 32。B 必须来自实际 `hyper_input_normed.shape[0]`，不能一概等于请求数：speculative、DP gather、ragged verify 的请求数和 token 行数可能不同。本文完整模型路径分析限定于默认非 speculative decode；其他模式须在目标版本重新核对行映射。

M=0 的测试不意味着 B=0，也不意味着生产端可以跳过整个 graph；idle/TP collective 的参与规则仍由原框架决定。

Graph replay 不重新执行 Python forward，所以不会在本轮 M=3 时自动把 T=4 kernel 换成 T=3 kernel。不能给 B=4 的 raw launcher 换成只有 3 行存储的 X，也不能仅重绑 Python tensor 变量来修改已捕获地址。

## 2. 固定数学与 Tensor 契约

固定 `C=4, H=2560, K=C*H=10240, R=320`。接收已经 normalized 的 residual，RMSNorm、GR write 和 TP 通信均在此 kernel 外部。

```text
A[t] = SiLU((X[t] @ W_down.T) / 4)
G[t] = sigmoid(A[t] @ W_up.T)
Y[t,h] = mean_c(G[t,c*H+h] * X[t,c*H+h])
```

| 参数 | Shape / dtype | 备注 |
| --- | --- | --- |
| X | contiguous BF16 [B,10240] | 原始 c*H+h 顺序；不 reorder |
| 原始 W_down | BF16 [320,10240] | nn.Linear 的 [out,in] 顺序 |
| 原始 W_up | BF16 [10240,320] | 原始输出行 c*H+h |
| Packed WD / WU | 各为 contiguous BF16 flat [3276800] | 见下一节 |
| P | contiguous FP32 flat [4*T_pad*320] | 四份线性 partial |
| Y | contiguous BF16 [B,2560] | 输出自然 h 顺序 |

均在同一 ROCm device，推理用途，不支持 autograd。公开参数检查/生产 gating 必须保留 dtype、shape、stride、device 校验。开发代码在非 gfx942 上仅 warning，不构成 gfx950 已验证的生产支持声明。

模型名包含 FP8 不代表 HC 权重是 FP8；本模型的 HC 参数实际是 BF16。不要因为 TP=2 再把这里的 H/K/R 除以 2；若目标模块实际本地权重形状不同，应拒绝此路径或保留 fallback。

## 3. Load Model 时只打包一次

### 精确布局，不得更改

只对 W_up 做逻辑行 reorder：

```python
up_hc = w_up.reshape(4, 2560, 320).permute(1, 0, 2).contiguous().reshape(10240, 320)
# up_hc[h*4+c, r] == w_up[c*2560+h, r]
```

然后两份权重使用同一个 preshuffle：

```python
def preshuffle(weight):
    n, k = weight.shape
    return weight.reshape(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(-1)

packed_down = preshuffle(w_down)
packed_up = preshuffle(up_hc)
```

| Weight | Packed 5D 存储形状 | 字节数 |
| --- | --- | ---: |
| W_down | [20,320,4,16,8] | 6,553,600 |
| W_up | [640,10,4,16,8] | 6,553,600 |

映射是 `packed[n//16,k//32,(k//8)%4,n%16,k%8] = W[n,k]`；W_up 这里的 W 指 up_hc。Up epilogue 会把 h*C+c 的 gate 对回 X 的 c*H+h，调用方不做 X reorder 或 Y unshuffle。

**这就是 prefill/decode 共用的既定格式，不允许为 decode 改 packing。** 如果 prefill 已准备相同格式的 tensor，decode 直接共享其指针，不重复 preshuffle、重复 reorder 或额外分配另一份 weight。

### Loader 生命周期

1. 等 HC 的原始权重真正加载完成，且加载后的量化/设备迁移/TP 处理已稳定，再准备 packed buffers。
2. 覆盖所有 use_mix 的 HC 模块，包括 attention、MLP、最终 mixer 和实际使用的 MTP 模块。
3. 用 module/device/weight-version/layout-version 标识 packed cache，**不要把 B 加入打包缓存键**。一个权重版本供全部 B=1..32 共用。
4. 保持 packed buffers 存活到所有使用它们的 graph 被销毁。可用非 persistent buffer 保存派生布局，避免把 packed 值写回普通 checkpoint 参数名。
5. 保留原始 nn.Linear 权重，除非目标仓库的所有 fallback/其他消费者均已改造。直接覆盖原始 `.weight` 的数值顺序会破坏 F.linear fallback，即使 shape 看起来没变。
6. Reload、权重替换或 device 迁移时，使相关 packed cache 和 graph 失效并重新准备；不能让旧 graph 继续指向已释放的 packed storage。

建议生产端维护一个明确的布局标记，例如 `gr_read_bf16_hc_interleaved_preshuffle_v1`。这是交接建议，不是本机 SGLang 已存在的字段。双重打包无法仅靠 flat shape 检测，必须由 loader/共享缓存的所有权避免。

本机 SGLang 的 `model_loader/loader.py` 在加载后遍历的是 `module.quant_method.process_weights_after_loading(module)`，不是自动调用每个模块上同名方法。因此不要只给 GatedResidual 加一个 `process_weights_after_loading()` 就假定它会被执行。请在目标版本真实执行的 load/post-load 阶段显式接上，且必须早于 graph warmup/capture。

`CombinedPaddedGRRead` 构造函数接收原始权重，并会自己打包。**生产 loader 已经提供 packed weights 后，不要再用它构造每个 bucket。** 下节展示不依赖原始权重、不构造旧 control 的编译方式。

## 4. 按 B 准备编译 Handle 和 P/Y

| 物理 B | Down BM / BN / 每 wave BK | Global split / waves | Down | Up BM / BN / BK | P，FP32 |
| --- | --- | --- | --- | --- | --- |
| 1..6 | 16 / 16 / 128 | 4 / 4 | U2 | 16 / 128 / 64 | [4,16,320]，80 KiB |
| 7..15 | 同上 | 4 / 4 | U2 | 16 / 64 / 64 | 同上 |
| 16 | 同上 | 4 / 4 | U2 | 16 / 128 / 64 | 同上 |
| 17..25 | 32 / 16 / 128 | 4 / 4 | U2 | 16 / 128 / 64 | [4,32,320]，160 KiB |
| 26..32 | 同上 | 4 / 4 | U1 | 16 / 128 / 64 | 同上 |

使用源码里的 default_config，不另抄一份手工调参表作为生产分发。参考实现与当前 Small/Large 的 up 配置一致：

```python
from dataclasses import replace
import torch
import flydsl.compiler as flyc
import prefetch_up
from small_batch import default_config as small_config
from large_down import default_config as large_config, pair_launcher

def compile_from_packed(B, packed_down, packed_up):
    assert 1 <= B <= 32
    base = replace(prefetch_up.default_config(B), hidden_pad=4, prefetch_low=False)
    if B <= 16:
        cfg = small_config(B)
        down, up = cfg.down, cfg.up_config(base)
    else:
        down = large_config(B)
        up = replace(base, down_mode="partial", down_n=64, split_k=4)
    down.validate()
    up.validate()
    T_pad = 16 if B <= 16 else 32
    device = packed_down.device
    P = torch.empty(4 * T_pad * 320, dtype=torch.float32, device=device)
    Y = torch.empty(B, 2560, dtype=torch.bfloat16, device=device)
    example_x = torch.empty(B, 10240, dtype=torch.bfloat16, device=device)
    with torch.cuda.device(device):
        launch = flyc.compile(
            pair_launcher(B, down, up),
            example_x.view(-1), packed_down, packed_up, P, Y.view(-1),
            torch.cuda.current_stream(device),
        )
    return launch, P, Y
```

这个示例省略生产输入验证；验证要求见第 2 节。测试文件 `test_graph_buckets.py:compile_bucket` 额外使用 guard storage，实际调用的是同一 launcher/config 路径。不要把测试里的 guard allocation 搬进生产。

`example_x` 只用于编译参数规格；执行时传真正的 normalized X。不要在每次 forward 或 capture 内重新打包、编译或临时创建 P/Y。可在明确的预热阶段为需要的 B 准备 cache；若在 capture 内遇到未准备的 B，应明确报错或使用预先准备好的 fallback，不能偷偷 JIT/allocate。

每次完整 GR 的低层调用：

```python
launch(X.view(-1), packed_down, packed_up, P, Y.view(-1),
       torch.cuda.current_stream(X.device))
# Return Y[B,2560] to the next model operation, not Y[:M].
```

在 SGLang 中，只替换 `GatedResidual.mix` 的数学计算分支。保留前面的 hc_norm，返回值仍为 `mixed_input, (hyper_input, hyper_input_normed)`；combine 后续还会使用 residual tuple。不能把 X/normalized residual 的 storage 当作 Y/P 覆盖。

## 5. SGLang 如何用大的 Capture 跑小的 Batch

本机 `/opt/sglang` 为 `21d0d512ea452a59490aa6585a42d721ef9fb18d`，相关 runner/registry/model 文件与该 HEAD 一致。以下是源码与 registry probe 的事实，不是一次整模型线上 trace：

```text
普通 decode，raw_bs=3，captured_req_width=1
  -> _pad_to_bucket(3, capture_bs) 选择 4
  -> padded_num_tokens=4；capture forward 的 tensor shape 是 4 行
  -> fill_from 只更新有效前缀，按各 slot 的策略处理尾部
  -> replay 已捕获的 B=4 graph，GR 仍执行 4 行
  -> 整个模型 graph 完成后，runner 裁 next_token_logits[:3]
```

脚本默认非 speculative、max decode batch=32 时，常见 buckets 为 `[1,2,4,8,12,16,24,32]`。本次测试没有仅限这些 buckets，而是验证了每个 B=1..32。

**SGLang 并不保证所有 padded tensor 都清零：**

| Registry slot | 小 batch replay 的尾部策略 |
| --- | --- |
| input_ids | FOREACH_COPY，只覆盖前缀，保留尾部旧值 |
| positions / out_cache_loc / req_pool_indices | ZERO |
| seq_lens / seq_lens_cpu | backend 提供的 sentinel |

input_ids/positions 的 padding 策略不能等价为“每层 GR 的 normalized X 尾部必为零”。Embedding、norm、上游 kernel 有各自行为；不要基于零尾行假设来实现 GR。

### 4 Capture / 3 Live 的正确约定

- GR 调用的 X 必须仍有 B=4 行可读，Y 有 4 行可写，P 是 [4,16,320]，不是 [4,3,320]。
- 前三行是有效 normalized X，第四行是 dummy row。kernel 仍计算它，但它不能影响前三行。
- 模型 graph 内返回完整 Y[4,2560] 给下游，**不要在 GR 内动态裁成 3 行**。有效输出的裁剪由 runner/消费边界按正确 token 映射处理。
- standalone 演示可以在 replay 后检查 `Y[:3]`；这只是测试观察，不意味着生产 GR 接口改成 3 行。

其他重要分界同理：

| B / M | 必须采用的配置 |
| --- | --- |
| 8 / 6 | 按 B8，Small 的 up BN64，不是按 M6 选 BN128 |
| 16 / 7 | 按 B16，up BN128，不是按 M7 选 BN64 |
| 24 / 16 | 按 B24，Large BM32、P[4,32,320] |
| 32 / 25 | 按 B32，用 U1，不是按 M25 用 U2 |

### 为什么无效行不会污染有效行

此算子的所有运算都按 token 行独立：down/up 的归约只沿 K/R、global split 或 C，不沿 token/M 维归约。CTA 内 wave split-K 的 LDS 合并也对应同一行。改变 X[j] 只应改变 Y[j]，不改变另一行 Y[i]。

因此本 GR 的有效前缀正确性不要求 X[M:B] 为零，不需要为了这个算子额外加一次清尾 kernel。保留有限旧值的尾行已经针对 100 对真实 HC 权重、全部 B/M 组合验证。

NaN 尾行的压力测试也通过了有效行检查，但 **不允许据此宣称整模型可安全放任 NaN**：下游 attention/MoE/metadata 索引的 masking 和数值条件必须维持原 SGLang 契约。测试也不承诺 dummy Y 的值；它可能是旧输入对应的结果或 NaN，不能当成真实 token 消费。

若独立调用方需要 dummy Y=0，可在已有输入准备环节把对应 GR X 尾行设为 0。仅仅把 token id 设为 0，不等价于 GR X 为零。不要改变 SGLang 全局 padding 策略来掩盖 GR 集成错误。

## 6. P/Y 与 Graph 生命周期

这里有两种 padding，不能混淆：

- `[M:B]`：框架的 dummy token 行，当前 kernel 会正常计算，数值取决于该行 X。
- `[B:T_pad]`：kernel 内部 tile padding，buffer load 的边界/skip_padding 使用 B，而不是 M；在有限权重下 down 写零。

P 的 element offset 是 `(split*T_pad+row)*320+rank`。小 B stride=(5120,320,1)，大 B stride=(10240,320,1)。LDS hidden_pad4 不改变全局 P 的 320 列 stride。

**每次完整调用的 down 覆盖四份 P 的全部 T_pad*R，随后同 stream 的 up 才读取。** P 不需要先清零，没有跨 replay 累加状态或 barrier counter。不能只在 capture 前清一次 P，然后让 replay 依赖旧内容。本次测试在每次 replay 前把整个 P/Y 填 NaN，以检查覆盖行为。

生命周期要求：

1. P/Y 与对应 B/config 绑定，正常 Torch allocation 或至少 16-byte 对齐；不与 X、weights 或彼此重叠。
2. 保持 graph 引用的 X/P/Y/packed weights 地址与编译 handle 存活。释放/重新分配 cache 后必须重建使用它的 graph。
3. 获取 capture 当时的 current stream，不缓存 capture 前的 stream。其他 stream 的输入生产必须通过 event/wait 排序。
4. 只读 packed weights 可共享；并发执行的 graph/stream 必须使用独立 P/Y。初次集成按 HC module + B + graph execution context 分配，避免共享可写 scratch。
5. Sequential workspace 复用必须证明上一次所有消费者已完成；尤其 Y 在下一个模块仍可能被使用，不可仅因两个 kernel 已 launch 就重用。
6. Kernel 返回不意味着 GPU 已同步。正常同-stream消费即可；host 检查要同步。不要为了计时/正确性在生产每次调用强加同步。


### 额外 Padding 总表：不要让调用方重复补齐

| 层级 | B=4、M=3 的例子 | B=24、M=17 的例子 | 谁负责 |
| --- | --- | --- | --- |
| 框架 graph dummy rows | X/Y 保留 B=4 行，第 3 行是 dummy | X/Y 保留 B=24 行，第 17..23 行是 dummy | SGLang graph bucket/输入准备；不要求这些 GR X 行必为零 |
| P 的内部 M tile padding | T_pad=16，P[4,16,320]，第 4..15 行是内部 padding | T_pad=32，P[4,32,320]，第 24..31 行是内部 padding | 调用方按完整容量分配；down 每次写入，有限权重下内部 padding 为零 |
| Up 的 hidden LDS padding | 每个 CTA 的 BF16 high/low 都用 [16,324] 存储逻辑 [16,320] | 同左；up 的 BM 仍然是 16 | kernel 内部 SharedAllocator；调用方不分配这些 LDS 数组 |
| Weight 的额外维度 padding | 无 | 无 | 只做既定 reorder/preshuffle，WD/WU 各 3276800 个 BF16 元素 |
| 测试 guard | X/P/Y 前后各 16 个元素的 canary | 同左 | 仅 test_graph_buckets.py 检查越界用，不是生产接口要求 |

行号按 0-based 描述。**M..B 的 dummy rows 与 B..T_pad 的内部 padding 是两回事。**
B4/M3 时，P[:,3,:] 会计算第四行 X，可能非零；P[:,4:16,:] 才是内部补齐的零行。

生产分配应为：

```text
X: BF16 [B,10240]             不要求外部补到 T_pad
P: FP32 [4*T_pad*320] flat    必须按 T_pad 分配，rank stride=320
Y: BF16 [B,2560]              不补到 T_pad
WD/WU: 各 BF16 [3276800] flat 不补到 324，也不随 B/M 改变

T_pad = 16 if B <= 16 else 32
```

不能把 P 的 320 改成 324，也不能仅把 P 放大却按错误 stride 解释它。16-byte base alignment 是内存对齐约束，不是要求额外增加 Tensor 的逻辑维度。

## 7. 本地已经做了哪些验证

新增独立测试：[test_graph_buckets.py](/opt/pyhip/tests/flydsl/gr_read/test_graph_buckets.py)。不依赖 SGLang、也不构造 CombinedPaddedGRRead。每对原始权重只打包一次，全部 B 共享其指针；每个 B 只捕获一次，然后 M=B..0..B 变化，期间不 recapture。

| 记录 | 覆盖 | 结果 |
| --- | --- | --- |
| E76 synthetic | 两对权重、B1..32、每个 M、zero/stale/NaN 三种尾部 | 6528 次 replay 检查，6336 次非空 FP64 检查，全过；最大 scaled error 0.25951 |
| E77 checkpoint | 全部 100 对真实 HC 权重，含 MTP；B1..32、每个 M、stale 尾部 | 108800 次 replay 检查，105600 次非空 FP64 检查，全过；最大 scaled error 0.30575 |
| SGLang registry probe | 实际 build_decode_registry/fill_from，CPU，B1..32、M=B..0..B | 1088 次 fill 检查；input_ids 保留尾部、索引清零、seq_lens sentinel、地址稳定 |

非空 M 的每个值在下降/上升阶段分别检查；M=0 的检查计入 replay 数，不计入非空 FP64 数。E77 的 100 对权重每一对都覆盖 B4/M3，不是只测试第一层。Synthetic 的 NaN dummy 输出确实被观察到，同时有效行保持正确。

同时检查：P/Y NaN 覆盖、内部 padding、X 未被修改、X/P/Y 前后 guard 未被写坏、packed weight 指针共享且字节不变。使用原 FP64 容差 `rtol=1e-2, atol=5e-3`，没有放宽。

这些是独立 GR 和真实 registry 行为的验证，不是已集成 SGLang 的整模型验证，也不是性能 benchmark。所有 GPU 结果来自 MI308X/gfx942、80 CU、FlyDSL 0.3.1、Torch 2.12.0+ROCm 7.2.4。

原始结果和源码快照在本机 `/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/`：

- `e76_bucket1_32_contract_synthetic_20260917.jsonl`
- `e77_bucket1_32_contract_checkpoint_stale_20260917.jsonl`
- `sglang_registry_padding_20260917.json`
- `probe_sglang_registry_padding_20260917.py`

## 8. 另一台机器复现与集成验收

先同步完整 `gr_read/` 目录，包括本文和新增 `test_graph_buckets.py`。源码模块存在间接导入依赖；不要只复制两个 kernel 函数后假定导入链已经齐全。验证后再按目标仓库的正常模块边界适配导入与 loader hook。

```bash
cd /opt/pyhip/tests/flydsl/gr_read
export HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2

# 固定 T 的最终入口、分段、graph 和输入契约。
python3 test_final_entry.py --graph --check-contract

# 所有 B=1..32，以及每个 bucket 内所有有效 M；无模型。
python3 test_graph_buckets.py --synthetic --buckets {1..32} --weights 2 \
  --output /tmp/gr_bucket_synthetic_new.jsonl

# 全部真实 HC 权重，以 SGLang 风格的旧尾行检验所有 B/M。
python3 test_graph_buckets.py --buckets {1..32} --weights 100 --tails stale \
  --model-path /models/Qwen3.8-Flash-Next-PTPC-FP8 \
  --output /tmp/gr_bucket_checkpoint_new.jsonl
```

output 可以省略，已有路径不能覆盖。只检查用户举例时用 `--buckets 4`，但最终验收必须保留完整 1..32。没有模型的机器只能完成 synthetic 检查，不将其写成 checkpoint 验收。

集成后还必须在目标 SGLang 分支验证：

- 加载时的 packed tensor 与上述格式逐元素/字节一致；重复 warmup/capture/replay 不再次打包；prefill/decode 共享同一格式/版本的只读权重。
- Capture/warmup 时记录 GR 实际 B、配置和 P/Y 容量。3 请求选 B4、25 请求选 B32 等情况不能误用 M 的配置。
- 同一 bucket 连续切换 live batch 大小，比较有效 token 的 logits/模型输出；不要只测固定 batch 的 kernel reference。
- 核对目标版本的 attention/MoE/DP/speculative padding，保持既有 metadata/counter/输出裁剪约定；本文 CPU registry probe 不替代这些 GPU 路径验证。
- 维持 unsupported dtype/shape/B>32 和空输入的正确 fallback。已有 GatedResidual 的空输入分支应保留，不调用 B=0 kernel。
- 不删除 BF16 high/low 补偿。FP64 数学检查通过不等于与旧 BF16 中间舍入路径逐 bit 相同，仍需模型级质量评估。
- 最后再测完整服务性能；不要把本文件的 correctness run 或旧独立 benchmark 当作已经获得的生产收益。

## 9. SGLang 源码定位

以下位置以本机 `21d0d512` 为准，目标机器换了版本应重新核对，不能只复制行号：

| 文件 | 定位 | 说明 |
| --- | --- | --- |
| `python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py` | `_pad_to_bucket`, 136 | 选择不小于 raw size 的最小 bucket |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | `capture_prepare`, 885/895 | 按 capture tokens 取静态输入视图 |
| 同上 | `load_batch`, 1336/1342/1347 | raw count 与 padded count 分离，registry 填充 |
| 同上 | `execute`, 1489 | 完整 replay 后裁最终 logits 的有效前缀 |
| `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py` | `reset_padding`, 230；`build_decode_registry`, 565 | 各 buffer 的不同 tail policy |
| `python/sglang/srt/layers/hyperconnection.py` | `GatedResidual.mix`, 231/240/287 | norm 后数学分支，保留 residual tuple；无 runtime M 参数 |
| `python/sglang/srt/models/qwen4_exp.py` | 1310/1333/1658 | 按 tensor 行数做 embedding/HC 准备，没有给 GR 单独传有效行数 |
| `python/sglang/srt/model_loader/loader.py` | 996/1017 | 权重加载及 quant_method post-load 生命周期 |

源码核对没有修改 SGLang。基线版本与相关文件 SHA256 已随 registry probe JSON 留存。
