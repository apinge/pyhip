# Qwen3.8 GR Read on MI308X

Current algorithm: pyhip `a716010`, documented on 2026-09-17.
Standalone kernel development only; no SGLang production dispatch is changed.

The recommended entries are **SmallBatchGRRead for T1..16** and
**LargeDownGRRead for T17..32**. Both reuse the same packed weights and launch
two GPU kernels. Calling `CombinedPaddedGRRead` alone still runs the frozen
control, not the current selection.

Only the final selected workflow is documented here; experimental drivers and
old entry-point commands are intentionally omitted.

Chinese instructions:
[benchmark, accuracy and interfaces](/opt/qwen3.8-flash-next-doc/33-GR_read_使用指南_Benchmark精度校验与算法接口_2026-09-17.md).
Algorithm overview: [report 30](/opt/qwen3.8-flash-next-doc/30-GR_read_当前算法简述_2026-09-17.md).

For framework integration with weights packed at model load, read
[INTEGRATION.md](INTEGRATION.md). It covers the distinction between captured B,
live M and internal tile padding, the actual SGLang padding policies, and a
tested direct-from-packed launcher recipe. Do not repack once per bucket.

## Quick Start

Run from this directory in **Bash**. Select an idle physical GPU; change 2 for
your machine. ROCm PyTorch still uses the `torch.cuda` API.

```bash
cd /opt/pyhip/tests/flydsl/gr_read
export HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2
```

Validated environment: MI308X/gfx942, 80 CUs, Python 3.10.12, FlyDSL 0.3.1,
PyTorch 2.12.0+ROCm 7.2.4, Triton 3.7.1. SGLang is not required.
Copy the complete `gr_read/` directory, including `baselines/`.
Synthetic runs need no model, old results, or files under the local report directory.

### Benchmark All Selected T1..32 Without a Model

```bash
python3 bench_small_batch.py --selected --synthetic --rows {1..16} \
  --weights 100 --rounds 3 --samples 7 --seed 202

python3 bench_large_down.py --selected --synthetic --rows {17..32} \
  --weights 100 --rounds 3 --samples 7 --seed 202 --torch-baseline
```

These are two sequential runs, not a single combined T1..32 process.
Keep `--selected`: omitting it runs the script's tuning grid instead.
Small-T automatically includes the colleague's bundled `626c6413` Triton
implementation. Large-T includes Torch compile only with `--torch-baseline`;
omit that flag for a FlyDSL-only comparison. These scripts do not accept the
old driver's `--baselines` or `--combined-host` flags.

| Script / T | Read this row, Full graph column | Other rows |
| --- | --- | --- |
| small, 1..16 | `ours_optimized` | `frozen` is old FlyDSL; `triton3stage` is three-stage Triton from commit 626c6413 |
| large, 17..25 | `m32_k128_w4_pf1_i1_s4_u2` | `control` is old FlyDSL; optional `torch_compile` |
| large, 26..32 | `m32_k128_w4_pf1_i1_s4` | Same controls; no suffix means U1, not no prefetch |

A quick functional smoke can use `--rows 1 7 16` / `--rows 17 25 26 32`
and `--weights 2 --rounds 1 --samples 1`. Do not report those small-working-set
numbers as performance gains versus the 100-weight runs.

### Columns, Timing and Output

| Column | Meaning |
| --- | --- |
| Full graph | Microseconds per complete GR read, including both GPU kernels and their graph dependency |
| Down only | Separate graph repeating down only; a diagnostic, not down's timeline slice in the full graph |
| Up only | Separate graph repeating up on prepared P; a diagnostic |
| Eager wall | Host wall time per complete call, including submission and batch-end synchronization |
| `-` | Not measured, not zero |

**Use Full graph for comparison. Do not add Down only and Up only.**
Small-T needs `--stages` to measure isolated stages; large-T always measures them.
Both need `--eager` for eager wall time. Eager and graph can have different winners.

With 100 weight pairs, capture rotates all pairs twice (200 complete calls),
and each sample replays that graph three times. `3 rounds x 7 samples` gives
21 samples per backend/stage; the table reports their median. Measurement order
is shuffled each round. Compilation, weight preparation and FP64 references
are outside the timed region. Do not run concurrent GPU jobs or ATT/PMC while
collecting the main performance table. Record GPU, CU count and package versions.

Print-only is the default. To keep raw samples, metadata and source snapshots,
append a **new** output path to each command:

```text
--output /tmp/gr_small_synthetic_new.jsonl
--output /tmp/gr_large_synthetic_new.jsonl
```

No historical result is read. Existing output/snapshot names are not overwritten.
A larger number of weights changes the cache working set; preserve the weight
count and timing method for matched comparisons. T is actual input rows, not
necessarily service concurrency: 25 requests may use a T32 graph bucket.

### Real HC Weights and Full Acceptance

Only change the data source, not the timing method. The current local checkpoint
is below; use the directory available on your machine, containing
`model.safetensors.index.json` and its referenced shards.

```bash
MODEL=/models/Qwen3.8-Flash-Next-PTPC-FP8

python3 bench_small_batch.py --selected --rows {1..16} \
  --weights 100 --rounds 3 --samples 7 --seed 202 \
  --model-path "$MODEL" --output /tmp/gr_small_checkpoint_new.jsonl

python3 bench_large_down.py --selected --rows {17..32} \
  --weights 100 --rounds 3 --samples 7 --seed 202 \
  --model-path "$MODEL" --torch-baseline --output /tmp/gr_large_checkpoint_new.jsonl
```

Omit `--output` for print-only. `--synthetic` and `--model-path` are mutually
exclusive in these two scripts. With neither flag, both default to synthetic.
An explicitly missing checkpoint is an error; it is not silently replaced by
random weights. The loader reads only HC weight tensors, not the whole model.

The 100 checkpoint pairs include main-model and MTP HC weights. Inputs are
generated BF16 tensors, not captured production activations. Synthetic checks
alone do not satisfy real-weight acceptance or establish model-level quality.

## Accuracy Validation

The formula is evaluated in FP64 by `support.reference(x, w_down, w_up)`,
using **original, unpacked weights**. BF16 output must pass
`torch.allclose(actual.double(), expected, rtol=1e-2, atol=5e-3)`.
Equivalently, every element's scaled error
`abs(actual-expected)/(5e-3 + 1e-2*abs(expected))` must be <= 1.
Keep FP32 accumulation and the BF16 high/low activation compensation.

### Plain Python Checks

The main correctness/debug entry is a plain Python script, without pytest or
a checkpoint. By default it covers **every T from 1 to 32**:

```bash
python3 test_final_entry.py --graph --check-contract
```

It automatically constructs the final Small/Large reader, checks the selected
configuration, FP64 output and linear down result, split4 P shape/padding,
packed-weight immutability, and separate down/up calls. `--graph` adds
changed-input replay and zero-input checks; `--check-contract` checks rejection
of T=0/invalid rows and wrong dtype/device/shape/strides. Output names the actual
reader, GPU kernels, prefetch unroll, up BN and P shape.

To step through one final case, break after construction and before the call:

```bash
python3 test_final_entry.py --rows 7 --graph --debug
```

Supplementary per-backend regression, including additional configurations:

```bash
python3 test_small_batch.py --rows {1..16}
python3 test_large_down.py --selected --rows {17..32}
```

Both test seeds 101/202, FP64 output, separate down/up calls, capture/replay with
changed inputs, zero input, NaN-poisoned P overwrite, zero padded rows, shared
weight pointers and independent P/Y. Small-T additionally tests default plus
BN64/128 configurations and packed-weight byte equality. Both check range
rejection; the large test also checks frozen T1/8/16 controls.
Exceptions or nonzero exit are failures. Do not use `python -O`, which removes
assertions. These are correctness/debug calls, not performance measurements.

To test a larger captured graph with fewer live rows, including every B=1..32
and M=B..0..B, run the dedicated contract test:

```bash
python3 test_graph_buckets.py --synthetic --buckets {1..32} --weights 2
```

This captures once per bucket and changes the live prefix without recapture;
it tests zero/stale/NaN tails, poisoned P/Y, guards and shared packed weights.
The all-100-checkpoint command and results are in INTEGRATION.md. It is not a
whole-model SGLang test or a performance benchmark.

The real-weight benchmark commands above check every pair before and after
changed-input graph replay. Each selected range must pass 1,600 initial and
1,600 changed-input checks, totaling 3,200 of each across both runs.

| Selected backend | JSONL correctness fields | Required |
| --- | --- | --- |
| `ours_optimized` | `checks.initial`, `checks.changed` | `passed_pairs == total_pairs == 100` per T, empty `failures` |
| large `m32_...` | `checks.initial`, `checks.changed_replay` | `passed_pairs == total_pairs == 100`, `enforced == true` per T |

FlyDSL failures abort. Colleague/Torch errors are only recorded; a successful
benchmark exit does not mean every comparison backend passed FP64. Do not
weaken our tolerance to match an external baseline.

### Recorded Validation

These are recorded runs, not claims that every command is rerun whenever
this README is edited:

| Record | Coverage | Result |
| --- | --- | --- |
| E74, 2026-09-17 | Selected T1..16, real 100 HC pairs, seed 202 | 1,600 initial + 1,600 changed checks passed; max scaled error 0.30545 |
| E55, 2026-09-16 | Selected T17..32, real 100 HC pairs, seed 101 | 1,600 initial + 1,600 changed checks passed |
| a716010 pre-commit | Small T1..16 and large T17..32 plain Python tests | Passed |
| Final-entry documentation update | `test_final_entry.py --graph --check-contract`, all T1..32 | FP64 output/down, stages, replay, zero input and input rejection passed |

E74's selected T1..16 full graphs beat frozen and colleague controls. T1..7
latency fell 6.7-18.2% versus frozen; eager wall did not improve at every T.
Large-T was not newly benchmarked in E74. Raw data:
[E74](/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e74_small_selected_t1_t16_checkpoint_seed202_20260917.jsonl),
[E55](/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e55_selected_full_rerun_20260916.jsonl).
Do not merge these dates into a claimed single fresh T1..32 run. Optimization,
IR/ISA and ATT evidence: [report 32](/opt/qwen3.8-flash-next-doc/32-GR_read_T8到16固定WeightLayout优化与ATT_2026-09-17.md).
Reports and raw files are local evidence, not runtime dependencies.

## Algorithm and Call Chain

Fixed dimensions: `C=4, H=2560, K=10240, R=320`.

```text
X[T,K]
  -> down: X @ W_down.T, split across K
  -> P[4,T_pad,R] FP32 linear partials
  -> up: sum global partials -> /4 -> SiLU -> BF16 high/low
         -> up GEMM -> sigmoid -> pair with original X -> mean over C
  -> Y[T,H] BF16
```

| T | Entry | Down BM / BN / per-wave BK | Global splits / waves | Prefetch | Up BM / BN / BK | P |
| --- | --- | --- | --- | --- | --- | --- |
| 1..6 | SmallBatchGRRead | 16 / 16 / 128 | 4 / 4 | U2 | 16 / 128 / 64 | [4,16,320] |
| 7..15 | SmallBatchGRRead | 16 / 16 / 128 | 4 / 4 | U2 | 16 / 64 / 64 | [4,16,320] |
| 16 | SmallBatchGRRead | 16 / 16 / 128 | 4 / 4 | U2 | 16 / 128 / 64 | [4,16,320] |
| 17..25 | LargeDownGRRead | 32 / 16 / 128 | 4 / 4 | U2 | 16 / 128 / 64 | [4,32,320] |
| 26..32 | LargeDownGRRead | 32 / 16 / 128 | 4 / 4 | U1 | 16 / 128 / 64 | [4,32,320] |

All selected down grids are `(1,20,4)`, 80 CTAs, 256 threads per CTA. Each
global split covers K=2560; four waves divide that work and reduce locally in
LDS. Down writes linear partials, **not SiLU results**. U1/U2 are one-/two-beat
prefetch loops, not one/two GPU launches.

Up uses 4 waves, `hidden_pad=4` and high/low compensation. Up grids are
`(1,80,1)` for T1..6/T16, `(1,160,1)` for T7..15, and `(2,80,1)` for T17..32.
Each up CTA reads every split and R column for its own M tile, not other M
tiles. N CTAs repeat these reads; logical bytes are not measured HBM traffic.

```text
CombinedPaddedGRRead(T, original_WD, original_WU)
  -> prepare the unchanged packed weights
  -> also construct the frozen control
SmallBatchGRRead(prepared) [1 <= T <= 16]
  or LargeDownGRRead(prepared) [17 <= T <= 32]
  -> choose default_config(T); allocate independent P/Y; compile handles
reader(x)
  -> _check_input
  -> compiled large_down.pair_launcher
       -> down_launcher -> down_wave_splitk_pipeline
       -> prefetch_up._launchers -> up_gate
```

No new third stage, cross-CTA spin barrier, RMSNorm, GR write, or TP
communication is included. Do not halve the HC matrix dimensions merely
because the surrounding model uses TP=2.

## Weight and Workspace Interface

### Public Preparation

| Argument | Required shape and dtype |
| --- | --- |
| x | Contiguous normalized BF16 [T,10240], 1 <= T <= 32 |
| Original w_down | BF16 [320,10240] |
| Original w_up | BF16 [10240,320], stream-major row c*H+h |
| y | Contiguous BF16 [T,2560], natural hidden order |

All tensors must share one ROCm device. Prepare outside the hot loop/capture.
The following is a complete synthetic example; replace the tensors with your
own originals when integrating with a caller:

```python
import torch
from combined_host import CombinedPaddedGRRead
from small_batch import SmallBatchGRRead
from large_down import LargeDownGRRead
from support import reference, synthetic

x, w_down, w_up = synthetic(17, seed=101)
T = x.shape[0]
assert 1 <= T <= 32
with torch.cuda.device(x.device), torch.inference_mode():
    prepared = CombinedPaddedGRRead(T, w_down, w_up)
    reader = SmallBatchGRRead(prepared) if T <= 16 else LargeDownGRRead(prepared)
    y = reader(x)
    assert torch.allclose(y.double(), reference(x, w_down, w_up),
                          rtol=1e-2, atol=5e-3)
```

Both selected readers share `prepared.w_down/w_up` by pointer but allocate
independent P/Y. They do not pack again. No caller-created scratch is needed.
The temporary control construction is setup overhead, not full-call latency.

**Never pass packed or already-interleaved weights into CombinedPaddedGRRead.**
There is no public `from_packed`, `packed=True`, `out=` or `workspace=` argument.
Reshaping packed storage back to the original shape does not undo the permutation.
Use compiled handles below for already-packed weights and caller-owned buffers.

T is fixed at construction, including within one padded-row bucket. A physical
X[32,10240] needs T32/U1, not T25/U2. Do not externally pad X merely because P
has padded rows. Selected readers reject T=0 and T>32; only the frozen original
entry handles T=0 as an empty result.

### Exact Packed Weight Format

Weight preprocessing is globally unchanged and shared with prefill.
Continuing the example above, this reproduces it for inspection/low-level use:

```python
from prefetch_up import preshuffle_weight

C, H, R = 4, 2560, 320
K = C * H
with torch.cuda.device(x.device), torch.inference_mode():
    WU_hc = w_up.reshape(C, H, R).permute(1, 0, 2).contiguous().reshape(K, R)
    WD_packed = preshuffle_weight(w_down)
    WU_packed = preshuffle_weight(WU_hc)
    assert torch.equal(WD_packed, reader.w_down)
    assert torch.equal(WU_packed, reader.w_up)
```

1. Only W_up is logically reordered:
   `WU_hc[h*C+c,r] = w_up[c*H+h,r]`.
2. Both matrices use the same physical preshuffle:
   `W.reshape(N//16,16,K_in//32,4,8).permute(0,2,3,1,4).contiguous().view(-1)`.

Thus `packed[n//16,k//32,(k//8)%4,n%16,k%8] == W[n,k]`.
For W_up, W here means WU_hc. This is a permutation, not quantization.

| Packed weight | 5D storage | Flat launcher argument | Bytes |
| --- | --- | --- | --- |
| WD | [20,320,4,16,8] | BF16 [3276800] | 6,553,600 |
| WU | [640,10,4,16,8] | BF16 [3276800] | 6,553,600 |

**X is not reordered.** Up's epilogue matches gate[t,h*C+c] with X[t,c*H+h]
before reducing C. Do not perform a flat multiply of those different orders.
The format is independent of T, global split, BM, BN and U1/U2.

Pack once per layer/weight version/device. Constructors do not cache across
different T-specific readers. Changing original weights does not refresh
packed copies or graph addresses; rebuild/update affected resources.

### Intermediate Buffer

At the launcher boundary P is a contiguous flat **FP32** tensor.

| Selected T | View | Element strides | Elements / bytes |
| --- | --- | --- | --- |
| 1..16 | [4,16,320] | (5120,320,1) | 20,480 / 81,920 (80 KiB) |
| 17..32 | [4,32,320] | (10240,320,1) | 40,960 / 163,840 (160 KiB) |

Element offset is `(split_index*T_pad + row)*320 + rank`.
`P.view(4,T_pad,320).sum(0)[:T]` is the linear down result, before /4 or SiLU.
Never apply SiLU separately to each split. `hidden_pad=4` affects internal LDS
only: global P rank stride is **320, not 324**.

P and up configuration must both belong to the selected reader. Do not reuse
buffers/configuration from an unrelated implementation just because they have
the same member names.

Down overwrites P, including zero padding for finite inputs/weights. No
pre-zeroing or counter buffer is needed. Up must run after down with the same
X and P. Activation high/low arrays and logits are internal LDS, not additional
caller allocations. Y is [T,H], never [T_pad,H].

### Caller-Owned Buffers

Compiled handles expose these internal signatures, not a stable production C ABI:

```text
reader.down(X_flat, WD_packed, P_flat, stream)
reader.up(X_flat, WU_packed, P_flat, Y_flat, stream)
reader.dispatch(X_flat, WD_packed, WU_packed, P_flat, Y_flat, stream)
```

Continuing the selected-reader and packed-weight examples:

```python
T_pad = 16 if T <= 16 else 32
P = torch.empty(4 * T_pad * R, dtype=torch.float32, device=x.device)
Y = torch.empty((T, H), dtype=torch.bfloat16, device=x.device)
with torch.cuda.device(x.device), torch.inference_mode():
    reader.dispatch(x.view(-1), WD_packed, WU_packed, P, Y.view(-1),
                    torch.cuda.current_stream(x.device))
    assert torch.allclose(Y.double(), reference(x, w_down, w_up),
                          rtol=1e-2, atol=5e-3)
```

Read external **Y**, not `reader.output`: dispatch does not rebind owned buffers.
`run_down(x)` and `run_up(x)` operate on `reader.partial/output`; both return
None. For owned-buffer debugging, call them in order, then inspect
`reader.partial.view(4,T_pad,320)` and `reader.output`.

All arguments must match shape, dtype, contiguous layout, device and compiled
T/config. P/Y must not overlap each other, X or weights. Use normal Torch
allocations or at least 16-byte-aligned custom bases. Raw handles do not repeat
all wrapper checks. To compile without a temporary public reader, the internal
`down_launcher`, `prefetch_up._launchers` and `pair_launcher` need explicit,
mutually matching down/up configs; there is no public packed-weight factory yet.

### Lifetime, Streams and Graph Capture

Each reader owns mutable P/Y. Its output is overwritten on reuse; clone a
result before reusing that reader if it must be retained. Calls are asynchronous.
Read-only weights can be shared; concurrent executions require independent P/Y.

Prepare, compile, allocate and warm up before capture. Keep tensors and compiled
handles alive while pending work/graphs use them. Graphs keep captured addresses:
assigning a new tensor to a Python attribute does not retarget a graph. Use the
stream active inside capture; cross-stream producers need explicit event waits.

Continuing the example, capture and verify changed-input replay:

```python
with torch.cuda.device(x.device), torch.inference_mode():
    static_x = x.clone()
    next_x = x * 0.99 + 0.015625
    for _ in range(3):
        reader(static_x)
    torch.cuda.synchronize(x.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_y = reader(static_x)
    static_x.copy_(next_x)
    graph.replay()
    assert torch.allclose(static_y.double(), reference(static_x, w_down, w_up),
                          rtol=1e-2, atol=5e-3)
```

For external buffers, capture `reader.dispatch(...)` with external P/Y and the
current capture stream, then consume that external Y with correct ordering.

The Python examples above were executed on gfx942 at T=1/6/7/15/16/17/25/26/32,
including packed-weight equality, caller-owned buffers and changed-input graph
replay. These are interface checks, not another full performance run.

## Source Map and Portability

| File | Responsibility |
| --- | --- |
| `small_batch.py` | Selected T1..16 defaults, independent P/Y, public wrapper |
| `large_down.py` | Selected T17..32 defaults and shared down/pair launcher factories |
| `prefetch_up.py` | P reduction, SiLU, high/low, up GEMM and gate epilogue; original packing helpers |
| `combined_host.py` | Frozen compiled-host entries and input check; prepares weights for selected readers |
| `kernel.py` | Original implementation, shared constants/configuration/guards |
| `bench_small_batch.py`, `bench_large_down.py` | Current full-call benchmarks and real-weight checks |
| `test_final_entry.py` | Main final T1..32 plain Python correctness/debug entry; automatically selects Small/Large |
| `test_small_batch.py`, `test_large_down.py` | Supplementary per-backend configuration tests |
| `test_graph_buckets.py` | Load-once packing, raw packed launchers and larger-bucket/smaller-live graph correctness |
| `support.py` | FP64 reference, synthetic/checkpoint data, graph timing |
| `baselines/` | Unmodified bundled Triton sources, provenance and license |
| `analyze_trace_batch.py` | Supplied FlyDSL ATT analyzer wrapper; optional kernel-name filter |

Other ROCm architectures, including gfx950, warn instead of being rejected by
the GR architecture guard. Input/dtype/device/resource checks remain errors.
Compile source for the actual target; do not assume a gfx942 HSACO is portable
or force a gfx942 target when validating gfx950. No native gfx950 performance
or correctness result is claimed here. First run the selected-path FP64/replay
tests on that device. Triton baselines may have their own architecture limits.

There is no SGLang/AITER environment switch required for these standalone
FlyDSL calls. AITER attention/MoE configuration in a model launch is unrelated
to this standalone entry. Preserving the packing contract does not itself
integrate the kernel into a serving framework.

## Separate H64 Shared-Weight Entry

The new opt-in `h64_layout.H64GRRead` consumes the H64/stream/lane layout from
prefill commit `3f472e33`, sharing one packed weight pair with prefill. The old
entries described above are unchanged and cannot consume this new W_up.
See [H64_LAYOUT.md](H64_LAYOUT.md) for the exact load-time permutation, pointer
ownership, P padding, capture-B/replay-M contract and selected algorithm.

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python3 test_h64_layout.py
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python3 bench_h64_layout.py \
  --selected --check-no-regression --synthetic --rows {1..32} \
  --weights 100 --rounds 3 --samples 7
```

No model is needed for these commands. Compare the complete `Full graph`
column for `h64_selected` against `baseline_selected`; `--output` is optional.
