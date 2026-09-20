# H64 Shared-Weight Decode

This is the separate new-layout entry, `h64_layout.H64GRRead`, for T=1..32.
The existing `SmallBatchGRRead` / `LargeDownGRRead` entries keep the old
`h*C+c` layout and remain unchanged. Do not mix their packed W_up tensors.

## Weight ABI

Match prefill commit `3f472e33b08995903a2db9227c39d0a5a8840576` exactly:

```python
C, H, R = 4, 2560, 320
K = C * H

def preshuffle(weight):
    n, k = weight.shape
    return weight.detach().reshape(n // 16, 16, k // 32, 4, 8).permute(
        0, 2, 3, 1, 4
    ).contiguous().view(-1)

# Run once during model loading, not during forward/capture.
packed_down = preshuffle(w_down)  # Original BF16 [320,10240].
up_interleaved = w_up.detach().reshape(C, H // 64, 2, 4, 2, 4, R).permute(
    1, 0, 2, 4, 3, 5, 6
).contiguous().reshape(K, R)
packed_up = preshuffle(up_interleaved)  # Original w_up: BF16 [10240,320].
```

`prepare_weights(w_down, w_up)` implements this recipe for standalone use.
If prefill has already prepared the tensors, pass those tensors directly;
do not call the preparation helper again. Each packed tensor is contiguous
BF16 `[3276800]`, has a 16-byte-aligned base, and occupies 6.25 MiB. The pair
occupies 12.5 MiB per HC instance. No weight padding is added.

The layout identifier is `h64_stream_half_lane_preshuffle_v2`. Identical
shape/dtype does not distinguish this layout from old W_up. Track the layout
explicitly in the loader; invalidate old-layout graphs/plans on migration.

## Call and Buffer Ownership

```python
import torch
from h64_layout import H64GRRead

# These are the same tensor objects/storage that prefill reads.
reader = H64GRRead(bucket, packed_down, packed_up)
assert reader.w_down.data_ptr() == packed_down.data_ptr()
assert reader.w_up.data_ptr() == packed_up.data_ptr()

static_x = torch.zeros(bucket, 10240, device=packed_down.device, dtype=torch.bfloat16)
static_y = reader(static_x)  # BF16 [bucket,2560], overwritten on each call.
```

Construct and compile before graph capture. `H64GRRead` retains the weights
by reference without repacking or keeping original/old-layout copies. It owns:

| Buffer | Shape | Meaning |
| --- | --- | --- |
| `partial` | Flat FP32 `4*T_pad*320` | View `[4,T_pad,320]`, split-major linear down partials, before scale/SiLU |
| `output` | BF16 `[bucket,2560]` | Original hidden order, not the reordered weight-row order |

`T_pad=ceil(bucket/16)*16`: 16 for buckets 1..16, 32 for 17..32.
P storage is 80 KiB / 160 KiB. X and Y require only `bucket` rows, not T_pad.
Down overwrites all P entries, including zeroing internal rows `[bucket:T_pad]`.
Up consumes P directly and performs reduction, `/4 + SiLU`, BF16 high/low
compensation, up GEMM, sigmoid and four-stream mean. Two launches total.
Neither P nor Y needs clearing before a normal complete call.

P/Y must not alias X or weights. Concurrent calls need independent scratch
and outputs; multiple bucket readers may share the same read-only weights.
The returned Y is not a new allocation on each call. Consume or copy it before
the next call overwrites it. `run_up` alone requires valid P from matching down.

For framework-owned scratch, the compiled `reader.dispatch` accepts
`(flat_x, packed_down, packed_up, flat_p, flat_y, current_stream)` with the
same shapes/dtypes/alignment and bucket specialization. Prepare all storage
before capture and keep it alive for the graph lifetime.

## Capturing B and Replaying M <= B

```python
for _ in range(3):
    reader(static_x)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_y = reader(static_x)

# Example: bucket=4, actual_x has 3 rows. Keep the captured B-row allocation.
live = actual_x.shape[0]
assert 0 <= live <= bucket
static_x[:live].copy_(actual_x)
static_x[live:].zero_()
graph.replay()
live_y = static_y[:live]
```

The kernel specialization, full X allocation and row strides remain B-based.
Do not pass an M-row allocation to the B-row graph or change P's split stride
to M. Only Y[:M] is live. Rows are independent: stale/NaN padded X rows do not
contaminate live rows, but padded outputs are not valid service outputs.
Zero-fill tails for a predictable integration contract. Producer copies and
graph replay must be ordered on the same stream or by explicit events.

This is tested for every B=1..32, with M decreasing B..0 and increasing 1..B,
including zero/stale/NaN tails, P/Y poisoning before replay and memory guards.

## Selected Up Schedule

All rows use BM16, BN128, four waves, stream-plane layout, A-first MFMA,
128-bit weight loads, hidden LDS stride324 and FP32/high-low semantics.
The existing selected down pipeline is unchanged.

| T | Up BK | Skip P padding loads | Preload all W tile before P/SiLU |
| --- | ---: | --- | --- |
| 1..8 | 160 | Yes | Yes |
| 9..16 | 32 | No | Yes |
| 17..28 | 160 | Yes | No |
| 29..31 | 32 | Yes | No |
| 32 | 160 | Yes (no internal tail) | No |

These choices are gfx942 measurements, not promises of optimal performance on
other architectures. The constructor warns on other ROCm targets. It does not
select old-layout fallbacks or convert weights at runtime.

## Standalone Commands

From this directory, no model and no SGLang required:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 test_h64_layout.py

HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 bench_h64_layout.py --selected --check-no-regression --synthetic \
  --rows {1..32} --weights 100 --rounds 3 --samples 7 --seed 707
```

With checkpoint HC weights:

```bash
MODEL=/models/Qwen3.8-Flash-Next-FP8
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 test_h64_layout.py --weights 2 --model-path "$MODEL"

HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 bench_h64_layout.py --selected --check-no-regression \
  --rows {1..32} --weights 100 --rounds 3 --samples 7 \
  --model-path "$MODEL" --output /tmp/gr_h64_checkpoint.jsonl
```

Output files are optional and opened exclusively; use a fresh path for a new
run. Both benchmark and tests enforce the existing FP64 tolerance
`rtol=1e-2, atol=5e-3`. The benchmark checks every pair before and after input
changes. The dedicated test adds exhaustive bucket/live-row replay coverage.

The benchmark prints `baseline_selected` and `h64_selected`. **Compare Full
graph**, measured for the complete two-launch call. Negative Delta% is faster.
`--check-no-regression` rejects any requested row whose measured H64 median
exceeds its same-run baseline; it does not claim immunity to measurement noise.
`--stages` adds separately timed Down/Up, which must not be added as an E2E
estimate. `--eager` is diagnostic only, not the graph acceptance metric.

All runtime X/WD/WU/P/Y addresses are matched in the serialized A/B benchmark.
Only the benchmark retains old/new source payloads and switches WU outside
timing. This is not a production second packed weight or runtime conversion.
Every run checks `rocm-smi`; an occupied GPU is rejected so it can be retried
when idle. No clock/power setting is changed.

Experimental flags are for reproducing tuning, not framework integration.
`--selected --preload-weights 0 1` compares the two preload policies. Without
`--selected`, `--strategy`, `--up-n`, `--up-k`, `--up-waves`,
`--weight-copy-bits`, `--b-first`, and `--skip-padding` define an explicit sweep.
Rejected remap/register variants are retained for reproducibility only.
