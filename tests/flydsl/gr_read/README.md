# Qwen3.8 GR Read on MI308X

Standalone FlyDSL experiment. No SGLang production dispatch is changed.

## Review and Debug Start Here

The final path is `CombinedPaddedGRRead` -> `_padded_pair_launcher` ->
`prefetch_up._launchers` -> one selected down kernel and `up_gate`.
Construction chooses the T bucket; a prepared reader does not change T at call time.

For a final-entry-only example with explicit inputs, FP64 reference, intermediate
checks and an optional breakpoint:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python3 test_final_entry.py --rows 1 17 --graph
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python3 test_final_entry.py --rows 17 --debug
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python3 test_final_entry.py --graph --check-contract
```

Unlike the older pytest-only files, `test_final_entry.py` is a plain Python script
using `assert torch.allclose(...)`, with no pytest dependency. By default it checks
T=1/16/17/24; `--graph` adds replay checks for the selected rows and
`--check-contract` adds empty/invalid input checks. This is not the full checkpoint
acceptance suite, and its synchronized stage checks are not performance measurements.
See the [call-chain and manual-test guide](/opt/qwen3.8-flash-next-doc/22-GR_read_最终版调用链与手写单测_2026-09-14.md).

## Latest Optimized Entry

`CombinedPaddedGRRead` in `combined_host.py` is the latest validated standalone
candidate (E38). It keeps two GPU launches, uses a BF16 hidden LDS row stride of
324 for the logical width 320, and submits both kernels through one compiled
host entry. The selected configuration is `hidden_pad=4, prefetch_low=False`.
Weights, global tensor shapes and high/low activation compensation are unchanged.

`kernel.GRRead` and `selected_configs.json` retain the E17 reference defaults;
they are not silently replaced. From this directory, use the optimized entry as:

```python
from combined_host import CombinedPaddedGRRead

reader = CombinedPaddedGRRead(x.shape[0], w_down, w_up)
y = reader(x)
```

All 2,400 checkpoint/row combinations and changed-input graph replay checks
passed the original FP64 tolerance. E38 full-call graph medians are 15.270 us at
T=1 and 35.677 us at T=24, versus 18.697 / 44.386 us for the same-run E17 reference.
Ordinary eager timings are separate: at T=1 the padded combined entry is
20.329 us; the unpadded `CombinedHostGRRead` is 19.981 us. This small eager
difference is not evidence that the unpadded GPU kernels are faster.

See the current [optimization and trace report](/opt/qwen3.8-flash-next-doc/21-GR_read_ROCm_LDS_padding与HostLauncher优化_2026-09-10.md)
and [raw E38 results](/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e38_optimized_full_checkpoint.jsonl).

## Contract

- GPU: MI308X / gfx942. Measured with 80 compute units.
- Input: contiguous BF16 normalized residual `X[T,10240]`, `1 <= T <= 24`.
- Original weights: BF16 `W_down[320,10240]`, `W_up[10240,320]`.
- Output: BF16 `Y[T,2560]`. Zero rows return an empty output; rows above 24 are rejected.
- Formula: `mean_C(sigmoid(silu(X @ W_down.T / 4) @ W_up.T) * X)` for `C=4`.
- RMSNorm and GR write are outside this kernel.

The implementation uses two launches, with a measured row-count split:

- T=1..16: down writes 16 global FP32 K partials; up performs ordered reduction,
  SiLU, up GEMM and sigmoid/multiply/stream mean (the previous accepted path).
- T=17..24: down uses 256 threads / 4 waves to split K **within one CTA**, reduces
  the wave partials in LDS and applies `/4 + SiLU`. Up consumes the completed
  FP32 activation and fuses up GEMM with sigmoid/multiply/stream mean.

The fused-down path follows the wave split-K approach in `../test_gemm.py`.
There is no cross-CTA counter or spin barrier. Each wave processes a K512 tile;
four waves cover K2048 per iteration, with five iterations for K=10240. Weight
preparation interleaves the four up-projection streams and uses the colleague's
BF16 preshuffle layout. The T=17..24 intermediate shrinks from 640 KiB to 40 KiB.
The fusion is available explicitly for every T=1..24, but screening did not
justify replacing the small-row default.

**The default preserves the FP32 SiLU result as BF16 high/low components.** The up
projection accumulates both components using BF16 MFMA. This is necessary to pass
the original FP64 tolerance on every tested checkpoint weight pair, including a
recorded MTP rounding case. It does not bitwise reproduce an intermediate rounded
to one BF16 value. Inputs, weights and final output remain BF16; accumulators are
FP32. The diagnostic `Config(compensate_hidden=False)` reproduces the original
intermediate rounding and does not pass the complete acceptance dataset.

`kernel.GRRead(rows, wd, wu)` selects the E17 reference configuration. Explicit
`Config(...)` arguments are experimental. The optimized entry above fixes the
validated LDS padding and host submission options itself.

## Files

| File | Purpose |
| --- | --- |
| `kernel.py` | FlyDSL kernels, preparation, shape/device guards, fixed configuration |
| `test_gr_read.py` | BF16 correctness, graph replay, cache-key isolation, recorded rounding regression |
| `test_final_entry.py` | Plain Python final-entry checks with `assert torch.allclose`, graph replay and `--debug` |
| `support.py` | FP64 reference, exact local Triton baseline, checkpoint loading and timing |
| `benchmark.py` | All 100 checkpoint pairs, all 24 row counts, randomized order, optional previous FlyDSL baseline |
| `selected_configs.json` | Accepted per-row configurations with activation compensation enabled |
| `tune.py` | Explicit, recorded configuration sweeps, including `--family wave_fused` |
| `quick_bench.py` | Single-weight screening, optional individual-stage timings |
| `dump_ir.py` | FlyDSL MLIR/LLVM/ISA and tuned Triton TTIR/TTGIR/LLVM/ISA |
| `analyze_ir.py` | Static instruction/resource summary |
| `diagnose.py` | Reproduce the seed-101 MTP rounding case |
| `smoke.py` | Small compilation/correctness probe |
| `prefetch_up.py` | Isolated E17-derived kernel with low-fragment prefetch and hidden LDS padding controls |
| `combined_host.py` | Compiled host entries; `CombinedPaddedGRRead` uses the validated padding |
| `three_stage.py` | Independent reduce/SiLU control experiment; not the recommended candidate |
| `register_gate.py` | Wave/register epilogue experiment, slower in screening and not adopted |
| `bench_three_stage.py` | Matched graph/eager benchmark for all new candidates and optional external baselines |
| `test_three_stage.py`, `test_register_gate.py`, `test_combined_host.py`, `test_prefetch_up.py` | Candidate correctness, graph, layout and cache-key coverage |
| `profile_case.py` | Rotating real-weight workload for rocprofv3 discovery, ATT and separate PMC jobs |
| `analyze_trace_batch.py` | Invokes the provided FlyDSL skill analyzer and preserves per-dispatch reports |
| `dump_experiment_ir.py` | MLIR/LLVM/ISA snapshots for new candidates |

## Environment and Use

Validated with FlyDSL 0.3.1, PyTorch 2.12.0+ROCm 7.2.4, Triton 3.7.1, Python 3.10.
The machine already has the required packages. FlyDSL was not upgraded.

Run commands from this directory and select an available physical GPU. GPU 2 was
used for the recorded experiment.

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 -m pytest -q test_gr_read.py test_three_stage.py test_register_gate.py test_combined_host.py test_prefetch_up.py

HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 smoke.py --rows 24

HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 benchmark.py --configs selected_configs.json \
  --previous-kernel /opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e10_full_checkpoint_compensated.kernel.py \
  --output /opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/new_full_checkpoint.jsonl

HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 dump_ir.py --rows 24 \
  --output /opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/new_ir_t24
```

Benchmark output files and IR directories must be new. Existing experiment results
are not overwritten. The benchmark checks that the local Triton kernel matches
commit `8cf5501b6913f57a2e7c8dcee52b625fc8ab23c3` before executing it.
Its GPU kernel is `_hc_mix_persistent_kernel`; `fused_hc_mix` is the Python
wrapper. The optional previous FlyDSL snapshot is timed in the same randomized
backend order, not compared against a historical timing from another run.

The checkpoint tools assume the model is at `/models/Qwen3.8-Flash-Next-FP8`
and the SGLang baseline checkout is at `/opt/sglang`. The linked full report,
benchmark outputs, IR dumps and checkpoint-derived `.pt` samples are local
experiment artifacts and are not included in this repository.

The MTP regression test skips when its checkpoint-derived sample is absent.
On a machine with the model and baseline above, recreate the expected sample
with the following command, provided the output directory does not already exist:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 diagnose.py --rows 10 --seed 101 \
  --output /opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e08_t10_diagnosis
```

`diagnose.py` deliberately disables activation compensation to reproduce the old
failure and generate `98_flydsl.pt`; the accepted default kernel remains unchanged.

Python use from this directory:

```python
from kernel import GRRead

reader = GRRead(x.shape[0], w_down, w_up)
y = reader(x)
```

Prepare the reader before graph capture. Each reader owns its workspace/output;
subsequent calls overwrite `y`. Calls sharing a reader must be serialized. The
launch uses the current stream at invocation time, including the capture stream.
The reference and benchmark input generators use actual checkpoint weights with
synthetic normalized inputs, not recorded model activations.

To repeat the E38 optimized comparison, use a new output filename:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 bench_three_stage.py \
  --rows 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
  --weights 100 --rounds 3 --samples 7 --combined-host --baselines \
  --prefetch-kernel ./prefetch_up.py --hidden-pad 4 --reduce-threads 128 --reduce-vec 1 \
  --output /opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/new_optimized_full_checkpoint.jsonl
```

The eager measurements include the ordinary Python/runtime submission path and
batch-boundary synchronization. They are not pure hardware launch-cost measurements
or single-request synchronized latency. Graph timings do not substitute for eager.
The padded candidate is still limited to T<=24; it is not a prefill implementation.

## Historical E17 Results

The accepted kernel passed all 2,400 `(T, checkpoint weight pair)` combinations and
changed-input graph replay checks with the unchanged BF16 tolerances against FP64:
`rtol=1e-2, atol=5e-3`.

In the E17 rotating-weight benchmark, T=1..16 is 1.60-2.15x faster than the tuned
Triton baseline (geometric mean 1.843x) and effectively unchanged from the previous
FlyDSL version. T=17..24 is 1.040x faster than the previous FlyDSL (about 3.86%
lower latency), but its geometric-mean speedup against Torch compile is only
0.969x: Torch still wins 5 of 8 row counts. These are isolated GR read timings,
not model throughput results. The 8-weight screening overstated the benefit;
the conclusions here use all 100 checkpoint weight pairs.

Historical fused-down report:
[17-GR_read_down_SiLU_splitK_MI308X_2026-09-09.md](/opt/qwen3.8-flash-next-doc/17-GR_read_down_SiLU_splitK_MI308X_2026-09-09.md).

Historical V1/E10 report, including the CuTe source locations and the independent
gfx942 design rationale:
[16-GR_read_FlyDSL_MI308X_2026-09-09.md](/opt/qwen3.8-flash-next-doc/16-GR_read_FlyDSL_MI308X_2026-09-09.md).

Raw accepted run:
[e17_full_checkpoint_fused_down.jsonl](/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e17_full_checkpoint_fused_down.jsonl).

Final T=24 IR comparison:
[isa_summary_e17.json](/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/isa_summary_e17.json).
The later 2/4-accumulator probe did not improve T=24 and was not adopted; its
source snapshots, raw measurements and IR remain in the local result directory.
