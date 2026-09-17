# Qwen3.8 GR Read on MI308X

Standalone FlyDSL experiment. No SGLang production dispatch is changed.

## Review and Debug Start Here

The original/default path is `CombinedPaddedGRRead` -> `_padded_pair_launcher` ->
`prefetch_up._launchers` -> one selected down kernel and `up_gate`.
Construction chooses the T bucket; a prepared reader does not change T at call time.
The newer opt-in T=17..32 path is `LargeDownGRRead` in `large_down.py`;
it does not replace this default. See [Accuracy Validation](#accuracy-validation)
for separate commands testing both paths.

For a final-entry-only example with explicit inputs, FP64 reference, intermediate
checks and an optional breakpoint:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python3 test_final_entry.py --rows 1 17 --graph
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python3 test_final_entry.py --rows 17 --debug
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python3 test_final_entry.py --graph --check-contract
```

Unlike the older pytest-only files, `test_final_entry.py` is a plain Python script
using `assert torch.allclose(...)`, with no pytest dependency. By default it checks
T=1/16/17/24/25/32; `--graph` adds replay checks for the selected rows and
`--check-contract` adds empty/invalid input checks. This is not the full checkpoint
acceptance suite, and its synchronized stage checks are not performance measurements.
See the [call-chain and manual-test guide](/opt/qwen3.8-flash-next-doc/22-GR_read_最终版调用链与手写单测_2026-09-14.md).

## Frozen Padded Entry

`CombinedPaddedGRRead` in `combined_host.py` is the validated E38 standalone
entry, retained as the small-T path and large-T control. It keeps two GPU launches, uses a BF16 hidden LDS row stride of
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

The historical E38 run's 2,400 checkpoint/row combinations and changed-input graph replay checks
passed the original FP64 tolerance. E38 full-call graph medians are 15.270 us at
T=1 and 35.677 us at T=24, versus 18.697 / 44.386 us for the same-run E17 reference.
Ordinary eager timings are separate: at T=1 the padded combined entry is
20.329 us; the unpadded `CombinedHostGRRead` is 19.981 us. This small eager
difference is not evidence that the unpadded GPU kernels are faster.

See the current [optimization and trace report](/opt/qwen3.8-flash-next-doc/21-GR_read_ROCm_LDS_padding与HostLauncher优化_2026-09-10.md)
and [raw E38 results](/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e38_optimized_full_checkpoint.jsonl).

## Contract

- Validated GPU: MI308X / gfx942, measured with 80 compute units. Other ROCm
  architectures (including gfx950) emit `RuntimeWarning` and continue for
  review/testing; this is not a claim of validated cross-architecture correctness.
- Input: contiguous BF16 normalized residual `X[T,10240]`, `1 <= T <= 32`.
- Original weights: BF16 `W_down[320,10240]`, `W_up[10240,320]`.
- Output: BF16 `Y[T,2560]`. Zero rows return an empty output; rows above 32 are rejected.
- Formula: `mean_C(sigmoid(silu(X @ W_down.T / 4) @ W_up.T) * X)` for `C=4`.
- RMSNorm and GR write are outside this kernel.

The frozen padded entry uses two launches, with a measured row-count split:

- T=1..16: down writes 16 global FP32 K partials; up performs ordered reduction,
  SiLU, up GEMM and sigmoid/multiply/stream mean (the previous accepted path).
- T=17..32: down uses 256 threads / 4 waves to split K **within one CTA**, reduces
  the wave partials in LDS and applies `/4 + SiLU`. Up consumes the completed
  FP32 activation and fuses up GEMM with sigmoid/multiply/stream mean.

The fused-down path follows the wave split-K approach in `../test_gemm.py`.
There is no cross-CTA counter or spin barrier. Each wave processes a K512 tile;
four waves cover K2048 per iteration, with five iterations for K=10240. Weight
preparation interleaves the four up-projection streams and uses the colleague's
BF16 preshuffle layout. The T=17..32 intermediate shrinks from 640 KiB to 40 KiB.
The fusion is available explicitly for every T=1..32, but screening did not
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

## Accuracy Validation

Run the following commands in **Bash**, from this directory. `{1..32}` expands
to all 32 row counts; `{17..32}` expands to all 16 large-T row counts. Select an
idle ROCm GPU; the recorded results below are from gfx942. Synthetic checks
do not need SGLang or a model checkpoint.

```bash
cd /opt/pyhip/tests/flydsl/gr_read
export HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2
```

### Reference and Pass Criteria

`support.reference(x, w_down, w_up)` evaluates the full formula in FP64 using
the **original, unpacked** weights. Do not pass `reader.w_down` / `reader.w_up`
to the reference: those tensors have already been reordered/preshuffled.
For BF16 output the acceptance check is:

```python
assert torch.allclose(actual.double(), expected, rtol=1e-2, atol=5e-3)
```

Equivalently, every element must have
`abs(actual-expected)/(5e-3 + 1e-2*abs(expected)) <= 1`.
This is a tolerance check, not bitwise equality. Keep FP32 accumulation and
high/low compensation enabled; a BF16-intermediate Torch result is not the
FP64 reference. Intermediate checks in `test_final_entry.py` use their own,
tighter tolerances.

### Plain Python Checks

These are ordinary Python scripts with assertions, not pytest tests. An
`AssertionError`, traceback, or nonzero exit is a failure. The default row lists
are only subsets; use the ranges below for full coverage.

```bash
# Original/default entry, including the frozen T<=16 path and the old large-T control.
python3 test_final_entry.py --rows {1..32} --graph --check-contract

# New opt-in large-T entry: U2 for T17..25, U1 for T26..32.
python3 test_large_down.py --selected --rows {17..32}
```

The second command checks both seeds 101/202, FP64 output, separate down/up
calls, graph capture and changed-input replay, zero inputs, NaN-poisoned
workspace overwrite, zero padding, shared weight pointers, and independent
output/workspace allocations. It also checks the original T=1/8/16 entry and
that the new entry rejects small T. `test_final_entry.py` alone does **not**
test `LargeDownGRRead`.

For one case that is easy to step through:

```bash
python3 test_final_entry.py --rows 16 --graph --debug
python3 -m pdb test_large_down.py --selected --rows 17
```

A minimal manual output check for the currently selected path is:

```python
import torch
from combined_host import CombinedPaddedGRRead
from large_down import LargeDownGRRead
from support import reference, synthetic

x, w_down, w_up = synthetic(17, seed=101)
control = CombinedPaddedGRRead(x.shape[0], w_down, w_up)
reader = control if x.shape[0] <= 16 else LargeDownGRRead(control)
expected = reference(x, w_down, w_up)
actual = reader(x)
assert torch.allclose(actual.double(), expected, rtol=1e-2, atol=5e-3)
```

### Existing Pytest Regression

Explicitly list the test files. The repository's `pytest.ini` sets
`python_files=*.py`; pointing pytest at the whole `gr_read/` directory also
collects command-line benchmark scripts and can fail their imports before
any numerical test executes. Do not use that collection failure as evidence
that a GPU numerical check failed, or silently call it a passing suite.

```bash
python3 -m pytest -q \
  test_gr_read.py test_three_stage.py test_register_gate.py \
  test_combined_host.py test_prefetch_up.py
```

On this machine the explicit suite passed **366 tests, zero skips**, on
2026-09-17. The plain Python scripts above are not executed by this pytest
command and must be run separately. On another machine, an absent recorded
MTP `.pt` fixture can cause a skip; that is not a passed MTP regression.

### Full Real-Weight Acceptance

Synthetic tests are not full checkpoint acceptance. For the selected T1..32
combination, run both commands below using all 100 HC weight pairs. Input
activations are generated; weights come from the checkpoint. These commands
also measure performance, but the FP64/replay assertions remain mandatory.

```bash
# Frozen small-T entry; no baseline accuracy exemptions.
python3 bench_bandwidth.py --rows {1..16} --weights 100 \
  --model-path /models/Qwen3.8-Flash-Next-PTPC-FP8 \
  --output /tmp/gr_small_accuracy_new.jsonl

# Selected large-T entry plus the frozen control; no Torch baseline requested.
python3 bench_large_down.py --selected --rows {17..32} --weights 100 \
  --model-path /models/Qwen3.8-Flash-Next-PTPC-FP8 \
  --output /tmp/gr_large_accuracy_new.jsonl
```

Use new output filenames, or omit `--output` to print only. Do not use
`--synthetic` for checkpoint acceptance. Both commands check every weight pair
before/after changed-input graph replay; FlyDSL mismatches abort. Expect
1,600 initial + 1,600 changed-input checks per range, totaling 3,200 of each
for the selected T1..32 paths. `bench_bandwidth.py` records `initial_fp64` and
`changed_input_fp64`; `bench_large_down.py` records `checks.initial` and
`checks.changed_replay` for each backend.

The latest complete real-weight rerun passed **3,200/3,200 initial and
3,200/3,200 changed-input checks** for selected FlyDSL. The full results and
commands used are in the [T1..32 report](/opt/qwen3.8-flash-next-doc/29-GR_read_T1到32完整Benchmark复跑_2026-09-16.md).
This does not validate captured production activations or model-level throughput.

### Baselines and CPU Checks

The optional extended Triton script tests synthetic data at ROWS=32; it is
not the supported SGLang T<=16 wrapper and is not selected FlyDSL:

```bash
python3 test_extended_triton.py --rows {17..32}
python3 test_benchmark_paths.py
python3 test_bandwidth.py
python3 test_architecture_warnings.py
```

The last three checks are CPU-only: bundled source hash / CLI paths, byte-count
units, and architecture-warning contracts using mocked device properties.
They do not prove GPU numerical correctness or actual gfx950 compilation.

When benchmark flags `--baselines` or `--torch-baseline` are enabled, baseline
accuracy failures are recorded without aborting; an exit code of zero does
**not** mean every baseline passed FP64. In the latest real-weight rerun,
Torch passed 3,179/3,200 initial and 3,167/3,200 changed-input cases; supported
tuned Triton passed 1,599/1,600 initial and 1,600/1,600 changed-input cases.
Do not relax the FlyDSL tolerance to hide these failures.

### Pre-commit Verification: 2026-09-17

| Check | Executed coverage | Result |
| --- | --- | --- |
| Explicit pytest suite | The five files listed above, including recorded MTP cases | 366 passed, zero skips |
| `test_final_entry.py` | T1..32, `--graph --check-contract` | Passed |
| `test_large_down.py` | T17..32, `--selected`; includes frozen T1/8/16 checks | Passed |
| `test_extended_triton.py` | T17..32 synthetic FP64, replay, padding and counters | Passed |
| `test_benchmark_paths.py` | Bundled source and standalone CLI paths | Passed |
| `test_bandwidth.py` | T1..32 byte counts and units | Passed |

The successful pytest [JUnit record](/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e59_precommit_explicit_regression_20260917.xml)
is retained locally. The initial whole-directory collection attempt failed on
benchmark-script imports; the explicit command above is the verified invocation.
The full real-weight figures above are from E55/E56 on 2026-09-16, not another
checkpoint benchmark rerun during this pre-commit check.

## Files

| File | Purpose |
| --- | --- |
| `kernel.py` | FlyDSL kernels, preparation, shape/device guards, fixed configuration |
| `test_gr_read.py` | BF16 correctness, graph replay, cache-key isolation, recorded rounding regression |
| `test_final_entry.py` | Plain Python final-entry checks with `assert torch.allclose`, graph replay and `--debug` |
| `large_down.py` | Opt-in T17..32 global/wave split-K down pipeline with the original weight format |
| `test_large_down.py` | Plain Python selected large-T FP64, graph, padding, workspace and frozen-boundary checks |
| `bench_large_down.py` | Matched large-T control/selected comparison and all-pair real-weight accuracy checks |
| `support.py` | FP64 reference, bundled Triton baseline loading, checkpoint loading and timing |
| `baselines/hc_mix_triton.py` | Unmodified `8cf5501b` Triton baseline, bundled with upstream Apache-2.0 license |
| `test_benchmark_paths.py` | Plain Python CPU checks for bundled baseline, loader cache and configurable input paths |
| `bench_bandwidth.py` | Final `combined_padded` T=1..32 graph timing and logical effective bandwidth; no model required by default |
| `test_bandwidth.py` | CPU-only byte-count and cudaPerf-compatible GB/s conversion checks |
| `test_architecture_warnings.py` | CPU-only mock checks: gfx950 warnings, native gfx942 silence, unchanged hard input/resource guards |
| `benchmark.py` | Checkpoint correctness and matched timings; supports rows through 32 and optional previous FlyDSL baseline |
| `experimental_triton.py` | Opt-in ROWS=32 probe of the original tuned Triton kernel; not its production wrapper |
| `test_extended_triton.py` | Plain Python FP64, changed-input graph and barrier-counter checks for the Triton extension |
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

### Reviewing on gfx950

GR read's `kernel.GRRead` and `prefetch_up.GRRead` now warn rather than reject
other ROCm architectures. This also covers `CombinedPaddedGRRead` and
`LargeDownGRRead`. No weight reorder, preshuffle, tile selection, accumulation,
or high/low compensation changed. Shape, BF16, ROCm-device, and the existing
64 KiB LDS configuration checks remain hard errors.

Compile the source on the target GPU; do not force `ARCH=gfx942` or assume a
previously generated gfx942 HSACO can be loaded on gfx950. First run the plain
Python FP64/replay checks above. The architecture warning is advisory; actual
compiler/runtime instruction errors and numerical mismatches remain failures.

The opt-in `experimental_triton.py` also warns on a non-80-CU/gfx942 device;
it keeps its fixed 80-CTA launch and original tuning. This is for review, not a
retuned gfx950 baseline. The bundled `baselines/hc_mix_triton.py` source/hash
is unchanged and retains its original architecture-dependent dispatch.

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

The checkpoint tools default to `/models/Qwen3.8-Flash-Next-FP8`.
Both `bench_three_stage.py` and `benchmark.py` accept `--model-path` for another
location. The tuned Triton source is bundled under `baselines/`, so no SGLang
checkout or installation is needed. `--triton-kernel /path/to/hc_mix_triton.py`
can select an explicit source file; the same `8cf5501b` SHA256 check still applies.
See `baselines/README.md` for provenance and licensing. The linked full report,
benchmark outputs, IR dumps and checkpoint-derived `.pt` samples are local
experiment artifacts and are not included in this repository.

The MTP regression test skips when its checkpoint-derived sample is absent.
On a machine with the model at the default path, recreate the expected sample
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

For a standalone benchmark without model files or SGLang, use `--synthetic`.
This exercises the same candidates, including the final padded entry, and
prints results directly:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 bench_three_stage.py --synthetic \
  --rows 6 --weights 100 --rounds 3 --samples 7 --combined-host \
  --prefetch-kernel ./prefetch_up.py --hidden-pad 4 --reduce-threads 128 --reduce-vec 1
```

Here `--weights 100` generates and rotates 100 random BF16 down/up weight pairs
at the model's fixed shapes. `--seed` controls generation; the synthetic weight
scale is 0.02. `--model-path` is ignored in this explicit mode. The table and
optional JSONL record `source=synthetic` / `weight_source`; these are screening
results, not real-checkpoint acceptance. There is no automatic fallback from
missing model files to random weights. Add `--baselines` for the bundled
Triton/Torch comparison, also without SGLang.

For real-checkpoint benchmarking, omit `--synthetic` and provide the model
directory with `--model-path` if it is not at the default location. No output
file, previous result directory or V1 snapshot is required:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 bench_three_stage.py \
  --rows 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
  --weights 100 --rounds 3 --samples 7 --combined-host --baselines \
  --prefetch-kernel ./prefetch_up.py --hidden-pad 4 --reduce-threads 128 --reduce-vec 1
```

Results print as a table with Graph and Eager wall medians in us per complete
GR read. Add `--output /path/to/new_results.jsonl` only when JSONL samples,
metadata and source snapshots are needed; existing files are never overwritten.

For a checkpoint stored elsewhere, append `--model-path /path/to/Qwen3.8-Flash-Next-FP8`.
The default Triton baseline follows the repository location, not `/opt/sglang`.
Sync the `baselines/` directory along with the scripts when moving to another
machine. A `--baselines` run with only T=17..32 uses Torch compile and does not
load Triton. Run `python3 test_benchmark_paths.py` for CPU-only path checks.

SGLang is optional for all these standalone kernel checks. Without
`--baselines`, `bench_three_stage.py` does not load the Triton baseline or
prepare the Torch compile comparison. Real-weight benchmarking still needs the
checkpoint; standalone `--synthetic` benchmarking does not. For a final-entry
correctness/debug check with random weights, use
`python3 test_final_entry.py --rows 6 --graph`. For synthetic E17/configuration
screening without model files or baseline loading, use:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 quick_bench.py --rows 1 6 --skip-baselines \
  --split-k 16 --down-n 64 --waves 4 --compensate-hidden
```

This quick benchmark targets `kernel.GRRead`, not the final padded entry, and
uses one synthetic weight pair per T; do not compare its cache-hot timings to
the rotating-checkpoint acceptance results.

The optional `--previous-kernel /path/to/v1_kernel.py` adds an earlier `GRRead`
implementation as the `v1` backend. It loads Python source, reruns that kernel in
the same benchmark, and records a source snapshot when `--output` is supplied;
it never reads old timing data.
To reproduce the original E38 backend set, explicitly pass the preserved
`e10_full_checkpoint_compensated.kernel.py` snapshot with this option.

The eager measurements include the ordinary Python/runtime submission path and
batch-boundary synchronization. They are not pure hardware launch-cost measurements
or single-request synchronized latency. Graph timings do not substitute for eager.
The padded candidate is limited to T<=32; it is not a general prefill implementation.

## Rows 25 Through 32

The 17..24 algorithm is reused without changing GPU kernel bodies or tuning
parameters: T=17..32 all use `block_m=16`, `padded_rows=32`, down grid `(2,20,1)`,
up grid `(2,80,1)`, and a 40 KiB FP32 activation buffer. The reader and CLI limits
now permit 32 rows. Default two-kernel selection is unchanged for T<=24.

The added 800 real-weight/row combinations and changed-input replay checks
passed the unchanged FP64 tolerance; the expanded regression suite passed 366
tests. The final two-kernel graph path is about 30 us at T=25..32 in the same-run
comparison, but Torch compile is faster at 6 of the 8 row counts and has some
FP64 tolerance failures. See the [extension report](/opt/qwen3.8-flash-next-doc/24-GR_read_T25到32复用与基线比较_2026-09-16.md)
and [raw results](/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/rows32_full_checkpoint_20260916.jsonl).

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 test_final_entry.py --rows 24 25 26 27 28 29 30 31 32 --graph --check-contract

HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 bench_three_stage.py --synthetic --rows 24 25 26 27 28 29 30 31 32 \
  --weights 100 --rounds 3 --samples 7 --combined-host --baselines --extended-triton \
  --prefetch-kernel ./prefetch_up.py --hidden-pad 4 --reduce-threads 128 --reduce-vec 1
```

Omit `--synthetic` to use real checkpoint weights. `--extended-triton` is an
explicit experiment: it launches the unchanged baseline kernel at `ROWS=32`
with the existing gfx942 tuning parameters and labels it `triton_extended32`.
It does not alter the bundled source, the supported `tuned_triton` T<=16
baseline, or SGLang dispatch. Its buffers are prepared before timing; its eager
host path therefore differs from the original allocating Triton wrapper.
Without this flag, T>16 continues to compare only against Torch compile.
An explicitly selected old V1 source may still reject rows above 24.

## Effective Bandwidth

`bench_bandwidth.py` measures only the final `CombinedPaddedGRRead` entry for
T=1..32. It defaults to 100 synthetic weight pairs and console output, so no
model, SGLang installation, baseline source or output file is required:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 python3 bench_bandwidth.py
```

For the real-weight measurement matching the PTPC launch script:

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 bench_bandwidth.py --model-path /models/Qwen3.8-Flash-Next-PTPC-FP8 \
  --output /tmp/gr_read_bandwidth_new.jsonl
```

The GB/s conversion matches `pyhip.cudaPerf`: `bytes / latency_us / 1000`.
The byte model counts the full GR operator's external BF16 tensors once:
`Read = X + W_down + W_up = 13107200 + 20480*T` bytes and
`I/O = Read + Y = 13107200 + 25600*T` bytes. Like the CK GEMM example, the
Read column excludes output stores; I/O adds only the final Y stores.
The denominator uses the existing full-call graph event timer, with 100 weight
pairs rotated twice per graph, 3 replays/sample and 21 samples by default.

These are logical effective bandwidths, not measured HBM traffic: workspace
transactions, duplicated CTA/wave loads, LDS accesses and cache behavior are
not counted. `Scratch KiB` is buffer capacity, not traffic added to the numerator.
Weight preparation and correctness checks are outside timing. Use
`python3 test_bandwidth.py` to check byte counts and units without a GPU.

The full PTPC checkpoint run passed 3,200 initial and 3,200 changed-input FP64
checks. Logical Read bandwidth was 1032.5 GB/s at T=1, 664.3 GB/s at T=16,
and 448-458 GB/s at T=17..32. See the [bandwidth report](/opt/qwen3.8-flash-next-doc/26-GR_read_最优FlyDSL_T1到32有效带宽_2026-09-16.md)
for all rows, the byte model and raw samples; these are not HBM counters.

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

## Large-T Down Experiments (2026-09-16)

`large_down.py` is an opt-in candidate for T=17..32, not a change to the default
entry or the T<=16 path. Both weights keep the original preshuffle format;
W_up still uses the logical `c*H+h -> h*C+c` reorder. The candidate shares the
control's weight pointers, with separate P/Y allocations:

```python
from combined_host import CombinedPaddedGRRead
from large_down import LargeDownGRRead

control = CombinedPaddedGRRead(rows, w_down, w_up)
candidate = LargeDownGRRead(control)
y = candidate(x)
# Debug the two stages separately:
candidate.run_down(x)
partials = candidate.partial.view(4, 32, 320)  # linear sums, before /4 and SiLU
candidate.run_up(x)
```

For `global_split=1`, down still fuses SiLU and P is FP32 `[32,320]`.
For `global_split=2/4`, down writes linear partials; the existing up partial
path reduces all splits, then applies `/4 + SiLU`, high/low conversion, the
up GEMM and gated mean. Both cases launch exactly two GPU kernels. The new
GPU kernel is `down_wave_splitk_pipeline`; up still uses `up_gate`.
The opt-in default uses BM32/BK128/4 waves/global split=4. T=17..25 uses
two-beat prefetch; T=26..32 uses one-beat prefetch. Explicit `DownConfig`
overrides remain available for experiments.

Plain Python tests and a synthetic-weight benchmark (no checkpoint needed):

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 test_large_down.py --selected
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 bench_large_down.py --selected --rows 17 24 32 --eager --torch-baseline
```

Add `--model-path /models/Qwen3.8-Flash-Next-PTPC-FP8` for real HC weights.
`--output /tmp/new_run.jsonl` is optional and refuses to overwrite an existing
file. Default timing uses 100 weights, 3 rounds, 7 samples, 200 full calls per
graph and 3 replays per sample; candidate/stage order is randomized per round.
`Full graph` is the complete GR call. `Down only`/`Up only` are isolated graph
repeats and must not be added to reconstruct full latency. `Eager wall` is
full-call host wall time, not an isolated device-kernel time.

All FlyDSL candidates must pass the original FP64 tolerance. The optional
Torch baseline records failures without aborting, matching the historical
benchmark convention. Source snapshots and raw samples accompany JSONL output;
the benchmark never reads historical result files. Findings and negative
experiments are recorded in the [large-down report](/opt/qwen3.8-flash-next-doc/28-GR_read_大T_Down预取与GlobalSplitK实验_2026-09-16.md).
