# Qwen3.8 GR Read on MI308X

Standalone FlyDSL experiment. No SGLang production dispatch is changed.

## Contract

- GPU: MI308X / gfx942. Measured with 80 compute units.
- Input: contiguous BF16 normalized residual `X[T,10240]`, `1 <= T <= 24`.
- Original weights: BF16 `W_down[320,10240]`, `W_up[10240,320]`.
- Output: BF16 `Y[T,2560]`. Zero rows return an empty output; rows above 24 are rejected.
- Formula: `mean_C(sigmoid(silu(X @ W_down.T / 4) @ W_up.T) * X)` for `C=4`.
- RMSNorm and GR write are outside this kernel.

The implementation uses two launches: split-K down partials, then ordered partial
reduction + SiLU + up GEMM + sigmoid/multiply/stream mean. Weight preparation
interleaves the four up-projection streams and uses the BF16 preshuffle layout
from `../test_gemm.py`.

**The default preserves the FP32 SiLU result as BF16 high/low components.** The up
projection accumulates both components using BF16 MFMA. This is necessary to pass
the original FP64 tolerance on every tested checkpoint weight pair, including a
recorded MTP rounding case. It does not bitwise reproduce an intermediate rounded
to one BF16 value. Inputs, weights and final output remain BF16; accumulators are
FP32. The diagnostic `Config(compensate_hidden=False)` reproduces the original
intermediate rounding and does not pass the complete acceptance dataset.

`GRRead(rows, wd, wu)` selects the accepted configuration. Explicit `Config(...)`
arguments are experimental; use `default_config(rows)` or `selected_configs.json`
for the accepted settings.

## Files

| File | Purpose |
| --- | --- |
| `kernel.py` | FlyDSL kernels, preparation, shape/device guards, fixed configuration |
| `test_gr_read.py` | BF16 correctness, graph replay, cache-key isolation, recorded rounding regression |
| `support.py` | FP64 reference, exact local Triton baseline, checkpoint loading and timing |
| `benchmark.py` | All 100 checkpoint pairs, all 24 row counts, randomized benchmark order |
| `selected_configs.json` | Accepted configuration with activation compensation enabled |
| `tune.py` | Explicit, recorded configuration sweeps |
| `quick_bench.py` | Single-weight screening, optional individual-stage timings |
| `dump_ir.py` | FlyDSL MLIR/LLVM/ISA and tuned Triton TTIR/TTGIR/LLVM/ISA |
| `analyze_ir.py` | Static instruction/resource summary |
| `diagnose.py` | Reproduce the seed-101 MTP rounding case |
| `smoke.py` | Small compilation/correctness probe |

## Environment and Use

Validated with FlyDSL 0.3.1, PyTorch 2.12.0+ROCm 7.2.4, Triton 3.7.1, Python 3.10.
The machine already has the required packages. FlyDSL was not upgraded.

Run commands from this directory and select an available physical GPU. GPU 2 was
used for the recorded experiment.

```bash
HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 -m pytest -q test_gr_read.py

HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 smoke.py --rows 24

HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 benchmark.py --configs selected_configs.json \
  --output /opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/new_full_checkpoint.jsonl

HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
  python3 dump_ir.py --rows 1 \
  --output /opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/new_ir_t1
```

Benchmark output files and IR directories must be new. Existing experiment results
are not overwritten. The benchmark checks that the local Triton kernel matches
commit `8cf5501b6913f57a2e7c8dcee52b625fc8ab23c3` before executing it.

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

## Results

The accepted kernel passed all 2,400 `(T, checkpoint weight pair)` combinations and
changed-input graph replay checks with the unchanged BF16 tolerances against FP64:
`rtol=1e-2, atol=5e-3`.

In the recorded rotating-weight benchmark, T=1..16 is 1.60-2.15x faster than the
tuned Triton baseline (geometric mean 1.848x). T=17..24 is mostly slower than the
Torch compile fallback; its geometric-mean speedup is 0.932x. These are isolated
GR read timings, not model throughput results.

Detailed Chinese report:
[16-GR_read_FlyDSL_MI308X_2026-09-09.md](/opt/qwen3.8-flash-next-doc/16-GR_read_FlyDSL_MI308X_2026-09-09.md).

Raw accepted run:
[e10_full_checkpoint_compensated.jsonl](/opt/qwen3.8-flash-next-doc/gr_read_flydsl_results/e10_full_checkpoint_compensated.jsonl).
