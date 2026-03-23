## How to get asm and ir from flydsl

```bash
export PYTHONPATH=/root/workspace/FlyDSL
export FLYDSL_DUMP_IR=1
export FLYDSL_DUMP_DIR=/root/workspace/pyhip/archive/flydsl/asm_cache
export FLYDSL_RUNTIME_ENABLE_CACHE=0 # not hit pkl cache
python3 test_rmsnorm.py
```

## IR ref

https://mlir.llvm.org/docs/Dialects/ROCDLDialect/


## perf test on MI308

```
========================================================================================================================
Perf Compare (gpu us): FlyDSL vs AIter Triton vs aiter.rms_norm (CK)
========================================================================================================================
op         shape              dtype  FlyDSL(gpu us) AIter triton(us) aiter.rms_norm(us)
rmsnorm    32768x8192         bf16            427.8          1,099.9              847.9
rmsnorm    32768x4096         bf16            234.4            707.2              334.0
rmsnorm    8000x4096          bf16             49.3            177.2               83.0
rmsnorm    8x4096             bf16             18.2             44.6                9.7
rmsnorm    4x4096             bf16             18.5             43.8                9.9
rmsnorm    2x4096             bf16             18.2             44.1                9.8
rmsnorm    1x4096             bf16             18.2             43.8                9.6
rmsnorm    32768x256          bf16             92.2             40.7               24.2
rmsnorm    8000x256           bf16             25.1             94.8                9.7
rmsnorm    8x256              bf16             18.0             42.7                9.7
rmsnorm    4x256              bf16             18.4             43.2               10.0
rmsnorm    2x256              bf16             18.1             42.4                9.9
rmsnorm    1x256              bf16             18.0             44.0                9.7
========================================================================================================================

```