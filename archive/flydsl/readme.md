## How to get asm and ir from flydsl

```bash
export PYTHONPATH=/root/workspace/FlyDSL
export FLYDSL_DUMP_IR=1
export FLYDSL_DUMP_DIR=/root/workspace/pyhip/archive/flydsl/asm_cache
export FLYDSL_RUNTIME_ENABLE_CACHE=0 # not hit pkl cache
python3 test_rmsnorm.py
```