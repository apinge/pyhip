# perf on MI308

aiter version: 0.1.10.post4.dev64+g04ac4cf6e
rocm7.1.0
```
python test_gemma_norm.py  -d bf16
[aiter] WARNING: NUMA balancing is enabled, which may cause errors. It is recommended to disable NUMA balancing by running "sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'" for more details: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing
[aiter] import [module_aiter_enum] under /root/workspace/aiter_20260302/aiter/jit/module_aiter_enum.so

start gemma rmsnorm test(no residual) ---
 Kernel gemma_rmsnorm_bf16(['std::bfloat16_t*', 'std::bfloat16_t const*', 'std::bfloat16_t const*', 'unsigned int', 'float'])  /root/workspace/pyhip/archive/gemma_norm/gemma_norm.cpp : /root/workspace/pyhip/archive/gemma_norm/gemma_norm.co : _Z18gemma_rmsnorm_bf16PDF16bPKDF16bS1_jf 

start gemma rmsnorm fuse add test
 Kernel gemma_fused_add_rmsnorm_bf16(['std::bfloat16_t*', 'std::bfloat16_t*', 'std::bfloat16_t const*', 'unsigned int', 'float'])  /root/workspace/pyhip/archive/gemma_norm/gemma_norm.cpp : /root/workspace/pyhip/archive/gemma_norm/gemma_norm.co : _Z28gemma_fused_add_rmsnorm_bf16PDF16bS_PKDF16bjf 
All tests passed.

--- perf (torch / gemma(ours) / aiter) ---
 python test_gemma_norm.py  -d bf16 
[aiter] WARNING: NUMA balancing is enabled, which may cause errors. It is recommended to disable NUMA balancing by running "sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'" for more details: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing
[aiter] import [module_aiter_enum] under /root/workspace/aiter_20260302/aiter/jit/module_aiter_enum.so

start gemma rmsnorm test(no residual) ---
 Kernel gemma_rmsnorm_bf16(['std::bfloat16_t*', 'std::bfloat16_t const*', 'std::bfloat16_t const*', 'unsigned int', 'float'])  /root/workspace/pyhip/archive/gemma_norm/gemma_norm.cpp : /root/workspace/pyhip/archive/gemma_norm/gemma_norm.co : _Z18gemma_rmsnorm_bf16PDF16bPKDF16bS1_jf 

start gemma rmsnorm fuse add test
 Kernel gemma_fused_add_rmsnorm_bf16(['std::bfloat16_t*', 'std::bfloat16_t*', 'std::bfloat16_t const*', 'unsigned int', 'float'])  /root/workspace/pyhip/archive/gemma_norm/gemma_norm.cpp : /root/workspace/pyhip/archive/gemma_norm/gemma_norm.co : _Z28gemma_fused_add_rmsnorm_bf16PDF16bS_PKDF16bjf 
All tests passed.

--- perf (torch / gemma(ours) / aiter) ---
[aiter] import [module_rmsnorm] under /root/workspace/aiter_20260302/aiter/jit/module_rmsnorm.so
[perf rmsnorm] dim=(1, 256) dtype=bf16: torch 35.56 us, gemma(ours) 6.74 us 0.00 TFLOPS (vs torch 5.27x), aiter 9.81 us (vs torch 3.63x), gemma/aiter 1.45x
[perf rmsnorm] dim=(1, 4096) dtype=bf16: torch 38.91 us, gemma(ours) 6.75 us 0.00 TFLOPS (vs torch 5.77x), aiter 11.07 us (vs torch 3.51x), gemma/aiter 1.64x
[perf rmsnorm] dim=(2, 256) dtype=bf16: torch 37.72 us, gemma(ours) 6.72 us 0.00 TFLOPS (vs torch 5.61x), aiter 9.72 us (vs torch 3.88x), gemma/aiter 1.44x
[perf rmsnorm] dim=(2, 4096) dtype=bf16: torch 39.20 us, gemma(ours) 6.72 us 0.00 TFLOPS (vs torch 5.83x), aiter 11.02 us (vs torch 3.56x), gemma/aiter 1.64x
[perf rmsnorm] dim=(128, 256) dtype=bf16: torch 37.15 us, gemma(ours) 6.85 us 0.02 TFLOPS (vs torch 5.43x), aiter 10.05 us (vs torch 3.70x), gemma/aiter 1.47x
[perf rmsnorm] dim=(128, 4096) dtype=bf16: torch 65.68 us, gemma(ours) 7.14 us 0.29 TFLOPS (vs torch 9.20x), aiter 12.13 us (vs torch 5.41x), gemma/aiter 1.70x
[perf rmsnorm] dim=(256, 256) dtype=bf16: torch 38.79 us, gemma(ours) 6.78 us 0.04 TFLOPS (vs torch 5.72x), aiter 10.31 us (vs torch 3.76x), gemma/aiter 1.52x
[perf rmsnorm] dim=(256, 4096) dtype=bf16: torch 81.08 us, gemma(ours) 9.12 us 0.46 TFLOPS (vs torch 8.89x), aiter 13.54 us (vs torch 5.99x), gemma/aiter 1.49x
[perf rmsnorm] dim=(8000, 256) dtype=bf16: torch 95.02 us, gemma(ours) 19.05 us 0.43 TFLOPS (vs torch 4.99x), aiter 15.96 us (vs torch 5.95x), gemma/aiter 0.84x
[perf rmsnorm] dim=(8000, 4096) dtype=bf16: torch 529.87 us, gemma(ours) 119.71 us 1.09 TFLOPS (vs torch 4.43x), aiter 95.51 us (vs torch 5.55x), gemma/aiter 0.80x
[perf rmsnorm] dim=(8294, 256) dtype=bf16: torch 83.04 us, gemma(ours) 19.45 us 0.44 TFLOPS (vs torch 4.27x), aiter 15.76 us (vs torch 5.27x), gemma/aiter 0.81x
[perf rmsnorm] dim=(8294, 4096) dtype=bf16: torch 542.70 us, gemma(ours) 124.15 us 1.09 TFLOPS (vs torch 4.37x), aiter 98.36 us (vs torch 5.52x), gemma/aiter 0.79x
[perf rmsnorm] dim=(33176, 256) dtype=bf16: torch 195.23 us, gemma(ours) 58.25 us 0.58 TFLOPS (vs torch 3.35x), aiter 33.20 us (vs torch 5.88x), gemma/aiter 0.57x
[perf rmsnorm] dim=(33176, 4096) dtype=bf16: torch 2266.56 us, gemma(ours) 480.49 us 1.13 TFLOPS (vs torch 4.72x), aiter 345.65 us (vs torch 6.56x), gemma/aiter 0.72x
[aiter] import [module_rmsnorm_quant] under /root/workspace/aiter_20260302/aiter/jit/module_rmsnorm_quant.so
[perf fused_add_rmsnorm] dim=(1, 256) dtype=bf16: torch 44.33 us, gemma(ours) 13.66 us 0.00 TFLOPS (vs torch 3.25x), aiter 16.03 us (vs torch 2.76x), gemma/aiter 1.17x
[perf fused_add_rmsnorm] dim=(1, 4096) dtype=bf16: torch 46.70 us, gemma(ours) 15.46 us 0.00 TFLOPS (vs torch 3.02x), aiter 15.99 us (vs torch 2.92x), gemma/aiter 1.03x
[perf fused_add_rmsnorm] dim=(2, 256) dtype=bf16: torch 45.07 us, gemma(ours) 14.12 us 0.00 TFLOPS (vs torch 3.19x), aiter 15.92 us (vs torch 2.83x), gemma/aiter 1.13x
[perf fused_add_rmsnorm] dim=(2, 4096) dtype=bf16: torch 48.58 us, gemma(ours) 15.38 us 0.00 TFLOPS (vs torch 3.16x), aiter 15.91 us (vs torch 3.05x), gemma/aiter 1.03x
[perf fused_add_rmsnorm] dim=(128, 256) dtype=bf16: torch 45.95 us, gemma(ours) 14.27 us 0.01 TFLOPS (vs torch 3.22x), aiter 16.03 us (vs torch 2.87x), gemma/aiter 1.12x
[perf fused_add_rmsnorm] dim=(128, 4096) dtype=bf16: torch 77.56 us, gemma(ours) 21.34 us 0.12 TFLOPS (vs torch 3.63x), aiter 19.10 us (vs torch 4.06x), gemma/aiter 0.90x
[perf fused_add_rmsnorm] dim=(256, 256) dtype=bf16: torch 45.95 us, gemma(ours) 14.47 us 0.02 TFLOPS (vs torch 3.17x), aiter 16.12 us (vs torch 2.85x), gemma/aiter 1.11x
[perf fused_add_rmsnorm] dim=(256, 4096) dtype=bf16: torch 95.73 us, gemma(ours) 29.95 us 0.18 TFLOPS (vs torch 3.20x), aiter 250.80 us (vs torch 0.38x), gemma/aiter 8.37x
[perf fused_add_rmsnorm] dim=(8000, 256) dtype=bf16: torch 109.16 us, gemma(ours) 36.80 us 0.28 TFLOPS (vs torch 2.97x), aiter 26.76 us (vs torch 4.08x), gemma/aiter 0.73x
[perf fused_add_rmsnorm] dim=(8000, 4096) dtype=bf16: torch 708.49 us, gemma(ours) 356.56 us 0.46 TFLOPS (vs torch 1.99x), aiter 216.56 us (vs torch 3.27x), gemma/aiter 0.61x
[perf fused_add_rmsnorm] dim=(8294, 256) dtype=bf16: torch 107.42 us, gemma(ours) 40.12 us 0.26 TFLOPS (vs torch 2.68x), aiter 27.21 us (vs torch 3.95x), gemma/aiter 0.68x
[perf fused_add_rmsnorm] dim=(8294, 4096) dtype=bf16: torch 731.55 us, gemma(ours) 371.19 us 0.46 TFLOPS (vs torch 1.97x), aiter 226.10 us (vs torch 3.24x), gemma/aiter 0.61x
[perf fused_add_rmsnorm] dim=(33176, 256) dtype=bf16: torch 249.38 us, gemma(ours) 112.22 us 0.38 TFLOPS (vs torch 2.22x), aiter 71.37 us (vs torch 3.49x), gemma/aiter 0.64x
[perf fused_add_rmsnorm] dim=(33176, 4096) dtype=bf16: torch 3079.18 us, gemma(ours) 1490.89 us 0.46 TFLOPS (vs torch 2.07x), aiter 956.59 us (vs torch 3.22x), gemma/aiter 0.64x
```