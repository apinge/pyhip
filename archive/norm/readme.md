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

# perf on MI350

aiter Version: 0.1.10.post4.dev0+g6a0e7b26c.d20260301

rocm 7.2.0
```
python3 test_gemma_norm.py  -d bf16
[aiter] import [module_aiter_enum] under /sgl-workspace/aiter/aiter/jit/module_aiter_enum.so

start gemma rmsnorm test(no residual) ---
 Kernel gemma_rmsnorm_bf16(['std::bfloat16_t*', 'std::bfloat16_t const*', 'std::bfloat16_t const*', 'unsigned int', 'float'])  /root/workspace/pyhip/archive/norm/gemma_norm.cpp : /root/workspace/pyhip/archive/norm/gemma_norm.co : _Z18gemma_rmsnorm_bf16PDF16bPKDF16bS1_jf 

start gemma rmsnorm fuse add test
 Kernel gemma_fused_add_rmsnorm_bf16(['std::bfloat16_t*', 'std::bfloat16_t*', 'std::bfloat16_t const*', 'unsigned int', 'float'])  /root/workspace/pyhip/archive/norm/gemma_norm.cpp : /root/workspace/pyhip/archive/norm/gemma_norm.co : _Z28gemma_fused_add_rmsnorm_bf16PDF16bS_PKDF16bjf 
All tests passed.

--- perf (torch / gemma(ours) / aiter) ---
[aiter] import [module_rmsnorm] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm.so
[aiter] type hints mismatch, override to --> rmsnorm2d_fwd(input: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex, use_model_sensitive_rmsnorm: int | typing.SupportsIndex = 0) -> torch.Tensor
[perf rmsnorm] dim=(1, 256) dtype=bf16: torch 26.28 us, gemma(ours) 6.40 us 0.00 TFLOPS (vs torch 4.10x), aiter 8.74 us (vs torch 3.01x), gemma/aiter 1.36x
[perf rmsnorm] dim=(1, 4096) dtype=bf16: torch 28.11 us, gemma(ours) 6.26 us 0.00 TFLOPS (vs torch 4.49x), aiter 9.39 us (vs torch 2.99x), gemma/aiter 1.50x
[perf rmsnorm] dim=(2, 256) dtype=bf16: torch 25.77 us, gemma(ours) 6.21 us 0.00 TFLOPS (vs torch 4.15x), aiter 8.45 us (vs torch 3.05x), gemma/aiter 1.36x
[perf rmsnorm] dim=(2, 4096) dtype=bf16: torch 30.83 us, gemma(ours) 6.28 us 0.01 TFLOPS (vs torch 4.91x), aiter 9.32 us (vs torch 3.31x), gemma/aiter 1.48x
[perf rmsnorm] dim=(128, 256) dtype=bf16: torch 28.62 us, gemma(ours) 6.32 us 0.02 TFLOPS (vs torch 4.53x), aiter 9.18 us (vs torch 3.12x), gemma/aiter 1.45x
[perf rmsnorm] dim=(128, 4096) dtype=bf16: torch 38.04 us, gemma(ours) 6.34 us 0.33 TFLOPS (vs torch 6.00x), aiter 9.88 us (vs torch 3.85x), gemma/aiter 1.56x
[perf rmsnorm] dim=(256, 256) dtype=bf16: torch 29.84 us, gemma(ours) 6.70 us 0.04 TFLOPS (vs torch 4.46x), aiter 9.36 us (vs torch 3.19x), gemma/aiter 1.40x
[perf rmsnorm] dim=(256, 4096) dtype=bf16: torch 43.94 us, gemma(ours) 6.50 us 0.65 TFLOPS (vs torch 6.76x), aiter 10.13 us (vs torch 4.34x), gemma/aiter 1.56x
[perf rmsnorm] dim=(8000, 256) dtype=bf16: torch 51.50 us, gemma(ours) 8.21 us 1.00 TFLOPS (vs torch 6.27x), aiter 10.55 us (vs torch 4.88x), gemma/aiter 1.28x
[perf rmsnorm] dim=(8000, 4096) dtype=bf16: torch 276.77 us, gemma(ours) 34.75 us 3.77 TFLOPS (vs torch 7.96x), aiter 40.42 us (vs torch 6.85x), gemma/aiter 1.16x
[perf rmsnorm] dim=(8294, 256) dtype=bf16: torch 53.95 us, gemma(ours) 8.15 us 1.04 TFLOPS (vs torch 6.62x), aiter 11.11 us (vs torch 4.86x), gemma/aiter 1.36x
[perf rmsnorm] dim=(8294, 4096) dtype=bf16: torch 295.51 us, gemma(ours) 36.02 us 3.77 TFLOPS (vs torch 8.20x), aiter 43.01 us (vs torch 6.87x), gemma/aiter 1.19x
[perf rmsnorm] dim=(33176, 256) dtype=bf16: torch 91.53 us, gemma(ours) 18.33 us 1.85 TFLOPS (vs torch 4.99x), aiter 16.90 us (vs torch 5.41x), gemma/aiter 0.92x
[perf rmsnorm] dim=(33176, 4096) dtype=bf16: torch 1285.84 us, gemma(ours) 119.02 us 4.57 TFLOPS (vs torch 10.80x), aiter 125.99 us (vs torch 10.21x), gemma/aiter 1.06x
[aiter] import [module_rmsnorm_quant] under /sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
[aiter] type hints mismatch, override to --> add_rmsnorm(out: torch.Tensor, input: torch.Tensor, residual_in: torch.Tensor, residual_out: torch.Tensor, weight: torch.Tensor, epsilon: float | typing.SupportsIndex) -> None
[perf fused_add_rmsnorm] dim=(1, 256) dtype=bf16: torch 30.85 us, gemma(ours) 10.87 us 0.00 TFLOPS (vs torch 2.84x), aiter 12.18 us (vs torch 2.53x), gemma/aiter 1.12x
[perf fused_add_rmsnorm] dim=(1, 4096) dtype=bf16: torch 33.16 us, gemma(ours) 12.07 us 0.00 TFLOPS (vs torch 2.75x), aiter 12.70 us (vs torch 2.61x), gemma/aiter 1.05x
[perf fused_add_rmsnorm] dim=(2, 256) dtype=bf16: torch 31.04 us, gemma(ours) 11.95 us 0.00 TFLOPS (vs torch 2.60x), aiter 13.09 us (vs torch 2.37x), gemma/aiter 1.10x
[perf fused_add_rmsnorm] dim=(2, 4096) dtype=bf16: torch 36.70 us, gemma(ours) 12.36 us 0.00 TFLOPS (vs torch 2.97x), aiter 13.36 us (vs torch 2.75x), gemma/aiter 1.08x
[perf fused_add_rmsnorm] dim=(128, 256) dtype=bf16: torch 35.61 us, gemma(ours) 12.04 us 0.01 TFLOPS (vs torch 2.96x), aiter 13.03 us (vs torch 2.73x), gemma/aiter 1.08x
[perf fused_add_rmsnorm] dim=(128, 4096) dtype=bf16: torch 44.76 us, gemma(ours) 12.69 us 0.21 TFLOPS (vs torch 3.53x), aiter 13.61 us (vs torch 3.29x), gemma/aiter 1.07x
[perf fused_add_rmsnorm] dim=(256, 256) dtype=bf16: torch 35.69 us, gemma(ours) 12.50 us 0.03 TFLOPS (vs torch 2.86x), aiter 13.26 us (vs torch 2.69x), gemma/aiter 1.06x
[perf fused_add_rmsnorm] dim=(256, 4096) dtype=bf16: torch 52.14 us, gemma(ours) 13.29 us 0.39 TFLOPS (vs torch 3.92x), aiter 14.12 us (vs torch 3.69x), gemma/aiter 1.06x
[perf fused_add_rmsnorm] dim=(8000, 256) dtype=bf16: torch 56.20 us, gemma(ours) 18.92 us 0.54 TFLOPS (vs torch 2.97x), aiter 17.60 us (vs torch 3.19x), gemma/aiter 0.93x
[perf fused_add_rmsnorm] dim=(8000, 4096) dtype=bf16: torch 381.30 us, gemma(ours) 109.36 us 1.50 TFLOPS (vs torch 3.49x), aiter 103.62 us (vs torch 3.68x), gemma/aiter 0.95x
[perf fused_add_rmsnorm] dim=(8294, 256) dtype=bf16: torch 60.70 us, gemma(ours) 18.84 us 0.56 TFLOPS (vs torch 3.22x), aiter 17.86 us (vs torch 3.40x), gemma/aiter 0.95x
[perf fused_add_rmsnorm] dim=(8294, 4096) dtype=bf16: torch 398.75 us, gemma(ours) 113.33 us 1.50 TFLOPS (vs torch 3.52x), aiter 106.88 us (vs torch 3.73x), gemma/aiter 0.94x
[perf fused_add_rmsnorm] dim=(33176, 256) dtype=bf16: torch 115.62 us, gemma(ours) 40.34 us 1.05 TFLOPS (vs torch 2.87x), aiter 34.57 us (vs torch 3.34x), gemma/aiter 0.86x
[perf fused_add_rmsnorm] dim=(33176, 4096) dtype=bf16: torch 1697.81 us, gemma(ours) 448.06 us 1.52 TFLOPS (vs torch 3.79x), aiter 444.46 us (vs torch 3.82x), gemma/aiter 0.99x
perf done.
```