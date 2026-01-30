# LeetGPU implemented with pyhip

Programs in this folder demonstrate basic examples of leetGPU implemented using the pyhip.
leetGPU provides generic prototypes for common GPU-related problems; the problem descriptions and corresponding test cases can be found at [AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges/tree/main).

Under the hood, pyhip uses HIP kernels to solve these problems. Since the platform requires that leetGPU submissions be implemented in CUDA, but I primarily develop and debug on the HIP platform, you may notice (especially in the .cpp files) a mix of CUDA and HIP code. In such cases, the CUDA call entries are often commented out, serving only to meet the submission requirements, while the HIP code is used for actual testing during development.

To maintain broader portability, I've tried to write the code in a more general style, rather than relying heavily on HIP- or CUDA-specific extensions.



## Golden Solution

### triton 
- [meta-pytorch/tritonbench](https://github.com/meta-pytorch/tritonbench)
### GEMM
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

### Attention 
- [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)


## Other Reference


- https://github.com/ROCm/rocPRIM

- https://github.com/ROCm/rocPRIM/tree/develop_deprecated/benchmark

- https://github.com/NVIDIA/cub
