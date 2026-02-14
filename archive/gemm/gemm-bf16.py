"""
BF16 GEMM via pyhip: D = alpha * (A @ B) + beta * C.
Kernel from gemm-bf16-naive.cpp (MFMA 16x16x16 bf16，原始无优化).
Layout: A MxK col-major (lda=M), B KxN row-major (ldb=N), C/D MxN col-major (ldc=ldd=M).
"""

import torch
import pyhip

if torch.cuda.is_available():
    torch.set_default_device("cuda")
torch.manual_seed(0)

# Compile kernel (no extra defines needed)
hip = pyhip.module("gemm-bf16-naive.cpp")
kernel = hip.sgemm_bf16_d

# Problem size (must be multiples of 16)
M, N, K = 256, 256, 256
alpha, beta = 2.1, 2.1

# A: MxK col-major -> store as (K, M) so A[col, row] = linear index col*M+row
# B: KxN row-major -> (K, N)
# C, D: MxN col-major -> (N, M)
A = torch.randn(K, M, dtype=torch.bfloat16)
B = torch.randn(K, N, dtype=torch.bfloat16)
C = torch.randn(N, M, dtype=torch.float32)
D = torch.empty(N, M, dtype=torch.float32)

lda, ldb, ldc, ldd = M, N, M, M

# Grid/block: 64x64 per workgroup -> grid = [ceil(M/64), ceil(N/64)], block = [256, 4]
grid = [(M + 63) // 64, (N + 63) // 64]
block = [256, 4]

# Launch: sgemm_bf16_d(m, n, k, a, b, c, d, lda, ldb, ldc, ldd, alpha, beta)
kernel(
    grid,
    block,
    M, N, K,
    A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(),
    lda, ldb, ldc, ldd,
    alpha, beta,
)

# Reference: D_ref = alpha * (A.T @ B) + beta * C.T  (math layout A (M,K), B (K,N), C (M,N))
ref = alpha * torch.nn.functional.linear(A.T.to(torch.float32), B.T.to(torch.float32)) + beta * C.T

# D is (N,M) col-major = (M,N).T -> compare with ref (M,N) via D.T
pass_flag = torch.allclose(ref, D.T, atol=1e-2, rtol=1e-2)
print("PASS" if pass_flag else "FAILED")
if not pass_flag:
    print("max |ref - D.T|:", (ref - D.T).abs().max().item())
