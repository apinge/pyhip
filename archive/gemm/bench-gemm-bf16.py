"""
BF16 GEMM 性能测试，模仿 gemm-ilp.py。
D = alpha*(A@B) + beta*C，kernel: gemm-bf16-naive.cpp sgemm_bf16_d (MFMA 16x16x16 bf16，原始无优化).
"""

import sys
import torch
import pyhip

torch.set_printoptions(linewidth=300)
if torch.cuda.is_available():
    torch.set_default_device("cuda")
torch.manual_seed(0)

cur_gpu_device = torch.cuda.get_device_name()
num_CU = torch.cuda.get_device_properties().multi_processor_count
NUM_XCD = 8 if num_CU > 80 else 4
print(f"{torch.get_default_device()=} with {num_CU=} {NUM_XCD=}")

hip = pyhip.module("gemm-bf16-naive.cpp")
gemm = hip.sgemm_bf16_d

# 问题规模（M,N,K 均为 16 的倍数）
M = 8192
N = 8192
K = 8192
# 保证 16 的倍数
M = (M + 15) // 16 * 16
N = (N + 15) // 16 * 16
K = (K + 15) // 16 * 16

print(f" {M}x{N}x{K} ")

alpha, beta = 2.1, 2.1
lda, ldb, ldc, ldd = M, N, M, M

# 布局: A (K,M) 列优先 MxK, B (K,N) 行优先 KxN, C/D (N,M) 列优先 MxN
A = torch.randn(K, M, dtype=torch.bfloat16)
B = torch.randn(K, N, dtype=torch.bfloat16)
C = torch.randn(N, M, dtype=torch.float32)

DATA_CLONES = 4
As = [torch.clone(A) for _ in range(DATA_CLONES)]
Bs = [torch.clone(B) for _ in range(DATA_CLONES)]
Cs = [torch.clone(C) for _ in range(DATA_CLONES)]
Ds = [torch.empty(N, M, dtype=torch.float32) for _ in range(DATA_CLONES)]

grid = [(M + 63) // 64, (N + 63) // 64]
block = [256, 4]

# Flops: 2*M*N*K
flops = 2 * M * N * K
# Bytes: read A(M*K*2) + B(K*N*2) + C(M*N*4), write D(M*N*4)
bytes_io = M * K * 2 + K * N * 2 + M * N * 4 + M * N * 4

if len(sys.argv) == 1:
    # bf16 linear 性能（hipblaslt 等），不改动
    for i in range(4):
        di = i % DATA_CLONES
        with pyhip.cudaPerf(flops, name=f"torch-bf16-linear-{di}"):
            ref = alpha * torch.nn.functional.linear(As[di].T, Bs[di].T) + beta * Cs[di].T

    D = torch.empty(N, M, dtype=torch.float32)
    gemm(
        grid, block,
        M, N, K,
        A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(),
        lda, ldb, ldc, ldd,
        alpha, beta,
    )
    # 正确性用 float32 参考单独算，不影响上面 bf16 性能段
    ref_f32 = alpha * torch.nn.functional.linear(A.T.to(torch.float32), B.T.to(torch.float32)) + beta * C.T
    pass_flag = torch.allclose(ref_f32, D.T, atol=1e-2, rtol=1e-2)
    if not pass_flag:
        print("ref_f32 sample:", ref_f32[0, :4])
        print("D.T sample:", D.T[0, :4])
else:
    pass_flag = None

for i in range(4):
    di = i % DATA_CLONES
    with pyhip.cudaPerf(flops, bytes_io, name=f"gemm_bf16_{di}"):
        gemm(
            grid, block,
            M, N, K,
            As[di].data_ptr(), Bs[di].data_ptr(), Cs[di].data_ptr(), Ds[di].data_ptr(),
            lda, ldb, ldc, ldd,
            alpha, beta,
        )

if len(sys.argv) == 1:
    print("PASS" if pass_flag else "FAILED")
