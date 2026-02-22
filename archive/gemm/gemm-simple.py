"""
MFMA kernels 调用与 ref 对比。来源:
https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
所有矩阵 row-major，C = A @ B。
"""
import torch
import pyhip

if torch.cuda.is_available():
    torch.set_default_device("cuda")
torch.manual_seed(0)


def get_current_arch():
    return torch.cuda.get_device_properties().gcnArchName


hip = pyhip.module("gemm-simple.cpp")

def run_and_check(name, A, B, C, kernel, grid, block, atol=1e-2, rtol=1e-2):
    """C 为输出，ref = A @ B (float)，与 C 对比。"""
    kernel(grid, block, A.data_ptr(), B.data_ptr(), C.data_ptr())
    if "A_BT" in name:
        ref = torch.mm(A.to(torch.float32), B.to(torch.float32).T)
    else:
        ref = torch.mm(A.to(torch.float32), B.to(torch.float32))
    ok = torch.allclose(ref, C, atol=atol, rtol=rtol)
    print(f"{name}: {'PASS' if ok else 'FAILED'}")
    if not ok:
        print("  max |ref - C|:", (ref - C).abs().max().item())
        print(ref)
        print(C)
    return ok

# ---------------------------------------------------------------------------
# FP32 32x32x2: A 32x2, B 2x32, C 32x32
# ---------------------------------------------------------------------------
def test_mfma_fp32_32x32x2():
    A = torch.randn(32, 2, dtype=torch.float32, device="cuda")
    B = torch.randn(2, 32, dtype=torch.float32, device="cuda")
    C = torch.zeros(32, 32, dtype=torch.float32, device="cuda")
    run_and_check("mfma_fp32_32x32x2_fp32", A, B, C, hip.mfma_fp32_32x32x2_fp32, [1], [64])

# ---------------------------------------------------------------------------
# FP16 16x16x16: A 16x16, B 16x16, C 16x16
# ---------------------------------------------------------------------------
def test_mfma_fp32_16x16x16_fp16():
    A = torch.randn(16, 16, dtype=torch.float16, device="cuda")
    B = torch.randn(16, 16, dtype=torch.float16, device="cuda")
    C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    run_and_check("mfma_fp32_16x16x16_fp16", A, B, C, hip.mfma_fp32_16x16x16_fp16, [1], [64])

# ---------------------------------------------------------------------------
# FP16 16x16x16: A 16x16, B 16x16, C 16x16 
# C= A*(B^T)
# ---------------------------------------------------------------------------
def test_mfma_fp32_16x16x16_fp16_A_BT():
    A = torch.randn(16, 16, dtype=torch.float16, device="cuda")
    B = torch.randn(16, 16, dtype=torch.float16, device="cuda")
    C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    run_and_check("mfma_fp32_16x16x16_fp16_A_BT", A, B, C, hip.mfma_fp32_16x16x16_fp16_A_BT, [1], [64])

# ---------------------------------------------------------------------------
# BF16 16x16x16: A 16x16, B 16x16, C 16x16
# ---------------------------------------------------------------------------
def test_sgemm_bf16_simple():
    M = N = K = 16
    A = (torch.arange(M * K, device="cuda", dtype=torch.float32) % 256).reshape(M, K).to(torch.bfloat16)
    B = (torch.arange(N * K, device="cuda", dtype=torch.float32) % 256).reshape(N, K).to(torch.bfloat16)
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    run_and_check("sgemm_bf16_simple", A, B, C, hip.sgemm_bf16_simple, [1], [64])

# ---------------------------------------------------------------------------
# FP32 32x32x8: A 32x8, B 8x32, C 32x32
# ./matrix_calculator.py --architecture cdna3 --instruction v_mfma_f32_32x32x8_f16 --detail-instruction

# ---------------------------------------------------------------------------

def test_sgemm_fp32_32x32x8_fp16_A_BT():
    M = N = 32
    K = 8
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    run_and_check("mfma_fp32_32x32x8_fp16_A_BT", A, B, C, hip.mfma_fp32_32x32x8_fp16_A_BT, [1], [64])

# ---------------------------------------------------------------------------
# FP32 32x32x8: A 32x8, B 8x32, C 32x32
# ---------------------------------------------------------------------------

def test_sgemm_fp32_32x32x8_fp16():
    M = N = 32
    K = 8
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    run_and_check("mfma_fp32_32x32x8_fp16", A, B, C, hip.mfma_fp32_32x32x8_fp16, [1], [64])

# ---------------------------------------------------------------------------
# FP8 16x16x32 A_BT: A 16x32, B 16x32, C = A @ B^T 16x16
# ---------------------------------------------------------------------------
def test_mfma_fp32_16x16x32_fp8_fp8_A_BT():
    if not hasattr(torch, "float8_e4m3fnuz"):
        print("mfma_fp32_16x16x32_fp8_fp8_A_BT: SKIP (no torch.float8_e4m3fnuz)")
        return
    try:
        arch = get_current_arch()
        if "gfx942" in arch:
            A = torch.randn(16, 32, device="cuda").to(torch.float8_e4m3fnuz)
            B = torch.randn(16, 32, device="cuda").to(torch.float8_e4m3fnuz)
        elif "gfx950" in arch:
            A = torch.randn(16, 32, device="cuda").to(torch.float8_e4m3fn)
            B = torch.randn(16, 32, device="cuda").to(torch.float8_e4m3fn)
        else:
            print("mfma_fp32_16x16x32_fp8_fp8_A_BT: SKIP (arch not gfx942/gfx950)")
            return
    except Exception as e:
        print("mfma_fp32_16x16x32_fp8_fp8_A_BT: SKIP", e)
        return
    C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
    run_and_check("mfma_fp32_16x16x32_fp8_fp8_A_BT", A, B, C, hip.mfma_fp32_16x16x32_fp8_fp8_A_BT, [1], [64])

# ---------------------------------------------------------------------------
# FP8 32x32x16: A 32x16, B 16x32, C 32x32
# ---------------------------------------------------------------------------
def test_mfma_fp32_32x32x16_fp8():
    if not hasattr(torch, "float8_e4m3fnuz"):
        print("mfma_fp32_32x32x16_fp8_fp8: SKIP (no torch.float8_e4m3fnuz)")
        return
    try:
        arch = get_current_arch()
        if "gfx942" in arch:
            A = torch.randn((32,16), device="cuda").to(torch.float8_e4m3fnuz) # for cdna3
            B = torch.randn((16,32), device="cuda").to(torch.float8_e4m3fnuz) # for cnda3
        elif "gfx950" in arch:
            A = torch.randn((32,16), device="cuda").to(torch.float8_e4m3fn) # for cdna4
            B = torch.randn((16,32), device="cuda").to(torch.float8_e4m3fn) # for cnda4
        else:
            print("mfma_fp32_32x32x16_fp8_fp8: SKIP (no torch.float8_e4m3fnuz or torch.float8_e4m3fn)")
            return
        print(A.is_contiguous())
        print(B.is_contiguous())
    except Exception as e:
        print("mfma_fp32_32x32x16_fp8_fp8: SKIP", e)
        return
    C = torch.zeros(32, 32, dtype=torch.float32, device="cuda")
    run_and_check("mfma_fp32_32x32x16_fp8_fp8", A, B, C, hip.mfma_fp32_32x32x16_fp8_fp8, [1], [64])

# ---------------------------------------------------------------------------
# 全部运行
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    test_mfma_fp32_32x32x2()
    test_mfma_fp32_16x16x16_fp16()
    test_mfma_fp32_16x16x16_fp16_A_BT()
    test_sgemm_fp32_32x32x8_fp16_A_BT()
    test_sgemm_bf16_simple()
    test_mfma_fp32_16x16x32_fp8_fp8_A_BT()
    test_mfma_fp32_32x32x16_fp8()
