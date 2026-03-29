import os

import pyhip
import torch

torch.set_default_device("cuda")

_DIR = os.path.dirname(os.path.abspath(__file__))
hip = pyhip.module(os.path.join(_DIR, "AB_swap.cpp"))


def torch_ref(A: torch.Tensor, B: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Out = (A @ B) @ D，与 kernel 约定一致；用 fp32 累加再回 fp16。"""
    return (A @ B) @ D # 结果是dtype fp16


def main():
    torch.manual_seed(0)
    A = torch.randn(16, 32, dtype=torch.float16)
    B = torch.randn(32, 16, dtype=torch.float16)
    D = torch.randn(16, 16, dtype=torch.float16)
    Out = torch.empty(16, 16, dtype=torch.float16)
    ref = torch_ref(A, B, D)

    hip.ab_swap([1], [64], A.data_ptr(), B.data_ptr(), D.data_ptr(), Out.data_ptr())
    torch.cuda.synchronize()
    
    ok = torch.allclose(ref, Out, atol=1e-2, rtol=1e-2)
    print("PASS" if ok else "FAIL", "| max|ref-out|:", (ref - Out).abs().max().item())


if __name__ == "__main__":
    main()
