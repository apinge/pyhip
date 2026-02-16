"""
调用 test_data_dep kernel：LDS + readfirstlane + __syncthreads，out[tid] = base + (255 - tid)。
"""
import torch
import pyhip

torch.set_default_device("cuda")

hip = pyhip.module("test_data_dep.cpp")

def main():
    n = 256
    base = 42
    shared_mem = n * 4  # 256 ints
    expected = torch.tensor([base + (255 - i) for i in range(n)], dtype=torch.int32)

    # C 版本
    out = torch.empty(n, dtype=torch.int32)
    hip.test_data_dep([1], [n], out.data_ptr(), base, sharedMemBytes=shared_mem)
    torch.cuda.synchronize()
    assert out.equal(expected), f"test_data_dep: out={out.tolist()}\nexpected={expected.tolist()}"
    print("PASS: test_data_dep -> out[tid] = base + (255 - tid)")

    # 内联汇编版本（与 .s 中 _Z13test_data_depPii 对应）
    out_asm = torch.empty(n, dtype=torch.int32)
    hip.test_data_dep_asm([1], [n], out_asm.data_ptr(), base, sharedMemBytes=shared_mem)
    torch.cuda.synchronize()
    assert out_asm.equal(expected), f"test_data_dep_asm: out={out_asm.tolist()}\nexpected={expected.tolist()}"
    print("PASS: test_data_dep_asm -> out[tid] = base + (255 - tid)")
    assert out.equal(out_asm), "C vs asm mismatch"
    print("PASS: C and asm outputs match.")

if __name__ == "__main__":
    main()
