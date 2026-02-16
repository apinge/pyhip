"""
测试 __builtin_amdgcn_readfirstlane：wave 内所有线程得到的是 first lane 的值。
两个 kernel 语义一致：out[tid] = readfirstlane(base + (tid & 63))，即每 wave 内全写 first lane 的值。
"""
import torch
import pyhip

torch.set_default_device("cuda")

hip = pyhip.module("test_readfirstlane.cpp")

def main():
    n = 64 * 4
    base = 42
    # 每个 wave 的 first lane 是 tid&63==0，值为 base+0=base，故全 block 应全是 base
    expected = torch.full((n,), base, dtype=torch.int32)

    # 1) C 版本 kernel
    out_c = torch.empty(n, dtype=torch.int32)
    hip.test_readfirstlane([1], [n], out_c.data_ptr(), base)
    torch.cuda.synchronize()
    assert out_c.equal(expected), f"test_readfirstlane: out={out_c.tolist()}, expected all {base}"
    print("PASS: test_readfirstlane (C) -> all elements =", base)

    # 2) 内联汇编版本 kernel（与 C 等价，用于对照 ISA）
    out_asm = torch.empty(n, dtype=torch.int32)
    hip.test_readfirstlane_asm([1], [n], out_asm.data_ptr(), base)
    torch.cuda.synchronize()
    assert out_asm.equal(expected), f"test_readfirstlane_asm: out={out_asm.tolist()}, expected all {base}"
    print("PASS: test_readfirstlane_asm (inline asm) -> all elements =", base)

    # 3) 两 kernel 结果一致
    assert out_c.equal(out_asm), f"C vs asm mismatch: C={out_c.tolist()}, asm={out_asm.tolist()}"
    print("PASS: C kernel and asm kernel outputs match.")

    # 4) 新 kernel test_readfirstlane_asm_2（操作数绑定 out/base，无 s_load）
    out_asm2 = torch.empty(n, dtype=torch.int32)
    hip.test_readfirstlane_asm_2([1], [n], out_asm2.data_ptr(), base)
    torch.cuda.synchronize()
    assert out_asm2.equal(expected), f"test_readfirstlane_asm_2: out={out_asm2.tolist()}, expected all {base}"
    print("PASS: test_readfirstlane_asm_2 -> all elements =", base)
    assert out_c.equal(out_asm2), f"C vs asm_2 mismatch"
    print("PASS: C kernel and asm_2 kernel outputs match.")

if __name__ == "__main__":
    main()
