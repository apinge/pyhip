import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_basic():
    @pyhip.jit()
    def kernel(J, pA:"int*", cnt:"int"):
        acc = J.new_gpr("a", 4,name="acc")
        s_idx = J.new_gpr('s',1, dtype="u32", name="s_idx")
        s_temp0 = J.new_gpr('s',1,name="s_temp0")
        s_temp = J.new_gpr('s',2, align=2,name="s_temp")
        s_temp2 = J.new_gpr('s',2, align=2,name="s_temp2")
        vtemp = J.new_gpr('v',2, dtype="u32", align=2,name="vtemp")

        for i in range(4):
            J.v_accvgpr_write_b32(acc[i], 0)

        s_idx[0] = 0
        J.Label("bb0")

        #J.s_lshl_b32(s_temp[1],1, s_idx)
        J.s_lshl_b32(s_temp0,s_idx,2)
        #s_temp[0] = s_idx[0] << 2

        s_temp[:] = pA[:] + s_temp0[0]
        J.s_store_dword(cnt, s_temp, 0, mod="glc")

        J.s_add_u32(s_temp2[0], pA[0], s_temp0)

        J.s_addk_i32(s_idx, 1)
        J.s_cmp_lt_i32(s_idx, cnt)
        J.s_cbranch_scc0(mod="bb1")
        J.s_branch(mod="bb0")

        J.Label("bb1")

    A = torch.ones(64, dtype=torch.int)
    CNT = 31
    kernel([1],[64], A.data_ptr(), CNT)
    torch.cuda.synchronize()
    ref = torch.ones(64, dtype=torch.int)
    ref[:CNT] = CNT
    torch.testing.assert_close(A, ref)

def test_basic1():
    magic_value = 0x123463
    @pyhip.jit()
    def kernel(J, pA:"int*", mv:"int"):
        idx = J.gpr("su32", mv*8)
        idx4 = J.gpr(4, "su32", mv*9, mv*10)
        J.s_store_dword(idx, pA, 0, mod="glc")
        J.s_store_dwordx4(idx4, pA, 4, mod="glc")

    A = torch.ones(5, dtype=torch.int)
    kernel([1],[64], A.data_ptr(), magic_value)
    torch.cuda.synchronize()
    assert A[0] == magic_value*8, A[0]
    assert A[1] == magic_value*9, A[1]
    assert A[2] == magic_value*10, A[2]
    assert A[3] == magic_value*10, A[3]
    assert A[4] == magic_value*10, A[4]

def test_basic2():
    @pyhip.jit()
    def kernel(J, pA:"int*", cnt:"int"):
        s_idx = J.new_gpr('s',1, dtype="u32", name="s_idx")
        s_offset = J.new_gpr('s',1,dtype="u32", name="s_offset")
        s_temp = J.new_gpr('s',2, dtype="u32", align=2,name="s_temp")

        J.s_waitcnt(mod=f"lgkmcnt({0})")

        s_idx[0] = 0
        J.Label("bb0")

        s_offset[0] = s_idx[0] << 2

        s_temp[0] = pA[0] + s_offset[0]
        J.s_addc_u32(s_temp[1], pA[1], 0)

        J.s_store_dword(cnt, s_temp, 0, mod="glc")

        J.s_addk_i32(s_idx, 1)
        J.s_cmp_lt_i32(s_idx, cnt)
        J.s_cbranch_scc0(mod="bb1")
        J.s_branch(mod="bb0")

        J.Label("bb1")
        J.s_waitcnt(mod=f"lgkmcnt({0})")

    A = torch.ones(64, dtype=torch.int)
    CNT = 56
    kernel([1],[64], A.data_ptr(), CNT)
    torch.cuda.synchronize()
    ref = torch.ones(64, dtype=torch.int)
    ref[:CNT] = CNT
    torch.testing.assert_close(A, ref)

class Buffer:
    def __init__(self, J):
        self.J = J
        self.desc = J.new_gpr('s', 4, align=4)
        self.base = self.desc[0:1]
        self.range = self.desc[2]
        self.config = self.desc[3]
        J.s_mov_b32(self.config, 0x00020000)

    def setup(self, base, range):
        self.base[0] = base[0]
        self.base[1] = base[1]
        self.range[0] = range[0]

    def load_dwordx4(self, vdst, voffset, soffset, offset12=0):
        # vdst,     vaddr,           srsrc, soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset12:{offset12}"
        self.J.buffer_load_dwordx4(vdst, voffset, self.desc, soffset, mod=mod)

    def store_dwordx4(self, vdata, voffset, soffset, offset12=0):
        # vdata,    vaddr,        srsrc,  soffset          idxen offen offset12 sc0 nt sc1
        assert isinstance(offset12 , int) # must be compile time constant
        mod = f"offen"
        if offset12 > 0:
            mod += f" offset12:{offset12}"
        self.J.buffer_store_dwordx4(vdata, voffset, self.desc, soffset, mod=mod)

def test_vadd():
    @pyhip.jit()
    def kernel(J, pA:"int*", pB:"int*", cnt:"int"):
        vtemp =J.new_gpr('v',1) 
        v4 = J.new_gpr('v', 4, dtype="u32",align=2)

        buff_a = Buffer(J)
        buff_b = Buffer(J)

        size = J.gpr(cnt[0] * 4)
        buff_a.setup(pA, size)
        buff_b.setup(pB, size)
        
        vtemp[0] = J.threadIdx.x[0] << 4
        buff_a.load_dwordx4(v4, vtemp[0], 0)
        J.s_waitcnt(mod=f"vmcnt(0)")

        for i in range(4):
            # J.v_add_u32(v4[i], 102, v4[i])
            v4[i] = 102 + v4[i]

        buff_b.store_dwordx4(v4, vtemp[0], 0)
        J.s_waitcnt(mod=f"vmcnt(0)")

    A = torch.ones(64*4*8, dtype=torch.int)
    B = torch.ones(64*4*8, dtype=torch.int)
    kernel([1],[64], A.data_ptr(), B.data_ptr(), 64*4)
    torch.cuda.synchronize()
    ref = torch.full((256,), 103, dtype=torch.int)
    if not torch.allclose(ref, B[:256]):
        print(B)
    torch.testing.assert_close(B[:256], ref)


class FlatBuffer:
    """
    hipcc 更倾向于生成global_load global_store 而非buffer_load buffer_store
    观察到 compiler支持以下两种寻址模式
    
    1. base+offset 模式：
        - buffer 持有一个 base 寄存器（saddr 或 v[lo:hi]）
        - 每次 load/store 时，传入的是 offset，FlatBuffer 会做 base + offset
        e.g global_load_dwordx4 v[4:7], v8, s[6:7] i.e. v[4:7] = load_dwordx4_from_addr(v8 + s[6:7])
    
    2. absolute address 模式：
        - FlatBuffer 不再计算偏移，load/store 时传入已经计算好的 64-bit 全局地址
        - 适合多 buffer 共享同一偏移的情况（例如 thread index 左移乘元素大小）
        e.g. global_load_dwordx4 v[0:3], v[0:1], off
    """
    def __init__(self, J, mode="base_offset"):
        self.J = J
        self.base = None
        self.mode = mode   # "base_offset" or "absolute"

    def setup(self, base):
        self.base = base #saddr s[:]

    def _make_abs_addr(self, vaddr32):
        """
        把 32-bit vaddr 扩展成 64-bit，
        然后做 base + offset
        返回 v[lo:hi]
        这个函数仅供参考 
        vector add 这种多个buffer公用一个偏移的情况 
        倾向于把 base的偏移的部分 拉到公共的地方计算 
        否则容易重复计算
        """
        J = self.J

        vaddr64 = J.new_gpr('v', 2, align=2)

        # sign extend 32 -> 64
        vaddr64[0] = vaddr32
        J.v_ashrrev_i32(vaddr64[1], 31, vaddr32)

        # 乘 element size (x4 = 16B)
        J.v_lshlrev_b64(vaddr64, 4, vaddr64)

        # base + offset
        J.v_lshl_add_u64(vaddr64, self.base, 0, vaddr64)

        return vaddr64

    def load_dwordx4(self, vdst, vaddr):
        if self.mode == "base_offset":
            self.J.global_load_dwordx4(vdst, vaddr, self.base)
        else:
            #vaddr64 = self._make_abs_addr(vaddr)
            self.J.global_load_dwordx4(vdst, vaddr, "off")

    def store_dwordx4(self, vaddr, vdata):
        if self.mode == "base_offset":
            self.J.global_store_dwordx4(vaddr, vdata, self.base)
        else:
            #vaddr64 = self._make_abs_addr(vaddr)
            self.J.global_store_dwordx4(vaddr, vdata, "off")


def test_vadd_flat_buffer():
    @pyhip.jit()
    def vector_add_flat_buffer(J, N:"int", PA:"int*", PB:"int*", PC:"int*"):
        """
        vector add use flat buffer 
        equal to hip version
        using intx4 = __attribute__((vector_size(4 * sizeof(int)))) int;
        __global__ void vector_add(int N, const intx4* A, const intx4* B, intx4* C) {
        int i = threadIdx.x;
            C[i] = A[i] + B[i] ;
         }

        """
        vtemp =J.new_gpr('v',1) 
        v4_a = J.new_gpr('v', 4, dtype="u32",align=2)
        v4_b = J.new_gpr('v', 4, dtype="u32",align=2)

        buff_a = FlatBuffer(J)
        buff_b = FlatBuffer(J)
        buff_c = FlatBuffer(J)

        buff_a.setup(PA)
        buff_b.setup(PB)
        buff_c.setup(PC)
        
        vtemp[0] = J.threadIdx.x[0] << 4
        buff_a.load_dwordx4(v4_a, vtemp[0])
        buff_b.load_dwordx4(v4_b, vtemp[0])
        J.s_waitcnt(mod=f"vmcnt(0)")

        for i in range(4):
            v4_a[i] = v4_a[i] + v4_b[i]

        buff_c.store_dwordx4(vtemp[0],v4_a)
        J.s_waitcnt(mod=f"vmcnt(0)")

    pass
    A = torch.randint(0, 256, (64,), dtype=torch.int32, device="cuda")
    B = torch.randint(0, 256, (64,), dtype=torch.int32, device="cuda")
    C = torch.empty_like(A)
    N = A.shape[0]
    vector_add_flat_buffer([1], [64], N, A.data_ptr(), B.data_ptr(), C.data_ptr())
    torch.cuda.synchronize()
    ref = A + B
    assert torch.equal(C, ref), f"C != ref (A+B)"
    print("test_vadd_flat_buffer PASS: C == ref (A+B)")

def test_vadd_flat_buffer_absolute():
    @pyhip.jit()
    def vector_flat_buffer_absolute(J, N:"int", PA:"int*", PB:"int*", PC:"int*"):
        vtemp =J.new_gpr('v',1) 
        v4_a = J.new_gpr('v', 4, dtype="u32",align=2)
        v4_b = J.new_gpr('v', 4, dtype="u32",align=2)

        buff_a = FlatBuffer(J, mode="absolute")
        buff_b = FlatBuffer(J, mode="absolute")
        buff_c = FlatBuffer(J, mode="absolute")
        # 计算size s_lshl_b32 s5, s5, 0x2
        #  size = N*4
        # size = J.gpr(N[0] * 4)
        buff_a.setup(PA)
        buff_b.setup(PB)
        buff_c.setup(PC)
        
        vtemp[0] = J.threadIdx.x[0]

        """
        把 32-bit vaddr 扩展成 64-bit，
        然后做 base + offset
        返回 v[lo:hi]
        """
        vaddr64 = J.new_gpr('v', 2, align=2)
        vaddr64_a = J.new_gpr('v', 2, align=2)
        vaddr64_b = J.new_gpr('v', 2, align=2)
        vaddr64_c = J.new_gpr('v', 2, align=2)
        # sign extend 32 -> 64
        vaddr64[0] = vtemp[0]
        J.v_ashrrev_i32(vaddr64[1], 31, vtemp[0])

        # 乘 element size (x4 = 16B)
        J.v_lshlrev_b64(vaddr64, 4, vaddr64)

        # base + offset
        J.v_lshl_add_u64(vaddr64_a, buff_a.base, 0, vaddr64)
        J.v_lshl_add_u64(vaddr64_b, buff_b.base, 0, vaddr64)
        J.v_lshl_add_u64(vaddr64_c, buff_c.base, 0, vaddr64)

        buff_a.load_dwordx4(v4_a, vaddr64_a)
        J.s_nop(0)
        buff_b.load_dwordx4(v4_b, vaddr64_b)
        J.s_waitcnt(mod=f"vmcnt(0)")

        for i in range(4):
            v4_a[i] = v4_a[i] + v4_b[i]

        buff_c.store_dwordx4(vaddr64_c,v4_a)
        J.s_waitcnt(mod=f"vmcnt(0)")

    pass
    A = torch.randint(0, 256, (64,), dtype=torch.int32, device="cuda")
    B = torch.randint(0, 256, (64,), dtype=torch.int32, device="cuda")
    C = torch.empty_like(A)
    N = A.shape[0]
    vector_flat_buffer_absolute([1], [64], N, A.data_ptr(), B.data_ptr(), C.data_ptr())
    torch.cuda.synchronize()
    ref = A + B
    assert torch.equal(C, ref), f"C != ref (A+B)"
    print("test_vadd_flat_buffer_absolute PASS: C == ref (A+B)")

if __name__ == "__main__":
    test_vadd_flat_buffer()
    test_vadd_flat_buffer_absolute()
    test_basic1()
    assert 0
    test_basic()
    test_basic2()
    test_vadd()
