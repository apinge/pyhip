"""Qwen3.8 GR read using gfx942 MFMA and two FlyDSL launches.

Layout/fragment usage follows ../test_gemm.py and FlyDSL's tiled GEMM examples.
The SiLU and gated-mean epilogues follow SGLang's CuTe HC mix implementation.
This module has no SGLang dispatch integration.
The default keeps the SiLU activation as BF16 high/low components so the
full checkpoint passes the existing FP64 tolerance without changing weights.
"""

from dataclasses import dataclass
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

HC = 4
HS = 2560
K = HC * HS
R = 320
MAX_ROWS = 24
LOG2E = 1.4426950408889634


@dataclass(frozen=True)
class Config:
    split_k: int = 16
    block_m: int = 16
    down_n: int = 64
    up_n: int = 128
    block_k: int = 64
    waves: int = 4
    skip_padding: bool = True
    preshuffle: bool = True
    fast_math: bool = True
    compensate_hidden: bool = False

    def validate(self):
        if self.split_k <= 0 or K % (self.split_k * self.block_k):
            raise ValueError("split_k * block_k must divide 10240")
        if self.block_m not in (16, 32):
            raise ValueError("block_m must be 16 or 32")
        if self.waves not in (1, 2, 4):
            raise ValueError("waves must be 1, 2, or 4")
        if R % self.down_n or K % self.up_n:
            raise ValueError("GEMM output tiles must divide their output widths")
        if self.down_n % (16 * self.waves) or self.up_n % (16 * self.waves):
            raise ValueError("N tiles must contain each wave's MFMA tile")
        if R % self.block_k or self.block_k % 16:
            raise ValueError("block_k must be a multiple of 16 dividing 320")
        lds_bytes = self.block_m * R * 2 * (2 if self.compensate_hidden else 1) + self.block_m * self.up_n * 4
        if lds_bytes > 65536:
            raise ValueError("up kernel exceeds gfx942's 64 KiB LDS capacity")


@cache
def _launchers(rows: int, dtype: torch.dtype, config: Config):
    bm, dn, un, bk = config.block_m, config.down_n, config.up_n, config.block_k
    split, waves = config.split_k, config.waves
    # FlyDSL 0.3.1 hashes scalar closure values, not fields of config objects.
    skip_padding, preshuffle, fast_math = config.skip_padding, config.preshuffle, config.fast_math
    compensate_hidden = config.compensate_hidden
    padded_rows = (rows + bm - 1) // bm * bm
    threads = 64 * waves
    elem = fx.BFloat16

    @fx.struct
    class UpShared:
        hidden: fx.Array[elem, bm * R, 16]
        hidden_low: fx.Array[elem, bm * R if compensate_hidden else 1, 16]
        logits: fx.Array[fx.Float32, bm * un, 16]

    @flyc.kernel
    def down_partial(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor):
        tid = fx.thread_idx.x
        im, jn, sk = fx.block_idx
        x = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1))), max_size=False)
        if fx.const_expr(preshuffle):
            w_layout = fx.make_layout(((16, R // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512)))
        else:
            w_layout = fx.make_layout((R, K), (K, 1))
        w = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(W), w_layout), max_size=False)
        p = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(P),
                fx.make_layout((padded_rows, R, split), (R, 1, padded_rows * R)),
            ),
            max_size=False,
        )
        a_tile = fx.flat_divide(x, fx.make_tile(bm, bk))[None, None, im, None]
        b_tile = fx.flat_divide(w, fx.make_tile(dn, bk))[None, None, jn, None]
        c_tile = fx.flat_divide(p[None, None, sk], fx.make_tile(bm, dn))[None, None, im, jn]
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, elem))
        tiled = fx.make_tiled_mma(mma, fx.make_layout((1, waves, 1), (0, 1, 0)))
        thr = tiled.thr_slice(tid)
        copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), elem)
        ca = fx.make_tiled_copy_A(copy_ab, tiled).get_slice(tid)
        cb = fx.make_tiled_copy_B(copy_ab, tiled).get_slice(tid)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        cc = fx.make_tiled_copy_C(copy_c, tiled).get_slice(tid)
        fa = thr.make_fragment_A(a_tile[None, None, 0])
        fb = thr.make_fragment_B(b_tile[None, None, 0])
        fc = thr.make_fragment_C(c_tile)
        fc.fill(0)
        ga, gb = ca.partition_S(a_tile), cb.partition_S(b_tile)
        ra, rb = ca.retile(fa), cb.retile(fb)
        for ki, state in range(fx.Index(0), fx.Index(K // split // bk), fx.Index(1), init=[fc.load()]):
            fc.store(state[0])
            kt = sk * (K // split // bk) + fx.Int32(ki)
            fx.copy(copy_ab, ga[None, None, None, kt], ra)
            fx.copy(copy_ab, gb[None, None, None, kt], rb)
            fx.gemm(mma, fc, fa, fb, fc)
            result = yield [fc.load()]
        fc.store(result)
        fx.copy(copy_c, cc.retile(fc), cc.partition_D(c_tile))

    @flyc.kernel
    def up_gate(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        im, jn, _ = fx.block_idx
        shared = fx.SharedAllocator().allocate(UpShared).peek()
        h = shared.hidden.view(fx.make_layout((bm, R), (R, 1)))
        c = shared.logits.view(fx.make_layout((bm, un), (un, 1)))
        p = fx.rocdl.make_buffer_tensor(P, max_size=False)
        p4 = fx.flat_divide(p, fx.make_tile(4))
        h4 = fx.flat_divide(shared.hidden.view(fx.make_layout(bm * R, 1)), fx.make_tile(4))
        if fx.const_expr(compensate_hidden):
            h_low = shared.hidden_low.view(fx.make_layout((bm, R), (R, 1)))
            h_low4 = fx.flat_divide(shared.hidden_low.view(fx.make_layout(bm * R, 1)), fx.make_tile(4))
        copy_p = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        copy_h = fx.make_copy_atom(fx.UniversalCopy64b(), elem)
        fp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        fh = fx.make_rmem_tensor(fx.make_layout(4, 1), elem)
        # Each up CTA reconstructs its small low-rank input from ordered partials.
        for i in range_constexpr(bm * R // (threads * 4)):
            ix = tid + i * threads
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            if fx.const_expr(skip_padding and rows % bm != 0):
                if im * bm + ix // (R // 4) < rows:
                    for s in range_constexpr(split):
                        offset = (s * padded_rows + im * bm) * (R // 4) + ix
                        fx.copy(copy_p, p4[None, offset], fp)
                        acc = acc + fp.load()
            else:
                for s in range_constexpr(split):
                    offset = (s * padded_rows + im * bm) * (R // 4) + ix
                    fx.copy(copy_p, p4[None, offset], fp)
                    acc = acc + fp.load()
            z = acc * 0.25
            if fx.const_expr(fast_math):
                values = []
                for j in range_constexpr(4):
                    exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-z[j] * LOG2E)))
                    inverse = fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent)))
                    values.append(z[j] * inverse)
                activated = fx.Vector.from_elements(values, fx.Float32)
            else:
                activated = z / (1.0 + (-z * LOG2E).exp2())
            fh.store(activated.to(elem))
            fx.copy(copy_h, fh, h4[None, ix])
            if fx.const_expr(compensate_hidden):
                low = activated - activated.to(elem).to(fx.Float32)
                fh.store(low.to(elem))
                fx.copy(copy_h, fh, h_low4[None, ix])
        fx.gpu.barrier()

        if fx.const_expr(preshuffle):
            w_layout = fx.make_layout(((16, K // 16), (8, 4, R // 32)), ((8, 16 * R), (1, 128, 512)))
        else:
            w_layout = fx.make_layout((K, R), (R, 1))
        w = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(W), w_layout), max_size=False)
        a_tile = fx.flat_divide(h, fx.make_tile(bm, bk))[None, None, 0, None]
        if fx.const_expr(compensate_hidden):
            a_low_tile = fx.flat_divide(h_low, fx.make_tile(bm, bk))[None, None, 0, None]
        b_tile = fx.flat_divide(w, fx.make_tile(un, bk))[None, None, jn, None]
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, elem))
        tiled = fx.make_tiled_mma(mma, fx.make_layout((1, waves, 1), (0, 1, 0)))
        thr = tiled.thr_slice(tid)
        copy_a = fx.make_copy_atom(fx.UniversalCopy64b(), elem)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), elem)
        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        ca = fx.make_tiled_copy_A(copy_a, tiled).get_slice(tid)
        cb = fx.make_tiled_copy_B(copy_b, tiled).get_slice(tid)
        cc = fx.make_tiled_copy_C(copy_c, tiled).get_slice(tid)
        fa = thr.make_fragment_A(a_tile[None, None, 0])
        fb = thr.make_fragment_B(b_tile[None, None, 0])
        fc = thr.make_fragment_C(c)
        fc.fill(0)
        ga, gb = ca.partition_S(a_tile), cb.partition_S(b_tile)
        if fx.const_expr(compensate_hidden):
            ga_low = ca.partition_S(a_low_tile)
        ra, rb = ca.retile(fa), cb.retile(fb)
        for ki in range_constexpr(R // bk):
            fx.copy(copy_a, ga[None, None, None, ki], ra)
            fx.copy(copy_b, gb[None, None, None, ki], rb)
            fx.gemm(mma, fc, fa, fb, fc)
            if fx.const_expr(compensate_hidden):
                fx.copy(copy_a, ga_low[None, None, None, ki], ra)
                fx.gemm(mma, fc, fa, fb, fc)
        fx.copy(copy_c, cc.retile(fc), cc.partition_D(c))
        fx.gpu.barrier()

        x = fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1)))
        y = fx.make_view(fx.get_iter(Y), fx.make_layout((rows, HS), (HS, 1)))
        for i in range_constexpr(bm * (un // HC) // threads):
            index = tid + i * threads
            row_local = index // (un // HC)
            col_local = index % (un // HC)
            row = im * bm + row_local
            col = jn * (un // HC) + col_local
            if row < rows:
                total = fx.Float32(0.0)
                for g in range_constexpr(HC):
                    logit = fx.memref_load(c, (row_local, col_local * HC + g))
                    if fx.const_expr(fast_math):
                        exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-logit * LOG2E)))
                        gate = fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent)))
                    else:
                        gate = 1.0 / (1.0 + (-logit * LOG2E).exp2())
                    value = fx.memref_load(x, (row, g * HS + col)).to(fx.Float32)
                    total = total + gate * value
                fx.memref_store((total * 0.25).to(elem), y, (row, col))

    @flyc.jit
    def launch_down(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, stream: fx.Stream):
        down_partial(X, W, P).launch(grid=(padded_rows // bm, R // dn, split), block=(threads, 1, 1), stream=stream)

    @flyc.jit
    def launch_up(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        up_gate(X, W, P, Y).launch(grid=(padded_rows // bm, K // un, 1), block=(threads, 1, 1), stream=stream)

    return launch_down, launch_up


def preshuffle_weight(weight):
    """BF16 layout used by the colleague's GEMM: [N/16,K/32,4,16,8]."""
    n, k = weight.shape
    return weight.reshape(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(-1)


def default_config(rows):
    """Fixed MI308X choice after the compensated e09 sweep; no runtime tuning."""
    return Config(block_m=16, compensate_hidden=True)


class GRRead:
    """Prepared weights and workspace for sequential calls on the current stream."""

    def __init__(self, rows, w_down, w_up, config=None):
        config = default_config(rows) if config is None else config
        config.validate()
        if not 0 <= rows <= MAX_ROWS:
            raise ValueError("GR read supports 0..24 rows")
        if w_down.shape != (R, K) or w_up.shape != (K, R):
            raise ValueError("expected W_down[320,10240] and W_up[10240,320]")
        if w_down.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
            raise ValueError("Qwen3.8 GR read weights must be BF16")
        if w_down.device != w_up.device or not w_down.is_cuda or torch.version.hip is None:
            raise ValueError("weights must be on the same ROCm device")
        props = torch.cuda.get_device_properties(w_down.device)
        if props.gcnArchName.split(":", 1)[0] != "gfx942":
            raise ValueError("this experiment targets gfx942")
        self.rows, self.config, self.dtype, self.device = rows, config, w_down.dtype, w_down.device
        up_interleaved = w_up.reshape(HC, HS, R).permute(1, 0, 2).contiguous().reshape(K, R)
        if config.preshuffle:
            self.w_down = preshuffle_weight(w_down)
            self.w_up = preshuffle_weight(up_interleaved)
        else:
            self.w_down = w_down.contiguous().view(-1)
            self.w_up = up_interleaved.view(-1)
        padded = (rows + config.block_m - 1) // config.block_m * config.block_m
        self.partial = torch.empty((config.split_k * padded * R,), dtype=torch.float32, device=self.device)
        self.output = torch.empty((rows, HS), dtype=self.dtype, device=self.device)
        self.down = self.up = None
        if rows:
            x = torch.empty((rows, K), dtype=self.dtype, device=self.device)
            ld, lu = _launchers(rows, self.dtype, config)
            stream = torch.cuda.current_stream(self.device)
            self.down = flyc.compile(ld, x.view(-1), self.w_down, self.partial, stream)
            self.up = flyc.compile(lu, x.view(-1), self.w_up, self.partial, self.output.view(-1), stream)

    def __call__(self, x):
        if x.shape != (self.rows, K) or x.dtype != self.dtype or x.device != self.device:
            raise ValueError("input must match the prepared rows, dtype and device")
        if not x.is_contiguous():
            raise ValueError("input must be contiguous")
        if self.rows:
            stream = torch.cuda.current_stream(self.device)
            self.down(x.view(-1), self.w_down, self.partial, stream)
            self.up(x.view(-1), self.w_up, self.partial, self.output.view(-1), stream)
        return self.output
