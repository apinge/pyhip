"""Experimental up epilogue using wave-local stream gathers, still two launches."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

if __package__:
    from .kernel import HC, HS, LOG2E, GRRead, K, R
else:
    from kernel import HC, HS, LOG2E, GRRead, K, R


@cache
def _register_up_launcher(rows, config, ordered):
    bm, un, bk, waves = config.block_m, config.up_n, config.block_k, config.waves
    split = config.split_k
    skip_padding, preshuffle = config.skip_padding, config.preshuffle
    compensate_hidden, fast_math, down_mode = config.compensate_hidden, config.fast_math, config.down_mode
    padded_rows = (rows + bm - 1) // bm * bm
    threads = waves * 64
    elem = fx.BFloat16

    @fx.struct
    class Shared:
        hidden: fx.Array[elem, bm * R, 16]
        hidden_low: fx.Array[elem, bm * R if compensate_hidden else 1, 16]

    @flyc.kernel
    def up_gate_register(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        im, jn, _ = fx.block_idx
        shared = fx.SharedAllocator().allocate(Shared).peek()
        h = shared.hidden.view(fx.make_layout((bm, R), (R, 1)))
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
        for i in range_constexpr(bm * R // (threads * 4)):
            ix = tid + i * threads
            if fx.const_expr(down_mode == "partial"):
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
            else:
                fx.copy(copy_p, p4[None, im * bm * (R // 4) + ix], fp)
                activated = fp.load()
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
        ca = fx.make_tiled_copy_A(copy_a, tiled).get_slice(tid)
        cb = fx.make_tiled_copy_B(copy_b, tiled).get_slice(tid)
        fa = thr.make_fragment_A(a_tile[None, None, 0])
        fb = thr.make_fragment_B(b_tile[None, None, 0])
        # MFMA C: row=4*(lane//16)+v+16*mr; col=lane%16+16*wave+16*waves*nr.
        fc = fx.make_rmem_tensor(
            fx.make_layout(((4, 1), bm // 16, un // (16 * waves)), ((1, 0), 4, 4 * (bm // 16))), fx.Float32
        )
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

        x = fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1)))
        y = fx.make_view(fx.get_iter(Y), fx.make_layout((rows, HS), (HS, 1)))
        for nr in range_constexpr(un // (16 * waves)):
            n = jn * un + wave * 16 + lane % 16 + nr * waves * 16
            col, stream_id = n // HC, lane % HC
            for mr in range_constexpr(bm // 16):
                for v in range_constexpr(4):
                    row = im * bm + mr * 16 + (lane // 16) * 4 + v
                    # A row predicate is uniform within every four-lane stream group.
                    if row < rows:
                        logit = fx.memref_load(fc, ((v, 0), mr, nr))
                        if fx.const_expr(fast_math):
                            exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-logit * LOG2E)))
                            gate = fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent)))
                        else:
                            gate = 1.0 / (1.0 + (-logit * LOG2E).exp2())
                        piece = gate * fx.memref_load(x, (row, stream_id * HS + col)).to(fx.Float32)
                        if fx.const_expr(ordered):
                            total = fx.Float32(0.0)
                            bits = fx.arith.bitcast(fx.T.i32, fx.arith.unwrap(piece))
                            for g in range_constexpr(HC):
                                source = fx.Int32((lane // HC * HC + g) * 4)
                                peer = fx.rocdl.ds_bpermute(fx.T.i32, fx.arith.unwrap(source), fx.arith.unwrap(bits))
                                total = total + fx.Float32(fx.arith.bitcast(fx.T.f32, peer))
                        else:
                            total = piece
                            for delta in range_constexpr(2):
                                source = fx.Int32((lane ^ (1 << delta)) * 4)
                                bits = fx.arith.bitcast(fx.T.i32, fx.arith.unwrap(total))
                                peer = fx.rocdl.ds_bpermute(fx.T.i32, fx.arith.unwrap(source), fx.arith.unwrap(bits))
                                total = total + fx.Float32(fx.arith.bitcast(fx.T.f32, peer))
                        if stream_id == 0:
                            fx.memref_store((total * 0.25).to(elem), y, (row, col))

    @flyc.jit
    def launch(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        up_gate_register(X, W, P, Y).launch(grid=(padded_rows // bm, K // un, 1), block=(threads, 1, 1), stream=stream)

    return launch


class RegisterGateGRRead(GRRead):
    def __init__(self, rows, w_down, w_up, config=None, ordered=True):
        super().__init__(rows, w_down, w_up, config)
        self.ordered_gate_reduce = ordered
        if rows:
            x = torch.empty((rows * K,), device=self.device, dtype=self.dtype)
            launch = _register_up_launcher(rows, self.config, ordered)
            self.up = flyc.compile(
                launch, x, self.w_up, self.partial, self.output.view(-1), torch.cuda.current_stream(self.device)
            )
