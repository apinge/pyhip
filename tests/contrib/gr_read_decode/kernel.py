# SPDX-License-Identifier: MIT
"""Selected H64 GR read decode for T=1..32; test-local, no model dependency.

Extracted from pyhip be1555f (h64_layout.py + large_down.py). Only the
selected two-launch path remains. Dimensions and launch choices are fixed.
"""

from functools import cache
import warnings

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
import torch

HC, HS, K, R = 4, 2560, 10240, 320
LOG2E = 1.4426950408889634


def preshuffle_weight(weight):
    n, k = weight.shape
    return weight.detach().reshape(n // 16, 16, k // 32, 4, 8).permute(
        0, 2, 3, 1, 4
    ).contiguous().view(-1)


def prepare_weights(w_down, w_up):
    """Pack original logical matrices once, using the prefill H64 layout."""
    if w_down.shape != (R, K) or w_up.shape != (K, R):
        raise ValueError("expected W_down[320,10240] and W_up[10240,320]")
    if w_down.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
        raise ValueError("weights must be BF16")
    if w_down.device != w_up.device:
        raise ValueError("weights must share a device")
    interleaved = w_up.detach().reshape(HC, HS // 64, 2, 4, 2, 4, R).permute(
        1, 0, 2, 4, 3, 5, 6
    ).contiguous().reshape(K, R)
    return preshuffle_weight(w_down), preshuffle_weight(interleaved)


@cache
def down_launcher(rows):
    (bm, bk, waves) = (16 if rows <= 16 else 32, 128, 4)
    (split, dn) = (4, 16)
    prefetch_unroll = 2 if rows <= 25 else 1
    iterations = K // waves // bk // split
    padded_rows = (rows + bm - 1) // bm * bm

    @fx.struct
    class DownShared:
        partials: fx.Array[fx.Float32, bm * dn * waves, 16]

    @flyc.kernel
    def down_wave_splitk_pipeline(X: fx.Tensor, W: fx.Tensor, A: fx.Tensor):
        tid = fx.thread_idx.x
        (lane, wave) = (tid % 64, tid // 64)
        (im, jn, sk) = fx.block_idx
        x = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1))), max_size=False)
        w_layout = fx.make_layout(((16, R // 16), (8, 4, K // 32)), ((8, 16 * K), (1, 128, 512)))
        w = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(W), w_layout), max_size=False)
        shared = fx.SharedAllocator().allocate(DownShared).peek()
        partials = shared.partials.view(fx.make_layout((bm, dn, waves), (dn, 1, bm * dn)))
        a_tile = fx.flat_divide(w, fx.make_tile(dn, bk))[None, None, jn, None]
        b_tile = fx.flat_divide(x, fx.make_tile(bm, bk))[None, None, im, None]
        c_tile = fx.select(partials[None, None, wave], [1, 0])
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tiled = fx.make_tiled_mma(mma, fx.make_layout((1, 1, 1), (0, 0, 0)), (None, None, fx.make_layout((4, 4, 2), (1, 8, 4))))
        thr = tiled.thr_slice(lane)
        copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        ca = fx.make_tiled_copy_A(copy_ab, tiled).get_slice(lane)
        cb = fx.make_tiled_copy_B(copy_ab, tiled).get_slice(lane)
        cc = fx.make_tiled_copy_C(copy_c, tiled).get_slice(lane)
        fa = thr.make_fragment_A(a_tile[None, None, 0])
        fb = thr.make_fragment_B(b_tile[None, None, 0])
        fc = thr.make_fragment_C(c_tile)
        (ga, gb) = (ca.partition_S(a_tile), cb.partition_S(b_tile))
        (ra, rb) = (ca.retile(fa), cb.retile(fb))
        fc.fill(0)
        next_a = thr.make_fragment_A(a_tile[None, None, 0])
        next_b = thr.make_fragment_B(b_tile[None, None, 0])
        (next_ra, next_rb) = (ca.retile(next_a), cb.retile(next_b))
        first = sk * (K // split // bk) + wave
        fx.copy(copy_ab, ga[None, None, None, first], ra)
        fx.copy(copy_ab, gb[None, None, None, first], rb)
        for (ki, state) in range(fx.Index(1), fx.Index(iterations), fx.Index(prefetch_unroll), init=[fa.load(), fb.load(), fc.load()]):
            fa.store(state[0])
            fb.store(state[1])
            fc.store(state[2])
            kt = sk * (K // split // bk) + fx.Int32(ki) * waves + wave
            fx.copy(copy_ab, ga[None, None, None, kt], next_ra)
            fx.copy(copy_ab, gb[None, None, None, kt], next_rb)
            fx.gemm(mma, fc, fa, fb, fc)
            if fx.const_expr(prefetch_unroll == 2):
                second = kt + waves
                fx.copy(copy_ab, ga[None, None, None, second], ra)
                fx.copy(copy_ab, gb[None, None, None, second], rb)
                fx.gemm(mma, fc, next_a, next_b, fc)
                (carried_a, carried_b) = (fa.load(), fb.load())
            else:
                (carried_a, carried_b) = (next_a.load(), next_b.load())
            result = (yield [carried_a, carried_b, fc.load()])
        fa.store(result[0])
        fb.store(result[1])
        fc.store(result[2])
        fx.gemm(mma, fc, fa, fb, fc)
        fx.copy(copy_c, cc.retile(fc), cc.partition_D(c_tile))
        fx.gpu.barrier()
        out = fx.make_view(fx.get_iter(A), fx.make_layout((padded_rows, R, split), (R, 1, padded_rows * R)))
        for i in range_constexpr((bm * dn + waves * 64 - 1) // (waves * 64)):
            index = tid + i * waves * 64
            if index < bm * dn:
                (row, col) = (index // dn, index % dn)
                total = fx.Float32(0.0)
                for s in range_constexpr(waves):
                    total = total + fx.memref_load(partials, (row, col, s))
                fx.memref_store(total, out, (im * bm + row, jn * dn + col, sk))

    @flyc.jit
    def launch(X: fx.Tensor, W: fx.Tensor, A: fx.Tensor, stream: fx.Stream):
        down_wave_splitk_pipeline(X, W, A).launch(grid=(padded_rows // bm, R // dn, split), block=(waves * 64, 1, 1), stream=stream)
    return launch


@cache
def up_launcher(rows):
    (bm, un, split, waves) = (16, 128, 4, 4)
    bk = 32 if 9 <= rows <= 16 or 29 <= rows <= 31 else 160
    skip_padding = not 9 <= rows <= 16
    preload_weights = rows <= 16
    (hn, hidden_stride) = (un // HC, R + 4)
    padded_rows = (rows + bm - 1) // bm * bm
    (threads, elem) = (waves * 64, fx.BFloat16)

    @fx.struct
    class UpShared:
        hidden: fx.Array[elem, bm * hidden_stride, 16]
        hidden_low: fx.Array[elem, bm * hidden_stride, 16]
        logits: fx.Array[fx.Float32, bm * un, 16]

    @flyc.kernel
    def up_gate_h64(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        (lane, wave) = (tid % 64, tid // 64)
        (im, jn, _) = fx.block_idx
        shared = fx.SharedAllocator().allocate(UpShared).peek()
        h = shared.hidden.view(fx.make_layout((bm, R), (hidden_stride, 1)))
        c = shared.logits.view(fx.make_layout((bm, un), (un, 1)))
        c_tile = c
        p = fx.rocdl.make_buffer_tensor(P, max_size=False)
        p4 = fx.flat_divide(p, fx.make_tile(4))
        h4 = shared.hidden.view(fx.make_layout((4, (R // 4, bm)), (1, (4, hidden_stride))))
        h_low = shared.hidden_low.view(fx.make_layout((bm, R), (hidden_stride, 1)))
        h_low4 = shared.hidden_low.view(fx.make_layout((4, (R // 4, bm)), (1, (4, hidden_stride))))
        copy_p = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        copy_h = fx.make_copy_atom(fx.UniversalCopy64b(), elem)
        fp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        fh = fx.make_rmem_tensor(fx.make_layout(4, 1), elem)
        w_layout = fx.make_layout(((16, hn // 16, HC, 64 // hn, HS // 64), (8, 4, R // 32)), ((8, 16 * R, 64 * R, hn * R, 64 * HC * R), (1, 128, 512)))
        w = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(W), w_layout), max_size=False)
        a_tile = fx.flat_divide(h, fx.make_tile(bm, bk))[None, None, 0, None]
        a_low_tile = fx.flat_divide(h_low, fx.make_tile(bm, bk))[None, None, 0, None]
        b_tile = fx.flat_divide(w, fx.make_tile(un, bk))[None, None, jn, None]
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, elem))
        wave_layout = fx.make_layout((1, waves, 1), (0, 1, 0))
        tiled = fx.make_tiled_mma(mma, wave_layout, (None, None, fx.make_layout((4, 4, 2), (1, 8, 4))))
        thr = tiled.thr_slice(tid)
        copy_a = fx.make_copy_atom(fx.UniversalCopy64b(), elem)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem)
        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        ca = fx.make_tiled_copy_A(copy_a, tiled).get_slice(tid)
        cb = fx.make_tiled_copy_B(copy_b, tiled).get_slice(tid)
        fa = thr.make_fragment_A(a_tile[None, None, 0])
        fb = thr.make_fragment_B(b_tile[None, None, 0])
        cc = fx.make_tiled_copy_C(copy_c, tiled).get_slice(tid)
        fc = thr.make_fragment_C(c_tile)
        fc.fill(0)
        (ga, gb) = (ca.partition_S(a_tile), cb.partition_S(b_tile))
        ga_low = ca.partition_S(a_low_tile)
        (ra, rb) = (ca.retile(fa), cb.retile(fb))
        if fx.const_expr(preload_weights):
            weight_fragments = [thr.make_fragment_B(b_tile[None, None, 0]) for _ in range_constexpr(R // bk)]
            for ki in range_constexpr(R // bk):
                fx.copy(copy_b, gb[None, None, None, ki], cb.retile(weight_fragments[ki]))
            fx.rocdl.sched_barrier(0)
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
            values = []
            for j in range_constexpr(4):
                exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-z[j] * LOG2E)))
                inverse = fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent)))
                values.append(z[j] * inverse)
            activated = fx.Vector.from_elements(values, fx.Float32)
            fh.store(activated.to(elem))
            fx.copy(copy_h, fh, h4[None, ix])
            low = activated - activated.to(elem).to(fx.Float32)
            fh.store(low.to(elem))
            fx.copy(copy_h, fh, h_low4[None, ix])
        fx.gpu.barrier()
        for ki in range_constexpr(R // bk):
            fx.copy(copy_a, ga[None, None, None, ki], ra)
            if fx.const_expr(preload_weights):
                weight_fragment = weight_fragments[ki]
            else:
                fx.copy(copy_b, gb[None, None, None, ki], rb)
                weight_fragment = fb
            fx.gemm(mma, fc, fa, weight_fragment, fc)
            fx.copy(copy_a, ga_low[None, None, None, ki], ra)
            fx.gemm(mma, fc, fa, weight_fragment, fc)
        fx.copy(copy_c, cc.retile(fc), cc.partition_D(c_tile))
        fx.gpu.barrier()
        x = fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1)))
        y = fx.make_view(fx.get_iter(Y), fx.make_layout((rows, HS), (HS, 1)))
        for i in range_constexpr(bm * hn // threads):
            index = tid + i * threads
            row_local = index // hn
            col_local = index % hn
            row = im * bm + row_local
            v_h = jn % (64 // hn) * hn + col_local
            col = jn // (64 // hn) * 64 + v_h // 32 * 32 + v_h // 4 % 4 * 8 + v_h // 16 % 2 * 4 + v_h % 4
            if row < rows:
                total = fx.Float32(0.0)
                for g in range_constexpr(HC):
                    ci = col_local + g * hn
                    logit = fx.memref_load(c, (row_local, ci))
                    exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-logit * LOG2E)))
                    gate = fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent)))
                    value = fx.memref_load(x, (row, g * HS + col)).to(fx.Float32)
                    total = total + gate * value
                fx.memref_store((total * 0.25).to(elem), y, (row, col))

    @flyc.jit
    def launch_up(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        up_gate_h64(X, W, P, Y).launch(grid=(padded_rows // bm, K // un, 1), block=(threads, 1, 1), stream=stream)
    return launch_up


@cache
def pair_launcher(rows):
    down, up = down_launcher(rows), up_launcher(rows)

    @flyc.jit
    def launch(X: fx.Tensor, WD: fx.Tensor, WU: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        if fx.const_expr(rows > 0):
            down(X, WD, P, stream)
            up(X, WU, P, Y, stream)

    return launch


class GRReadDecode:
    """Test helper: retain packed weights, own scratch/output, compile once."""

    def __init__(self, rows, packed_down, packed_up):
        if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 32:
            raise ValueError("decode supports T=1..32")
        for weight in (packed_down, packed_up):
            if weight.shape != (K * R,) or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
                raise ValueError("expected flat contiguous BF16 packed weights")
            if weight.data_ptr() % 16:
                raise ValueError("packed weight base must be 16-byte aligned")
        if not packed_down.is_cuda or packed_down.device != packed_up.device or torch.version.hip is None:
            raise ValueError("packed weights must share a ROCm device")
        self.rows, self.device = rows, packed_down.device
        props = torch.cuda.get_device_properties(self.device)
        if props.gcnArchName.split(":")[0] != "gfx942":
            warnings.warn(f"GR read decode was tuned on gfx942; running on {props.gcnArchName}", RuntimeWarning)
        self.w_down, self.w_up = packed_down, packed_up
        self.padded_rows = (rows + 15) // 16 * 16
        self.partial = torch.empty(4 * self.padded_rows * R, dtype=torch.float32, device=self.device)
        self.output = torch.empty((rows, HS), dtype=torch.bfloat16, device=self.device)
        with torch.cuda.device(self.device):
            # flyc.compile executes once: all example buffers must be valid.
            x = torch.zeros(rows * K, dtype=torch.bfloat16, device=self.device)
            self.dispatch = flyc.compile(
                pair_launcher(rows), x, self.w_down, self.w_up, self.partial,
                self.output.view(-1), torch.cuda.current_stream(self.device),
            )

    def __call__(self, x):
        if x.shape != (self.rows, K) or x.dtype != torch.bfloat16 or x.device != self.device:
            raise ValueError("input must match the prepared rows, dtype and device")
        if not x.is_contiguous():
            raise ValueError("input must be contiguous")
        self.dispatch(x.view(-1), self.w_down, self.w_up, self.partial, self.output.view(-1),
                      torch.cuda.current_stream(self.device))
        return self.output
