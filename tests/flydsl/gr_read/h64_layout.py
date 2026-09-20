"""Decode candidate consuming the prefill H64/stream packed weights.

Weights are supplied prepacked and retained by reference. Up uses stream-plane
weight tiles, 128-bit loads and a per-T preload/padding schedule. Down and
FP32/high-low semantics are unchanged. This does not replace the old-layout entry.
"""

from dataclasses import replace
from functools import cache
import warnings

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

if __package__:
    from . import prefetch_up as reference_up
    from .combined_host import _check_input
    from .large_down import default_config as large_config, down_launcher
    from .small_batch import default_config as small_config
else:
    import prefetch_up as reference_up
    from combined_host import _check_input
    from large_down import default_config as large_config, down_launcher
    from small_batch import default_config as small_config

HC, HS, K, R, LOG2E = 4, 2560, 10240, 320, reference_up.LOG2E
LAYOUT_ID = "h64_stream_half_lane_preshuffle_v2"


def prepare_up_weight(w_up):
    if w_up.shape != (K, R) or w_up.dtype != torch.bfloat16:
        raise ValueError("expected original BF16 W_up[10240,320]")
    interleaved = w_up.detach().reshape(HC, HS // 64, 2, 4, 2, 4, R).permute(
        1, 0, 2, 4, 3, 5, 6
    ).contiguous().reshape(K, R)
    return reference_up.preshuffle_weight(interleaved)


def prepare_weights(w_down, w_up):
    if w_down.shape != (R, K) or w_down.dtype != torch.bfloat16:
        raise ValueError("expected original BF16 W_down[320,10240]")
    if w_down.device != w_up.device:
        raise ValueError("weights must share a device")
    return reference_up.preshuffle_weight(w_down.detach()), prepare_up_weight(w_up)


def default_configs(rows):
    if not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 32:
        raise ValueError("H64 decode supports T=1..32")
    base = replace(reference_up.default_config(rows), hidden_pad=4, prefetch_low=False)
    if rows <= 16:
        config = small_config(rows)
        down, up = config.down, config.up_config(base)
    else:
        down, up = large_config(rows), replace(base, down_mode="partial", down_n=64, split_k=4)
    # Down overwrites internal padding with zero; full P loads remove divergent
    # load predicates for T9..15. Small T still benefits from skipping those loads.
    up = replace(up, up_n=128, block_k=32 if 9 <= rows <= 16 or 29 <= rows <= 31 else 160,
                 skip_padding=not 9 <= rows <= 16)
    return down, up


@cache
def up_launcher(rows, config, strategy="plane", weight_copy_bits=128, b_first=False, preload_weights=False):
    config.validate()
    if not config.preshuffle or not config.compensate_hidden or config.down_mode != "partial":
        raise ValueError("H64 up requires preshuffle, linear partials and high/low compensation")
    bm, un, bk, split, waves = config.block_m, config.up_n, config.block_k, config.split_k, config.waves
    if strategy not in ("remap", "plane", "register") or weight_copy_bits not in (64, 128):
        raise ValueError("invalid H64 strategy or weight copy width")
    hn = un // HC
    if strategy != "remap" and hn not in (16, 32, 64):
        raise ValueError("plane/register up requires BN64/128/256")
    if strategy == "register" and hn != 16 * waves:
        raise ValueError("register up requires BN=64*waves to keep all streams in each lane")
    if strategy == "register" and not config.fast_math:
        raise ValueError("experimental register epilogue requires fast_math=True")
    register_gate = strategy == "register"
    skip_padding, preshuffle, fast_math = config.skip_padding, config.preshuffle, config.fast_math
    compensate_hidden, prefetch_low = config.compensate_hidden, config.prefetch_low
    down_mode, hidden_stride = config.down_mode, R + config.hidden_pad
    padded_rows = (rows + bm - 1) // bm * bm
    threads, elem = waves * 64, fx.BFloat16

    @fx.struct
    class UpShared:
        hidden: fx.Array[elem, bm * hidden_stride, 16]
        hidden_low: fx.Array[elem, bm * hidden_stride, 16]
        if not register_gate:
            logits: fx.Array[fx.Float32, bm * un, 16]

    @flyc.kernel
    def up_gate_h64(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        lane, wave = tid % 64, tid // 64
        im, jn, _ = fx.block_idx
        shared = fx.SharedAllocator().allocate(UpShared).peek()
        h = shared.hidden.view(fx.make_layout((bm, R), (hidden_stride, 1)))
        if fx.const_expr(not register_gate):
            c = shared.logits.view(fx.make_layout((bm, un), (un, 1)))
            c_tile = fx.select(c, [1, 0]) if b_first else c
        p = fx.rocdl.make_buffer_tensor(P, max_size=False)
        p4 = fx.flat_divide(p, fx.make_tile(4))
        h4 = shared.hidden.view(fx.make_layout((4, (R // 4, bm)), (1, (4, hidden_stride))))
        if fx.const_expr(compensate_hidden):
            h_low = shared.hidden_low.view(fx.make_layout((bm, R), (hidden_stride, 1)))
            h_low4 = shared.hidden_low.view(fx.make_layout((4, (R // 4, bm)), (1, (4, hidden_stride))))
        copy_p = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        copy_h = fx.make_copy_atom(fx.UniversalCopy64b(), elem)
        fp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        fh = fx.make_rmem_tensor(fx.make_layout(4, 1), elem)
        if fx.const_expr(preshuffle):
            if fx.const_expr(strategy == "remap"):
                # Keep logical n=h*C+c while reading the shared prefill H64 layout.
                w_layout = fx.make_layout(
                    ((HC, 4, 2, 4, 2, HS // 64), (8, 4, R // 32)),
                    ((64 * R, 8, 16 * R, 32, 32 * R, 64 * HC * R), (1, 128, 512)),
                )
            else:
                # Each logical tile has four planes of hn adjacent physical rows.
                w_layout = fx.make_layout(
                    ((16, hn // 16, HC, 64 // hn, HS // 64), (8, 4, R // 32)),
                    ((8, 16 * R, 64 * R, hn * R, 64 * HC * R), (1, 128, 512)),
                )
        else:
            w_layout = fx.make_layout((K, R), (R, 1))
        w = fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(W), w_layout), max_size=False)
        a_tile = fx.flat_divide(h, fx.make_tile(bm, bk))[None, None, 0, None]
        if fx.const_expr(compensate_hidden):
            a_low_tile = fx.flat_divide(h_low, fx.make_tile(bm, bk))[None, None, 0, None]
        b_tile = fx.flat_divide(w, fx.make_tile(un, bk))[None, None, jn, None]
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, elem))
        wave_layout = fx.make_layout((waves, 1, 1), (1, 0, 0)) if b_first else fx.make_layout((1, waves, 1), (0, 1, 0))
        if fx.const_expr(weight_copy_bits == 128):
            tiled = fx.make_tiled_mma(mma, wave_layout,
                                      (None, None, fx.make_layout((4, 4, 2), (1, 8, 4))))
        else:
            tiled = fx.make_tiled_mma(mma, wave_layout)
        thr = tiled.thr_slice(tid)
        copy_a = fx.make_copy_atom(fx.UniversalCopy64b(), elem)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy128b() if weight_copy_bits == 128 else fx.rocdl.BufferCopy64b(), elem)
        copy_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        if fx.const_expr(b_first):
            ca = fx.make_tiled_copy_B(copy_a, tiled).get_slice(tid)
            cb = fx.make_tiled_copy_A(copy_b, tiled).get_slice(tid)
            fa = thr.make_fragment_B(a_tile[None, None, 0])
            fb = thr.make_fragment_A(b_tile[None, None, 0])
        else:
            ca = fx.make_tiled_copy_A(copy_a, tiled).get_slice(tid)
            cb = fx.make_tiled_copy_B(copy_b, tiled).get_slice(tid)
            fa = thr.make_fragment_A(a_tile[None, None, 0])
            fb = thr.make_fragment_B(b_tile[None, None, 0])
        cc = fx.make_tiled_copy_C(copy_c, tiled).get_slice(tid)
        if fx.const_expr(register_gate):
            if fx.const_expr(b_first):
                fc = fx.make_rmem_tensor(
                    fx.make_layout(((4, 1), un // (16 * waves), bm // 16),
                                   ((1, 0), 4, 4 * (un // (16 * waves)))), fx.Float32)
            else:
                fc = fx.make_rmem_tensor(
                    fx.make_layout(((4, 1), bm // 16, un // (16 * waves)),
                                   ((1, 0), 4, 4 * (bm // 16))), fx.Float32)
        else:
            fc = thr.make_fragment_C(c_tile)
        if fx.const_expr(compensate_hidden and prefetch_low):
            fa_low = thr.make_fragment_B(a_tile[None, None, 0]) if b_first else thr.make_fragment_A(a_tile[None, None, 0])
        fc.fill(0)
        ga, gb = ca.partition_S(a_tile), cb.partition_S(b_tile)
        if fx.const_expr(compensate_hidden):
            ga_low = ca.partition_S(a_low_tile)
        ra, rb = ca.retile(fa), cb.retile(fb)
        if fx.const_expr(compensate_hidden and prefetch_low):
            ra_low = ca.retile(fa_low)
        if fx.const_expr(preload_weights):
            weight_fragments = [thr.make_fragment_A(b_tile[None, None, 0]) if b_first
                                else thr.make_fragment_B(b_tile[None, None, 0])
                                for _ in range_constexpr(R // bk)]
            for ki in range_constexpr(R // bk):
                fx.copy(copy_b, gb[None, None, None, ki], cb.retile(weight_fragments[ki]))
            fx.rocdl.sched_barrier(0)

        # Each up CTA reconstructs its small low-rank input from ordered partials.
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

        for ki in range_constexpr(R // bk):
            fx.copy(copy_a, ga[None, None, None, ki], ra)
            if fx.const_expr(compensate_hidden and prefetch_low):
                fx.copy(copy_a, ga_low[None, None, None, ki], ra_low)
            if fx.const_expr(preload_weights):
                weight_fragment = weight_fragments[ki]
            else:
                fx.copy(copy_b, gb[None, None, None, ki], rb)
                weight_fragment = fb
            if fx.const_expr(b_first):
                fx.gemm(mma, fc, weight_fragment, fa, fc)
            else:
                fx.gemm(mma, fc, fa, weight_fragment, fc)
            if fx.const_expr(compensate_hidden):
                if fx.const_expr(prefetch_low):
                    if fx.const_expr(b_first):
                        fx.gemm(mma, fc, weight_fragment, fa_low, fc)
                    else:
                        fx.gemm(mma, fc, fa_low, weight_fragment, fc)
                else:
                    fx.copy(copy_a, ga_low[None, None, None, ki], ra)
                    if fx.const_expr(b_first):
                        fx.gemm(mma, fc, weight_fragment, fa, fc)
                    else:
                        fx.gemm(mma, fc, fa, weight_fragment, fc)
        if fx.const_expr(not register_gate):
            fx.copy(copy_c, cc.retile(fc), cc.partition_D(c_tile))
            fx.gpu.barrier()

        x = fx.make_view(fx.get_iter(X), fx.make_layout((rows, K), (K, 1)))
        y = fx.make_view(fx.get_iter(Y), fx.make_layout((rows, HS), (HS, 1)))
        if fx.const_expr(register_gate and b_first):
            v_h = (jn % (64 // hn)) * hn + wave * 16 + lane // 16 * 4
            col = (jn // (64 // hn)) * 64 + v_h // 32 * 32 + (v_h // 4 % 4) * 8 + (v_h // 16 % 2) * 4 + v_h % 4
            x4 = fx.flat_divide(fx.rocdl.make_buffer_tensor(X, max_size=False), fx.make_tile(4))
            y4 = fx.flat_divide(fx.rocdl.make_buffer_tensor(Y, max_size=False), fx.make_tile(4))
            xy_copy = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), elem)
            xy_frag = fx.make_rmem_tensor(fx.make_layout(4, 1), elem)
            for mr in range_constexpr(bm // 16):
                row = im * bm + mr * 16 + lane % 16
                if row < rows:
                    total = fx.Vector.filled(4, 0.0, fx.Float32)
                    for g in range_constexpr(HC):
                        fx.copy(xy_copy, x4[None, (row * K + g * HS + col) // 4], xy_frag)
                        gates = []
                        for v in range_constexpr(4):
                            logit = fx.memref_load(fc, ((v, 0), g, mr))
                            exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-logit * LOG2E)))
                            gates.append(fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent))))
                        total = total + fx.Vector.from_elements(gates, fx.Float32) * xy_frag.load().to(fx.Float32)
                    xy_frag.store((total * 0.25).to(elem))
                    fx.copy(xy_copy, xy_frag, y4[None, (row * HS + col) // 4])
        elif fx.const_expr(register_gate):
            v_h = (jn % (64 // hn)) * hn + wave * 16 + lane % 16
            col = (jn // (64 // hn)) * 64 + v_h // 32 * 32 + (v_h // 4 % 4) * 8 + (v_h // 16 % 2) * 4 + v_h % 4
            for mr in range_constexpr(bm // 16):
                for v in range_constexpr(4):
                    row = im * bm + mr * 16 + lane // 16 * 4 + v
                    if row < rows:
                        total = fx.Float32(0.0)
                        for g in range_constexpr(HC):
                            logit = fx.memref_load(fc, ((v, 0), mr, g))
                            exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-logit * LOG2E)))
                            gate = fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent)))
                            value = fx.memref_load(x, (row, g * HS + col)).to(fx.Float32)
                            total = total + gate * value
                        fx.memref_store((total * 0.25).to(elem), y, (row, col))
        else:
            for i in range_constexpr(bm * hn // threads):
                index = tid + i * threads
                row_local = index // hn
                col_local = index % hn
                row = im * bm + row_local
                if fx.const_expr(strategy == "remap"):
                    col = jn * hn + col_local
                else:
                    v_h = (jn % (64 // hn)) * hn + col_local
                    col = (jn // (64 // hn)) * 64 + v_h // 32 * 32 + (v_h // 4 % 4) * 8 + (v_h // 16 % 2) * 4 + v_h % 4
                if row < rows:
                    total = fx.Float32(0.0)
                    for g in range_constexpr(HC):
                        ci = col_local * HC + g if strategy == "remap" else col_local + g * hn
                        logit = fx.memref_load(c, (row_local, ci))
                        if fx.const_expr(fast_math):
                            exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-logit * LOG2E)))
                            gate = fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent)))
                        else:
                            gate = 1.0 / (1.0 + (-logit * LOG2E).exp2())
                        value = fx.memref_load(x, (row, g * HS + col)).to(fx.Float32)
                        total = total + gate * value
                    fx.memref_store((total * 0.25).to(elem), y, (row, col))


    @flyc.jit
    def launch_up(X: fx.Tensor, W: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        up_gate_h64(X, W, P, Y).launch(
            grid=(padded_rows // bm, K // un, 1), block=(threads, 1, 1), stream=stream
        )

    return launch_up


@cache
def pair_launcher(rows, down_config, up_config, strategy="plane", weight_copy_bits=128, b_first=False, preload_weights=False):
    down = down_launcher(rows, down_config)
    up = up_launcher(rows, up_config, strategy, weight_copy_bits, b_first, preload_weights)

    @flyc.jit
    def launch(X: fx.Tensor, WD: fx.Tensor, WU: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        if fx.const_expr(rows > 0):
            down(X, WD, P, stream)
            up(X, WU, P, Y, stream)

    return launch


class H64GRRead:
    """Prepared decoder owning P/Y, sharing one existing packed WD/WU pair."""

    def __init__(self, rows, packed_down, packed_up, *, up_config=None, strategy="plane", weight_copy_bits=128, b_first=False, preload_weights=None):
        self.down_config, selected_up = default_configs(rows)
        preload_weights = rows <= 16 if preload_weights is None else preload_weights
        self.up_config = selected_up if up_config is None else up_config
        self.strategy, self.weight_copy_bits = strategy, weight_copy_bits
        self.b_first = b_first
        self.preload_weights = preload_weights
        self.up_config.validate()
        if self.up_config.block_m != 16 or self.up_config.split_k != 4:
            raise ValueError("selected down requires up BM16 and split4")
        for weight in (packed_down, packed_up):
            if weight.shape != (K * R,) or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
                raise ValueError("expected contiguous BF16 flat packed weights")
            if weight.data_ptr() % 16:
                raise ValueError("packed weight base must be 16-byte aligned")
        if not packed_down.is_cuda or packed_down.device != packed_up.device or torch.version.hip is None:
            raise ValueError("packed weights must share a ROCm device")
        self.rows, self.device, self.dtype = rows, packed_down.device, packed_down.dtype
        props = torch.cuda.get_device_properties(self.device)
        if props.gcnArchName.split(":")[0] != "gfx942":
            warnings.warn(f"H64 decode is being validated on gfx942, not {props.gcnArchName}", RuntimeWarning)
        self.w_down, self.w_up = packed_down, packed_up
        self.weight_layout = LAYOUT_ID
        padded = 16 if rows <= 16 else 32
        self.partial = torch.empty(4 * padded * R, dtype=torch.float32, device=self.device)
        self.output = torch.empty((rows, HS), dtype=self.dtype, device=self.device)
        with torch.cuda.device(self.device):
            x = torch.zeros(rows * K, dtype=self.dtype, device=self.device)
            stream = torch.cuda.current_stream(self.device)
            self.down = flyc.compile(down_launcher(rows, self.down_config), x, self.w_down, self.partial, stream)
            self.up = flyc.compile(up_launcher(rows, self.up_config, strategy, weight_copy_bits, b_first, preload_weights), x, self.w_up, self.partial, self.output.view(-1), stream)
            self.dispatch = flyc.compile(
                pair_launcher(rows, self.down_config, self.up_config, strategy, weight_copy_bits, b_first, preload_weights),
                x, self.w_down, self.w_up, self.partial, self.output.view(-1), stream,
            )

    def run_down(self, x):
        _check_input(self, x)
        self.down(x.view(-1), self.w_down, self.partial, torch.cuda.current_stream(self.device))

    def run_up(self, x):
        _check_input(self, x)
        self.up(x.view(-1), self.w_up, self.partial, self.output.view(-1), torch.cuda.current_stream(self.device))

    def __call__(self, x):
        _check_input(self, x)
        self.dispatch(x.view(-1), self.w_down, self.w_up, self.partial, self.output.view(-1),
                      torch.cuda.current_stream(self.device))
        return self.output
