"""Experimental down-partial / reduce-SiLU / up path; not a default dispatch."""

from dataclasses import replace
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

if __package__:
    from .kernel import LOG2E, Config, GRRead, R, _launchers
else:
    from kernel import LOG2E, Config, GRRead, R, _launchers


@cache
def _reduce_launcher(rows, padded_rows, split, threads, vec):
    @flyc.kernel
    def reduce_silu(P: fx.Tensor, A: fx.Tensor):
        index = fx.block_idx.x * threads + fx.thread_idx.x
        p = fx.rocdl.make_buffer_tensor(P, max_size=False)
        a = fx.rocdl.make_buffer_tensor(A, max_size=False)
        pv = fx.flat_divide(p, fx.make_tile(vec))
        av = fx.flat_divide(a, fx.make_tile(vec))
        if fx.const_expr(vec == 4):
            atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        elif fx.const_expr(vec == 2):
            atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Float32)
        else:
            atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        fragment = fx.make_rmem_tensor(fx.make_layout(vec, 1), fx.Float32)
        acc = fx.Vector.filled(vec, 0.0, fx.Float32)
        if index < padded_rows * R // vec:
            if index < rows * R // vec:
                # Match V1's per-element split order without inter-CTA synchronization.
                for s in range_constexpr(split):
                    fx.copy(atom, pv[None, s * padded_rows * R // vec + index], fragment)
                    acc = acc + fragment.load()
            z = acc * 0.25
            values = []
            for i in range_constexpr(vec):
                exponent = fx.Float32(fx.rocdl.exp2(fx.T.f32, fx.arith.unwrap(-z[i] * LOG2E)))
                inverse = fx.Float32(fx.rocdl.rcp(fx.T.f32, fx.arith.unwrap(1.0 + exponent)))
                values.append(z[i] * inverse)
            fragment.store(fx.Vector.from_elements(values, fx.Float32))
            fx.copy(atom, fragment, av[None, index])

    @flyc.jit
    def launch(P: fx.Tensor, A: fx.Tensor, stream: fx.Stream):
        reduce_silu(P, A).launch(
            grid=((padded_rows * R + threads * vec - 1) // (threads * vec), 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch


class ThreeStageGRRead(GRRead):
    """Reuse both accepted GEMMs, adding only a standalone FP32 activation kernel."""

    def __init__(self, rows, w_down, w_up, reduce_threads=64, reduce_vec=4):
        if reduce_threads not in (64, 128, 256) or reduce_vec not in (1, 2, 4):
            raise ValueError("reducer requires 64/128/256 threads and vector width 1/2/4")
        super().__init__(rows, w_down, w_up, Config(compensate_hidden=True))
        self.reduce_threads, self.reduce_vec = reduce_threads, reduce_vec
        padded = (rows + self.config.block_m - 1) // self.config.block_m * self.config.block_m
        self.activation = torch.empty((padded * R,), dtype=torch.float32, device=self.device)
        self.reduce = None
        if rows:
            stream = torch.cuda.current_stream(self.device)
            reducer = _reduce_launcher(rows, padded, self.config.split_k, reduce_threads, reduce_vec)
            self.reduce = flyc.compile(reducer, self.partial, self.activation, stream)
            up_config = replace(self.config, down_mode="wave_splitk")
            _, up = _launchers(rows, self.dtype, up_config)
            x = torch.empty((rows * 10240,), dtype=self.dtype, device=self.device)
            self.up = flyc.compile(up, x, self.w_up, self.activation, self.output.view(-1), stream)

    def run_reduce(self):
        self.reduce(self.partial, self.activation, torch.cuda.current_stream(self.device))

    def run_projection(self, x):
        self.up(x.view(-1), self.w_up, self.activation, self.output.view(-1), torch.cuda.current_stream(self.device))

    def run_up(self, x):
        self.run_reduce()
        self.run_projection(x)
