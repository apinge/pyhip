"""One compiled host entry for two/three existing GPU launches, without fusion."""

from dataclasses import replace
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

if __package__:
    from . import prefetch_up as padded_impl
    from .kernel import Config, GRRead, K, _launchers, default_config
    from .three_stage import ThreeStageGRRead, _reduce_launcher
else:
    import prefetch_up as padded_impl
    from kernel import Config, GRRead, K, _launchers, default_config
    from three_stage import ThreeStageGRRead, _reduce_launcher


@cache
def _pair_launcher(rows):
    down, up = _launchers(rows, torch.bfloat16, default_config(rows))

    @flyc.jit
    def launch(X: fx.Tensor, WD: fx.Tensor, WU: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        # Keep row specialization explicit in this enclosing JIT cache key.
        if fx.const_expr(rows > 0):
            down(X, WD, P, stream)
            up(X, WU, P, Y, stream)

    return launch


@cache
def _triple_launcher(rows, threads, vec):
    config = Config(compensate_hidden=True)
    padded = (rows + config.block_m - 1) // config.block_m * config.block_m
    down, _ = _launchers(rows, torch.bfloat16, config)
    _, up = _launchers(rows, torch.bfloat16, replace(config, down_mode="wave_splitk"))
    reduce = _reduce_launcher(rows, padded, config.split_k, threads, vec)

    @flyc.jit
    def launch(X: fx.Tensor, WD: fx.Tensor, WU: fx.Tensor, P: fx.Tensor, A: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        if fx.const_expr(rows > 0 and threads * vec > 0):
            down(X, WD, P, stream)
            reduce(P, A, stream)
            up(X, WU, A, Y, stream)

    return launch


@cache
def _padded_pair_launcher(rows, padding):
    config = replace(padded_impl.default_config(rows), hidden_pad=padding, prefetch_low=False)
    down, up = padded_impl._launchers(rows, torch.bfloat16, config)

    @flyc.jit
    def launch(X: fx.Tensor, WD: fx.Tensor, WU: fx.Tensor, P: fx.Tensor, Y: fx.Tensor, stream: fx.Stream):
        if fx.const_expr(rows > 0 and padding >= 0):
            down(X, WD, P, stream)
            up(X, WU, P, Y, stream)

    return launch


def _check_input(reader, x):
    if x.shape != (reader.rows, K) or x.dtype != reader.dtype or x.device != reader.device:
        raise ValueError("input must match the prepared rows, dtype and device")
    if not x.is_contiguous():
        raise ValueError("input must be contiguous")


class CombinedHostGRRead(GRRead):
    def __init__(self, rows, w_down, w_up):
        super().__init__(rows, w_down, w_up)
        self.dispatch = None
        if rows:
            x = torch.empty(rows * K, device=self.device, dtype=self.dtype)
            self.dispatch = flyc.compile(
                _pair_launcher(rows),
                x,
                self.w_down,
                self.w_up,
                self.partial,
                self.output.view(-1),
                torch.cuda.current_stream(self.device),
            )

    def __call__(self, x):
        _check_input(self, x)
        if self.rows:
            self.dispatch(
                x.view(-1),
                self.w_down,
                self.w_up,
                self.partial,
                self.output.view(-1),
                torch.cuda.current_stream(self.device),
            )
        return self.output


class CombinedThreeStageGRRead(ThreeStageGRRead):
    def __init__(self, rows, w_down, w_up, reduce_threads=128, reduce_vec=1):
        super().__init__(rows, w_down, w_up, reduce_threads, reduce_vec)
        self.dispatch = None
        if rows:
            x = torch.empty(rows * K, device=self.device, dtype=self.dtype)
            self.dispatch = flyc.compile(
                _triple_launcher(rows, reduce_threads, reduce_vec),
                x,
                self.w_down,
                self.w_up,
                self.partial,
                self.activation,
                self.output.view(-1),
                torch.cuda.current_stream(self.device),
            )

    def __call__(self, x):
        _check_input(self, x)
        if self.rows:
            self.dispatch(
                x.view(-1),
                self.w_down,
                self.w_up,
                self.partial,
                self.activation,
                self.output.view(-1),
                torch.cuda.current_stream(self.device),
            )
        return self.output


class CombinedPaddedGRRead(padded_impl.GRRead):
    """Two GPU kernels, padded hidden LDS, one compiled host entry."""

    def __init__(self, rows, w_down, w_up, hidden_pad=4):
        config = replace(padded_impl.default_config(rows), hidden_pad=hidden_pad, prefetch_low=False)
        super().__init__(rows, w_down, w_up, config)
        self.dispatch = None
        if rows:
            x = torch.empty(rows * K, device=self.device, dtype=self.dtype)
            self.dispatch = flyc.compile(
                _padded_pair_launcher(rows, hidden_pad),
                x,
                self.w_down,
                self.w_up,
                self.partial,
                self.output.view(-1),
                torch.cuda.current_stream(self.device),
            )

    def __call__(self, x):
        _check_input(self, x)
        if self.rows:
            self.dispatch(
                x.view(-1),
                self.w_down,
                self.w_up,
                self.partial,
                self.output.view(-1),
                torch.cuda.current_stream(self.device),
            )
        return self.output
