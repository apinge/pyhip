"""T1..16 split-K pipeline with the original packed-weight contract."""

from dataclasses import dataclass, replace

import flydsl.compiler as flyc
import torch

if __package__:
    from . import prefetch_up
    from .combined_host import CombinedPaddedGRRead, _check_input
    from .large_down import DownConfig, down_launcher, pair_launcher
else:
    import prefetch_up
    from combined_host import CombinedPaddedGRRead, _check_input
    from large_down import DownConfig, down_launcher, pair_launcher


@dataclass(frozen=True)
class SmallConfig:
    down: DownConfig = DownConfig(block_k=128, global_split=4, prefetch_unroll=2)
    up_n: int = 128
    up_waves: int = 4

    @property
    def name(self):
        return f"small_{self.down.name}_un{self.up_n}_uw{self.up_waves}"

    def validate(self):
        self.down.validate()
        if self.down.block_m != 16:
            raise ValueError("small-batch down must retain BM16")
        self.up_config(prefetch_up.Config(compensate_hidden=True, hidden_pad=4)).validate()

    def up_config(self, base):
        return replace(
            base, down_mode="partial" if self.down.global_split > 1 else "wave_splitk",
            split_k=self.down.global_split, up_n=self.up_n, waves=self.up_waves,
            compensate_hidden=True, prefetch_low=False,
        )


def default_config(rows):
    if not 1 <= rows <= 16:
        raise ValueError("small-batch pipeline requires T=1..16")
    # Full-call graph timings choose BN128 at both ends of the small-T range.
    return SmallConfig(up_n=128 if rows <= 6 or rows == 16 else 64)


class SmallBatchGRRead:
    """Two launches, original packed weights, independent split-major FP32 P/Y."""

    def __init__(self, prepared, config=None):
        if not isinstance(prepared, CombinedPaddedGRRead):
            raise TypeError("prepared must be a CombinedPaddedGRRead")
        if not 1 <= prepared.rows <= 16:
            raise ValueError("small-batch pipeline experiment requires T=1..16")
        config = default_config(prepared.rows) if config is None else config
        if not prepared.config.preshuffle or prepared.config.hidden_pad != 4:
            raise ValueError("requires the original preshuffled pad4 control")
        config.validate()
        self.rows, self.dtype, self.device = prepared.rows, prepared.dtype, prepared.device
        self.config, self.up_config = config, config.up_config(prepared.config)
        self.w_down, self.w_up = prepared.w_down, prepared.w_up
        self.partial = torch.empty(config.down.global_split * 16 * 320, dtype=torch.float32, device=self.device)
        self.output = torch.empty_like(prepared.output)
        x = torch.empty(self.rows * 10240, dtype=self.dtype, device=self.device)
        stream = torch.cuda.current_stream(self.device)
        ld = down_launcher(self.rows, config.down)
        _, lu = prefetch_up._launchers(self.rows, self.dtype, self.up_config)
        self.down = flyc.compile(ld, x, self.w_down, self.partial, stream)
        self.up = flyc.compile(lu, x, self.w_up, self.partial, self.output.view(-1), stream)
        self.dispatch = flyc.compile(
            pair_launcher(self.rows, config.down, self.up_config),
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
