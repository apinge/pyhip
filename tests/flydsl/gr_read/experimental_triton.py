"""Opt-in ROWS=32 probe of the unchanged tuned Triton kernel, not production dispatch."""

import torch


class ExtendedTritonGRRead:
    """Use the 8cf5501b tuning options beyond its supported <=16-row wrapper."""

    def __init__(self, rows, w_down, w_up, baseline):
        if not 17 <= rows <= 32:
            raise ValueError("extended Triton experiment requires 17..32 rows")
        if w_down.shape != (320, 10240) or w_up.shape != (10240, 320):
            raise ValueError("expected W_down[320,10240] and W_up[10240,320]")
        if w_down.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
            raise ValueError("extended Triton experiment requires BF16 weights")
        if w_down.device != w_up.device or not w_down.is_cuda or torch.version.hip is None:
            raise ValueError("weights must be on the same ROCm device")
        if not w_down.is_contiguous() or not w_up.is_contiguous():
            raise ValueError("weights must be contiguous")
        props = torch.cuda.get_device_properties(w_down.device)
        if props.gcnArchName.split(":", 1)[0] != "gfx942" or props.multi_processor_count != 80:
            raise ValueError("extended Triton experiment requires an 80-CU gfx942 device")
        self.rows, self.device = rows, w_down.device
        self.options = dict(baseline._GFX942_MIX_CONFIG, ROWS=32, HC=4)
        self.kernel = baseline._hc_mix_persistent_kernel
        self.partial = torch.empty((32, 320), dtype=torch.float32, device=self.device)
        self.output = torch.empty((rows, 2560), dtype=torch.bfloat16, device=self.device)
        self.counters = baseline._get_counters(self.device)
        self.arguments = (w_down, w_up, self.partial, self.output, self.counters, 10240, 320, 2560, rows, 80, 0.25)
        x = torch.empty((rows, 10240), dtype=torch.bfloat16, device=self.device)
        self.compiled = self.kernel.warmup(x, *self.arguments, **self.options, grid=(80,))
        self.compiled._init_handles()
        self.resources = {
            "shared_bytes": self.compiled.metadata.shared,
            "n_regs": self.compiled.n_regs,
            "n_spills": self.compiled.n_spills,
            "num_warps": self.compiled.metadata.num_warps,
            "grid": 80,
            "rows_tile": 32,
        }
        if self.resources["shared_bytes"] > 65536:
            raise ValueError("extended Triton experiment exceeds gfx942 per-CTA LDS capacity")

    def __call__(self, x):
        if x.shape != (self.rows, 10240) or x.dtype != torch.bfloat16 or x.device != self.device:
            raise ValueError("input must match the prepared rows, BF16 dtype and device")
        if not x.is_contiguous():
            raise ValueError("input must be contiguous")
        self.kernel[(80,)](x, *self.arguments, **self.options)
        return self.output
