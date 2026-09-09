"""Small, bounded compilation/correctness probe; no serving integration."""

import argparse

import torch
import torch.nn.functional as F
from kernel import GRRead


def reference(x, wd, wu):
    x = x.double()
    t = F.silu(F.linear(x, wd.double()) / 4)
    g = torch.sigmoid(F.linear(t, wu.double()))
    return (g.reshape(-1, 4, 2560) * x.reshape(-1, 4, 2560)).mean(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(0)
    x = torch.randn(args.rows, 10240, dtype=torch.bfloat16, device="cuda")
    wd = torch.randn(320, 10240, dtype=x.dtype, device=x.device) * 0.02
    wu = torch.randn(10240, 320, dtype=x.dtype, device=x.device) * 0.02
    print("Compiling", args.rows, flush=True)
    kernel = GRRead(args.rows, wd, wu)
    y = kernel(x)
    torch.cuda.synchronize()
    down_ref = F.linear(x.double(), wd.double())
    down = kernel.partial.reshape(kernel.config.split_k, -1, 320).sum(0)[: args.rows]
    print("down max error", (down.double() - down_ref).abs().max().item(), flush=True)
    ref = reference(x, wd, wu)
    print("output max error", (y.double() - ref).abs().max().item(), flush=True)
    torch.testing.assert_close(y.double(), ref, rtol=1e-2, atol=5e-3)
    print("PASS", flush=True)
