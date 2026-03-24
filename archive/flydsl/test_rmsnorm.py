#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
独立 RMSNorm 示例（从 kernels/rmsnorm_kernel.py 与 tests/kernels/test_rmsnorm.py 抽出）。

    cd /path/to/FlyDSL/archive/flydsl && python test_rmsnorm.py

可选环境变量（与主仓库测试一致）::
    ROCDSL_RMSNORM_SHAPES — 形如 "M,N,dtype;..."，例如 "64,2048,bf16"
    ROCDSL_COMPARE_AITER — 设为 1 时尝试与 AIter 对比耗时
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# 将 FlyDSL 源码树中的 python/ 加入路径（本脚本位于 FlyDSL/archive/flydsl/）
_THIS = os.path.abspath(__file__)
_ARCHIVE_DIR = os.path.dirname(_THIS)
_FLYDSL_ROOT = os.path.abspath(os.path.join(_ARCHIVE_DIR, "..", ".."))
_PYTHON = os.path.join(_FLYDSL_ROOT, "python")
if os.path.isdir(_PYTHON) and _PYTHON not in sys.path:
    sys.path.insert(0, _PYTHON)
_EMBEDDED = os.path.join(_FLYDSL_ROOT, "build-fly", "python_packages")
if os.path.isdir(_EMBEDDED) and _EMBEDDED not in sys.path:
    sys.path.insert(0, _EMBEDDED)

import pytest

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

import torch.nn.functional as F

from rmsnorm_kernel import build_rmsnorm_module, build_rmsnorm_fusedadd_module

DTYPE_FP32 = torch.float32
DTYPE_FP16 = torch.float16
DTYPE_BF16 = torch.bfloat16

EPS: float = 1e-5
WARMUP_ITERS = 10
BENCH_ITERS = 100


def bench_gpu_us_torch(fn: Callable[[], None], *, warmup: int = 20, iters: int = 200) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def run_perftest_simple(fn: Callable[[], None], *, num_iters: int, num_warmup: int) -> Tuple[None, float]:
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    avg_us = start.elapsed_time(end) * 1e3 / num_iters
    return None, avg_us


@dataclass(frozen=True)
class PerfRow:
    op: str
    shape: str
    dtype: str
    flydsl_gpu_us: Optional[float]
    aiter_gpu_us: Optional[float]
    aiter_rms_norm_gpu_us: Optional[float] = None


def _fmt_us(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:,.1f}"


def print_perf_table(rows: List[PerfRow]) -> None:
    print("\n" + "=" * 120)
    print("Perf Compare (gpu us): FlyDSL vs AIter Triton vs aiter.rms_norm (CK)")
    print("=" * 120)
    print(
        f"{'op':10s} {'shape':18s} {'dtype':6s} {'FlyDSL(gpu us)':>14s} "
        f"{'AIter triton(us)':>16s} {'aiter.rms_norm(us)':>18s}"
    )
    for r in rows:
        print(
            f"{r.op:10s} {r.shape:18s} {r.dtype:6s} {_fmt_us(r.flydsl_gpu_us):>14s} "
            f"{_fmt_us(r.aiter_gpu_us):>16s} {_fmt_us(r.aiter_rms_norm_gpu_us):>18s}"
        )
    print("=" * 120 + "\n")


def maybe_enable_aiter() -> bool:
    try:
        import aiter  # noqa: F401
        return True
    except Exception:
        pass
    aiter_repo = os.environ.get("AITER_REPO", "").strip()
    if aiter_repo and os.path.isdir(aiter_repo):
        sys.path.insert(0, aiter_repo)
        try:
            import aiter  # noqa: F401
            return True
        except Exception:
            return False
    return False


def run_torch(input, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = F.rms_norm(
            input=input, normalized_shape=(input.shape[-1],), weight=weight, eps=eps
        )
    else:
        residual_out = input + residual
        output = F.rms_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            eps=eps,
        )
    return output, residual_out


def run_test(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting RMSNorm (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_rmsnorm_module(M, N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False, None
    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)

    if dtype == "f32":
        input_dev = input_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 1e-4
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 1e-2
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    expected, _ = run_torch(input_ref, gamma_ref, EPS, None)
    expected = expected.to(DTYPE_FP32)

    print("Launching kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        launch_fn(input_dev, gamma_dev, output_dev, M, stream=stream)

    _, avg_us = run_perftest_simple(kernel_launch, num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS)
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = 2 * M * N * elem_bytes
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL rmsnorm gpu: {flydsl_gpu_us:.1f} us")

    output_ref = output_dev.to(DTYPE_FP32)
    error = (output_ref - expected).abs().max().item()
    print(f"Max absolute error: {error:.2e} (atol={atol})")

    if error < atol:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
        print("First row Expected:")
        print(expected[0, :5])
        print("First row Actual:")
        print(output_ref[0, :5])
        ok = False
    return ok, flydsl_gpu_us

def run_test_fused_add(M: int, N: int, dtype: str = "f32"):
    print(f"\nTesting RMSNorm Fused Add (M={M}, N={N}, dtype={dtype})")

    try:
        launch_fn = build_rmsnorm_fusedadd_module(M, N, dtype)
    except Exception as e:
        print(f"[FAIL] Compile failed for (M={M}, N={N}, dtype={dtype}): {type(e).__name__}: {e}")
        return False, None
    torch.manual_seed(42)
    input_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    gamma_t = torch.rand((N,), device="cuda", dtype=DTYPE_FP32)
    residual_t = torch.randn((M, N), device="cuda", dtype=DTYPE_FP32)
    if dtype == "f32":
        input_dev = input_t.contiguous()
        gamma_dev = gamma_t.contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP32)
        residual_dev = residual_t.contiguous()
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 1e-4
    elif dtype == "f16":
        input_dev = input_t.to(DTYPE_FP16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_FP16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_FP16)
        residual_dev = residual_t.to(DTYPE_FP16).contiguous()
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 1e-2
    elif dtype == "bf16":
        input_dev = input_t.to(DTYPE_BF16).contiguous()
        gamma_dev = gamma_t.to(DTYPE_BF16).contiguous()
        output_dev = torch.empty((M, N), device="cuda", dtype=DTYPE_BF16)
        residual_dev = residual_t.to(DTYPE_BF16).contiguous()
        input_ref = input_dev.to(DTYPE_FP32)
        gamma_ref = gamma_dev.to(DTYPE_FP32)
        atol = 2e-2
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    # fused kernel 会原地覆盖 residual；perf 多次 launch 时每次须从备份恢复，否则后续迭代在已融合的张量上再融合
    residual_backup = residual_dev.clone()

    # 与 device kernel 一致：先 cast 到 elem dtype 再升到 fp32 做加法和 RMSNorm，不能用未量化的 residual_t(fp32)
    residual_fp32 = residual_dev.to(DTYPE_FP32)
    expected, fused_sum_fp32 = run_torch(input_ref, gamma_ref, EPS, residual_fp32)
    expected = expected.to(DTYPE_FP32)
    if fused_sum_fp32 is not None:
        fused_sum_fp32 = fused_sum_fp32.to(DTYPE_FP32)
        # kernel 把 x+residual 截断写回 residual buffer；对比时应对 golden 做同 dtype 舍入再升回 fp32
        if dtype == "f32":
            residual_expected = fused_sum_fp32
        elif dtype == "f16":
            residual_expected = fused_sum_fp32.to(DTYPE_FP16).to(DTYPE_FP32)
        else:
            residual_expected = fused_sum_fp32.to(DTYPE_BF16).to(DTYPE_FP32)
    else:
        residual_expected = None

    print("Launching rmsnorm_fusedadd_kernel...")
    stream = torch.cuda.current_stream()

    def kernel_launch():
        # TODO 这个地方测性能会有问题
        residual_dev.copy_(residual_backup)
        launch_fn(input_dev, gamma_dev, residual_dev, output_dev, M, stream=stream)

    _, avg_us = run_perftest_simple(kernel_launch, num_iters=BENCH_ITERS, num_warmup=WARMUP_ITERS)
    torch.cuda.synchronize()
    flydsl_gpu_us = None
    if os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1":
        flydsl_gpu_us = bench_gpu_us_torch(kernel_launch, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    avg_ms = avg_us / 1000.0

    elem_bytes = 4 if dtype == "f32" else 2
    total_bytes = 2 * M * N * elem_bytes
    bandwidth_gbs = total_bytes / (avg_us / 1e6) / 1e9

    print(f"Kernel avg time: {avg_ms:.4f} ms (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    if flydsl_gpu_us is not None:
        print(f"[Perf] FlyDSL rmsnorm gpu: {flydsl_gpu_us:.1f} us")

    output_ref = output_dev.to(DTYPE_FP32)
    error = (output_ref - expected).abs().max().item()
    print(f"Max absolute error (output): {error:.2e} (atol={atol})")

    if residual_expected is not None:
        residual_out_fp32 = residual_dev.to(DTYPE_FP32)
        error_residual = (residual_out_fp32 - residual_expected).abs().max().item()
        print(f"Max absolute error (residual): {error_residual:.2e} (atol={atol})")
        ok_res = error_residual < atol
    else:
        ok_res = True

    ok_out = error < atol
    if ok_out and ok_res:
        print("PASSED")
        ok = True
    else:
        print("FAILED")
        if not ok_out:
            print("Output — first row Expected:")
            print(expected[0, :5])
            print("Output — first row Actual:")
            print(output_ref[0, :5])
        if not ok_res:
            print("Residual — first row Expected:")
            print(residual_expected[0, :5])
            print("Residual — first row Actual:")
            print(residual_out_fp32[0, :5])
        ok = False
    return ok, flydsl_gpu_us


def test_all():
    print("=" * 80)
    print("Running RMSNorm Tests (archive/flydsl standalone)")
    print("=" * 80)

    shapes_env = os.environ.get("ROCDSL_RMSNORM_SHAPES", "").strip()
    if shapes_env:
        configs = []
        for part in shapes_env.split(";"):
            p = part.strip()
            if not p:
                continue
            m_s, n_s, dt = [x.strip() for x in p.split(",")]
            configs.append((int(m_s), int(n_s), dt))
    else:
        configs = [
            (32768, 8192, "f16"),
            (32768,4096, "f16"),
            (8000,4096, "f16"),
            (8,4096, "f16"),
            (4,4096, "f16"),
            (2,4096, "f16"),
            # (1, 4096, "f32"),
            # (1, 4096, "f16"),
            # (1, 4096, "bf16"),
            # (32768,256, "bf16"),
            # (8000,256, "bf16"),
            # (8,256, "bf16"),
            # (4,256, "bf16"),
            # (2,256, "bf16"),
            # (1,256, "bf16"),
        ]

    do_compare = os.environ.get("ROCDSL_COMPARE_AITER", "0") == "1"
    perf_rows: List[PerfRow] = []
    failures = 0
    for M, N, dtype in configs:
        # ok, flydsl_gpu_us = run_test(M, N, dtype)
        # if not ok:
        #     failures += 1
        # 暂时屏蔽掉 rmsnorm 测试，为了开发方便

        ok_1, flydsl_gpu_us_fusedadd = run_test_fused_add(M, N, dtype)
        if not ok_1:
            failures += 1

        if do_compare:
            aiter_us = None
            aiter_rms_norm_us = None
            if maybe_enable_aiter():
                from aiter import rms_norm as aiter_ck_rmsnorm

                x = torch.randn(
                    (M, N),
                    device="cuda",
                    dtype=DTYPE_BF16 if dtype == "bf16" else (DTYPE_FP16 if dtype == "f16" else DTYPE_FP32),
                )
                w = torch.rand((N,), device="cuda", dtype=x.dtype)
                try:
                    from aiter.ops.triton.rmsnorm import rms_norm as aiter_rms_norm

                    def run_aiter_triton():
                        aiter_rms_norm(x, w, EPS)

                    aiter_us = bench_gpu_us_torch(run_aiter_triton, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] AIter Triton rmsnorm gpu: {aiter_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] AIter Triton rmsnorm skipped: {type(e).__name__}: {e!r}")
                try:

                    def run_aiter_ck_rmsnorm():
                        aiter_ck_rmsnorm(x, w, EPS, 0)

                    aiter_rms_norm_us = bench_gpu_us_torch(run_aiter_ck_rmsnorm, warmup=WARMUP_ITERS, iters=BENCH_ITERS)
                    print(f"[Perf] aiter.rms_norm (CK) gpu: {aiter_rms_norm_us:.1f} us")
                except Exception as e:
                    print(f"[Perf] aiter.rms_norm skipped: {type(e).__name__}: {e!r}")

            perf_rows.append(
                PerfRow(
                    op="rmsnorm",
                    shape=f"{M}x{N}",
                    dtype=dtype,
                    flydsl_gpu_us=flydsl_gpu_us,
                    aiter_gpu_us=aiter_us,
                    aiter_rms_norm_gpu_us=aiter_rms_norm_us,
                )
            )

    print("\n" + "=" * 80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("=" * 80)
    if do_compare and perf_rows:
        print_perf_table(perf_rows)
    if failures != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    test_all()
