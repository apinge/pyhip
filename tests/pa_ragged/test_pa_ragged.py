# SPDX-License-Identifier: MIT
"""
正确性：仅 pyhip `run_pa_ragged_bf16` 与 Torch golden `allclose`。aiter 只做 cudaPerf 性能对比，不比正确性。

    ctx_len=8192, num_seqs=1, num_heads=(32,4), head_dim=128, block_size=1, bf16, NHD

pyhip：BUF_COPY 轮换 + cudaPerf。aiter：`AITER_RUN_COUNT` 次 cudaPerf、单 buffer。
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import torch
from einops import rearrange

_ROOT_PYHIP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT_PYHIP, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _aiter_root in ("/opt/aiter",):
    if os.path.isdir(_aiter_root) and _aiter_root not in sys.path:
        sys.path.insert(0, _aiter_root)

from pyhip import cudaPerf

from pa_ragged_pyhip import HEAD_DIM, HK, HQ, KV_PART_SIZE, run_pa_ragged_bf16

# pyhip：多 buffer 轮换 + cudaPerf；与 gemm_splitk 一致
BUF_COPY = 32
RUN_COUNT = 10

# aiter：只做性能对比，次数尽量少（单 buffer、3 轮计时）
AITER_BUF_COPY = 1
AITER_RUN_COUNT = 3

# 与 /opt/aiter/op_tests/test_pa_ragged.py run_aiter 中 _PARTITION_SIZE_ROCM 一致
AITER_PARTITION_SIZE_ROCM = 256


def _has_aiter_pa_ragged() -> bool:
    """仅在需要时 import aiter；导入前固定 cuda:0 并完成同步，避免与 pyhip 队列交错。

    说明（常见崩溃「Memory access fault by GPU node-2」）：
    - ROCm 里「node-2」常对应 HSA Agent 编号，并不等于 PyTorch 的 cuda:2；很多机器上首块 GPU 就是 agent 2。
    - 全量 `import aiter` 会拉 JIT、注册大量 torch.ops，并可能触发分配/同步；若此时 NUMA balancing 仍开启，
      aiter 也会在 get_module 前打 NUMA 警告——与 MI300 调优文档一致，错误内存放置可导致 GPU fault（属系统/驱动层，
      非单条 Python 语句能「修掉」）。
    - 此处通过 set_device(0) + synchronize，排除「当前上下文非 0 号卡」与「pyhip 未完成就接着加载 aiter」两类可修复因素。
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.synchronize()
        import aiter  # noqa: F401
    except ImportError:
        return False
    #return hasattr(torch.ops.aiter, "paged_attention_ragged")
    return False # aiter老是core dump 禁用

# ---------------------------------------------------------------------------
# Reference（同 pa/op_tests/test_pa_ragged.py run_torch_new）
# ---------------------------------------------------------------------------


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
    logits_soft_cap: float = 0.0,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    if 0 < logits_soft_cap:
        attn_weights = logits_soft_cap * torch.tanh(attn_weights / logits_soft_cap)
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def torch_paged_decode_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_queries_per_kv: int,
    logits_soft_cap: float = 0.0,
) -> torch.Tensor:
    """key/value_cache: [num_blocks, HK, block_size, head_dim]。"""
    output = torch.zeros_like(query)
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: List[torch.Tensor] = []
        values_lst: List[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, :, block_offset, :]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        out = ref_masked_attention(q, keys, values, scale, None, logits_soft_cap)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out)
    return output


def _nhd_stride_fix(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    h, d = t.size(2), t.size(3)
    if t.size(1) == 1 and t.stride(1) != h * d:
        return t.reshape(*t.shape)
    return t


def _build_pa_case(
    *,
    device: str,
    ctx_len: int,
    num_seqs: int,
    seed: int,
):
    """构造一组 query / K/V / 元数据 + golden；与 __main__ 典型配置一致。"""
    num_query_heads, num_kv_heads = HQ, HK
    head_size = HEAD_DIM
    block_size = 1
    dtype = torch.bfloat16

    torch.manual_seed(seed)
    max_seq_len = ctx_len
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs
    num_queries_per_kv = num_query_heads // num_kv_heads

    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype, device=device)
    query.uniform_(-1.0, 1.0)

    x = 16 // dtype.itemsize
    key_5d = torch.empty(
        num_blocks, num_kv_heads, head_size // x, block_size, x, dtype=dtype, device=device
    )
    val_5d = torch.empty(
        num_blocks, num_kv_heads, head_size, block_size, dtype=dtype, device=device
    )
    key_5d.uniform_(-1.0, 1.0)
    val_5d.uniform_(-1.0, 1.0)

    key_h = rearrange(key_5d, "b h d1 s d2 -> b h s (d1 d2)")
    val_h = rearrange(val_5d, "b h d s -> b h s d")
    key_nhd = rearrange(key_h, "b h s d -> b s h d")
    val_nhd = rearrange(val_h, "b h s d -> b s h d")
    key_nhd = _nhd_stride_fix(key_nhd)
    val_nhd = _nhd_stride_fix(val_nhd)

    block_tables = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=num_seqs,
    )
    seq_lens = torch.full((num_seqs,), fill_value=ctx_len, dtype=torch.int, device=device)

    def convert_to_kv_indptr_last_page_lens(context_length: int) -> torch.Tensor:
        def get_num_blocks(cl: int) -> int:
            return (cl + block_size - 1) // block_size

        num_blocks_list = [get_num_blocks(context_length) for _ in range(num_seqs)]
        return torch.tensor([0] + num_blocks_list, dtype=torch.int32, device=device).cumsum(0)

    def convert_to_page_indices(bt: torch.Tensor, kv_indptr_: torch.Tensor) -> torch.Tensor:
        elements_per_row = kv_indptr_[1:] - kv_indptr_[:-1]
        col_indices = torch.arange(bt.size(1), device=bt.device).expand(bt.size(0), -1)
        return bt[col_indices < elements_per_row.unsqueeze(1)]

    kv_indptr = convert_to_kv_indptr_last_page_lens(ctx_len)
    kv_page_indices = convert_to_page_indices(block_tables, kv_indptr)

    # 与 aiter test `convert_to_kv_indptr_last_page_lens` 一致（供 torch.ops.aiter.paged_attention_ragged）
    def _last_page_len(cl: int) -> int:
        r = cl % block_size
        return r if r > 0 else block_size

    kv_last_page_lens = torch.tensor(
        [_last_page_len(ctx_len) for _ in range(num_seqs)],
        dtype=torch.int32,
        device=device,
    )

    scale = float(1.0 / (head_size**0.5))
    golden = torch_paged_decode_reference(
        query,
        key_h,
        val_h,
        block_tables,
        seq_lens,
        scale,
        num_queries_per_kv,
        0.0,
    )

    max_num_partitions = (max_seq_len + KV_PART_SIZE - 1) // KV_PART_SIZE
    # 近似 FLOPs：两段大 matmul（QK + AV）~ 4 * seq * HQ * head * ctx_len
    flops = float(4 * num_seqs * HQ * head_size * ctx_len)
    ele = 2
    q_bytes = num_seqs * num_query_heads * head_size * ele
    kv_plane = num_blocks * block_size * num_kv_heads * head_size * ele
    meta_bytes = kv_indptr.numel() * 4 + kv_page_indices.numel() * 4
    rw_bytes = int(q_bytes + 2 * kv_plane + q_bytes + meta_bytes)

    return (
        query,
        key_h,
        val_h,
        key_nhd,
        val_nhd,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        golden,
        max_num_partitions,
        scale,
        rw_bytes,
        flops,
    )


def _perf_buffers_for_case(
    device: str,
    num_seqs: int,
    max_num_partitions: int,
    query: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nstat = num_seqs * HQ * max_num_partitions
    out = torch.empty_like(query)
    workspace_f2 = torch.empty(2 * nstat, dtype=torch.float32, device=device)
    tmp_out_seg = torch.empty(
        (num_seqs, HQ, max_num_partitions, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        memory_format=torch.contiguous_format,
    )
    qk_stub = torch.empty(1, dtype=torch.float32, device=device)
    return out, workspace_f2, tmp_out_seg, qk_stub


def _aiter_workspace_numel(
    num_seqs: int,
    num_heads: int,
    max_num_partitions: int,
    head_size: int,
    dtype: torch.dtype,
) -> int:
    """与 /opt/aiter/op_tests/test_pa_ragged.py run_aiter 中 workspace_buffer 一致。"""
    nbytes_elem = torch.finfo(dtype).bits // 8
    return (
        num_seqs * num_heads * max_num_partitions * head_size * nbytes_elem
        + 2 * (num_seqs * num_heads * max_num_partitions) * 4
    )


def _run_aiter_paged_attention_ragged(
    output: torch.Tensor,
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    block_size: int,
    max_num_partitions: int,
    scale: float,
    *,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    kv_cache_layout: str = "HND",
    logits_soft_cap: float = 0.0,
) -> None:
    """封装 `torch.ops.aiter.paged_attention_ragged`，对齐 op_tests/test_pa_ragged.py:374-394。"""
    torch.ops.aiter.paged_attention_ragged(
        output,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        block_size,
        max_num_partitions,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        None,
        AITER_PARTITION_SIZE_ROCM,
    )


def _benchmark_pa_ragged(buf_cases: List[dict], run_count: int, perf_name: str) -> None:
    """gemm_splitk：cudaPerf 包 kernel；BUF_COPY 轮换 i；平均时丢弃第 1 次（与 tflops_res[1:] 一致）。"""
    i = 0
    tflops_res: List[float] = []
    latencies: List[float] = []
    bw_res: List[float] = []
    for _ in range(run_count):
        c = buf_cases[i]
        with cudaPerf(c["flops"], c["rw_bytes"], name=f"{perf_name}[{c['tag']}]") as p:
            run_pa_ragged_bf16(
                c["out"],
                c["query"],
                c["key_nhd"],
                c["val_nhd"],
                c["kv_indptr"],
                c["kv_page_indices"],
                c["max_num_partitions"],
                int(c["query"].stride(0)),
                workspace_f2=c["workspace_f2"],
                tmp_out_seg_buf=c["tmp_out_seg"],
                qk_stub=c["qk_stub"],
            )
        i = (i + 1) % len(buf_cases)
        tflops_res.append(p.tflops())
        latencies.append(p.dt())
        bw_res.append(p.bw())
    if run_count > 1:
        avg_tf = sum(tflops_res[1:]) / len(tflops_res[1:])
        avg_lat_us = sum(latencies[1:]) / len(latencies[1:]) * 1e6
        avg_bw = sum(bw_res[1:]) / len(bw_res[1:])
        print(
            f"{perf_name}: avg (drop 1st) TFLOPS={avg_tf:.3f}, latency_us={avg_lat_us:.3f}, GB/s={avg_bw:.1f}"
        )
    elif run_count == 1:
        print(f"{perf_name}: TFLOPS={tflops_res[0]:.3f}, latency_us={latencies[0]*1e6:.3f}, GB/s={bw_res[0]:.1f}")


def _benchmark_aiter_pa_ragged(buf_cases: List[dict], run_count: int, perf_name: str) -> None:
    i = 0
    tflops_res: List[float] = []
    latencies: List[float] = []
    bw_res: List[float] = []
    for _ in range(run_count):
        c = buf_cases[i]
        with cudaPerf(c["flops"], c["rw_bytes"], name=f"{perf_name}[{c['tag']}]") as p:
            _run_aiter_paged_attention_ragged(
                c["out"],
                c["workspace"],
                c["query"],
                c["key_h"],
                c["val_h"],
                c["kv_indptr"],
                c["kv_page_indices"],
                c["kv_last_page_lens"],
                int(c["block_size"]),
                int(c["max_num_partitions"]),
                float(c["scale"]),
                k_scale=c["k_scale"],
                v_scale=c["v_scale"],
                alibi_slopes=c.get("alibi_slopes"),
                kv_cache_dtype=c.get("kv_cache_dtype", "auto"),
                kv_cache_layout=c.get("kv_cache_layout", "HND"),
                logits_soft_cap=float(c.get("logits_soft_cap", 0.0)),
            )
        i = (i + 1) % len(buf_cases)
        tflops_res.append(p.tflops())
        latencies.append(p.dt())
        bw_res.append(p.bw())
    if run_count > 1:
        avg_tf = sum(tflops_res[1:]) / len(tflops_res[1:])
        avg_lat_us = sum(latencies[1:]) / len(latencies[1:]) * 1e6
        avg_bw = sum(bw_res[1:]) / len(bw_res[1:])
        print(
            f"{perf_name}: avg (drop 1st) TFLOPS={avg_tf:.3f}, latency_us={avg_lat_us:.3f}, GB/s={avg_bw:.1f}"
        )
    elif run_count == 1:
        print(f"{perf_name}: TFLOPS={tflops_res[0]:.3f}, latency_us={latencies[0]*1e6:.3f}, GB/s={bw_res[0]:.1f}")


def compare_pa_ragged_to_torch() -> int:
    """返回 0 表示误差在阈值内，1 表示失败。正确性通过后可选 cudaPerf 性能（BUF_COPY 轮换）。"""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available", file=sys.stderr)
        return 1

    # 显式绑定 0 号卡，避免默认 cuda 上下文与后续 aiter 内部 `device="cuda"` 不一致。
    torch.cuda.set_device(0)
    device = "cuda:0"
    ctx_len = 8192
    num_seqs = 1

    (
        query,
        key_h,
        val_h,
        key_nhd,
        val_nhd,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        golden,
        max_num_partitions,
        scale,
        _rw_bytes,
        _flops,
    ) = _build_pa_case(device=device, ctx_len=ctx_len, num_seqs=num_seqs, seed=0)

    out = torch.empty_like(query)
    q_stride = int(query.stride(0))
    run_pa_ragged_bf16(
        out,
        query,
        key_nhd,
        val_nhd,
        kv_indptr,
        kv_page_indices,
        max_num_partitions,
        q_stride,
    )

    rtol, atol = 1e-2, 1e-2
    ok = torch.allclose(out, golden, rtol=rtol, atol=atol)
    if ok:
        print(f"pa_ragged vs torch reference: torch.allclose(rtol={rtol}, atol={atol}) -> PASS")
    else:
        diff = (out.float() - golden.float()).abs()
        print(
            f"pa_ragged vs torch reference: torch.allclose(rtol={rtol}, atol={atol}) -> FAIL, "
            f"max_abs_err={float(diff.max()):.6f}, mean_abs_err={float(diff.mean()):.6f}"
        )
        return 1

    block_size = 1

    buf_cases: List[dict] = []
    for b in range(BUF_COPY):
        (
            q_b,
            kh_b,
            vh_b,
            kn_b,
            vn_b,
            kvi_b,
            kvp_b,
            kvlp_b,
            _g,
            max_part_b,
            sc_b,
            rw_b,
            flops_b,
        ) = _build_pa_case(device=device, ctx_len=ctx_len, num_seqs=num_seqs, seed=b)
        o_b, w2_b, tmp_b, qk_b = _perf_buffers_for_case(device, num_seqs, max_part_b, q_b)
        buf_cases.append(
            {
                "query": q_b,
                "key_nhd": kn_b,
                "val_nhd": vn_b,
                "kv_indptr": kvi_b,
                "kv_page_indices": kvp_b,
                "max_num_partitions": max_part_b,
                "out": o_b,
                "workspace_f2": w2_b,
                "tmp_out_seg": tmp_b,
                "qk_stub": qk_b,
                "flops": flops_b,
                "rw_bytes": rw_b,
                "tag": f"{ctx_len=},{num_seqs=}",
            }
        )
    _benchmark_pa_ragged(buf_cases, RUN_COUNT, perf_name="pa_ragged_bf16")

    # pyhip cudaPerf 与 ctypes 路径跑完后先排空队列，再全量 import aiter，降低驱动侧交错风险。
    torch.cuda.synchronize()

    if _has_aiter_pa_ragged():
        aiter_cases: List[dict] = []
        for b in range(AITER_BUF_COPY):
            (
                q_b,
                kh_b,
                vh_b,
                _kn,
                _vn,
                kvi_b,
                kvp_b,
                kvlp_b,
                _g,
                max_part_b,
                sc_b,
                rw_b,
                flops_b,
            ) = _build_pa_case(device=device, ctx_len=ctx_len, num_seqs=num_seqs, seed=b)
            out_a = torch.empty_like(q_b)
            ws_a = torch.empty(
                _aiter_workspace_numel(num_seqs, HQ, max_part_b, HEAD_DIM, q_b.dtype),
                dtype=torch.uint8,
                device=device,
            )
            ks_b = torch.tensor([1.0], dtype=torch.float32, device=device)
            vs_b = torch.tensor([1.0], dtype=torch.float32, device=device)
            aiter_cases.append(
                {
                    "query": q_b,
                    "key_h": kh_b.contiguous(),
                    "val_h": vh_b.contiguous(),
                    "kv_indptr": kvi_b,
                    "kv_page_indices": kvp_b,
                    "kv_last_page_lens": kvlp_b,
                    "out": out_a,
                    "workspace": ws_a,
                    "block_size": block_size,
                    "max_num_partitions": max_part_b,
                    "scale": sc_b,
                    "k_scale": ks_b,
                    "v_scale": vs_b,
                    "alibi_slopes": None,
                    "kv_cache_dtype": "auto",
                    "kv_cache_layout": "HND",
                    "logits_soft_cap": 0.0,
                    "flops": flops_b,
                    "rw_bytes": rw_b,
                    "tag": f"{ctx_len=},{num_seqs=}",
                }
            )
        _benchmark_aiter_pa_ragged(
            aiter_cases, AITER_RUN_COUNT, perf_name="aiter_paged_attention_ragged"
        )

    return 0


def main() -> int:
    return compare_pa_ragged_to_torch()


if __name__ == "__main__":
    sys.exit(main())
