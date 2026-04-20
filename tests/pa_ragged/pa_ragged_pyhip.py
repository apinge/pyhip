"""
与 `conv_depthwise.py` 相同：`pyhip.module("pa_ragged_kernels.cpp", ...)` 编译 device-only `.co`，
在 Python 里顺序 launch `pa` → `pa_reduce`（对齐 aiter `pa_ragged.cpp.jinja` 双 kernel），无需单独 `hipcc -shared` / ctypes。

写死宏与 `/root/workspace/pa/csrc/cpp_itfs/pa/pa_ragged.cpp.jinja` 默认一致：
`HQ=32, HK=4, S=128, KV_PART_SIZE=256`。

期望张量：
  query: [num_seqs, 32, 128] bf16
  out:   同形状
  key_cache / value_cache: NHD [num_blocks, block_size, 4, 128] bf16
  kv_indptr: int32 [num_seqs+1]
  kv_page_indices: int32 展平页表

workspace 布局与 host 参考实现一致：先 max_logits 与 exp_sums 两段 float32，再 tmp_out_seg bf16
（此处拆成 `torch.empty(2*nstat)` + `tmp_out_seg` 三个张量，指针语义一致即可）。
"""
from __future__ import annotations

import os
from typing import Optional

import pyhip
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "pa_ragged_kernels.cpp")
mod = pyhip.module(_SRC, "-O2")
HQ = 32
HK = 4
HEAD_DIM = 128
KV_PART_SIZE = 256
NUM_THREADS = 256


def run_pa_ragged_bf16(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    max_num_partitions: int,
    q_stride: int,
    *,
    workspace_f2: Optional[torch.Tensor] = None,
    tmp_out_seg_buf: Optional[torch.Tensor] = None,
    qk_stub: Optional[torch.Tensor] = None,
) -> None:
    if query.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise TypeError("query/out expect bf16")
    device = query.device
    kv_indptr = kv_indptr.to(device=device, dtype=torch.int32)
    kv_page_indices = kv_page_indices.to(device=device, dtype=torch.int32)
    num_seqs, num_heads, head_dim = query.shape
    if num_heads != HQ or head_dim != HEAD_DIM:
        raise ValueError(f"need num_heads={HQ}, head_dim={HEAD_DIM}, got {num_heads}, {head_dim}")
    nstat = num_seqs * num_heads * max_num_partitions
  #  breakpoint()
    if workspace_f2 is None:
        f2 = torch.empty(2 * nstat, dtype=torch.float32, device=device)
    else:
        if workspace_f2.numel() != 2 * nstat or workspace_f2.dtype != torch.float32:
            raise ValueError("workspace_f2 must be float32 of size 2*nstat")
        f2 = workspace_f2
    exp_sums_buf = f2[:nstat]
    max_logits_buf = f2[nstat:]

    if tmp_out_seg_buf is None:
        tmp_out_seg = torch.empty(
            (num_seqs, num_heads, max_num_partitions, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            memory_format=torch.contiguous_format,
        )
    else:
        exp_shape = (num_seqs, num_heads, max_num_partitions, HEAD_DIM)
        if tmp_out_seg_buf.shape != exp_shape or tmp_out_seg_buf.dtype != torch.bfloat16:
            raise ValueError(f"tmp_out_seg_buf must be bf16 shape {exp_shape}")
        tmp_out_seg = tmp_out_seg_buf

    if qk_stub is None:
        qk_stub = torch.empty(1, dtype=torch.float32, device=device)
    #breakpoint()
    tmp_out_seg.zero_()
    max_logits_buf.zero_()
    exp_sums_buf.zero_()
    out.zero_()
   
  # 256x64 = 16384
    mod.pa(
        [num_seqs, HK, max_num_partitions],  # [num_seqs, 4, 64]
        [NUM_THREADS],
        query.data_ptr(), # [1, 32, 128] 
        key_cache.data_ptr(), #  ([16384, 1, 4, 128]
        value_cache.data_ptr(), # [16384, 1, 4, 128]
        kv_indptr.data_ptr(), # [2] [    0, 16384]
        kv_page_indices.data_ptr(), # shape [16384]
        tmp_out_seg.data_ptr(), # [1, 32, 64, 128] [seq, Q_head, partition, head_dim]
        qk_stub.data_ptr(), # [1]
        max_logits_buf.data_ptr(), # torch.Size([2048])  32*64 Q_head, partition
        exp_sums_buf.data_ptr(), # exp_sums_buf shape: torch.Size([2048]) 32*64
        int(q_stride) & 0xFFFFFFFF, #  4096 0xFFFFFFFF是32位掩码
    )
   # print(f"tmp_out_seg shape: {tmp_out_seg.shape} max_logits_buf shape: {max_logits_buf.shape} exp_sums_buf shape: {exp_sums_buf.shape}")
    mod.pa_reduce(
        [num_seqs, HQ],  # [num_seqs, HQ]
        [NUM_THREADS],
        kv_indptr.data_ptr(),
        tmp_out_seg.data_ptr(),
        max_logits_buf.data_ptr(),
        exp_sums_buf.data_ptr(),
        out.data_ptr(), # out shape: torch.Size([1, 32, 128])
        int(max_num_partitions) & 0xFFFFFFFF,
    )
    #print(f"out shape: {out.shape}")
