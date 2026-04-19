"""
通过 `pa_ragged_kernels.cpp` 内 **C++ host 入口** `pa_ragged::launch_bf16_two_kernels` 顺序 launch
`pa` 与 `pa_reduce`（与 aiter `pa_ragged.cpp.jinja` 中双 `<<<>>>` 一致），不再在 Python 里分两次 `pyhip.module` 调 kernel。

设备代码与 host 由 hipcc **全量**编成共享库 `libpa_ragged_bf16.so`，Python 用 ctypes 调 C ABI 符号 `pyhip_pa_ragged_launch_bf16`
（薄封装，内部仍为 C++ 实现；避免 pyhip.module 解析 device-only 时无法带 host launch）。

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

import ctypes
import os
import subprocess
from typing import Optional

import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "pa_ragged_kernels.cpp")
_PYHIP_CACHE = os.getenv("PYHIP_CACHE_DIR", os.path.expanduser("~/.pyhip"))
_SO_NAME = "libpa_ragged_bf16.so"
_SO_PATH = os.path.join(_PYHIP_CACHE, _SO_NAME)

_LIB: Optional[ctypes.CDLL] = None
_LAUNCH = None


def _compile_so_if_needed() -> None:
    os.makedirs(_PYHIP_CACHE, exist_ok=True)
    if os.path.isfile(_SO_PATH) and os.path.getmtime(_SO_PATH) >= os.path.getmtime(_SRC):
        return
    tmp = _SO_PATH + ".building"
    cmd = ["hipcc", "-std=c++20", "-fPIC", "-shared", "-O2", "-o", tmp, _SRC]
    subprocess.check_call(cmd)
    os.replace(tmp, _SO_PATH)


def _load_launch() -> None:
    global _LIB, _LAUNCH
    if _LAUNCH is not None:
        return
    _compile_so_if_needed()
    _LIB = ctypes.CDLL(_SO_PATH)
    fn = _LIB.pyhip_pa_ragged_launch_bf16
    fn.argtypes = [
        ctypes.c_void_p,  # stream
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int32
    _LAUNCH = fn


HQ = 32
HK = 4
HEAD_DIM = 128
KV_PART_SIZE = 256


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

    _load_launch()
    stream = torch.cuda.current_stream()
    # PyTorch 2: cuda_stream 为底层 stream 句柄（整数）
    stream_ptr = ctypes.c_void_p(stream.cuda_stream)
    err = _LAUNCH(
        stream_ptr,
        int(num_seqs) & 0xFFFFFFFF,
        int(max_num_partitions) & 0xFFFFFFFF,
        int(q_stride) & 0xFFFFFFFF,
        ctypes.c_void_p(query.data_ptr()),
        ctypes.c_void_p(key_cache.data_ptr()),
        ctypes.c_void_p(value_cache.data_ptr()),
        ctypes.c_void_p(kv_indptr.data_ptr()),
        ctypes.c_void_p(kv_page_indices.data_ptr()),
        ctypes.c_void_p(tmp_out_seg.data_ptr()),
        ctypes.c_void_p(qk_stub.data_ptr()),
        ctypes.c_void_p(max_logits_buf.data_ptr()),
        ctypes.c_void_p(exp_sums_buf.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
    )
    if err != 0:
        raise RuntimeError(f"pyhip_pa_ragged_launch_bf16 failed, hipError_t={err}")
