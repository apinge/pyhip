import torch
import pyhip

torch.cuda.set_device(4)
torch.set_default_device("cuda")

B = 10
HQ = 4
HK = 1
S = 256
KV_LEN = 8000
DT = torch.bfloat16
BLOCK_SIZE = 1
BLOCK_NUM = B * KV_LEN + 1000
BUF_COPY = 32
KV_PART_SIZE = 256

FP8_DT = torch.float8_e4m3fnuz

print(f"kvcache = {B * (HK * KV_LEN * S * 1 * 2) // 1024 // 1024:,} MB (fp8)")
workspace_buffer = torch.empty(
    (512 * HQ * 256 * S) * 4
    + 2 * (512 * HQ * 256) * 4,
    dtype=torch.uint8,
)

query = torch.randint(-2, 3, [B, HQ, S], dtype=DT)

key_caches = []
value_caches = []
kv_indptrs = []
kv_page_indices_ = []
kv_last_page_lens_ = []

k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

for _ in range(BUF_COPY):
    key_cache = torch.randint(-2, 3, [BLOCK_NUM, BLOCK_SIZE, HK, S], dtype=DT).to(FP8_DT)
    value_cache = torch.randint(-2, 3, [BLOCK_NUM, BLOCK_SIZE, HK, S], dtype=DT).to(FP8_DT)
    batch_start = [0] * (B + 1)
    for b in range(B):
        batch_start[b + 1] = (b + 1) * KV_LEN
    kv_indptr = torch.tensor(batch_start, dtype=torch.int32)
    kv_page_indices = torch.linspace(1, KV_LEN * B, KV_LEN * B, dtype=torch.int32)
    kv_last_page_lens = torch.ones([KV_LEN], dtype=torch.int32)
    key_caches.append(key_cache)
    value_caches.append(value_cache)
    kv_indptrs.append(kv_indptr)
    kv_page_indices_.append(kv_page_indices)
    kv_last_page_lens_.append(kv_last_page_lens)

scale = 1 / (S**0.5)


def test_aiter(
    query,
    key_cache,
    value_cache,
    scale,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    block_size=BLOCK_SIZE,
    max_num_partitions=256,
    alibi_slopes=None,
    kv_cache_dtype="fp8_e4m3",
    kv_cache_layout="NHD",
    logits_soft_cap=0.0,
    k_scale=k_scale,
    v_scale=v_scale,
    fp8_out_scale=None,
    partition_size=256,
    mtp=1,
):
    out = torch.empty([B, HQ, S], dtype=DT)
    torch.ops.aiter.paged_attention_ragged(
        out,
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
        fp8_out_scale,
        partition_size,
        mtp,
    )
    return out


import aiter

out = test_aiter(
    query=query,
    key_cache=key_caches[-1],
    value_cache=value_caches[-1],
    scale=scale,
    kv_indptr=kv_indptrs[-1],
    kv_page_indices=kv_page_indices_[-1],
    kv_last_page_lens=kv_last_page_lens_[-1],
)

i = 0
for _ in range(10):
    with pyhip.cudaPerf(
        # flops: B批 * 每批GQA重复次数(HQ/HK) * KV_LEN个token * S维 * 2(Q@K + Score@V各一次乘加)  * 2(乘和加各算一次op)
        B * HQ // HK * KV_LEN * S * 2 * 2,
        # bytes: B批 * HK个KV head * KV_LEN个token * S维 * 1字节(fp8) * 2(K+V)
        B * (HK * KV_LEN * S * 1 * 2),
        name="aiter_fp8",
    ):
        test_aiter(
            query=query,
            key_cache=key_caches[i],
            value_cache=value_caches[i],
            scale=scale,
            kv_indptr=kv_indptrs[i],
            kv_page_indices=kv_page_indices_[i],
            kv_last_page_lens=kv_last_page_lens_[i],
        )
    i = (i + 1) % BUF_COPY

print("done")
