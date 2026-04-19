/*
 * Host 入口：与 aiter pa_ragged.cpp.jinja 中 GOLDEN 路径一致，一次 host 调用顺序 launch
 *   pa  -> pa_reduce
 * 由 hipcc 与 pa_ragged_kernels.cpp 链成共享库，Python 仅 ctypes 调用本文件导出符号，
 * 不再通过 pyhip.module 分别 launch 两个 kernel。
 */
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#ifndef HQ
#define HQ 32
#define HK 4
#endif

// 与 pa_ragged_kernels.cpp 中定义一致（仅声明，定义在同链单元）
__global__ void __launch_bounds__(256, 2) pa(__bf16 *query, __bf16 *key_cache, __bf16 *value_cache,
                                             uint *kv_indptr, uint *kv_page_indices, __bf16 *out_seg,
                                             float *qk_ptr, float *max_out, float *sum_out, uint q_stride);

__global__ void __launch_bounds__(256, 2)
    pa_reduce(uint *kv_indptr, __bf16 *out_seg, float *max_out, float *sum_out, __bf16 *out, uint max_part);

extern "C" {

hipError_t pa_ragged_forward_bf16(void *stream, unsigned int num_seqs, unsigned int max_num_partitions,
                                  unsigned int q_stride, __bf16 *query, __bf16 *key_cache, __bf16 *value_cache,
                                  uint *kv_indptr, uint *kv_page_indices, __bf16 *tmp_out_seg, float *qk_ptr,
                                  float *max_logits_buf, float *exp_sums_buf, __bf16 *out,
                                  unsigned int max_part) {
    hipStream_t s = reinterpret_cast<hipStream_t>(stream);
    dim3 grid_pa(num_seqs, HK, max_num_partitions);
    dim3 grid_r(num_seqs, HQ);
    dim3 block(256);

    hipLaunchKernelGGL(pa, grid_pa, block, 0, s, query, key_cache, value_cache, kv_indptr, kv_page_indices,
                       tmp_out_seg, qk_ptr, max_logits_buf, exp_sums_buf, q_stride);

    hipLaunchKernelGGL(pa_reduce, grid_r, block, 0, s, kv_indptr, tmp_out_seg, max_logits_buf, exp_sums_buf, out,
                         max_part);

    return hipGetLastError();
}
}
