/*
 * Standalone Gemma norm entry points for HIP (fp16).
 * Extracted from FlashInfer csrc_rocm/norm.cu (gemma_rmsnorm, gemma_fused_add_rmsnorm).
 *
 * API: raw device pointers, fp16 only. Compile with hipcc -x hip.
 */
#include "gemma_norm_hip.cuh"

#include <hip/hip_runtime.h>

namespace gemma_norm {

extern "C" hipError_t gemma_rmsnorm_fp16(__half* output, const __half* input, const __half* weight,
                                          uint32_t batch_size, uint32_t hidden_size, double eps,
                                          bool enable_pdl, hipStream_t stream) {
  (void)enable_pdl;
  if (batch_size == 0 || hidden_size == 0) return hipSuccess;
  return GemmaRMSNorm(const_cast<__half*>(input), const_cast<__half*>(weight), output, batch_size,
                      hidden_size, hidden_size, hidden_size, static_cast<float>(eps), false,
                      stream);
}

extern "C" hipError_t gemma_fused_add_rmsnorm_fp16(__half* input, __half* residual,
                                                   const __half* weight, uint32_t batch_size,
                                                   uint32_t hidden_size, double eps,
                                                   bool enable_pdl, hipStream_t stream) {
  (void)enable_pdl;
  if (batch_size == 0 || hidden_size == 0) return hipSuccess;
  return GemmaFusedAddRMSNorm(input, residual, const_cast<__half*>(weight), batch_size,
                              hidden_size, hidden_size, hidden_size, static_cast<float>(eps), false,
                              stream);
}

}  // namespace gemma_norm
