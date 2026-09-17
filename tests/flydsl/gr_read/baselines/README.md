# Fixed Triton Baseline

`hc_mix_triton.py` is an unmodified copy of
`python/sglang/srt/layers/hc_mix_triton.py` from SGLang commit
`8cf5501b6913f57a2e7c8dcee52b625fc8ab23c3` (apinge's MI308X tuning):

https://github.com/apinge/sglang/blob/8cf5501b6913f57a2e7c8dcee52b625fc8ab23c3/python/sglang/srt/layers/hc_mix_triton.py

SHA256: `647a90bf2622e8e145e2b9784059afb9ed9593287f7ce54b980eb54e6b669854`.

This source is licensed under Apache-2.0; the upstream license is included in
`LICENSE`. It is benchmark source, not historical timing data. Do not reformat
or retune this frozen baseline. Its `fused_hc_mix` entry needs Torch and Triton,
not a SGLang checkout or installation. No SGLang production dispatch is changed.

## Three-stage Comparison: 626c6413

`hc_mix_triton_626c6413.py` is a separate, unmodified upstream snapshot:

https://github.com/sgl-project/sglang/blob/626c64132d7f197de2053232264aeabb37ad156b/python/sglang/srt/layers/hc_mix_triton.py

- Git blob: `c7e66d57de0faab54a20469529127dee4da29d14`.
- SHA256: `4a36c9cb92cfcccdd2232ffa6e3a9a5c086361c03d7f30fc3c83bd653739a394`.
- The same upstream Apache-2.0 `LICENSE` applies.
- This snapshot replaces the persistent barrier kernel with down/reduce/up
  launches and stores the SiLU activation in the input dtype (BF16 in our test).

The final-entry comparison is `bench_small_batch.py --selected`, where this
baseline is named `triton3stage` in printed tables and new JSONL results.
The source filename, commit and SHA256 remain pinned above. It does not replace
`hc_mix_triton.py`, the existing persistent baseline, or SGLang code.
