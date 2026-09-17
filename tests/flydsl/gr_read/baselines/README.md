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
