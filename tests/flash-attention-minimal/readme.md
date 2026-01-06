# Flash Attention Minimal

origin version from [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal/tree/main)

## Performance Data

On H20
```
attn values sanity check: True
dt = 1186.94401 us
dt = 1134.75204 us
dt = 1488.16001 us
Max shared memory: 49152, requested shared memory: 28672 \nMax shared memory: 49152, requested shared memory: 28672 \nMax shared memory: 49152, requested shared memory: 28672 \nMax shared memory: 49152, req
```
On MI308
```
Batch Size: 16, Num Heads 12, Sequence Length64, Head Dim 64
Max shared memory: 65536, requested shared memory: 28672
torch.Size([16, 12, 64, 64])
PASS
dt = 3097.69011 us
dt = 3085.08992 us
dt = 3088.33003 us

```