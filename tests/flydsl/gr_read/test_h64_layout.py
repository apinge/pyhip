"""Standalone H64 weight ABI and graph-bucket checks; uses ordinary asserts.

The kernel remains an opt-in candidate. No SGLang or checkpoint is required.
"""

import argparse
import hashlib
import importlib.metadata
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch

if __package__:
    from .bench_h64_layout import idle_preflight
    from .h64_layout import H64GRRead, LAYOUT_ID, default_configs, prepare_weights
    from .support import capture, checkpoint_pairs, synthetic
    from .test_graph_buckets import BucketCase, check_replays, guarded_tensor
else:
    from bench_h64_layout import idle_preflight
    from h64_layout import H64GRRead, LAYOUT_ID, default_configs, prepare_weights
    from support import capture, checkpoint_pairs, synthetic
    from test_graph_buckets import BucketCase, check_replays, guarded_tensor

C, H, K, R = 4, 2560, 10240, 320


def check_layout(wd, wu, packed_down, packed_up):
    def pack(w):
        n, k = w.shape
        return w.detach().reshape(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(-1)

    # Exact preparation recipe from prefill commit 3f472e33, not an old-layout conversion.
    interleaved = wu.detach().reshape(C, H // 64, 2, 4, 2, 4, R).permute(1, 0, 2, 4, 3, 5, 6).contiguous().reshape(K, R)
    assert torch.equal(packed_down, pack(wd))
    assert torch.equal(packed_up, pack(interleaved))
    original = torch.arange(K)
    c, h = original // H, original % H
    physical = 256 * (h // 64) + 64 * c + 32 * (h // 32 % 2) + 16 * (h // 4 % 2) + 4 * (h // 8 % 4) + h % 4
    reordered = original.reshape(C, H // 64, 2, 4, 2, 4).permute(1, 0, 2, 4, 3, 5).reshape(-1)
    assert torch.equal(reordered[physical], original)
    assert torch.equal(physical.sort().values, original)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", nargs="+", type=int, default=list(range(1, 33)))
    parser.add_argument("--weights", type=int, default=1)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--up-n", type=int)
    parser.add_argument("--up-k", type=int)
    parser.add_argument("--skip-padding", type=int, choices=[0, 1])
    parser.add_argument("--preload-weights", type=int, choices=[0, 1])
    parser.add_argument("--strategy", choices=["remap", "plane", "register"], default="plane")
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument("--b-first", type=int, choices=[0, 1], default=0)
    parser.add_argument("--weight-copy-bits", type=int, choices=[64, 128], default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not 1 <= b <= 32 for b in args.buckets) or args.weights < 1:
        parser.error("buckets must be 1..32 and weights must be positive")
    preflight = idle_preflight()
    pairs = checkpoint_pairs(args.weights, args.model_path) if args.model_path else (
        (f"synthetic_{i}", *synthetic(1, seed=args.seed + i)[1:]) for i in range(args.weights)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    count = replays = 0
    with args.output.open("x") if args.output else nullcontext() as log:
        def emit(record):
            if log:
                log.write(json.dumps(record) + "\n")
                log.flush()

        sources = {}
        for name in ("test_h64_layout.py", "h64_layout.py", "test_graph_buckets.py", "large_down.py", "support.py"):
            content = Path(__file__).with_name(name).read_bytes()
            sources[name] = hashlib.sha256(content).hexdigest()
            if log:
                with args.output.with_name(args.output.stem + "." + name).open("xb") as snapshot:
                    snapshot.write(content)
        emit({"type": "environment", "preflight": preflight, "layout": LAYOUT_ID,
              "torch": torch.__version__, "hip": torch.version.hip, "flydsl": importlib.metadata.version("flydsl"),
              "source_sha256": sources,
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
        for name, wd, wu in pairs:
            pd, pu = prepare_weights(wd, wu)
            check_layout(wd, wu, pd, pu)
            pd_before, pu_before = pd.clone(), pu.clone()
            for bucket in args.buckets:
                _, config = default_configs(bucket)
                overrides = {k: v for k, v in (("up_n", args.up_n), ("block_k", args.up_k),
                             ("skip_padding", None if args.skip_padding is None else bool(args.skip_padding))) if v is not None}
                config = replace(config, waves=args.waves, **overrides)
                reader = H64GRRead(bucket, pd, pu, up_config=config, strategy=args.strategy,
                                   weight_copy_bits=args.weight_copy_bits, b_first=bool(args.b_first),
                                   preload_weights=args.preload_weights)
                x, sx = guarded_tensor((bucket, K), torch.bfloat16, pd.device)
                reader.partial, sp = guarded_tensor(reader.partial.shape, torch.float32, pd.device)
                reader.output, sy = guarded_tensor((bucket, H), torch.bfloat16, pd.device)
                case = BucketCase(bucket, (bucket + 15) // 16 * 16, reader.down_config, config,
                                  x, reader.partial, reader.output, pd, pu, (sx, sp, sy), reader.dispatch)
                assert reader.w_down.data_ptr() == pd.data_ptr()
                assert reader.w_up.data_ptr() == pu.data_ptr()
                x.zero_()
                graph, _ = capture([lambda: reader(x)])
                results = [check_replays(case, graph, wd, wu, tail, args.seed + bucket * 1000 + count)
                           for tail in ("zero", "stale", "nan")]
                assert torch.equal(pd, pd_before) and torch.equal(pu, pu_before)
                replays += sum(r["replays_checked"] for r in results)
                emit({"type": "result", "weight": name, "bucket": bucket, "tails": results,
                      "exact_prefill_layout": True, "shared_weight_pointers": True, "weights_unchanged": True})
                print(f"{name}: B={bucket}, M={bucket}..0..{bucket}, zero/stale/NaN tails PASS", flush=True)
                del graph, reader, case
            count += 1
        assert count == args.weights
        emit({"type": "summary", "passed": True, "weight_pairs": count, "replay_checks": replays})
        print(f"PASS: {count} packed-once pairs, {replays} graph replay checks", flush=True)


if __name__ == "__main__":
    main()
