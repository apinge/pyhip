"""Packed-weight and padded CUDA Graph contract checks; no SGLang required.

Example: python3 test_graph_buckets.py --buckets 4
Captures B rows once, then replays with M=B..0..B live rows without recapture.
This is an integration reference test, not a production serving adapter.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import flydsl.compiler as flyc
import torch

if __package__:
    from . import prefetch_up
    from .large_down import DownConfig, default_config as large_config, pair_launcher
    from .small_batch import default_config as small_config
    from .support import TOLERANCES, capture, checkpoint_pairs, reference, synthetic
else:
    import prefetch_up
    from large_down import DownConfig, default_config as large_config, pair_launcher
    from small_batch import default_config as small_config
    from support import TOLERANCES, capture, checkpoint_pairs, reference, synthetic

C, H, R = 4, 2560, 320
K = C * H
GUARD = 16
SENTINEL = 97.0


def pack_original_weights(w_down, w_up):
    """Model-load recipe; inputs are original logical BF16 matrices."""
    if w_down.shape != (R, K) or w_up.shape != (K, R):
        raise ValueError("expected original W_down[320,10240] and W_up[10240,320]")
    if w_down.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
        raise ValueError("HC weights must be BF16")
    if not w_down.is_cuda or w_down.device != w_up.device or torch.version.hip is None:
        raise ValueError("weights must be on the same ROCm device")
    up_hc = w_up.reshape(C, H, R).permute(1, 0, 2).contiguous().reshape(K, R)
    return prefetch_up.preshuffle_weight(w_down), prefetch_up.preshuffle_weight(up_hc)


def selected_configs(bucket):
    if not 1 <= bucket <= 32:
        raise ValueError("capture bucket must be 1..32")
    base = replace(prefetch_up.default_config(bucket), hidden_pad=4, prefetch_low=False)
    if bucket <= 16:
        config = small_config(bucket)
        down = config.down
        up = config.up_config(base)
    else:
        down = large_config(bucket)
        up = replace(base, down_mode="partial", down_n=64, split_k=4)
    down.validate()
    up.validate()
    assert down.global_split == up.split_k == 4
    return down, up


def guarded_tensor(shape, dtype, device):
    count = 1
    for dim in shape:
        count *= dim
    storage = torch.full((count + 2 * GUARD,), SENTINEL, dtype=dtype, device=device)
    tensor = storage[GUARD:-GUARD].view(shape)
    assert tensor.is_contiguous() and tensor.data_ptr() % 16 == 0
    return tensor, storage


@dataclass
class BucketCase:
    bucket: int
    padded_rows: int
    down_config: DownConfig
    up_config: prefetch_up.Config
    x: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    w_down: torch.Tensor
    w_up: torch.Tensor
    storage: tuple
    dispatch: object

    def run(self):
        self.dispatch(
            self.x.view(-1), self.w_down, self.w_up, self.partial,
            self.output.view(-1), torch.cuda.current_stream(self.x.device),
        )


def compile_bucket(bucket, packed_down, packed_up):
    """Compile from existing packed weights, without constructing a control."""
    down, up = selected_configs(bucket)
    for weight in (packed_down, packed_up):
        if weight.shape != (R * K,) or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            raise ValueError("packed weights must be contiguous BF16 flat tensors with R*K elements")
    if not packed_down.is_cuda or packed_up.device != packed_down.device:
        raise ValueError("packed weights must share a ROCm device")
    device = packed_down.device
    padded = (bucket + down.block_m - 1) // down.block_m * down.block_m
    x, x_storage = guarded_tensor((bucket, K), torch.bfloat16, device)
    partial, p_storage = guarded_tensor((4 * padded * R,), torch.float32, device)
    output, y_storage = guarded_tensor((bucket, H), torch.bfloat16, device)
    with torch.cuda.device(device):
        dispatch = flyc.compile(
            pair_launcher(bucket, down, up),
            x.view(-1), packed_down, packed_up, partial, output.view(-1),
            torch.cuda.current_stream(device),
        )
    return BucketCase(bucket, padded, down, up, x, partial, output, packed_down, packed_up,
                      (x_storage, p_storage, y_storage), dispatch)


def check_replays(case, graph, w_down, w_up, tail_mode, seed):
    generator = torch.Generator(device=case.x.device).manual_seed(seed)
    bucket = case.bucket
    # Start with a populated full bucket, so shrinking replays retain old rows.
    case.x.normal_(generator=generator)
    graph.replay()
    live_counts = list(range(bucket, -1, -1)) + list(range(1, bucket + 1))
    maximum = 0.0
    nan_tail_observed = False
    tol = TOLERANCES[torch.bfloat16]
    for live in live_counts:
        active = torch.randn(live, K, dtype=torch.bfloat16, device=case.x.device, generator=generator)
        case.x[:live].copy_(active)
        if tail_mode == "zero":
            case.x[live:].zero_()
        elif tail_mode == "nan":
            case.x[live:].fill_(float("nan"))
        # stale deliberately does not touch X[live:bucket], like a head-only fill.
        before = case.x.clone()
        case.partial.fill_(float("nan"))
        case.output.fill_(float("nan"))
        graph.replay()
        expected = reference(active, w_down, w_up)
        actual = case.output[:live].double()
        label = f"B={bucket}, M={live}, tail={tail_mode}"
        assert torch.allclose(actual, expected, **tol), f"{label}: live-row FP64 mismatch"
        if live:
            error = ((actual - expected).abs() / (tol["atol"] + tol["rtol"] * expected.abs())).max().item()
            maximum = max(maximum, error)
        partials = case.partial.view(4, case.padded_rows, R)
        assert torch.isfinite(partials[:, :live]).all(), f"{label}: live P not overwritten"
        assert torch.count_nonzero(partials[:, bucket:]) == 0, f"{label}: internal tile padding"
        if tail_mode == "zero":
            assert torch.count_nonzero(partials[:, live:]) == 0, f"{label}: zero tail P"
            assert torch.count_nonzero(case.output[live:]) == 0, f"{label}: zero tail Y"
        elif tail_mode == "stale":
            assert torch.isfinite(partials).all(), f"{label}: stale P not overwritten"
            assert torch.isfinite(case.output).all(), f"{label}: stale Y not overwritten"
        elif live < bucket:
            nan_tail_observed |= bool(torch.isnan(case.output[live:]).any().item())
        assert torch.allclose(case.x, before, rtol=0, atol=0, equal_nan=True), f"{label}: X mutated"
        for storage in case.storage:
            assert torch.all(storage[:GUARD] == SENTINEL), f"{label}: leading guard overwritten"
            assert torch.all(storage[-GUARD:] == SENTINEL), f"{label}: trailing guard overwritten"
    return {
        "tail": tail_mode, "live_counts": live_counts, "replays_checked": len(live_counts),
        "nonempty_fp64_checks": sum(m > 0 for m in live_counts), "max_scaled_error": maximum,
        "nan_tail_output_observed": nan_tail_observed,
        "live_fp64_passed": True, "scratch_overwrite_passed": True, "guards_passed": True,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", type=int, nargs="+", default=list(range(1, 33)))
    parser.add_argument("--weights", type=int, default=2)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--tails", nargs="+", choices=["zero", "stale", "nan"], default=["zero", "stale", "nan"])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not 1 <= b <= 32 for b in args.buckets) or args.weights < 1:
        parser.error("buckets must be 1..32 and weights must be positive")
    if args.model_path and not (args.model_path / "model.safetensors.index.json").is_file():
        parser.error("checkpoint index not found; use --synthetic without a model")
    if torch.version.hip is None or not torch.cuda.is_available():
        parser.error("a ROCm device is required")
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(":", 1)[0] != "gfx942":
        warnings.warn(f"Bucket contract was developed on gfx942; validating {props.gcnArchName}", RuntimeWarning)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs = checkpoint_pairs(args.weights, args.model_path) if args.model_path else (
        (f"synthetic_{index}", *synthetic(1, seed=args.seed + index)[1:]) for index in range(args.weights)
    )
    with args.output.open("x") if args.output else nullcontext() as log:
        def emit(record):
            if log:
                log.write(json.dumps(record) + "\n")
                log.flush()

        sources = {}
        for name in ("test_graph_buckets.py", "small_batch.py", "large_down.py", "prefetch_up.py", "support.py"):
            content = Path(__file__).with_name(name).read_bytes()
            sources[name] = hashlib.sha256(content).hexdigest()
            if log:
                with args.output.with_name(args.output.stem + "." + name).open("xb") as snapshot:
                    snapshot.write(content)
        emit({"type": "environment", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "gpu": props.name, "arch": props.gcnArchName, "compute_units": props.multi_processor_count,
              "torch": torch.__version__, "hip": torch.version.hip,
              "flydsl": importlib.metadata.version("flydsl"), "source_sha256": sources,
              "weight_source": "checkpoint" if args.model_path else "synthetic",
              "tolerance": TOLERANCES[torch.bfloat16], "timing": "correctness only; no performance claim"})
        pair_count = total_replays = total_fp64 = 0
        for index, (name, w_down, w_up) in enumerate(pairs):
            packed_down, packed_up = pack_original_weights(w_down, w_up)
            before_down, before_up = packed_down.clone(), packed_up.clone()
            for bucket in args.buckets:
                case = compile_bucket(bucket, packed_down, packed_up)
                assert case.w_down.data_ptr() == packed_down.data_ptr()
                assert case.w_up.data_ptr() == packed_up.data_ptr()
                case.x.zero_()
                graph, _ = capture([case.run])
                results = [check_replays(case, graph, w_down, w_up, tail, args.seed + index + bucket * 1000)
                           for tail in args.tails]
                assert torch.equal(packed_down, before_down)
                assert torch.equal(packed_up, before_up)
                total_replays += sum(r["replays_checked"] for r in results)
                total_fp64 += sum(r["nonempty_fp64_checks"] for r in results)
                emit({"type": "result", "weight": name, "bucket": bucket,
                      "down_config": asdict(case.down_config), "up_config": asdict(case.up_config),
                      "partial_shape": [4, case.padded_rows, R], "tails": results,
                      "packed_once_per_pair": True, "packed_pointers_shared": True, "packed_bytes_unchanged": True})
                if args.weights <= 4:
                    print(f"{name}: B={bucket}, M={bucket}..0..{bucket}, tails={','.join(args.tails)} PASS", flush=True)
                del graph, case
            pair_count += 1
            if args.weights > 4:
                print(f"[{pair_count}/{args.weights}] {name}: {len(args.buckets)} buckets PASS", flush=True)
        assert pair_count == args.weights, f"requested {args.weights} pairs, found {pair_count}"
        emit({"type": "summary", "weight_pairs": pair_count, "replays_checked": total_replays,
              "nonempty_fp64_checks": total_fp64, "passed": True})
        print(f"PASS: {pair_count} packed-once pairs, {total_replays} replay checks, {total_fp64} nonempty FP64 checks", flush=True)


if __name__ == "__main__":
    main()
