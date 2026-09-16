"""Full-checkpoint correctness and matched rotating-weight graph benchmark."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
import triton
from kernel import Config, GRRead
from support import (
    BASELINE_COMMIT,
    BASELINE_PATH,
    BASELINE_SHA256,
    MODEL_PATH,
    TOLERANCES,
    capture,
    checkpoint_pairs,
    load_flydsl_baseline,
    load_triton_baseline,
    reference,
    time_graph,
    torch_mix,
)


def choose_config(rows, buckets):
    for bucket in buckets:
        if rows <= bucket["max_rows"]:
            return Config(**bucket["config"])
    raise ValueError(f"no configured bucket for T={rows}")


def output_error(actual, expected):
    difference = (actual.double() - expected).abs()
    tolerance = TOLERANCES[torch.bfloat16]
    scaled = difference / (tolerance["atol"] + tolerance["rtol"] * expected.abs())
    return {"max_abs": difference.max().item(), "max_scaled": scaled.max().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=list(range(1, 25)))
    parser.add_argument("--weights", type=int, default=100)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--triton-kernel", type=Path, default=BASELINE_PATH)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--graph-repeats", type=int, default=2)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-kernel", type=Path)
    args = parser.parse_args()
    if any(not 1 <= row <= 24 for row in args.rows):
        raise ValueError("benchmark rows must be 1..24")
    if min(args.weights, args.rounds, args.samples, args.replays, args.graph_repeats) < 1:
        raise ValueError("benchmark counts must be positive")
    if not (args.model_path / "model.safetensors.index.json").is_file():
        parser.error(f"checkpoint index not found under {args.model_path}; set --model-path to the model directory")
    if not args.triton_kernel.is_file():
        parser.error(
            f"Triton source not found: {args.triton_kernel}; restore baselines/hc_mix_triton.py or set --triton-kernel"
        )
    source = args.triton_kernel.read_bytes()
    if hashlib.sha256(source).hexdigest() != BASELINE_SHA256:
        parser.error(
            f"Triton source {args.triton_kernel} differs from the verified 8cf5501b baseline; use the bundled copy"
        )
    with args.configs.open() as f:
        buckets = json.load(f)["buckets"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch_compiled = torch.compile(torch_mix, dynamic=False, fullgraph=True)
    baseline = load_triton_baseline(args.triton_kernel)
    previous = load_flydsl_baseline(args.previous_kernel) if args.previous_kernel else None
    rng = random.Random(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    props = torch.cuda.get_device_properties(0)
    with args.output.open("x") as log:
        metadata = {
            "type": "environment",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": props.name,
            "arch": props.gcnArchName,
            "compute_units": props.multi_processor_count,
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "flydsl": importlib.metadata.version("flydsl"),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "dtype": "bfloat16",
            "baseline_commit": BASELINE_COMMIT,
            "baseline_source": str(args.triton_kernel),
            "baseline_sha256": hashlib.sha256(source).hexdigest(),
            "model_path": str(args.model_path),
            "seed": args.seed,
            "configurations": buckets,
            "rounds": args.rounds,
            "samples": args.samples,
            "graph_replays_per_sample": args.replays,
            "graph_repeats_per_weight": args.graph_repeats,
            "reference": "same formula in FP64; synthetic BF16 normalized inputs, actual checkpoint HC weights",
            "measurement": "HIP events around graph replay; warmup, JIT, weight packing and reference excluded",
        }
        if previous:
            previous_source = args.previous_kernel.read_bytes()
            metadata["previous_flydsl_source"] = str(args.previous_kernel)
            metadata["previous_flydsl_sha256"] = hashlib.sha256(previous_source).hexdigest()
            args.output.with_name(args.output.stem + ".previous_kernel.py").write_bytes(previous_source)
        for name in ("kernel.py", "support.py", "benchmark.py", "hc_mix_triton.py"):
            content = source if name == "hc_mix_triton.py" else Path(__file__).with_name(name).read_bytes()
            destination = args.output.with_name(args.output.stem + "." + name)
            destination.write_bytes(content)
            metadata[name + "_sha256"] = hashlib.sha256(content).hexdigest()
        pairs = list(checkpoint_pairs(limit=args.weights, model_path=args.model_path))
        if len(pairs) != args.weights:
            raise RuntimeError(f"requested {args.weights} pairs, found {len(pairs)}")
        metadata["weight_names"] = [p[0] for p in pairs]
        metadata["weight_pairs"] = len(pairs)
        log.write(json.dumps(metadata) + "\n")
        log.flush()
        print(f"Loaded {len(pairs)} BF16 checkpoint pairs", flush=True)
        for rows in args.rows:
            config = choose_config(rows, buckets)
            print(f"T={rows}: preparing {len(pairs)} cases, {config}", flush=True)
            xs = [torch.randn(rows, 10240, dtype=torch.bfloat16, device="cuda", generator=generator) for _ in pairs]
            refs = [reference(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)]
            start = time.perf_counter()
            kernels = [GRRead(rows, wd, wu, config) for _, wd, wu in pairs]
            torch.cuda.synchronize()
            prepare_seconds = time.perf_counter() - start
            implementations = {
                "flydsl": [lambda k=k, x=x: k(x) for k, x in zip(kernels, xs)],
                "torch_compile": [
                    lambda x=x, wd=wd, wu=wu: torch_compiled(x, wd, wu) for x, (_, wd, wu) in zip(xs, pairs)
                ],
            }
            previous_kernels = None
            if previous:
                previous_kernels = [previous.GRRead(rows, wd, wu) for _, wd, wu in pairs]
                implementations["previous_flydsl"] = [lambda k=k, x=x: k(x) for k, x in zip(previous_kernels, xs)]
            if rows <= 16:
                implementations["tuned_triton"] = [
                    lambda x=x, wd=wd, wu=wu: baseline.fused_hc_mix(x, wd, wu, 4, 2560)
                    for x, (_, wd, wu) in zip(xs, pairs)
                ]
            graphs, results, retained_outputs = {}, {}, {}
            for name, functions in implementations.items():
                outputs = [None] * len(functions)

                def retained(i, function, outputs=outputs):
                    outputs[i] = function()
                    return outputs[i]

                calls = [lambda i=i, fn=fn, retained=retained: retained(i, fn) for i, fn in enumerate(functions)]
                graph, count = capture(calls, repeats=args.graph_repeats)
                graph.replay()
                errors = [output_error(out, ref) for out, ref in zip(outputs, refs)]
                if name in ("flydsl", "previous_flydsl"):
                    for out, ref in zip(outputs, refs):
                        torch.testing.assert_close(out.double(), ref, **TOLERANCES[torch.bfloat16])
                results[name] = {
                    "max_abs_error": max(e["max_abs"] for e in errors),
                    "max_scaled_error": max(e["max_scaled"] for e in errors),
                    "fp64_passed_pairs": sum(e["max_scaled"] <= 1.0 for e in errors),
                    "rounds": [],
                }
                graphs[name] = (graph, count)
                retained_outputs[name] = outputs
            for round_id in range(args.rounds):
                order = list(graphs)
                rng.shuffle(order)
                for name in order:
                    graph, count = graphs[name]
                    timing = time_graph(graph, count, samples=args.samples, replay_per_sample=args.replays)
                    results[name]["rounds"].append({"round": round_id, "order": order, **timing})
            for result in results.values():
                samples = [v for r in result["rounds"] for v in r["samples_us"]]
                result["median_us"] = statistics.median(samples)
                result["min_us"] = min(samples)
                result["max_us"] = max(samples)
            # Every captured FlyDSL case must see new input, not a stale result.
            for x in xs:
                x.mul_(0.99).add_(0.015625)
            fly_graph = graphs["flydsl"][0]
            for _ in range(20):
                fly_graph.replay()
            changed_errors = []
            for x, (_, wd, wu), out in zip(xs, pairs, retained_outputs["flydsl"]):
                changed_ref = reference(x, wd, wu)
                torch.testing.assert_close(out.double(), changed_ref, **TOLERANCES[torch.bfloat16])
                changed_errors.append(output_error(out, changed_ref))
            primary = "tuned_triton" if rows <= 16 else "torch_compile"
            record = {
                "type": "result",
                "rows": rows,
                "config": asdict(config),
                "weight_pairs": len(pairs),
                "prepare_including_compile_seconds": prepare_seconds,
                "prepared_weight_bytes": sum(k.w_down.numel() * 2 + k.w_up.numel() * 2 for k in kernels),
                "scratch_bytes_per_pair": kernels[0].partial.numel() * 4,
                "changed_input_graph_replays": 20,
                "changed_input_passed_pairs": len(changed_errors),
                "changed_input_max_scaled_error": max(e["max_scaled"] for e in changed_errors),
                "primary_baseline": primary,
                "results": results,
                "speedup": results[primary]["median_us"] / results["flydsl"]["median_us"],
                "speedup_vs_torch_compile": results["torch_compile"]["median_us"] / results["flydsl"]["median_us"],
            }
            if previous:
                record["speedup_vs_previous_flydsl"] = (
                    results["previous_flydsl"]["median_us"] / results["flydsl"]["median_us"]
                )
            log.write(json.dumps(record) + "\n")
            log.flush()
            medians = {name: round(value["median_us"], 3) for name, value in results.items()}
            print(
                f"T={rows}: {medians}, primary speedup={record['speedup']:.3f}x; all {len(pairs)} pairs passed",
                flush=True,
            )
            del (
                graphs,
                implementations,
                kernels,
                refs,
                xs,
                retained_outputs,
                outputs,
                calls,
                graph,
                fly_graph,
                previous_kernels,
            )


if __name__ == "__main__":
    main()
