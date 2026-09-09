"""Reference, baseline loading, checkpoint selection and graph timing."""

import importlib.util
import json
import statistics
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=2e-3, atol=1e-3),
}
MODEL_PATH = Path("/models/Qwen3.8-Flash-Next-FP8")
BASELINE_PATH = Path("/opt/sglang/python/sglang/srt/layers/hc_mix_triton.py")


def torch_mix(x, w_down, w_up):
    t = F.silu(F.linear(x, w_down) / 4)
    g = torch.sigmoid(F.linear(t, w_up))
    return (g.unflatten(-1, (4, 2560)) * x.unflatten(-1, (4, 2560))).mean(-2)


def reference(x, w_down, w_up):
    return torch_mix(x.double(), w_down.double(), w_up.double())


def synthetic(rows, dtype=torch.bfloat16, seed=0, scale=0.02):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 10240, device="cuda", dtype=dtype, generator=generator)
    wd = torch.randn(320, 10240, device="cuda", dtype=dtype, generator=generator) * scale
    wu = torch.randn(10240, 320, device="cuda", dtype=dtype, generator=generator) * scale
    return x, wd, wu


def load_triton_baseline():
    name = "gr_read_existing_triton_baseline"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BASELINE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def checkpoint_pairs(limit=None, model_path=MODEL_PATH):
    """Load only HC down/up tensors; never load the whole model."""
    from safetensors import safe_open

    with (model_path / "model.safetensors.index.json").open() as f:
        index = json.load(f)["weight_map"]
    suffix = ".input_mix_weight_down.weight"
    names = sorted(n for n in index if n.endswith(suffix))
    if limit is not None:
        names = names[:limit]
    for down_name in names:
        up_name = down_name[: -len(suffix)] + ".input_mix_weight_up.weight"
        with safe_open(model_path / index[down_name], framework="pt", device="cpu") as f:
            wd = f.get_tensor(down_name).to("cuda")
        with safe_open(model_path / index[up_name], framework="pt", device="cpu") as f:
            wu = f.get_tensor(up_name).to("cuda")
        yield down_name[: -len(suffix)], wd, wu


def capture(calls, repeats=1, warmup=3):
    for _ in range(warmup):
        for call in calls:
            call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                for call in calls:
                    call()
    for message in messages:
        if "graph is empty" in str(message.message).lower():
            raise RuntimeError("Empty graph: kernel launch did not use the capture stream")
        warnings.warn(str(message.message), message.category)
    return graph, len(calls) * repeats


def time_graph(graph, calls, samples=9, replay_per_sample=10):
    for _ in range(3):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        start.record()
        for _ in range(replay_per_sample):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (calls * replay_per_sample))
    return {"median_us": statistics.median(values), "samples_us": values}
