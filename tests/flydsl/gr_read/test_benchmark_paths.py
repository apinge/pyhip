"""CPU-only checks: python3 test_benchmark_paths.py (no pytest required)."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

if __package__:
    from .support import BASELINE_PATH, BASELINE_SHA256, load_triton_baseline
else:
    from support import BASELINE_PATH, BASELINE_SHA256, load_triton_baseline


def main():
    assert BASELINE_PATH == Path(__file__).resolve().parent / "baselines" / "hc_mix_triton.py"
    assert hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest() == BASELINE_SHA256
    baseline = load_triton_baseline()
    assert Path(baseline.__file__) == BASELINE_PATH
    assert baseline._GFX942_MIX_CONFIG["num_warps"] == 2
    assert baseline._GFX942_MIX_CONFIG["kpack"] == 2

    with tempfile.TemporaryDirectory(prefix="gr_read_paths_") as directory:
        root = Path(directory)
        first, second = root / "first.py", root / "second.py"
        first.write_text("marker = 1\n")
        second.write_text("marker = 2\n")
        a, b = load_triton_baseline(first), load_triton_baseline(second)
        assert a.marker == 1 and b.marker == 2
        assert a is not b
        assert load_triton_baseline(first) is a

        model = root / "model"
        model.mkdir()
        (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        for script in ("bench_three_stage.py", "benchmark.py"):
            command = [sys.executable, str(Path(__file__).with_name(script)), "--rows", "1"]
            if script == "benchmark.py":
                command += ["--configs", str(root / "unused.json"), "--output", str(root / "unused.jsonl")]
            else:
                command += ["--baselines"]
            cases = [
                (["--model-path", str(root / "missing_model")], "set --model-path"),
                (["--model-path", str(model), "--triton-kernel", str(root / "missing.py")], "set --triton-kernel"),
                (["--model-path", str(model), "--triton-kernel", str(first)], "verified 8cf5501b baseline"),
            ]
            if script == "bench_three_stage.py":
                cases.append(
                    (
                        [
                            "--synthetic",
                            "--model-path",
                            str(root / "missing_model"),
                            "--triton-kernel",
                            str(root / "missing.py"),
                        ],
                        "set --triton-kernel",
                    )
                )
            for options, expected in cases:
                completed = subprocess.run(command + options, capture_output=True, text=True, timeout=60)
                assert completed.returncode == 2, completed.stderr
                assert expected in completed.stderr, completed.stderr
                assert "Traceback" not in completed.stderr, completed.stderr
        assert not (root / "unused.jsonl").exists()

    assert not torch.cuda.is_initialized()
    print("PASS: bundled SHA, source-specific loader cache, model/Triton CLI errors and synthetic model-path bypass")


if __name__ == "__main__":
    main()
