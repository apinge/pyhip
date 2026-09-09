"""Summarize static ISA/resource differences; counts are not runtime counters."""

import argparse
import collections
import json
import re
from pathlib import Path


def analyze(path, metadata):
    source = path.read_text()
    ops = collections.Counter(re.findall(r"^\s+((?:[sv]|buffer|global|flat|ds|scratch)_[A-Za-z0-9_]+)\b", source, re.M))

    def count(prefix):
        return sum(v for k, v in ops.items() if k.startswith(prefix))

    def field(name):
        matches = re.findall(r"^\s*\." + re.escape(name) + r"\s*:?\s+(\d+)", source, re.M)
        return int(matches[-1]) if matches else None

    fixed = field("group_segment_fixed_size") or field("amdhsa_group_segment_fixed_size") or 0
    dynamic = metadata.get("triton_metadata", {}).get("shared", 0) if path.suffix == ".amdgcn" else 0
    return {
        "file": str(path),
        "vgpr_count_metadata": field("vgpr_count"),
        "sgpr_count_metadata": field("sgpr_count"),
        "next_free_vgpr_directive": field("amdhsa_next_free_vgpr"),
        "accum_offset_directive": field("amdhsa_accum_offset"),
        "fixed_lds_bytes": fixed,
        "dynamic_lds_bytes": dynamic,
        "private_segment_bytes": field("private_segment_fixed_size") or field("amdhsa_private_segment_fixed_size") or 0,
        "static_instruction_counts": {
            "mfma": count("v_mfma"),
            "buffer_load": count("buffer_load"),
            "global_load": count("global_load"),
            "lds_read": count("ds_read"),
            "lds_write": count("ds_write"),
            "s_barrier": count("s_barrier"),
            "atomic": sum(v for k, v in ops.items() if "atomic" in k),
            "exp": count("v_exp"),
            "rcp": count("v_rcp"),
            "division_fixup": count("v_div"),
            "scratch": count("scratch"),
        },
        "mfma_variants": {k: v for k, v in ops.items() if "mfma" in k},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = []
    for directory in args.directories:
        metadata = json.loads((directory / "metadata.json").read_text())
        files = sorted(directory.rglob("*final_isa.s")) + sorted(directory.rglob("*.amdgcn"))
        result.append(
            {
                "directory": str(directory),
                "rows": metadata["rows"],
                "config": metadata["config"],
                "kernels": [analyze(f, metadata) for f in files],
            }
        )
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
