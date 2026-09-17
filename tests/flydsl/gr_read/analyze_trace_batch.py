"""Run the supplied FlyDSL hotspot analyzer and retain per-dispatch summaries."""

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-regex", help="only analyze matching kernel names")
    parser.add_argument(
        "--analyzer",
        type=Path,
        default=Path("/opt/FlyDSL/.claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py"),
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("provided_hotspot_analyzer", args.analyzer)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    logs = args.output.with_suffix("")
    logs.mkdir(parents=True, exist_ok=False)
    result = {
        "analyzer": str(args.analyzer),
        "analyzer_sha256": hashlib.sha256(args.analyzer.read_bytes()).hexdigest(),
        "scope": "sampled wave instructions, not kernel wall time; occupancy is a resource limit, not achieved residency",
        "caveats": [
            "mixed vmcnt/lgkmcnt waits use the supplied analyzer's VMEM-wait category",
            "functools/decorator source locations are GPU debug attribution, not measured Python execution",
            "raw CSV VGPR/AGPR counts are retained alongside the analyzer's unified register interpretation",
        ],
        "dispatches": [],
    }
    for directory in args.directories:
        with (directory / "out_kernel_trace.csv").open() as trace:
            by_id = {row["Dispatch_Id"]: row for row in csv.DictReader(trace)}
        for dispatch in sorted(directory.glob("ui_output_agent_*_dispatch_*")):
            row = by_id[dispatch.name.rsplit("_", 1)[-1]]
            name = row["Kernel_Name"]
            if args.kernel_regex and not re.search(args.kernel_regex, name):
                continue
            command = [
                sys.executable,
                str(args.analyzer),
                str(dispatch),
                "--kernel",
                name,
                "--topk",
                "8",
                "--mode",
                "both",
                "--detail",
                "--context",
                "2",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            log_path = logs / f"{directory.name}_{dispatch.name}.txt"
            with log_path.open("x") as log:
                log.write(completed.stdout)
                log.write(completed.stderr)
            instructions = module.load_instructions(str(dispatch))
            hotspots = module.aggregate_by_source(instructions)
            meta = module.read_kernel_metadata(str(dispatch), kernel_filter=name)
            resources = module.detect_arch_and_reg_pressure(instructions, meta)
            stalls = collections.Counter()
            for inst in instructions:
                stalls[inst.stall_type] += inst.stall_cycles
            total = sum(stalls.values())
            record = {
                "directory": str(dispatch),
                "kernel": name,
                "trace_csv_row": row,
                "instruction_count": len(instructions),
                "source_mapped_instructions": sum(inst.source_loc != "<unknown>" for inst in instructions),
                "total_cycles": sum(inst.total_cycles for inst in instructions),
                "total_stall_cycles": total,
                "stall_cycles_by_type": dict(stalls),
                "stall_percent_by_type": {key: value * 100 / total if total else 0 for key, value in stalls.items()},
                "resources": resources,
                "top_source": [
                    {
                        "source": h.source_loc,
                        "stall_cycles": h.total_stall_cycles,
                        "percent_of_stall": h.total_stall_cycles * 100 / total if total else 0,
                        "dominant_type": h.dominant_stall_type,
                    }
                    for h in hotspots[:8]
                ],
                "text_report": str(log_path),
            }
            result["dispatches"].append(record)
            print(
                directory.name,
                name,
                "stalls",
                total,
                "types",
                {k: round(v, 1) for k, v in record["stall_percent_by_type"].items()},
                flush=True,
            )
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
