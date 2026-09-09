"""Measure fresh-process loading, GPU initialization and shutdown separately."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


PREFIX = "STARTUP_RECORD "


def worker():
    parts = {}
    tick = time.perf_counter()
    from . import benchmark_extended
    from threadpoolctl import threadpool_limits

    parts["scientific_module_loading"] = time.perf_counter() - tick
    tick = time.perf_counter()
    torch = benchmark_extended.require_cuda()
    import pyamgx

    parts["gpu_module_loading_and_device_check"] = time.perf_counter() - tick
    with threadpool_limits(limits=4):
        tick = time.perf_counter()
        torch.set_num_threads(4)
        torch.ones(1, device="cuda").sum().item()
        pyamgx.initialize()
        torch.cuda.synchronize()
        parts["gpu_runtime_initialization"] = time.perf_counter() - tick
        tick = time.perf_counter()
        pyamgx.finalize()
        torch.cuda.synchronize()
        parts["gpu_runtime_finalization"] = time.perf_counter() - tick
    print(PREFIX + json.dumps({"components_seconds": parts}), flush=True)


def run(output, repeats=5):
    from .reporting import environment, write_report
    from .validation import integer

    integer(repeats, "Repetitions", 1)
    output.mkdir(parents=True, exist_ok=False)
    specification = {
        "protocol": "fresh-process-startup-v1",
        "repetitions": repeats,
        "threads": 4,
        "scope": "fresh child process launch, benchmark and GPU imports, GPU initialization/finalization and process exit; no problem construction or optimization",
        "parent_interpretation": "launch/serialization/exit interval is the parent wall time minus the disjoint child intervals; parent launcher startup is outside the measured workload",
    }
    write_report(output / "protocol.json", specification)
    rows = []
    for repetition in range(repeats):
        tick = time.perf_counter()
        child = subprocess.run(
            [sys.executable, "-m", "deflation_example.benchmark_startup", "--worker"],
            capture_output=True,
            text=True,
            check=False,
        )
        total = time.perf_counter() - tick
        payloads = [
            line[len(PREFIX) :] for line in child.stdout.splitlines() if line.startswith(PREFIX)
        ]
        if child.returncode or len(payloads) != 1:
            rows.append(
                {
                    "repetition": repetition,
                    "success": False,
                    "returncode": child.returncode,
                    "total_seconds": total,
                }
            )
            continue
        parts = json.loads(payloads[0])["components_seconds"]
        parts["python_launch_serialization_and_exit"] = total - sum(parts.values())
        rows.append(
            {
                "repetition": repetition,
                "success": all(v >= 0 for v in parts.values()),
                "total_seconds": total,
                "components_seconds": parts,
            }
        )
    report = {
        "specification": specification,
        "environment": environment(),
        "rows": rows,
        "complete": len(rows) == repeats,
        "success": all(r["success"] for r in rows),
    }
    write_report(output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        worker()
    elif args.output is None:
        parser.error("--output is required")
    elif not run(args.output, args.repeats)["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
