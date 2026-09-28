"""Construction-inclusive fixed-quadratic CPU, CUDA and hybrid comparisons."""

import argparse
import json
import math
from pathlib import Path

from .coupled_qp_inexact_report import summarize as validate
from .reporting import file_sha256, write_report


def summarize(records):
    if not records:
        raise ValueError("Supply at least one execution record")
    rows, identity = [], None
    for record in records:
        checked = validate([record], [record["linear_tolerance"]])
        key = {
            k: v
            for k, v in checked["identity"].items()
            if k not in {"device", "reference_rank", "frozen_layout"}
        }
        key["linear_tolerance"] = record["linear_tolerance"]
        if identity is not None and key != identity:
            raise ValueError(
                "Match inputs, source, final accuracy, initialization and CPU environment"
            )
        identity = key
        row = dict(checked["rows"][0])
        row.update(
            device=record.get("device", "cpu"),
            rank=record.get("reference_rank", 0),
            frozen_layout=record.get("frozen_layout", "serial"),
            gpu=record.get("gpu"),
        )
        names = (
            "quadratic_seconds",
            "reference_construction_seconds",
            "preconditioner_construction_seconds",
            "solver_construction_seconds",
        )
        components = {name: record.get(name) for name in names}
        if record.get("status") == "complete" and any(
            value is None or not math.isfinite(value) or value < 0 for value in components.values()
        ):
            raise ValueError("Completed execution requires all construction timers")
        row["cost_components_seconds"] = components
        row["construction_inclusive_seconds"] = (
            sum(components.values()) if all(v is not None for v in components.values()) else None
        )
        # No ratio is assigned to a capped or failed computation.
        row["speedup"] = None
        rows.append(row)
    baselines = [r for r in rows if r["device"] == "cpu" and r["rank"] == 0]
    if len(baselines) == 1 and baselines[0].get("quadratic_verified"):
        base = baselines[0]["construction_inclusive_seconds"]
        for row in rows:
            if row.get("quadratic_verified") and row["construction_inclusive_seconds"] > 0:
                row["speedup"] = base / row["construction_inclusive_seconds"]
    return {
        "schema": "coupled-quadratic-execution-v1",
        "identity": identity,
        "rows": rows,
        "scope": "Single fixed-quadratic attempts; reconstruction is separate. No nonlinear or repeated-timing claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.records]
    report = summarize(records)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(
        args.output / "summary.json",
        {**report, "record_sha256": [file_sha256(p) for p in args.records]},
    )


if __name__ == "__main__":
    main()
