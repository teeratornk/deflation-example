"""Compare all declared accuracy replays without treating them as optimization results."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_trace import read_arrays
from .reporting import file_sha256, write_report


TOLERANCES = (1e-4, 1e-6, 1e-8, 1e-10)


def summarize(records, fields):
    rows, completed, binding = [], {}, None
    for tolerance in TOLERANCES:
        record = records.get(tolerance)
        if record is None:
            rows.append({"linear_tolerance": tolerance, "status": "missing"})
            continue
        key = [
            record.get(k)
            for k in ("source_record_sha256", "trace_sha256", "system_sha256", "quadratic_sha256")
        ]
        if any(v is None for v in key):
            raise ValueError("Every replay must identify its recorded inputs")
        key += [record["environment"]["git_head"], record["environment"]["cpu_model"]]
        if binding is not None and key != binding:
            raise ValueError("Replays must share inputs, numerical source and CPU")
        binding = key
        if record["linear_tolerance"] != tolerance:
            raise ValueError("Replay tolerance differs from its declared position")
        row = {
            "linear_tolerance": tolerance,
            "status": record["status"],
            "diagnostic": record.get("diagnostic"),
            "error": record.get("error"),
        }
        rows.append(row)
        d = record.get("diagnostic") or {}
        if record["status"] != "complete" or not d.get("linear_verified"):
            continue
        if not np.isfinite(d["original_residual"]) or not 0 <= d["original_residual"] <= tolerance:
            raise ValueError("A verified replay must meet its independent residual target")
        if tolerance not in fields:
            raise ValueError("Completed replays require saved candidates")
        completed[tolerance] = fields[tolerance]
    strict = completed.get(TOLERANCES[-1])
    if strict is not None:
        for row in rows:
            arrays = completed.get(row["linear_tolerance"])
            if arrays is None:
                continue
            for name in ("initial", "rhs", "partition"):
                if not np.array_equal(arrays[name], strict[name]):
                    raise ValueError(
                        "Replays differ in initial guess, right-hand side or bound assignments"
                    )
            row["next_partition_difference_from_strict"] = int(
                np.count_nonzero(arrays["next_partition"] != strict["next_partition"])
            )
            row["candidate_infinity_difference_from_strict"] = float(
                np.max(np.abs(arrays["candidate"] - strict["candidate"]))
            )
    return {
        "schema": "coupled-qp-accuracy-summary-v1",
        "rows": rows,
        "all_linear_replays_verified": len(completed) == len(TOLERANCES),
        "strict_comparison_available": strict is not None,
        "scope": "Fixed inactive equation; no quadratic/nonlinear convergence or complete-time claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        nargs=4,
        type=Path,
        required=True,
        help="Paths for 1e-4, 1e-6, 1e-8 and 1e-10, including missing paths",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, fields, hashes = {}, {}, {}
    for tolerance, path in zip(TOLERANCES, args.records, strict=True):
        if not path.exists():
            continue
        records[tolerance] = json.loads(path.read_text())
        hashes[str(tolerance)] = file_sha256(path)
        r = records[tolerance]
        if r["status"] == "complete":
            fields[tolerance] = read_arrays(path.parent, "fields.npz", r["fields_sha256"])
    summary = summarize(records, fields)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", {**summary, "record_sha256": hashes})


if __name__ == "__main__":
    main()
