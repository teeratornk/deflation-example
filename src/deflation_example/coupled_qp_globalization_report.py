"""Retain both fixed-quadratic outcomes without a nonlinear speedup claim."""

import argparse
import json
from pathlib import Path

import numpy as np

from .reporting import file_sha256, write_report


def summarize(records):
    rows, identity = [], None
    for method in ("pdas", "projected"):
        record = records.get(method)
        if record is None:
            rows.append({"method": method, "status": "missing", "quadratic_verified": False})
            continue
        if (
            record.get("schema") != "coupled-quadratic-globalization-v1"
            or record["method"] != method
        ):
            raise ValueError("Use one fixed-quadratic record for each declared procedure")
        key = {
            k: record[k]
            for k in (
                "trace_sha256",
                "quadratic_sha256",
                "quadratic",
                "linear_tolerance",
                "quadratic_tolerance",
                "initial",
                "restriction",
                "budget_seconds",
            )
        }
        key.update(source=record["environment"]["git_head"], cpu=record["environment"]["cpu_model"])
        if identity is not None and key != identity:
            raise ValueError("Compare matched input, source, hardware, accuracy and budget")
        identity = key
        verified = record.get("quadratic_verified", False)
        if verified:
            values = np.asarray(list(record["kkt"].values()))
            if (
                record["status"] != "complete"
                or record["quadratic_status"] != "converged"
                or len(values) != 5
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or values.max() > record["quadratic_tolerance"]
            ):
                raise ValueError("A verified quadratic must satisfy its recorded KKT test")
        rows.append(
            {
                k: record.get(k)
                for k in (
                    "method",
                    "status",
                    "quadratic_status",
                    "quadratic_verified",
                    "kkt",
                    "inner_iterations",
                    "quadratic_change",
                    "quadratic_seconds",
                    "reconstruction_seconds",
                    "preconditioner_construction_seconds",
                    "seconds",
                    "error",
                )
            }
        )
    successful = [r["method"] for r in rows if r.get("quadratic_verified")]
    return {
        "schema": "coupled-quadratic-globalization-summary-v1",
        "rows": rows,
        "verified_procedures": successful,
        "next_step": "verify_nonlinear_integration"
        if "projected" in successful
        else "inspect_quadratic_histories",
        "scope": "One intermediate quadratic per method, not nonlinear convergence or a repeated timing comparison. Failed attempts provide no completed-solve speedup.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdas", type=Path, required=True)
    parser.add_argument("--projected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, digests = {}, {}
    for method in ("pdas", "projected"):
        path = getattr(args, method)
        if path.exists():
            records[method] = json.loads(path.read_text())
            digests[method] = file_sha256(path)
    report = summarize(records)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", {**report, "record_sha256": digests})


if __name__ == "__main__":
    main()
