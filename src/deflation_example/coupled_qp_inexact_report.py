"""Compare declared reduced-direction tolerances at fixed quadratic KKT accuracy."""

import argparse
import json
import math
from pathlib import Path

from .coupled_qp_globalization_report import summarize as procedure_summary
from .reporting import file_sha256, write_report


def summarize(records, tolerances):
    """Retain missing/failed arms; never infer a speedup from an unsolved quadratic."""
    if (
        len(records) != len(tolerances)
        or not tolerances
        or len(set(tolerances)) != len(tolerances)
        or any(not math.isfinite(t) or not 0 < t < 1 for t in tolerances)
    ):
        raise ValueError("Declare distinct positive direction tolerances below one")
    rows, identity = [], None
    for record, target in zip(records, tolerances, strict=True):
        if record is None:
            rows.append(
                {"linear_tolerance": target, "status": "missing", "quadratic_verified": False}
            )
            continue
        if record["method"] != "projected" or record["linear_tolerance"] != target:
            raise ValueError("Each projected record must match its declared direction tolerance")
        # Reuse the existing schema and independent KKT validation.
        row = procedure_summary({"projected": record})["rows"][1]
        key = {
            k: record[k]
            for k in (
                "trace_sha256",
                "quadratic_sha256",
                "quadratic",
                "quadratic_tolerance",
                "initial",
                "restriction",
                "budget_seconds",
            )
        }
        key["environment"] = {
            k: record["environment"].get(k)
            for k in ("git_head", "source_sha256", "cpu_model", "python", "numpy", "scipy", "blas")
        }
        if identity is not None and key != identity:
            raise ValueError(
                "Match quadratic, final criterion, source, backend, hardware and budget"
            )
        identity = key
        history = record.get("history", [])
        completed = [h for h in history if h.get("linear_status") == "converged"]
        for item in completed:
            residual = item["linear_residual"]
            if not math.isfinite(residual) or not 0 <= residual <= target:
                raise ValueError("A completed direction must meet its declared fresh residual")
        steps = [s for h in history for s in h.get("projected_steps", [])]
        steps.extend(h["reduced_search"] for h in history if "reduced_search" in h)
        changes = [s["quadratic_change"] for s in steps if "quadratic_change" in s]
        if any(not math.isfinite(c) or c >= 0 for c in changes):
            raise ValueError("Every retained projected-search update must decrease the quadratic")
        if record.get("status") == "complete":
            if record["inner_iterations"] != sum(h.get("linear_iterations", 0) for h in history):
                raise ValueError("Iteration total differs from the complete history")
            if not math.isclose(
                sum(changes), record["quadratic_change"], rel_tol=1e-7, abs_tol=1e-8
            ):
                raise ValueError("Search decreases differ from the independently evaluated change")
        row.update(
            linear_tolerance=target,
            quadratic_tolerance=record["quadratic_tolerance"],
            completed_linear_solves=len(completed),
            reduced_updates=sum("quadratic_change" in h.get("reduced_search", {}) for h in history),
            projected_updates=sum(
                "quadratic_change" in s for h in history for s in h.get("projected_steps", [])
            ),
            largest_completed_linear_residual=max(
                (h["linear_residual"] for h in completed), default=None
            ),
        )
        rows.append(row)
    verified = [r["linear_tolerance"] for r in rows if r.get("quadratic_verified")]
    return {
        "schema": "coupled-quadratic-inexact-summary-v1",
        "identity": identity,
        "rows": rows,
        "verified_direction_tolerances": verified,
        "next_step": "verify_nonlinear_integration" if verified else "inspect_quadratic_histories",
        "scope": "Single attempts on one fixed intermediate quadratic. Direction tolerances vary; final quadratic KKT accuracy is fixed. No nonlinear convergence or repeated-timing speedup is inferred.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--tolerances", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, digests = [], []
    for path in args.records:
        records.append(json.loads(path.read_text()) if path.exists() else None)
        digests.append(file_sha256(path) if path.exists() else None)
    report = summarize(records, args.tolerances)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", {**report, "record_sha256": digests})


if __name__ == "__main__":
    main()
