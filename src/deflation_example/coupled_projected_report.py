"""Check complete projected trust-region records without changing their criteria."""

import argparse
import json
import math
from pathlib import Path

from .coupled_optimize import adjoint_acceptance, equation_acceptance
from .reporting import file_sha256, write_report

KKT_NAMES = {
    "primal_absolute",
    "stationarity",
    "dual_feasibility",
    "lower_complementarity",
    "upper_complementarity",
}


def _nonnegative(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _kkt(values, tolerance):
    return (
        _nonnegative(tolerance)
        and isinstance(values, dict)
        and set(values) == KKT_NAMES
        and all(_nonnegative(v) and v <= tolerance for v in values.values())
    )


def audit(record):
    """Distinguish final accuracy, complete coverage and elapsed-time accounting.

    Intermediate projected directions are checked at their declared tolerance;
    the final strict phase, original residuals, equations, adjoint and nonlinear
    KKT requirements remain independent checks. Capped and missing attempts are
    retained without assigning a speedup. This reads saved independent checks;
    it does not recompute the physical model from the saved fields.
    """
    if record is None:
        return {"status": "missing", "verified": False, "failures": ["missing_record"]}
    cfg = record["configuration"]
    if (
        record.get("schema") != "coupled-trust-development-v1"
        or cfg.get("qp_solver") != "projected"
        or cfg.get("trust_accuracy") != "adaptive_projected"
    ):
        raise ValueError("Use a projected trust-region record with adaptive_projected accuracy")
    failures, case_rows = [], []
    if any(
        cfg.get(name) != target
        for name, target in (
            ("inner_tolerance", 1e-10),
            ("qp_tolerance", 1e-10),
            ("nonlinear_tolerance", 1e-8),
        )
    ):
        failures.append("final_tolerances")
    cases, queries = record.get("cases", []), cfg["queries"]
    if not queries or len(cases) != len(queries):
        failures.append("target_coverage")
    for position, case in enumerate(cases):
        reasons, residuals, attempts = [], [], []
        if position >= len(queries) or (
            case.get("position") != position or case.get("target") != queries[position]["target"]
        ):
            reasons.append("target_identity")
        if case.get("status") != "converged" or case.get("verified") is not True:
            reasons.append("termination")
        if not _kkt(case.get("kkt"), 1e-8):
            reasons.append("final_kkt")
        equations = case.get("equations", [])
        if (
            len(equations) != cfg["slabs"]
            or {r.get("slab") for r in equations} != set(range(cfg["slabs"]))
            or not equation_acceptance(equations, cfg)
        ):
            reasons.append("equations_or_conservation")
        if not adjoint_acceptance(case.get("adjoint", {})):
            reasons.append("adjoint")
        if "history" not in case:
            reasons.append("missing_history")
        for outer in case.get("history", []):
            for attempt in outer["attempts"]:
                attempts.append(attempt)
                target, qtarget = attempt["linear_tolerance"], attempt["qp_tolerance"]
                if not _nonnegative(target) or not 1e-10 <= target <= 0.01:
                    reasons.append("inner_target")
                if not _nonnegative(qtarget) or not 0 < qtarget <= 0.01:
                    reasons.append("quadratic_target")
                if attempt["qp_status"] != "converged" or not _kkt(attempt["qp_kkt"], qtarget):
                    reasons.append("quadratic_accuracy")
                for step in attempt["qp_history"]:
                    if "linear_status" not in step:
                        if step.get("inactive", 0) > 0:
                            reasons.append("missing_inner_result")
                        continue
                    rho = step.get("linear_residual")
                    if (
                        step["linear_status"] != "converged"
                        or not _nonnegative(rho)
                        or not _nonnegative(target)
                        or rho > target
                    ):
                        reasons.append("original_residual")
                    if _nonnegative(rho):
                        residuals.append(rho)
        if attempts and (
            attempts[-1].get("strict_accuracy") is not True
            or attempts[-1]["linear_tolerance"] != 1e-10
            or not _nonnegative(attempts[-1]["qp_tolerance"])
            or not 0 < attempts[-1]["qp_tolerance"] <= 1e-10
        ):
            reasons.append("final_strict_phase")
        case_rows.append(
            {
                "position": position,
                "target": case.get("target"),
                "status": case.get("status"),
                "verified": not reasons,
                "failures": sorted(set(reasons)),
                "maximum_kkt": max(case["kkt"].values()) if case.get("kkt") else None,
                "maximum_intermediate_original_residual": max(residuals, default=None),
                "quadratic_attempts": len(attempts),
            }
        )
    if record.get("status") != "complete" or record.get("all_problems_verified") is not True:
        failures.append("sequence_termination")
    if any(not c["verified"] for c in case_rows):
        failures.append("case_verification")
    elapsed = record.get("attempt_seconds")
    prior = record.get("prior_attempt_seconds", 0)
    total = record.get("cumulative_attempt_seconds")
    components = record.get("components_seconds", {})
    timing_ok = (
        all(_nonnegative(v) for v in (elapsed, prior, total))
        and bool(components)
        and all(_nonnegative(v) for v in components.values())
        and math.isclose(sum(components.values()), elapsed, rel_tol=1e-9, abs_tol=1e-6)
        and math.isclose(elapsed + prior, total, rel_tol=1e-9, abs_tol=1e-6)
    )
    if not timing_ok:
        failures.append("elapsed_accounting")
    return {
        "status": record.get("status"),
        "verified": not failures,
        "failures": failures,
        "cases": case_rows,
        "cumulative_attempt_seconds": total,
        "numerical_source": record.get("environment", {}).get("git_head"),
        "scope": "Saved independent checks for complete nonlinear optimization; no repeated-timing or speedup claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the preceding verification report")
    record = json.loads(args.record.read_text()) if args.record.exists() else None
    write_report(
        args.output,
        {
            "schema": "coupled-projected-audit-v1",
            **audit(record),
            "record_sha256": file_sha256(args.record) if record is not None else None,
        },
    )


if __name__ == "__main__":
    main()
