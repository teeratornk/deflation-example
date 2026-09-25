"""Predeclared transition from two repair diagnostics to one common policy."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_optimize import adjoint_acceptance, equation_acceptance
from .reporting import file_sha256, write_report


def inner_accuracy(case, cfg):
    attempts = [a for h in case["history"] for a in h["attempts"]]
    for a in attempts:
        target = a["linear_tolerance"]
        if not np.isfinite(target) or not cfg["inner_tolerance"] <= target <= 1e-4:
            return False
        for step in a["qp_history"]:
            if step.get("candidate_retained", False) and step["linear_status"] != "empty":
                rho = step.get("linear_residual")
                if rho is None or not np.isfinite(rho) or rho < 0 or rho > target:
                    return False
    return not attempts or (
        attempts[-1]["strict_accuracy"] and attempts[-1]["linear_tolerance"] <= 1e-10
    )


def verified(record):
    cfg = record["configuration"]
    cases = record.get("cases", [])
    return (
        record.get("status") == "complete"
        and record.get("all_problems_verified") is True
        and len(cases) == len(cfg["queries"]) == 1
        and np.isfinite(record["cumulative_attempt_seconds"])
        and record["cumulative_attempt_seconds"] > 0
        and all(
            c["status"] == "converged"
            and c["verified"]
            and len(c["kkt"]) == 5
            and np.isfinite(list(c["kkt"].values())).all()
            and min(c["kkt"].values()) >= 0
            and max(c["kkt"].values()) <= 1e-8
            and equation_acceptance(c["equations"], cfg)
            and adjoint_acceptance(c["adjoint"])
            and inner_accuracy(c, cfg)
            for c in cases
        )
    )


def decision(records, fields):
    if len(records) != 2 or {r["configuration"]["trust_accuracy"] for r in records} != {
        "strict",
        "adaptive",
    }:
        raise ValueError("The repair gate requires both accuracy policies")
    identities = []
    for r in records:
        cfg = r["configuration"]
        if (
            r.get("schema") != "coupled-trust-development-v1"
            or cfg["flow_continuation"]
            or cfg["method"] != "jacobi"
            or cfg["inner_preconditioner"] != "frozen"
            or cfg["rank"] != 0
            or cfg["queries"] != [{"target": 7, "upper_K": 357.3}]
            or cfg["alpha"] != 1e-11
            or cfg["slabs"] != 64
            or any(
                cfg.get(k) != v
                for k, v in {
                    "horizon_s": 600.0,
                    "target_startup_s": 60.0,
                    "lower_K": 337.3,
                    "inner_tolerance": 1e-10,
                    "qp_tolerance": 1e-10,
                    "nonlinear_tolerance": 1e-8,
                    "equation_acceptance_tolerance": 1e-12,
                    "conservation_tolerance": 1e-6,
                }.items()
            )
        ):
            raise ValueError("Use the unchanged one-target frozen-preconditioner diagnostics")
        identities.append(
            (
                {k: v for k, v in cfg.items() if k != "trust_accuracy"},
                r["environment"]["source_sha256"],
                r["policy"],
                r.get("baseline_sha256"),
            )
        )
    if identities[0] != identities[1]:
        raise ValueError("Repair diagnostics differ beyond the intermediate accuracy policy")
    successes = [i for i, r in enumerate(records) if verified(r)]
    agreement = None
    if len(successes) == 2:
        a, b = fields
        if (
            a is None
            or b is None
            or a.shape != b.shape
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError("Verified diagnostics require matching saved states")
        oa, ob = (r["cases"][0]["objective"] for r in records)
        agreement = {
            "state_absolute_difference": float(np.max(np.abs(a - b))),
            "objective_relative_difference": abs(oa - ob) / max(abs(oa), abs(ob), 1e-30),
        }
        if not np.isfinite(list(agreement.values())).all() or max(agreement.values()) > 1e-6:
            return {"action": "stop_solution_disagreement", "agreement": agreement}
    if successes:
        selected = min(
            successes,
            key=lambda i: (
                records[i]["cumulative_attempt_seconds"],
                records[i]["configuration"]["trust_accuracy"],
            ),
        )
        return {
            "action": "evaluate_four_complete_sequences",
            "accuracy": records[selected]["configuration"]["trust_accuracy"],
            "continuation": False,
            "agreement": agreement,
            "scope": "One-target development policy selection; no speedup claim.",
        }
    flow_failures = []
    for r in records:
        cases = r.get("cases", [])
        trials = [
            t for c in cases for h in c["history"] for a in h["attempts"] for t in a["trials"]
        ]
        flow_failures.append(
            bool(cases)
            and cases[-1]["status"] == "trust_radius_exhausted"
            and bool(trials)
            and trials[-1]["status"].startswith("flow_")
        )
    return {
        "action": "evaluate_continuation_fallback" if all(flow_failures) else "stop_for_diagnosis",
        "scope": "Retain every unsuccessful outcome; time limits and missing records do not establish flow failure.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous gate decision")
    records = [json.loads(p.read_text()) for p in args.records]
    fields = []
    for p, r in zip(args.records, records, strict=True):
        if verified(r):
            with np.load(p.parent / "target-00.npz", allow_pickle=False) as a:
                fields.append(a["state"].copy())
        else:
            fields.append(None)
    result = decision(records, fields)
    result["record_sha256"] = [file_sha256(p) for p in args.records]
    write_report(args.output, result)


if __name__ == "__main__":
    main()
