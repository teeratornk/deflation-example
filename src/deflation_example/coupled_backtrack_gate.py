"""Bounded transition from the frozen adaptive diagnostic to two repair policies."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_trust_gate import verified
from .reporting import file_sha256, write_report


REQUIRED = {
    "alpha": 1e-11,
    "slabs": 64,
    "horizon_s": 600.0,
    "target_startup_s": 60.0,
    "lower_K": 337.3,
    "inner_tolerance": 1e-10,
    "qp_tolerance": 1e-10,
    "nonlinear_tolerance": 1e-8,
    "equation_acceptance_tolerance": 1e-12,
    "conservation_tolerance": 1e-6,
    "method": "jacobi",
    "inner_preconditioner": "frozen",
    "rank": 0,
    "trust_accuracy": "adaptive",
    "queries": [{"target": 7, "upper_K": 357.3}],
}


def check_configuration(record):
    if record.get("schema") != "coupled-trust-development-v1" or any(
        record["configuration"].get(k) != v for k, v in REQUIRED.items()
    ):
        raise ValueError(
            "Use the fixed one-target adaptive diagnostic and unchanged final accuracy"
        )


def complete(record):
    if not verified(record):
        return False
    value = record["cases"][0].get("objective")
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def decision(
    adaptive, repairs=None, fields=None, *, adaptive_terminal=False, repairs_terminal=(False, False)
):
    check_configuration(adaptive)
    if (
        adaptive["configuration"].get("trial_policy", "radius_rebuild") != "radius_rebuild"
        or adaptive["configuration"]["flow_continuation"]
    ):
        raise ValueError("The first record must be the original adaptive policy")
    if complete(adaptive):
        return {"action": "prepare_reference_screen", "selected": "original_adaptive"}
    if adaptive["status"] == "running" and not adaptive_terminal:
        return {"action": "wait_for_adaptive"}
    if repairs is None:
        if adaptive["status"].startswith("quadratic_"):
            return {
                "action": "stop_for_quadratic_diagnosis",
                "reason": "The quadratic solve failed before producing a direction; examine it before testing trial policies.",
            }
        return {"action": "run_two_repair_diagnostics", "hours_per_policy": 24}
    if len(repairs_terminal) != 2 or any(type(value) is not bool for value in repairs_terminal):
        raise ValueError("Supply a scheduler-termination flag for each repair")
    if len(repairs) != 2 or {r["configuration"]["flow_continuation"] for r in repairs} != {
        False,
        True,
    }:
        raise ValueError("Compare backtracking with and without residual-load continuation")
    identities = []
    changed_keys = {
        "flow_continuation",
        "trial_policy",
        "capture_trials",
        "optimizer_policy",
        "linear_heartbeat_seconds",
        "capture_linear_systems",
    }
    original_cfg = {k: v for k, v in adaptive["configuration"].items() if k not in changed_keys}
    for record in repairs:
        check_configuration(record)
        cfg = record["configuration"]
        if cfg.get("trial_policy") != "backtrack" or cfg.get("capture_trials") is not True:
            raise ValueError("Repair diagnostics must capture backtracking trials")
        if {k: v for k, v in cfg.items() if k not in changed_keys} != original_cfg:
            raise ValueError("Repair changes settings beyond the declared numerical policy")
        identities.append(
            (
                {k: v for k, v in cfg.items() if k != "flow_continuation"},
                record["environment"]["source_sha256"],
                record["baseline_sha256"],
            )
        )
        if record["baseline_sha256"] != adaptive["baseline_sha256"]:
            raise ValueError("Physical baselines differ")
    if identities[0] != identities[1]:
        raise ValueError("Repair diagnostics differ beyond flow continuation")
    if any(r["status"] == "running" and not done for r, done in zip(repairs, repairs_terminal)):
        return {"action": "wait_for_repairs"}
    successes = [i for i, r in enumerate(repairs) if complete(r)]
    agreement = None
    if len(successes) == 2:
        if fields is None or len(fields) != 2:
            raise ValueError("Both converged policies require their saved temperature fields")
        a, b = fields
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Temperature fields must be finite and match in shape")
        objectives = [r["cases"][0]["objective"] for r in repairs]
        agreement = {
            "state_absolute_difference": float(np.max(np.abs(a - b))),
            "objective_relative_difference": abs(objectives[0] - objectives[1])
            / max(abs(objectives[0]), abs(objectives[1]), 1e-30),
        }
        if max(agreement.values()) > 1e-6:
            return {"action": "stop_solution_disagreement", "agreement": agreement}
    if not successes:
        return {
            "action": "stop_for_numerical_diagnosis",
            "reason": "No verified complete target; do not launch timing repetitions.",
        }
    selected = min(
        successes,
        key=lambda i: (
            repairs[i]["cumulative_attempt_seconds"],
            repairs[i]["configuration"]["flow_continuation"],
        ),
    )
    return {
        "action": "prepare_reference_screen",
        "selected": "backtrack",
        "flow_continuation": repairs[selected]["configuration"]["flow_continuation"],
        "agreement": agreement,
        "scope": "One-target numerical policy selection; no reference speedup follows.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-record", type=Path, required=True)
    parser.add_argument(
        "--adaptive-terminal",
        action="store_true",
        help="Use only after scheduler termination is independently confirmed",
    )
    parser.add_argument("--repairs", type=Path, nargs=2)
    parser.add_argument(
        "--repairs-terminal",
        action="store_true",
        help="Use only after both repair jobs have independently confirmed scheduler termination",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing gate decision")
    paths = [args.adaptive_record, *(args.repairs or [])]
    records = [json.loads(p.read_text()) for p in paths]
    fields = []
    for path, record in zip(paths[1:], records[1:], strict=True):
        if complete(record):
            with np.load(path.parent / "target-00.npz", allow_pickle=False) as archive:
                fields.append(archive["state"].copy())
        else:
            fields.append(None)
    result = decision(
        records[0],
        records[1:] or None,
        fields,
        adaptive_terminal=args.adaptive_terminal,
        repairs_terminal=(args.repairs_terminal, args.repairs_terminal),
    )
    result["record_sha256"] = [file_sha256(p) for p in paths]
    write_report(args.output, result)


if __name__ == "__main__":
    main()
