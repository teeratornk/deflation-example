"""Predeclared transport/time-grid comparison of complete coupled optimization."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_optimize import adjoint_acceptance, equation_acceptance, load_problem
from .coupled_sequence import RestoredEvaluation
from .coupled_targets import desired_temperature
from .coupled_trust_check import check
from .coupled_trust_run import run
from .reporting import environment, file_sha256, write_report


PROTOCOL = {
    "identifier": "coupled-discretization-v1",
    "transport_forms": ["advective", "skew"],
    "time_slabs": [16, 32, 64],
    "targets": [7, 8, 9],
    "horizon_s": 600.0,
    "target_startup_s": 60.0,
    "initial_radius_K": 0.25,
    "minimum_radius_K": 1e-10,
    "outer_cap_per_target": 200,
    "sequence_budget_seconds": 28800,
    "inner_tolerance": 1e-10,
    "nonlinear_tolerance": 1e-8,
    "momentum_tolerance": 1e-12,
    "replay_subdivisions": [2, 4],
    "temperature_sensitivity_threshold_K": 0.05,
    "selection": "Coarsest grid passing complete optimization, independent derivatives, and both verified replay comparisons; prefer advective on a grid tie. Timing is excluded from selection.",
}


def configuration(source, form, slabs):
    if form not in PROTOCOL["transport_forms"] or slabs not in PROTOCOL["time_slabs"]:
        raise ValueError("Choose one of the six declared discretizations")
    cfg = copy.deepcopy(source["configuration"])
    if (
        source.get("status") in {None, "running"}
        or not cfg["transient"]
        or cfg["slabs"] != 16
        or cfg["horizon_s"] != PROTOCOL["horizon_s"]
        or cfg["target_startup_s"] != PROTOCOL["target_startup_s"]
        or [q["target"] for q in cfg["queries"]] != PROTOCOL["targets"]
        or len({q["upper_K"] for q in cfg["queries"]}) != 1
        or cfg["alpha"] != 1e-14
        or cfg["inner_preconditioner"] != "frozen"
        or cfg["frozen_sweeps"] != 3
    ):
        raise ValueError("The source must match the declared coupled study")
    for key in ("initial_state_snapshot", "initial_state_assessment"):
        cfg.pop(key, None)
    cfg.update(
        transport_form=form,
        slabs=slabs,
        initial_state_policy="physical_initial",
        initial_radius_K=PROTOCOL["initial_radius_K"],
        minimum_radius_K=PROTOCOL["minimum_radius_K"],
        method="jacobi",
        rank=0,
        device="cpu",
        nonlinear_cap=PROTOCOL["outer_cap_per_target"],
        optimization_only_budget=False,
        cooperative_flow_deadline=True,
        inner_tolerance=PROTOCOL["inner_tolerance"],
        nonlinear_tolerance=PROTOCOL["nonlinear_tolerance"],
        flow_tolerance=PROTOCOL["momentum_tolerance"],
        capture_trials=True,
        local_mass_diagnostics=True,
        discretization_protocol=PROTOCOL["identifier"],
    )
    return cfg


def verify_returned(record, output):
    """Evaluate final derivatives only for a complete, verified three-target sequence."""
    result = {
        "schema": "coupled-discretization-verification-v1",
        "all_verified": False,
        "cases": [],
    }
    if not record["all_problems_verified"]:
        result["status"] = "optimization_gate_failed"
        return result
    cfg = record["configuration"]
    with threadpool_limits(cfg["threads"]):
        problem, baseline = load_problem(cfg)
        if baseline["baseline_sha256"] != record["baseline_sha256"]:
            raise ValueError("The baseline changed before independent verification")
        for position, query in enumerate(cfg["queries"]):
            row = {"target": query["target"], "verified": False}
            result["cases"].append(row)
            try:
                field = output / f"target-{position:02d}.npz"
                with np.load(field, allow_pickle=False) as data:
                    guess = RestoredEvaluation(data["state"], data["velocity"], data["pressure"])
                desired = desired_temperature(
                    problem, query["target"], cfg["target_count"], cfg["target_startup_s"]
                )
                bounds = tuple(
                    (v - problem.temperature_offset) / problem.temperature_scale
                    for v in (cfg["lower_K"], query["upper_K"])
                )
                checks = check(problem, guess.state, guess, desired, bounds=bounds)
                row.update(checks, field_sha256=file_sha256(field))
                row["verified"] = bool(
                    checks["derivatives_passed"]
                    and np.isfinite(list(checks["kkt"].values())).all()
                    and max(checks["kkt"].values()) <= cfg["nonlinear_tolerance"]
                    and equation_acceptance(checks["equations"], cfg)
                    and adjoint_acceptance(checks["adjoint"])
                )
            except Exception as error:
                row.update(error_type=type(error).__name__, error=str(error))
            write_report(output / "independent-verification.json", result)
    result.update(all_verified=all(r["verified"] for r in result["cases"]), status="complete")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--transport", choices=PROTOCOL["transport_forms"], required=True)
    parser.add_argument("--slabs", type=int, choices=PROTOCOL["time_slabs"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_record.read_text())
    cfg = configuration(source, args.transport, args.slabs)
    cfg["discretization_source_record_sha256"] = file_sha256(args.source_record)
    expected_baseline = source["baseline_sha256"]

    def checked_loader(c):
        problem, baseline = load_problem(c)
        if baseline["baseline_sha256"] != expected_baseline:
            raise ValueError("The physical baseline changed")
        return problem, baseline

    record = run(
        cfg,
        args.output,
        problem_loader=checked_loader,
        budget_seconds=PROTOCOL["sequence_budget_seconds"],
    )
    write_report(
        args.output / "study-protocol.json",
        {
            **PROTOCOL,
            "environment": environment(),
            "source_record_sha256": file_sha256(args.source_record),
        },
    )
    verification = verify_returned(record, args.output)
    verification["optimization_record_sha256"] = file_sha256(args.output / "record.json")
    write_report(args.output / "independent-verification.json", verification)
    if not verification["all_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
