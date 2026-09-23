"""One-sided derivative and stabilization checks with an unchanged time prefix."""

import argparse
from copy import copy
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_control import FlowEvaluationError
from .coupled_derivatives import StabilizationBranchError
from .coupled_optimize import load_problem
from .coupled_review import read_snapshot
from .coupled_review_branches import branch_distances
from .coupled_review_derivatives import direction_checks
from .coupled_sequence import RestoredEvaluation
from .coupled_targets import desired_temperature
from .reporting import environment, write_report
from .validation import integer


def review_configuration(configuration, rule):
    """Declare a formulation change without modifying the checkpoint record."""
    result = dict(configuration)
    if rule is not None:
        if rule not in {"hard_min", "smooth_p8"}:
            raise ValueError("Choose hard_min or smooth_p8 streamline coefficients")
        result["streamline_rule"] = rule
    return result


def frozen_prefix_problem(problem, state, flows, first_slab):
    """Keep exactly the suffix equations when all preceding variables are fixed.

    The original quadrature weights and physical time steps are retained. The
    predecessor temperature and velocity become initial data; there is no temporal
    refinement, changed thermal model, or renormalization of the objective.
    """
    first_slab = integer(first_slab, "First suffix slab", 1)
    if not len(problem.steps) or first_slab >= problem.slabs:
        raise ValueError("Choose an interior slab of a transient problem")
    state = np.asarray(state, dtype=float)
    if state.shape != (problem.size,) or len(flows) != problem.slabs:
        raise ValueError("The complete preceding trajectory is required")
    result = copy(problem)
    result.initial = state.reshape(problem.slabs, -1)[first_slab - 1].copy()
    result.initial_flow = flows[first_slab - 1]
    result.slabs = problem.slabs - first_slab
    result.size = result.slabs * problem.spatial_size
    result.steps = problem.steps[first_slab:].copy()
    result.physical_steps = problem.physical_steps[first_slab:].copy()
    result.weights = problem.weights[first_slab * problem.spatial_size :].copy()
    result.evaluation_count = 0
    return result


def one_sided_trial(problem, evaluation, desired, index, cell, step):
    if not np.isfinite(step) or step == 0:
        raise ValueError("A one-sided perturbation must be finite and nonzero")
    trial = evaluation.state.copy()
    trial[index] += step
    try:
        candidate = problem.evaluate(trial, initial=evaluation)
        _, gradient = problem.objective_gradient(candidate, desired)
        switches = branch_distances(problem, [candidate.flows[0].velocity])[0]
        selected = next((r for r in switches["nearest_switches"] if r["mesh_cell"] == cell), None)
        return {
            "step": step,
            "status": "evaluated",
            "temperature_change_K": step * problem.temperature_scale,
            "objective_secant_slope": problem.objective_difference(candidate, evaluation, desired)
            / step,
            "trial_derivative": float(gradient[index]),
            "trial_weight_normalized_derivative": float(gradient[index] / problem.weights[index]),
            "selected_branch": selected,
            "nearest_switches": switches,
        }
    except FlowEvaluationError as failure:
        return {
            "step": step,
            "status": "flow_" + failure.result.status,
            "failed_slab": failure.slab,
            "metrics": failure.metrics,
            "flow_history": failure.result.history,
        }
    except StabilizationBranchError:
        return {"step": step, "status": "stabilization_branch_switch"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--slab", type=int, required=True, help="Zero-based first perturbed slab")
    parser.add_argument("--mesh-node", type=int, required=True)
    parser.add_argument("--cell", type=int, required=True, help="Global mesh cell near a switch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--streamline-rule",
        choices=("hard_min", "smooth_p8"),
        help="Explicit formulation override; saved state and flows are reevaluated",
    )
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument(
        "--centered",
        action="store_true",
        help="Centered source and objective derivative checks at 1e-2, 1e-3, 1e-4, 1e-5",
    )
    diagnostic.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        help="Positive magnitudes; both signs are evaluated in the supplied order",
    )
    args = parser.parse_args()
    if any(not np.isfinite(s) or s <= 0 for s in args.steps):
        raise ValueError("Step magnitudes must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    record, meta, arrays, manifest = read_snapshot(args.snapshot)
    selected = review_configuration(record["configuration"], args.streamline_rule)
    report = {
        "schema": "coupled-frozen-prefix-review-v1",
        "snapshot": manifest,
        "environment": environment(),
        "first_original_slab": args.slab,
        "mesh_node": args.mesh_node,
        "mesh_cell": args.cell,
        "status": "running",
        "source_streamline_rule": record["configuration"].get("streamline_rule", "hard_min"),
        "evaluated_streamline_rule": selected.get("streamline_rule", "hard_min"),
        "rows": [],
        "scope": "Exact suffix equations for a fixed preceding trajectory; one-sided derivatives are diagnostics, not altered optimization acceptance.",
    }
    write_report(args.output / "report.json", report)
    with threadpool_limits(8):
        cfg = {**selected, "baseline_directory": str(args.baseline)}
        full, baseline = load_problem(cfg)
        report["baseline_sha256"] = baseline["baseline_sha256"]
        flows = [
            FlowResult(v, p, "checkpoint", [])
            for v, p in zip(arrays["velocity"], arrays["pressure"], strict=True)
        ]
        problem = frozen_prefix_problem(full, arrays["state"], flows, args.slab)
        problem.evaluation_callback = lambda row: write_report(
            args.output / "evaluation-progress.json", row
        )
        offset = args.slab * full.spatial_size
        state = arrays["state"][offset:]
        saved = RestoredEvaluation(
            state, arrays["velocity"][args.slab :], arrays["pressure"][args.slab :]
        )
        evaluation = problem.evaluate(state, initial=saved)
        query = cfg["queries"][meta["position"]]
        desired = desired_temperature(
            full, query["target"], cfg["target_count"], cfg["target_startup_s"]
        )[offset:]
        indices = np.flatnonzero(problem.free == args.mesh_node)
        if len(indices) != 1:
            raise ValueError("The selected node must be a temperature degree of freedom")
        index = int(indices[0])
        _, gradient = problem.objective_gradient(evaluation, desired)
        report["analytic_derivative"] = float(gradient[index])
        report["weight"] = float(problem.weights[index])
        report["base_branches"] = branch_distances(problem, [evaluation.flows[0].velocity])[0]
        write_report(args.output / "report.json", report)
        if args.centered:
            direction = np.zeros(problem.size)
            direction[index] = 1

            def progress(checks):
                report["centered_checks"] = checks
                write_report(args.output / "report.json", report)

            report["centered_checks"] = direction_checks(
                problem, evaluation, desired, direction, progress
            )
            report["status"] = "review_complete"
            write_report(args.output / "report.json", report)
            return
        report["step_magnitudes"] = args.steps
        for magnitude in args.steps:
            for sign in (-1, 1):
                step = sign * magnitude
                report["rows"].append(
                    one_sided_trial(problem, evaluation, desired, index, args.cell, step)
                )
                write_report(args.output / "report.json", report)
        report["status"] = "review_complete"
        write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
