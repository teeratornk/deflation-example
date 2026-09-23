"""Checkpoint-based numerical diagnostics; these are not complete-run timings."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_bounds import temperature_bounds
from .coupled_control import FlowEvaluationError
from .coupled_derivatives import GaussNewtonOperator, StabilizationBranchError
from .coupled_optimize import load_problem, observe_linear_solves, solver_options
from .coupled_optimizer import box_kkt, box_quadratic
from .coupled_reference import configured_reference
from .coupled_secant import SecantGaussNewton
from .coupled_sequence import (
    CHECKPOINT_SCHEMA,
    RestoredEvaluation,
    configuration_digest,
    prepare_device,
)
from .coupled_targets import desired_temperature
from .reporting import environment, file_sha256, write_arrays, write_report
from .validation import integer


def snapshot(source, output, attempts=5):
    """Copy a checksum-consistent checkpoint while its producer may be running.

    An incomplete copy has no manifest and cannot be consumed by the reviewer.
    The source is never written, and an existing output is never overwritten.
    """
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for _ in range(integer(attempts, "Snapshot attempts", 1)):
        record = json.loads((source / "record.json").read_text())
        raw = (source / "checkpoint-latest.json").read_bytes()
        meta = json.loads(raw)
        if meta["schema"] != CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported checkpoint schema")
        if meta["configuration_sha256"] != configuration_digest(record["configuration"]):
            raise ValueError("Checkpoint and record configurations differ")
        if meta["method"] != record["configuration"]["method"]:
            raise ValueError("Checkpoint and record methods differ")
        shutil.copyfile(source / "checkpoint-latest.npz", output / "checkpoint-latest.npz")
        if file_sha256(output / "checkpoint-latest.npz") != meta["arrays_sha256"]:
            continue
        if (source / "checkpoint-latest.json").read_bytes() != raw:
            continue
        write_report(output / "checkpoint-latest.json", meta)
        write_report(output / "record.json", record)
        manifest = {
            "schema": "coupled-review-snapshot-v1",
            "source_case": source.name,
            "checkpoint_sha256": file_sha256(output / "checkpoint-latest.json"),
            "record_sha256": file_sha256(output / "record.json"),
            "arrays_sha256": meta["arrays_sha256"],
            "iteration": meta["iteration"],
        }
        write_report(output / "snapshot.json", manifest)
        return manifest
    raise RuntimeError("No checksum-consistent snapshot; incomplete output has no manifest")


def read_snapshot(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "snapshot.json").read_text())
    if manifest["schema"] != "coupled-review-snapshot-v1":
        raise ValueError("Unsupported snapshot schema")
    for filename, key in (
        ("record.json", "record_sha256"),
        ("checkpoint-latest.json", "checkpoint_sha256"),
        ("checkpoint-latest.npz", "arrays_sha256"),
    ):
        if file_sha256(directory / filename) != manifest[key]:
            raise ValueError(f"Snapshot checksum differs: {filename}")
    record = json.loads((directory / "record.json").read_text())
    meta = json.loads((directory / "checkpoint-latest.json").read_text())
    with np.load(directory / "checkpoint-latest.npz", allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    return record, meta, arrays, manifest


def optimality(problem, state, gradient, desired, lower, upper):
    """Apply the unchanged outer normalization at this specific state."""
    scaled = gradient / problem.weights
    scale = max(1.0, np.max(np.abs(state - desired)), np.max(np.abs(scaled - (state - desired))))
    return box_kkt(state, scaled, lower, upper, scale), float(scale)


def stationarity_locations(problem, state, gradient, lower, upper, scale, count=10):
    weights = np.asarray(problem.weights)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Tracking weights must be finite and positive")
    scaled = gradient / weights
    epsilon = 32 * np.finfo(float).eps * np.maximum(1, np.abs(state))
    low, high = state <= lower + epsilon, state >= upper - epsilon
    residual = scaled + np.where(high, np.maximum(-scaled, 0), 0)
    residual -= np.where(low, np.maximum(scaled, 0), 0)
    indices = np.argsort(-np.abs(residual), kind="stable")[:count]
    return [
        {
            "index": int(i),
            "slab": int(i // problem.spatial_size),
            "mesh_node": int(problem.free[i % problem.spatial_size]),
            "coordinates": problem.mesh.nodes[problem.free[i % problem.spatial_size]],
            "weight": weights[i],
            "gradient": gradient[i],
            "normalized_stationarity": abs(residual[i]) / scale,
            "active": "lower" if low[i] else "upper" if high[i] else "inactive",
        }
        for i in indices
    ]


def trial_report(problem, evaluation, desired, direction, length, lower, upper, continuation):
    tick = time.perf_counter()
    old = problem.flow_continuation
    problem.flow_continuation = continuation
    state = np.clip(evaluation.state + length * direction, lower, upper)
    try:
        candidate = problem.evaluate(state, initial=evaluation)
        objective, gradient = problem.objective_gradient(candidate, desired)
        kkt, scale = optimality(problem, candidate.state, gradient, desired, lower, upper)
        return {
            "step": length,
            "continuation": continuation,
            "status": "evaluated",
            "objective": objective,
            "objective_change": problem.objective_difference(candidate, evaluation, desired),
            "kkt": kkt,
            "kkt_scale": scale,
            "equations": problem.verify(candidate, local_mass=True),
            "seconds": time.perf_counter() - tick,
        }
    except FlowEvaluationError as failure:
        return {
            "step": length,
            "continuation": continuation,
            "status": "flow_" + failure.result.status,
            "failed_slab": failure.slab,
            "metrics": failure.metrics,
            "flow_history": failure.result.history,
            "seconds": time.perf_counter() - tick,
        }
    except StabilizationBranchError:
        return {
            "step": length,
            "continuation": continuation,
            "status": "stabilization_branch_switch",
            "seconds": time.perf_counter() - tick,
        }
    finally:
        problem.flow_continuation = old


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    record, meta, arrays, manifest = read_snapshot(args.snapshot)
    cfg = {**record["configuration"], "baseline_directory": str(args.baseline)}
    if args.mode == "step" and (
        cfg["method"] == "recycling" or cfg.get("reference_transfer", "full") != "full"
    ):
        raise ValueError(
            "Step replays require Jacobi or full-reference retention; transferred histories are not restored"
        )
    query = cfg["queries"][meta["position"]]
    report = {
        "schema": "coupled-numerical-review-v1",
        "snapshot": manifest,
        "source_environment": environment(),
        "configuration": record["configuration"],
        "status": "running",
        "mode": args.mode,
        "secants": args.secants,
        "scope": "Checkpoint diagnostics; timings exclude preceding optimization and are not complete-run comparisons.",
    }
    write_report(output / "review.json", report)
    problem, baseline = load_problem(cfg)
    report["baseline_sha256"] = baseline["baseline_sha256"]
    problem.evaluation_callback = lambda row: write_report(output / "evaluation-progress.json", row)
    if args.initial:
        state, restored = np.zeros(problem.size), None
    else:
        state = arrays["state"]
        restored = RestoredEvaluation(state, arrays["velocity"], arrays["pressure"])
    report["initial_state_policy"] = "original_zero_state" if args.initial else "saved_checkpoint"
    desired = desired_temperature(
        problem, query["target"], cfg["target_count"], cfg["target_startup_s"]
    )
    bounds = temperature_bounds(cfg, upper_K=query["upper_K"])
    lower = (
        bounds["optimization_lower_K"] - problem.temperature_offset
    ) / problem.temperature_scale
    upper = (
        bounds["optimization_upper_K"] - problem.temperature_offset
    ) / problem.temperature_scale
    state = np.clip(state, lower, upper)
    evaluation = problem.evaluate(state, initial=restored)
    objective, gradient = problem.objective_gradient(evaluation, desired)
    kkt, scale = optimality(problem, state, gradient, desired, lower, upper)
    report.update(
        objective=objective,
        kkt=kkt,
        kkt_scale=scale,
        weight_min=float(problem.weights.min()),
        weight_max=float(problem.weights.max()),
        stationarity_locations=stationarity_locations(
            problem, state, gradient, lower, upper, scale
        ),
        equations=problem.verify(evaluation, local_mass=True),
        adjoint=problem.verify_adjoint(evaluation, desired),
    )
    if not args.initial:
        report["objective_minus_checkpoint"] = objective - meta["objective"]
        report["maximum_velocity_change_on_recheck"] = float(
            np.max(np.abs(np.stack([f.velocity for f in evaluation.flows]) - arrays["velocity"]))
        )
    write_report(output / "review.json", report)
    write_arrays(
        output / "evaluation.npz",
        state=state,
        gradient=gradient,
        weights=problem.weights,
        control=evaluation.control,
        desired=desired,
    )
    if args.mode == "step":
        sampler, barrier, solver_class, device = prepare_device(
            cfg["device"], cfg["memory_interval"]
        )
        sampler.start()
        solver = None
        try:
            tick = time.perf_counter()
            reference = (
                configured_reference(problem, cfg, baseline)
                if cfg["method"] == "reference"
                else None
            )
            report["reference_seconds"] = time.perf_counter() - tick
            report["device"] = device
            solver = solver_class(
                cfg["method"],
                reference=reference,
                rank=cfg["rank"],
                window=cfg["recycle_window"],
                rtol=cfg["inner_tolerance"],
                maxiter=cfg["inner_cap"],
                cg_factor=0.1,
                residual_policy="refine",
                **solver_options(cfg, solver_class),
            )
            observe_linear_solves(solver, output / "linear-progress.json")
            H = GaussNewtonOperator(
                evaluation.jacobian,
                problem.weights,
                problem.alpha,
                0 if args.initial else meta["damping"],
            )
            if args.secants == "retained" and not args.initial:
                H = SecantGaussNewton(
                    H, list(zip(arrays["secant_steps"], arrays["secant_gradients"], strict=True))
                )
            qp = box_quadratic(
                H,
                gradient,
                problem.preconditioning_diagonal(
                    evaluation, 0 if args.initial else meta["damping"]
                ),
                lower - state,
                upper - state,
                solver,
                tolerance=cfg["qp_tolerance"],
                max_steps=cfg["qp_cap"],
            )
            qp_gradient = H @ qp.x + gradient
            normalized_qp = box_kkt(
                qp.x, qp_gradient / problem.weights, lower - state, upper - state, scale
            )
            report["qp"] = {
                "status": qp.status,
                "weighted_kkt": qp.kkt,
                "outer_normalized_kkt": normalized_qp,
                "history": qp.history,
                "direction_infinity_norm": float(np.max(np.abs(qp.x))),
                "directional_derivative": float(gradient @ qp.x),
                "secant_diagnostics": getattr(H, "secant_diagnostics", []),
            }
            write_arrays(output / "direction.npz", state=state, direction=qp.x)
            write_report(output / "review.json", report)
            report["trials"] = []
            if qp.status == "converged":
                for length in (1.0, 0.5, 0.25, 0.125):
                    row = trial_report(
                        problem, evaluation, desired, qp.x, length, lower, upper, False
                    )
                    report["trials"].append(row)
                    write_report(output / "review.json", report)
                    if args.continuation and row["status"].startswith("flow_"):
                        report["trials"].append(
                            trial_report(
                                problem, evaluation, desired, qp.x, length, lower, upper, True
                            )
                        )
                        write_report(output / "review.json", report)
        finally:
            if solver is not None:
                solver.close()
            barrier()
            report["memory"] = sampler.finish()
    report["status"] = "review_complete"
    write_report(output / "review.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot")
    capture.add_argument("--source", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("review")
    review.add_argument("--snapshot", type=Path, required=True)
    review.add_argument("--baseline", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--mode", choices=("inspect", "step"), default="inspect")
    review.add_argument("--secants", choices=("retained", "plain"), default="retained")
    review.add_argument(
        "--initial", action="store_true", help="Reconstruct the original zero-state first update"
    )
    review.add_argument(
        "--continuation",
        action="store_true",
        help="Replay failed trial evaluations with existing residual-load continuation",
    )
    review.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.source, args.output)
    else:
        with threadpool_limits(integer(args.threads, "Threads", 1)):
            run(args)


if __name__ == "__main__":
    main()
