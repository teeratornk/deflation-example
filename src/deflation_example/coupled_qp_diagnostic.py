"""Reproduce one rejected fixed-mask correction without advancing optimization."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator
from threadpoolctl import threadpool_limits

from .coupled_derivatives import GaussNewtonOperator
from .coupled_frozen_preconditioner import frozen_preconditioner_factory
from .coupled_optimize import load_problem, observe_linear_solves
from .coupled_optimizer import box_kkt
from .coupled_recovery import RecoveryStore, identity
from .coupled_secant import SecantGaussNewton
from .coupled_sequence import RestoredEvaluation
from .coupled_targets import desired_temperature
from .coupled_trust import intermediate_targets, optimality
from .reporting import environment, file_sha256, write_arrays, write_report
from .solvers import independent_residual
from .study_solvers import StudySolver


def read_previous(directory, fingerprint):
    """The previous slot is diagnostic input, not a verified restart checkpoint.

    Its bytes are hashed for the new report. Its retained nonlinear fields must
    equal those in the manifest-bound latest checkpoint, which is read normally.
    """
    directory = Path(directory)
    latest = RecoveryStore(directory, fingerprint).load()
    if latest is None:
        raise ValueError("A manifest-bound latest checkpoint is required")
    manifest = json.loads((directory / "latest.json").read_text())
    path = directory / f"state-{1 - manifest['slot']}.npz"
    digest = file_sha256(path)
    with np.load(path, allow_pickle=False) as data:

        def unpack(value):
            if isinstance(value, dict):
                if set(value) == {"__array__"}:
                    array = data[value["__array__"]].copy()
                    if array.dtype.hasobject or not np.isfinite(array).all():
                        raise ValueError("Invalid previous-slot array")
                    return array
                return {key: unpack(item) for key, item in value.items()}
            if isinstance(value, list):
                return [unpack(item) for item in value]
            return value

        previous = unpack(json.loads(str(data["metadata"])))
    before, after = previous["optimizer"], latest["optimizer"]
    if before is None or after is None or before["qp"] is None:
        raise ValueError("The previous slot must contain the unfinished quadratic")
    for key in ("state", "velocity", "pressure"):
        if not np.array_equal(before[key], after[key]):
            raise ValueError("Previous-slot nonlinear state differs from the latest checkpoint")
    if previous["baseline_sha256"] != latest["baseline_sha256"]:
        raise ValueError("Previous-slot baseline differs")
    if file_sha256(path) != digest:
        raise ValueError("Previous slot changed while reading")
    return previous, {"file": path.name, "sha256": digest, "manifest_bound": False}


def inspect_correction(
    H,
    gradient,
    diagonal,
    lower,
    upper,
    solver,
    qp,
    assess,
    factory=None,
    *,
    use_recorded_partition=False,
    progress=None,
):
    """Measure the existing rejection test and the candidate's subsequent mask."""
    x = np.asarray(qp["x"]).copy()

    def partition(state, derivative):
        projected = state - derivative / diagonal
        return (projected >= upper).astype(np.int8) - (projected <= lower).astype(np.int8)

    derivative = H @ x + gradient
    reconstructed = partition(x, derivative)
    mask = np.asarray(qp["partition"], dtype=np.int8)
    mismatch = int(np.count_nonzero(reconstructed != mask))
    if mismatch and not use_recorded_partition:
        raise ValueError("The saved mask does not repeat")
    indices = np.flatnonzero(mask == 0)
    if not len(indices):
        raise ValueError("The diagnostic requires an inactive system")
    fixed = np.where(mask < 0, lower, np.where(mask > 0, upper, 0.0))

    def action(z):
        full = np.zeros_like(x)
        full[indices] = np.asarray(z).ravel()
        return (H @ full)[indices]

    B = LinearOperator((len(indices),) * 2, matvec=action, rmatvec=action, dtype=float)
    B.diagonal = lambda: diagonal[indices].copy()
    if factory is not None:
        B.preconditioner = factory(indices)
    rhs = -(gradient + H @ fixed)[indices]
    defect = rhs - B @ x[indices]
    before = assess(x, derivative)
    preliminary = {
        "before_kkt": before,
        "saved_quadratic_kkt": qp.get("kkt"),
        "reconstructed_partition_changes": mismatch,
        "partition_policy": "recorded" if use_recorded_partition else "verified_repeated",
    }
    if progress is not None:
        progress(preliminary)
    result, timing = solver.solve(B, defect, indices, initial=np.zeros_like(defect))
    candidate = fixed.copy()
    candidate[indices] = x[indices] + result.x
    candidate_gradient = H @ candidate + gradient
    after = assess(candidate, candidate_gradient)
    next_mask = partition(candidate, candidate_gradient)
    fresh = independent_residual(B, candidate[indices], rhs)
    changed = int(np.count_nonzero(next_mask != mask))
    record = {
        **preliminary,
        "status": result.status,
        "iterations": result.iterations,
        "solved_equation_residual": result.residual,
        "original_system_residual": fresh,
        "before_kkt": before,
        "candidate_kkt": after,
        "mask_changes": changed,
        "newly_active": int(np.count_nonzero((mask == 0) & (next_mask != 0))),
        "newly_inactive": int(np.count_nonzero((mask != 0) & (next_mask == 0))),
        "existing_stagnation_test_rejects": max(after.values()) >= max(before.values()),
        "linear_checks_pass": result.status == "converged" and fresh <= solver.rtol,
        "timing": timing,
    }
    return record, {
        "before": x,
        "candidate": candidate,
        "partition": mask,
        "next_partition": next_mask,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--use-recorded-partition",
        action="store_true",
        help="Keep the saved inactive set and report changes after flow reconstruction",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_record = json.loads(args.record.read_text())
    cfg = source_record["configuration"]
    if cfg["method"] != "jacobi" or cfg["inner_preconditioner"] != "frozen":
        raise ValueError("This replay matches the rank-zero frozen-preconditioner diagnostic")
    fp = identity(cfg, source_record["environment"]["source_sha256"])
    saved, slot = read_previous(args.record.parent / "recovery", fp)
    opt = saved["optimizer"]
    result = {
        "schema": "coupled-fixed-mask-diagnostic-v1",
        "status": "rebuilding",
        "environment": environment(),
        "record_sha256": file_sha256(args.record),
        "previous_slot": slot,
        "scope": "One reconstructed correction; no optimizer restart, completed trajectory or performance comparison.",
    }
    write_report(args.output / "record.json", result)
    start = time.perf_counter()
    with threadpool_limits(cfg["threads"]):
        problem, baseline = load_problem(cfg)
        if baseline["baseline_sha256"] != saved["baseline_sha256"]:
            raise ValueError("Physical baseline differs")
        problem.evaluation_callback = lambda row: write_report(
            args.output / "evaluation-progress.json", row
        )
        guess = RestoredEvaluation(opt["state"], opt["velocity"], opt["pressure"])
        evaluation = problem.evaluate(opt["state"], initial=guess)
        desired = desired_temperature(
            problem,
            cfg["queries"][saved["position"]]["target"],
            cfg["target_count"],
            cfg["target_startup_s"],
        )
        objective, gradient = problem.objective_gradient(evaluation, desired)
        H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
        if opt["secants"]:
            H = SecantGaussNewton(H, opt["secants"])
        diagonal = problem.preconditioning_diagonal(evaluation, 0.0)
        lower = (cfg["lower_K"] - problem.temperature_offset) / problem.temperature_scale
        upper = (
            cfg["queries"][saved["position"]]["upper_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        kkt, scale = optimality(problem, evaluation, desired, gradient, lower, upper)
        delta = opt["radius_K"] / problem.temperature_scale
        lo, hi = (
            np.maximum(lower - evaluation.state, -delta),
            np.minimum(upper - evaluation.state, delta),
        )
        _, ltol, _ = intermediate_targets(
            max(kkt.values()), cfg["trust_accuracy"], cfg["qp_tolerance"], cfg["inner_tolerance"]
        )
        result.update(
            status="solving",
            reconstructed_objective=objective,
            retained_kkt=kkt,
            checkpoint_kkt=opt["kkt"],
            linear_tolerance=ltol,
        )
        write_report(args.output / "record.json", result)
        solver = StudySolver(
            "jacobi",
            rtol=ltol,
            maxiter=cfg["inner_cap"],
            cg_factor=0.1,
            refresh=cfg["inner_refresh"],
            residual_policy="refine",
        )
        observe_linear_solves(solver, args.output / "linear-progress.json")
        try:
            diagnostics, arrays = inspect_correction(
                H,
                gradient,
                diagonal,
                lo,
                hi,
                solver,
                opt["qp"],
                lambda x, g: box_kkt(x, g / problem.weights, lo, hi, scale),
                frozen_preconditioner_factory(problem, evaluation, sweeps=cfg["frozen_sweeps"]),
                use_recorded_partition=args.use_recorded_partition,
                progress=lambda row: write_report(args.output / "reconstruction.json", row),
            )
        finally:
            solver.close()
        write_arrays(args.output / "correction.npz", **arrays)
        result.update(
            status="complete", diagnostic=diagnostics, seconds=time.perf_counter() - start
        )
        write_report(args.output / "record.json", result)


if __name__ == "__main__":
    main()
