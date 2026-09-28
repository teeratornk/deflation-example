"""Matched accuracy replays of a recorded inactive system in an unfinished QP.

The quadratic, bound assignments, right-hand side and initial guess stay fixed.
Reconstruction and diagnostic costs are reported separately from the inner solve.
These replays do not advance or certify the nonlinear optimization.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator
from threadpoolctl import threadpool_limits

from .coupled_optimize import equation_acceptance, load_problem, observe_linear_solves
from .coupled_optimizer import box_kkt
from .coupled_qp_diagnostic import read_previous
from .coupled_recovery import identity
from .coupled_retention_replay import rebuild
from .coupled_trace import read_arrays, read_manifest
from .coupled_trust import optimality
from .reporting import environment, file_sha256, write_arrays, write_report
from .solvers import independent_residual
from .study_solvers import StudySolver
from .validation import positive_real


def partition(state, gradient, diagonal, lower, upper):
    projected = state - gradient / diagonal
    return (projected >= upper).astype(np.int8) - (projected <= lower).astype(np.int8)


def replay_system(
    H,
    gradient,
    diagonal,
    lower,
    upper,
    mask,
    rhs,
    initial,
    saved_solution,
    solver,
    assess,
    factory=None,
    *,
    reconstruction_tolerance=1e-8,
):
    """Solve the recorded state equation and measure its next active-set decision."""
    tolerance = positive_real(reconstruction_tolerance, "Reconstruction tolerance")
    g, d, lo, hi = (np.asarray(a, dtype=float) for a in (gradient, diagonal, lower, upper))
    mask = np.asarray(mask)
    if (
        g.ndim != 1
        or any(a.shape != g.shape for a in (d, lo, hi, mask))
        or H.shape != (len(g), len(g))
        or not np.isfinite([g, d, lo, hi]).all()
        or np.any(d <= 0)
        or np.any(lo >= hi)
        or not np.isin(mask, [-1, 0, 1]).all()
    ):
        raise ValueError("Use finite compatible quadratic data and a valid bound partition")
    indices = np.flatnonzero(mask == 0)
    b, x0, old = (np.asarray(a, dtype=float) for a in (rhs, initial, saved_solution))
    if not len(indices) or any(a.shape != indices.shape for a in (b, x0, old)):
        raise ValueError("The recorded vectors must match the nonempty inactive set")
    if not np.isfinite([b, x0, old]).all():
        raise ValueError("Recorded vectors must be finite")
    fixed = np.where(mask < 0, lo, np.where(mask > 0, hi, 0.0))

    def action(z):
        full = np.zeros_like(g)
        full[indices] = np.asarray(z).ravel()
        return (H @ full)[indices]

    B = LinearOperator((len(indices),) * 2, matvec=action, rmatvec=action, dtype=float)
    B.diagonal = lambda: d[indices].copy()
    reconstructed_rhs = -(g + H @ fixed)[indices]
    norm = np.linalg.norm(b)
    rhs_difference = (
        float(np.linalg.norm(reconstructed_rhs - b) / norm)
        if norm
        else float(np.linalg.norm(reconstructed_rhs - b))
    )
    if not np.isfinite(rhs_difference) or rhs_difference > tolerance:
        raise ValueError("Reconstructed right-hand side differs from the recorded system")
    before = fixed.copy()
    before[indices] = old
    before_gradient = H @ before + g
    old_next = partition(before, before_gradient, d, lo, hi)
    original_before = independent_residual(B, old, b)
    tick = time.perf_counter()
    if factory is not None:
        B.preconditioner = factory(indices)
    preconditioner_seconds = time.perf_counter() - tick
    result, timing = solver.solve(B, b.copy(), indices, initial=x0.copy())
    candidate = fixed.copy()
    candidate[indices] = result.x
    derivative = H @ candidate + g
    next_mask = partition(candidate, derivative, d, lo, hi)
    original = independent_residual(B, result.x, b)
    scaled_margin = np.minimum(
        np.abs(candidate - derivative / d - lo), np.abs(candidate - derivative / d - hi)
    )
    report = {
        "status": result.status,
        "iterations": result.iterations,
        "linear_tolerance": solver.rtol,
        "original_residual": original,
        "linear_verified": bool(
            result.status == "converged" and np.isfinite(original) and original <= solver.rtol
        ),
        "recorded_solution_recomputed_residual": original_before,
        "rhs_relative_difference": rhs_difference,
        "before_kkt": assess(before, before_gradient),
        "candidate_kkt": assess(candidate, derivative),
        "next_partition_changes": int(np.count_nonzero(next_mask != mask)),
        "newly_active": int(np.count_nonzero((mask == 0) & (next_mask != 0))),
        "newly_inactive": int(np.count_nonzero((mask != 0) & (next_mask == 0))),
        "next_partition_difference_from_recorded_solution": int(
            np.count_nonzero(next_mask != old_next)
        ),
        "candidate_infinity_difference_from_recorded_solution": float(
            np.max(np.abs(candidate - before))
        ),
        "minimum_partition_margin": float(np.min(scaled_margin)),
        "preconditioner_seconds": preconditioner_seconds,
        "timing": timing,
        "scope": "One fixed inactive equation; quadratic and nonlinear convergence unassessed.",
    }
    return report, {
        "candidate": candidate,
        "next_partition": next_mask,
        "recorded_candidate": before,
        "recorded_next_partition": old_next,
        "partition": mask.copy(),
        "initial": x0.copy(),
        "rhs": b.copy(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    args = parser.parse_args()
    positive_real(args.rtol, "Replay tolerance")
    source = json.loads(args.record.read_text())
    cfg = dict(source["configuration"])
    if source["status"] != "quadratic_active_set_cap" or cfg["method"] != "jacobi":
        raise ValueError("Use the completed rank-zero active-set-cap diagnostic")
    fp = identity(cfg, source["environment"]["source_sha256"])
    saved, slot = read_previous(args.record.parent / "recovery", fp)
    opt, trace = saved["optimizer"], args.record.parent / "inactive-trace-00"
    manifest = read_manifest(trace)
    if manifest["status"] != "complete" or manifest["baseline_sha256"] != source["baseline_sha256"]:
        raise ValueError("Use a complete trace of the same baseline")
    row = manifest["systems"][-1]
    qrow = manifest["quadratics"][row["quadratic"]]
    if (
        row["equation"] != "state"
        or row["status"] != "converged"
        or row["pdas_step"] + 1 != opt["qp"]["next_step"]
        or qrow["iteration"] != opt["iteration"]
    ):
        raise ValueError("The trace and preceding checkpoint must identify the same final solve")
    arrays = read_arrays(trace, qrow["file"], qrow["sha256"])
    data = read_arrays(trace, row["file"], row["sha256"])
    solved = read_arrays(trace, row["solution_file"], row["solution_sha256"])["x"]
    indices = np.flatnonzero(opt["qp"]["partition"] == 0)
    if (
        not np.array_equal(arrays["state"], opt["state"])
        or not np.array_equal(indices, data["indices"])
        or not np.array_equal(solved, opt["qp"]["x"][indices])
    ):
        raise ValueError("Checkpoint fields or partition do not match the recorded system")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-qp-accuracy-replay-v1",
        "status": "reconstructing",
        "environment": environment(),
        "source_record_sha256": file_sha256(args.record),
        "trace_sha256": file_sha256(trace / "manifest.json"),
        "previous_slot": slot,
        "system_sha256": row["sha256"],
        "quadratic_sha256": qrow["sha256"],
        "solution_sha256": row["solution_sha256"],
        "linear_tolerance": args.rtol,
        "scope": "New read-only diagnostic population; no change to earlier runs or final criteria.",
    }
    write_report(args.output / "record.json", report)
    start = time.perf_counter()
    try:
        with threadpool_limits(cfg["threads"]):
            cfg["baseline_directory"] = str(args.baseline)
            problem, baseline = load_problem(cfg)
            if baseline["baseline_sha256"] != source["baseline_sha256"]:
                raise ValueError("Physical baseline differs")
            problem.evaluation_callback = lambda r: write_report(args.output / "evaluation.json", r)
            evaluation, H, diagonal, gradient = rebuild(problem, arrays, qrow)
            equations = problem.verify(evaluation, local_mass=True)
            if not equation_acceptance(equations, cfg):
                raise ValueError("Reconstructed trajectory fails equation verification")
            lo, hi = arrays["lower"] - arrays["state"], arrays["upper"] - arrays["state"]
            _, scale = optimality(
                problem, evaluation, arrays["desired"], gradient, arrays["lower"], arrays["upper"]
            )
            report.update(
                status="solving",
                reconstruction_seconds=time.perf_counter() - start,
                equation_checks_passed=True,
                recorded_linear_residual=row["original_residual"],
            )
            write_report(args.output / "record.json", report)
            from .coupled_frozen_preconditioner import frozen_preconditioner_factory

            solver = StudySolver(
                "jacobi",
                rtol=args.rtol,
                maxiter=cfg["inner_cap"],
                cg_factor=0.1,
                refresh=cfg["inner_refresh"],
                residual_policy="refine",
            )
            observe_linear_solves(solver, args.output / "linear-progress.json")
            try:
                result, fields = replay_system(
                    H,
                    gradient,
                    diagonal,
                    lo,
                    hi,
                    opt["qp"]["partition"],
                    data["rhs"],
                    data["initial"],
                    solved,
                    solver,
                    lambda x, g: box_kkt(x, g / problem.weights, lo, hi, scale),
                    frozen_preconditioner_factory(
                        problem, evaluation, qrow["damping"], cfg["frozen_sweeps"]
                    ),
                )
            finally:
                solver.close()
            write_arrays(args.output / "fields.npz", **fields)
            report.update(
                status="complete",
                diagnostic=result,
                fields_sha256=file_sha256(args.output / "fields.npz"),
            )
    except Exception as error:
        report.update(status="diagnostic_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        write_report(args.output / "record.json", report)
    if not report["diagnostic"]["linear_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
