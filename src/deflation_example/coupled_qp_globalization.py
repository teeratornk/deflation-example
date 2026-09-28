"""Compare active-set updates on a recorded, unchanged coupled quadratic.

This diagnostic restarts the selected quadratic from a zero temperature step.
It neither restarts nor certifies the nonlinear optimization. Reconstruction,
preconditioner construction, and the complete quadratic solve have separate timers.
"""

import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .box_projected_cg import box_projected_cg
from .coupled_frozen_preconditioner import frozen_preconditioner_factory
from .coupled_optimize import equation_acceptance, load_problem, observe_linear_solves
from .coupled_optimizer import box_kkt, box_quadratic
from .coupled_retention_replay import rebuild
from .coupled_trace import read_arrays, read_manifest
from .coupled_trust import optimality
from .reporting import environment, file_sha256, write_arrays, write_report
from .study_solvers import StudySolver
from .validation import integer, positive_real


def relative_difference(actual, expected):
    norm = np.linalg.norm(expected)
    return float(np.linalg.norm(actual - expected) / norm if norm else np.linalg.norm(actual))


def validate_reconstruction(arrays, gradient, diagonal, tolerance=1e-8):
    """Bind recomputed derivative and active-set metric to the saved quadratic."""
    differences = {
        "gradient_relative_difference": relative_difference(gradient, arrays["gradient"]),
        "diagonal_relative_difference": relative_difference(diagonal, arrays["diagonal"]),
    }
    if any(not np.isfinite(value) or value > tolerance for value in differences.values()):
        raise ValueError("Reconstructed quadratic data differ from the trace")
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quadratic", type=int, default=2)
    parser.add_argument("--method", choices=["pdas", "projected"], required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--budget-seconds", type=float, default=7200)
    parser.add_argument("--restriction", choices=["product", "submatrix"], default="product")
    parser.add_argument("--device", choices=["cpu", "cuda", "hybrid"], default="cpu")
    parser.add_argument("--frozen-layout", choices=["serial", "block_diagonal"], default="serial")
    parser.add_argument(
        "--reference-initial", choices=["clipped_zero", "optimizer"], default="clipped_zero"
    )
    parser.add_argument("--initial-snapshot", type=Path)
    parser.add_argument("--initial-assessment", type=Path)
    parser.add_argument("--reference-rank", type=int, default=0)
    parser.add_argument("--reference-steps", type=int, default=48)
    args = parser.parse_args()
    positive_real(args.rtol, "Linear tolerance")
    positive_real(args.budget_seconds, "Quadratic time budget")
    integer(args.reference_rank, "Reference rank", 0)
    integer(args.reference_steps, "Reference construction steps", 1)
    if args.reference_rank > args.reference_steps:
        raise ValueError("Reference rank must not exceed its construction steps")
    if args.reference_initial == "optimizer" and (
        args.initial_snapshot is None or args.initial_assessment is None
    ):
        raise ValueError("Optimizer reference initialization requires snapshot and assessment")
    manifest = read_manifest(args.trace)
    if manifest["status"] != "complete" or not 0 <= args.quadratic < len(manifest["quadratics"]):
        raise ValueError("Select a recorded quadratic from a complete trace")
    row = manifest["quadratics"][args.quadratic]
    tolerance = positive_real(row["qp_tolerance"], "Recorded quadratic tolerance")
    arrays = read_arrays(args.trace, row["file"], row["sha256"])
    cfg = {**manifest["configuration"], "baseline_directory": str(args.baseline)}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-quadratic-globalization-v1",
        "status": "reconstructing",
        "environment": environment(),
        "trace_sha256": file_sha256(args.trace / "manifest.json"),
        "quadratic_sha256": row["sha256"],
        "quadratic": args.quadratic,
        "method": args.method,
        "linear_tolerance": args.rtol,
        "quadratic_tolerance": tolerance,
        "initial": "zero temperature step",
        "restriction": args.restriction,
        "budget_seconds": args.budget_seconds,
        "device": args.device,
        "reference_rank": args.reference_rank,
        "reference_initial": args.reference_initial,
        "frozen_layout": args.frozen_layout,
        "scope": "One unchanged quadratic; nonlinear optimality is not assessed.",
    }
    write_report(args.output / "record.json", report)
    start = time.perf_counter()
    try:
        with threadpool_limits(cfg["threads"]):
            problem, baseline = load_problem(cfg)
            if baseline["baseline_sha256"] != manifest["baseline_sha256"]:
                raise ValueError("Physical baseline differs")
            problem.evaluation_callback = lambda r: write_report(args.output / "evaluation.json", r)
            evaluation, H, diagonal, gradient = rebuild(problem, arrays, row)
            differences = validate_reconstruction(arrays, gradient, diagonal)
            equations = problem.verify(evaluation, local_mass=True)
            if not equation_acceptance(equations, cfg):
                raise ValueError("Reconstructed trajectory fails equation verification")
            lo, hi = arrays["lower"] - arrays["state"], arrays["upper"] - arrays["state"]
            if np.any(lo > 0) or np.any(hi < 0):
                raise ValueError("The declared initial zero step must satisfy the recorded bounds")
            _, scale = optimality(
                problem, evaluation, arrays["desired"], gradient, arrays["lower"], arrays["upper"]
            )
            report.update(
                reconstruction_seconds=time.perf_counter() - start,
                reconstruction=differences,
                equation_checks=equations,
                dimension=problem.size,
                maximum_steps=cfg["qp_cap"],
            )
            tick = time.perf_counter()
            factory = frozen_preconditioner_factory(
                problem,
                evaluation,
                row["damping"],
                cfg["frozen_sweeps"],
                restriction=args.restriction,
            )
            report["preconditioner_construction_seconds"] = time.perf_counter() - tick
            reference = None
            tick = time.perf_counter()
            if args.reference_rank:
                from .coupled_nominal_krylov import configured_krylov_reference

                guess = None
                if args.reference_initial == "optimizer":
                    from .coupled_initial_state import snapshot_initial_guess

                    guess, report["initial_state"] = snapshot_initial_guess(
                        {
                            **cfg,
                            "initial_state_snapshot": str(args.initial_snapshot),
                            "initial_state_assessment": str(args.initial_assessment),
                        },
                        problem,
                        baseline,
                        0,
                    )

                reference = configured_krylov_reference(
                    problem,
                    {
                        **cfg,
                        "device": "cpu",
                        "inner_preconditioner": "frozen",
                        "frozen_sweeps": 3,
                        "rank": args.reference_rank,
                        "reference_transfer": "full",
                        "reference_krylov_steps": args.reference_steps,
                        "reference_krylov_selection": "alternating_low_high",
                    },
                    initial_guess=guess,
                    verification_callback=lambda rows: write_report(
                        args.output / "reference-equations.json", {"equations": rows}
                    ),
                )
                from .coupled_retention_replay import save_reference

                report["reference"] = save_reference(args.output, "reference", reference)
            report["reference_construction_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter()
            solver_type = StudySolver
            extra = {}
            if args.device == "cuda":
                from .coupled_cuda_solver import CudaCoupledSolver

                solver_type = CudaCoupledSolver
                extra["frozen_layout"] = args.frozen_layout
            elif args.device == "hybrid":
                from .coupled_hybrid_solver import HybridCoupledSolver

                solver_type = HybridCoupledSolver
                extra.update(block_device="cpu", coarse_device="cuda")
            solver = solver_type(
                "reference" if reference is not None else "jacobi",
                rank=args.reference_rank,
                reference=reference,
                rtol=args.rtol,
                maxiter=cfg["inner_cap"],
                cg_factor=0.1,
                refresh=cfg["inner_refresh"],
                residual_policy="refine",
                **extra,
            )
            if args.device in {"cuda", "hybrid"}:
                import cupy as cp

                properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
                report["gpu"] = {
                    "model": properties["name"].decode(),
                    "cupy": cp.__version__,
                    "total_memory_bytes": properties["totalGlobalMem"],
                    "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                }
            report["solver_construction_seconds"] = time.perf_counter() - tick
            heartbeat = {"time": 0.0}

            def progress(event):
                now = time.perf_counter()
                if event["stage"] in {"initial", "final"} or now - heartbeat["time"] >= 30:
                    write_report(args.output / "iteration-progress.json", event)
                    heartbeat["time"] = now

            solver.progress_callback = progress
            observe_linear_solves(solver, args.output / "linear-progress.json")
            solve_start = time.perf_counter()
            solver.stop_requested = lambda: time.perf_counter() - solve_start >= args.budget_seconds
            report.update(status="solving")
            write_report(args.output / "record.json", report)

            def checkpoint(payload):
                write_arrays(args.output / "latest-state.npz", state=payload["x"])
                write_report(
                    args.output / "quadratic-progress.json",
                    {
                        "history": payload["history"],
                        "kkt": payload.get("kkt"),
                        "seconds": time.perf_counter() - solve_start,
                    },
                )

            def assess(x, g):
                return box_kkt(x, g / problem.weights, lo, hi, scale)

            procedure = box_quadratic if args.method == "pdas" else box_projected_cg
            options = (
                {"correction_policy": cfg["qp_correction_policy"]} if args.method == "pdas" else {}
            )
            try:
                result = procedure(
                    H,
                    gradient,
                    diagonal,
                    lo,
                    hi,
                    solver,
                    initial=np.zeros(problem.size),
                    tolerance=tolerance,
                    max_steps=cfg["qp_cap"],
                    preconditioner_factory=factory,
                    kkt_evaluator=assess,
                    checkpoint=checkpoint,
                    **options,
                )
            finally:
                solver.close()
            report["quadratic_seconds"] = time.perf_counter() - solve_start
            independent = assess(result.x, H @ result.x + gradient)
            verified = result.status == "converged" and max(independent.values()) <= tolerance
            write_arrays(args.output / "fields.npz", step=result.x, lower=lo, upper=hi)
            report.update(
                status="complete",
                quadratic_status=result.status,
                quadratic_verified=verified,
                kkt=independent,
                history=result.history,
                fields_sha256=file_sha256(args.output / "fields.npz"),
                inner_iterations=sum(r.get("linear_iterations", 0) for r in result.history),
                quadratic_change=float(gradient @ result.x + 0.5 * result.x @ (H @ result.x)),
            )
    except Exception as error:
        report.update(status="diagnostic_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        write_report(args.output / "record.json", report)
    if not report["quadratic_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
