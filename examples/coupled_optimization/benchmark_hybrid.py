"""Compare CPU and hybrid backends on the same saved coupled derivative.

Two manufactured right-hand sides exercise an inactive mask and subsequent
release of its constrained variables. These are kernel checks, not optimization
timings. Iteration caps and failed residual checks remain in the output.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from threadpoolctl import threadpool_limits

from deflation_example.axisymmetric_flow import FlowResult
from deflation_example.coupled_derivatives import GaussNewtonOperator
from deflation_example.coupled_hybrid_solver import HybridCoupledSolver
from deflation_example.coupled_optimize import load_problem
from deflation_example.coupled_reference import configured_reference
from deflation_example.coupled_saved import (
    load_saved_solution,
    require_matching_baseline,
    file_digest,
)
from deflation_example.reporting import environment, write_report
from deflation_example.solvers import independent_residual
from deflation_example.study_solvers import StudySolver
from deflation_example.validation import integer


def check_policy(H, diagonal, systems, reference, backend, method, rank, cap):
    """Charge solver creation, transfer, coarse processing and cleanup together."""
    solver_class = HybridCoupledSolver if backend == "hybrid" else StudySolver
    rows = []
    start = time.perf_counter()
    solver = solver_class(
        method,
        reference=reference,
        rank=0 if method == "jacobi" else rank,
        window=rank,
        rtol=1e-10,
        cg_factor=0.1,
        maxiter=cap,
        residual_policy="refine",
    )
    try:
        for indices, exact, rhs in systems:
            B = H.restrict(indices)
            B.diagonal = lambda: diagonal[indices]
            tick = time.perf_counter()
            result, timing = solver.solve(B, rhs, indices, initial=np.zeros(len(indices)))
            solve_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            residual = independent_residual(B, result.x, rhs)
            verification_seconds = time.perf_counter() - tick
            entries = asdict(result)
            del entries["x"]
            rows.append(
                {
                    "inactive_dofs": len(indices),
                    **entries,
                    "independent_relative_residual": residual,
                    "verified": result.status == "converged" and residual <= 1e-10,
                    "relative_solution_error": float(
                        np.linalg.norm(result.x - exact) / np.linalg.norm(exact)
                    ),
                    "solve_seconds": solve_seconds,
                    "additional_verification_seconds": verification_seconds,
                    "timing": timing,
                }
            )
    finally:
        solver.close()
    return {
        "backend": backend,
        "method": method,
        "requested_rank": 0 if method == "jacobi" else rank,
        "seconds": time.perf_counter() - start,
        "systems": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("jacobi", "reference", "recycling"), default="reference"
    )
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--cap", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = integer(args.rank, "Requested rank", 1)
    cap = integer(args.cap, "Shared inner iteration cap", 1)
    repeats = integer(args.repeats, "Repetitions", 1)
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        import cupy as cp

        tick = time.perf_counter()
        cp.zeros(1)
        cp.cuda.get_current_stream().synchronize()
        startup = time.perf_counter() - tick
        record, cfg, fields, digest = load_saved_solution(
            args.optimization, args.method, args.target_position
        )
        cfg.update(baseline_directory=str(args.baseline), rank=rank)
        problem, baseline = load_problem(cfg)
        require_matching_baseline(record, baseline)
        filename = (
            args.method + "-fields.npz"
            if args.target_position is None
            else f"target-{args.target_position:02d}.npz"
        )
        with np.load(args.optimization / filename, allow_pickle=False) as arrays:
            flows = tuple(
                FlowResult(v.copy(), p.copy(), "initial", [])
                for v, p in zip(arrays["velocity"], arrays["pressure"], strict=True)
            )
        evaluation = problem.evaluate(fields["state"], initial=SimpleNamespace(flows=flows))
        H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
        diagonal = problem.preconditioning_diagonal(evaluation)
        tick = time.perf_counter()
        reference = configured_reference(problem, cfg, baseline)
        construction = time.perf_counter() - tick
        lower = (cfg["lower_K"] - problem.temperature_offset) / problem.temperature_scale
        upper = (cfg["upper_K"] - problem.temperature_offset) / problem.temperature_scale
        initial_mask = np.flatnonzero(
            (fields["state"] > lower + 1e-10) & (fields["state"] < upper - 1e-10)
        )
        if not len(initial_mask):
            raise ValueError("The saved state has no inactive variables")
        rng = np.random.default_rng(782)
        exact = rng.normal(size=problem.size)
        systems = []
        for indices in (initial_mask, np.arange(problem.size)):
            systems.append((indices, exact[indices], H.restrict(indices) @ exact[indices]))
        # Warm both operator backends before any measured policy. Each policy
        # still pays its own lazy GPU factor upload and cleanup.
        from deflation_example.coupled_cuda import CudaControlJacobian, CudaGaussNewton

        tick = time.perf_counter()
        H @ exact
        device = CudaControlJacobian(evaluation.jacobian)
        try:
            CudaGaussNewton(device, problem.weights, problem.alpha).apply(exact)
            cp.cuda.get_current_stream().synchronize()
        finally:
            device.close()
        warmup = time.perf_counter() - tick
        metadata = {
            "schema": "coupled-hybrid-kernel-check-v1",
            "environment": environment(),
            "driver_sha256": file_digest(__file__),
            "optimization_field_sha256": digest,
            "baseline_sha256": baseline["baseline_sha256"],
            "rank": rank,
            "cap": cap,
            "repetitions": repeats,
            "seed": 782,
            "threads": args.threads,
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cupy": cp.__version__,
            "state_dofs": problem.size,
            "reference_description": reference.description,
            "reference_construction_seconds": construction,
            "process_startup_seconds": startup,
            "shared_warmup_seconds": warmup,
            "scope": "Manufactured fixed-derivative kernel checks with zero initial guesses; no nonlinear optimization speedup. Each policy includes transfer, setup, iteration, independent verification and cleanup. Shared model assembly, reference construction and warmup are outside those intervals.",
        }
        rows = []
        order = [
            (backend, method)
            for method in ("jacobi", "reference", "recycling")
            for backend in ("cpu", "hybrid")
        ]
        write_report(args.output / "record.json", {**metadata, "status": "running", "rows": rows})
        for repetition in range(repeats):
            for backend, method in order if repetition % 2 == 0 else order[::-1]:
                row = check_policy(H, diagonal, systems, reference, backend, method, rank, cap)
                rows.append({"repetition": repetition, **row})
                write_report(
                    args.output / "record.json", {**metadata, "status": "running", "rows": rows}
                )
        write_report(args.output / "record.json", {**metadata, "status": "complete", "rows": rows})


if __name__ == "__main__":
    main()
