"""Isolate the exact isothermal Gauss--Newton subproblem of a saved configuration.

At zero temperature rise, the source derivative has no velocity contribution.
The thermal normal operator therefore supplies the exact initial model without
constructing redundant momentum derivative factors. This is a diagnostic QP,
not a complete coupled optimization or a substitute for its measured cost.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from deflation_example.coupled_derivatives import ControlJacobian, GaussNewtonOperator
from deflation_example.coupled_optimize import load_problem, observe_linear_solves
from deflation_example.coupled_optimizer import box_quadratic
from deflation_example.mesh_control import build_mesh_control
from deflation_example.mesh_reference import build_mesh_reference
from deflation_example.coupled_targets import desired_temperature
from deflation_example.reporting import environment, write_report
from deflation_example.study_solvers import StudySolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--configuration-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=100)
    args = parser.parse_args()
    cfg = json.loads((args.configuration_run / "record.json").read_text())["configuration"]
    cfg["baseline_directory"] = str(args.baseline)
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(cfg["threads"]):
        problem, _ = load_problem(cfg)
        if np.any(problem.thermal_boundary) or np.any(problem.initial):
            raise ValueError(
                "This diagnostic requires zero initial temperature rise and boundary values"
            )
        frozen = build_mesh_control(
            problem.assembly,
            alpha=problem.alpha,
            time_steps=problem.steps if len(problem.steps) else None,
        )
        J = ControlJacobian(
            frozen.A,
            [sparse.csr_matrix((problem.spatial_size, 0))] * problem.slabs,
            sparse.csr_matrix((0, problem.spatial_size)),
            [None] * problem.slabs,
            [sparse.csr_matrix((0, 0))] * problem.slabs,
        )
        H = GaussNewtonOperator(J, problem.weights, problem.alpha)
        desired = desired_temperature(
            problem, cfg["query"], cfg["target_count"], cfg.get("target_startup_s", 0.0)
        )
        gradient = -frozen.load(desired)
        method = "reference" if args.rank else "jacobi"
        reference = None
        start = time.perf_counter()
        if args.rank:
            reference = build_mesh_reference(
                frozen,
                problem.assembly,
                sparse.eye(problem.spatial_size, format="csr"),
                args.rank,
                spatial_policy="scaled_schur",
                temporal_metric="jacobi",
            )
        construction = time.perf_counter() - start
        solver = StudySolver(
            method,
            rank=args.rank,
            reference=reference,
            rtol=1e-10,
            maxiter=50000,
            cg_factor=0.1,
            residual_policy="refine",
        )
        observe_linear_solves(solver, args.output / "linear-progress.json")
        metadata = {
            "environment": environment(),
            "configuration": {k: v for k, v in cfg.items() if k != "baseline_directory"},
            "rank": args.rank,
            "reference_seconds": construction,
            "scope": "Exact initial isothermal quadratic subproblem; coupled optimization is outside this timing interval.",
        }
        write_report(args.output / "record.json", {**metadata, "status": "running"})
        start = time.perf_counter()
        try:
            result = box_quadratic(
                H,
                gradient,
                frozen.H.diagonal(),
                (cfg["lower_K"] - problem.temperature_offset) / problem.temperature_scale,
                (cfg["upper_K"] - problem.temperature_offset) / problem.temperature_scale,
                solver,
                tolerance=cfg["qp_tolerance"],
                max_steps=cfg["qp_cap"],
            )
        finally:
            solver.close()
        write_report(
            args.output / "record.json",
            {
                **metadata,
                "status": result.status,
                "qp_seconds": time.perf_counter() - start,
                "kkt": result.kkt,
                "history": result.history,
            },
        )


if __name__ == "__main__":
    main()
