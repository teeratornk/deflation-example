"""Compare state and correction solves for one unchanged Oseen matrix.

This diagnostic uses the ordinary screening input. It neither changes the
screen nor selects a new nonlinear policy. Run on a compute node.
"""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from deflation_example.axisymmetric_flow import FlowResult
from deflation_example.coupled_resolution import forward_model
from deflation_example.fixed_point_study import prepare
from deflation_example.reporting import file_sha256, write_report

HERE = Path(__file__).resolve().parent


def fixed_matrix_comparison(matrix, rhs, initial, fixed, verify, corrections=5):
    """Keep the matrix, boundary values and factorization fixed in both routes."""
    free = np.setdiff1d(np.arange(matrix.shape[0]), fixed)
    block = matrix[free][:, free].tocsc()
    factor = splu(block)
    reduced_rhs = rhs[free] - matrix[free][:, fixed] @ initial[fixed]
    histories = {}
    for route in ("state", "correction"):
        state = initial.copy()
        if route == "state":
            state[free] = factor.solve(reduced_rhs)
        else:
            state[free] += factor.solve((rhs - matrix @ state)[free])
        rows = []
        for index in range(corrections + 1):
            residual = (rhs - matrix @ state)[free]
            rows.append(
                {
                    "corrections": index,
                    "linear_relative_residual": float(
                        np.linalg.norm(residual) / max(np.linalg.norm(reduced_rhs), 1e-30)
                    ),
                    "original_nonlinear_equations": verify(state),
                }
            )
            if index < corrections:
                state[free] += factor.solve(residual)
        histories[route] = rows
    return histories


def nonlinear_comparison(matrix_at, rhs, initial, fixed, verify, iterations=10):
    """Compare algebraically equivalent Oseen maps at identical final accuracy."""
    free = np.setdiff1d(np.arange(len(initial)), fixed)
    histories = {}
    for route in ("state", "correction"):
        start = time.perf_counter()
        x = initial.copy()
        checks = verify(x)
        rows = [{"iteration": 0, "checks": checks}]
        for iteration in range(1, iterations + 1):
            if max(checks.values()) <= 1e-12:
                break
            matrix = matrix_at(x)
            factor = splu(matrix[free][:, free].tocsc())
            if route == "state":
                x[free] = factor.solve(rhs[free] - matrix[free][:, fixed] @ x[fixed])
            else:
                x[free] += factor.solve((rhs - matrix @ x)[free])
            checks = verify(x)
            rows.append({"iteration": iteration, "checks": checks})
        histories[route] = {
            "status": "converged" if max(checks.values()) <= 1e-12 else "diagnostic_iteration_cap",
            "history": rows,
            "seconds": time.perf_counter() - start,
        }
    return histories


def run(data_root, output, threads):
    protocol = json.loads((HERE / "protocol.json").read_text())
    args = SimpleNamespace(
        optimization=data_root / protocol["optimization"],
        target_position=None,
        baseline=data_root / protocol["baseline"],
        fine_baseline=data_root / protocol["fine_baseline"],
        policy="relaxed100",
        family="momentum",
        threads=threads,
        repetition=0,
    )
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(threads):
        problem, _, metadata = prepare(args, 2)
        if metadata["optimization_field_sha256"] != protocol["optimization_field_sha256"]:
            raise ValueError("Diagnostic saved control differs from the protocol")
        flow = problem.flow
        model = forward_model(problem)
        state = problem.full_temperature(problem.initial)
        initial = problem.initial_flow
        dt = float(problem.physical_steps[0])
        force = model._force(state, True)
        load = flow.load(force) + flow.mass @ initial.velocity / dt
        rhs = np.r_[load[:, 0], load[:, 1], np.zeros(flow.np)]
        x = np.r_[initial.velocity[:, 0], initial.velocity[:, 1], initial.pressure].copy()
        fixed = np.r_[problem.boundary_indices, flow.nv + problem.boundary_indices]
        x[fixed] = np.r_[problem.boundary_values[:, 0], problem.boundary_values[:, 1]]
        if problem.pressure_gauge is not None:
            index, value = problem.pressure_gauge
            fixed = np.r_[fixed, 2 * flow.nv + index]
            x[2 * flow.nv + index] = value

        def verify(vector):
            candidate = FlowResult(
                np.column_stack((vector[: flow.nv], vector[flow.nv : 2 * flow.nv])),
                vector[2 * flow.nv :].copy(),
                "diagnostic",
                [],
            )
            return flow.verify(
                candidate,
                force,
                problem.boundary_indices,
                problem.boundary_values,
                previous=initial.velocity,
                time_step=dt,
                pressure_gauge=problem.pressure_gauge,
            )

        start = time.perf_counter()
        matrix = flow.operator(initial.velocity, dt)
        histories = fixed_matrix_comparison(matrix, rhs, x, fixed, verify)
        report = {
            **metadata,
            "status": "diagnostic_running",
            "diagnostic_source_sha256": file_sha256(Path(__file__)),
            "initial_equations": verify(x),
            "time_s": dt,
            "histories": histories,
        }
        write_report(output / "record.json", report)

        def matrix_at(vector):
            velocity = np.column_stack((vector[: flow.nv], vector[flow.nv : 2 * flow.nv]))
            return flow.operator(velocity, dt)

        sequences = nonlinear_comparison(matrix_at, rhs, x, fixed, verify)
        write_report(
            output / "record.json",
            {
                **report,
                "status": "diagnostic_complete",
                "nonlinear_sequences": sequences,
                "seconds": time.perf_counter() - start,
                "scope": "One frozen Oseen matrix with an identical factorization, followed by at most ten Oseen updates for each algebraically equivalent map. Physical data and final original-equation tolerance are unchanged. This diagnosis does not replace any declared screen or complete optimization result.",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    run(args.data_root, args.output, args.threads)


if __name__ == "__main__":
    main()
