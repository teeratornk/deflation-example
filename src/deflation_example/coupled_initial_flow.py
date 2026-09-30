"""Reproduce an initial momentum solve without changing its stopping criterion."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_flow_solve import solve_momentum
from .coupled_optimize import load_problem
from .reporting import environment, file_sha256, write_arrays, write_report


def diagnose(problem):
    temperature = problem.temperature_offset + problem.temperature_scale * problem.full_temperature(
        problem.initial
    )
    acceleration = problem.acceleration + problem.flow.buoyancy(
        temperature,
        problem.buoyancy_reference,
        problem.expansion,
    )
    dt = float(problem.physical_steps[0])
    start = time.perf_counter()
    result = solve_momentum(
        problem.flow,
        acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        initial=problem.initial_flow,
        previous=problem.initial_flow.velocity,
        time_step=dt,
        pressure_gauge=problem.pressure_gauge,
        tolerance=problem.flow_tolerance,
        max_iterations=problem.flow_cap,
        continuation=False,
    )
    checks = problem.flow.verify(
        result,
        acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        previous=problem.initial_flow.velocity,
        time_step=dt,
        pressure_gauge=problem.pressure_gauge,
    )
    flow = problem.flow
    x = np.r_[result.velocity[:, 0], result.velocity[:, 1], result.pressure]
    A = flow.operator(result.velocity, time_step=dt)
    force = flow.load(acceleration) + flow.mass @ problem.initial_flow.velocity / dt
    rhs = np.r_[force[:, 0], force[:, 1], np.zeros(flow.np)]
    free = problem.flow_free[problem.flow_free < 2 * flow.nv]
    fixed = np.setdiff1d(np.arange(flow.size), problem.flow_free)
    scale = max(
        float(np.linalg.norm(rhs[free]) + np.linalg.norm(A[free][:, fixed] @ x[fixed])), 1e-30
    )
    residual = A @ x - rhs
    # These use the same assembled coefficients and returned fields. They
    # diagnose cancellation only; neither substitutes for final verification.
    wide = A.astype(np.longdouble) @ x.astype(np.longdouble) - rhs.astype(np.longdouble)
    steady = flow.operator(result.velocity) @ x
    delta = flow.mass @ (result.velocity - problem.initial_flow.velocity) / dt
    load = flow.load(acceleration)
    split = steady + np.r_[delta[:, 0] - load[:, 0], delta[:, 1] - load[:, 1], np.zeros(flow.np)]
    return result, {
        "status": result.status,
        "original_equations": checks,
        "history": result.history,
        "seconds": time.perf_counter() - start,
        "time_step_s": dt,
        "momentum_tolerance": problem.flow_tolerance,
        "float64_momentum_residual": float(np.linalg.norm(residual[free]) / scale),
        "extended_accumulation_momentum_residual": float(np.linalg.norm(wide[free]) / scale),
        "split_storage_momentum_residual": float(np.linalg.norm(split[free]) / scale),
        "extended_mantissa_bits": int(np.finfo(np.longdouble).nmant),
        "maximum_velocity_change_m_s": float(
            np.abs(result.velocity - problem.initial_flow.velocity).max()
        ),
        "residual_scope": "Alternative residual evaluations diagnose rounding at the same returned fields. Original stopping and final verification remain unchanged.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.record.read_text())
    cfg = original["configuration"]
    if cfg.get("initial_state_policy") != "physical_initial" or not cfg["transient"]:
        raise ValueError("Select a physical-initial-state transient sequence")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": "coupled-initial-flow-diagnostic-v1",
        "environment": environment(),
        "source_record_sha256": file_sha256(args.record),
        "configuration": cfg,
        "status": "running",
    }
    write_report(args.output / "record.json", metadata)
    with threadpool_limits(cfg["threads"]):
        problem, baseline = load_problem(cfg)
        if baseline["baseline_sha256"] != original["baseline_sha256"]:
            raise ValueError("The recorded physical baseline changed")
        result, report = diagnose(problem)
    write_arrays(args.output / "result.npz", velocity=result.velocity, pressure=result.pressure)
    write_report(
        args.output / "record.json",
        {
            **metadata,
            **report,
            "result_sha256": file_sha256(args.output / "result.npz"),
        },
    )


if __name__ == "__main__":
    main()
