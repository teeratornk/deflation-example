"""Local momentum response at an immutable failed optimization trajectory.

The preceding time level is held fixed. These are single-slab diagnostics,
not a new trajectory optimization, a resolution study, or performance evidence.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_flow_solve import solve_momentum
from .coupled_optimize import load_problem
from .reporting import environment, file_sha256, write_arrays, write_report
from .validation import integer, positive_real


def temperature_direction(problem, state, preceding_target, slab):
    """One-kelvin infinity-norm direction; thermal boundary entries remain zero."""
    arrays = [np.asarray(a, dtype=float) for a in (state, preceding_target)]
    if any(a.shape != (problem.size,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Both temperature trajectories must be finite and complete")
    slab = integer(slab, "Zero-based slab", 0)
    if slab >= problem.slabs:
        raise ValueError("Slab is outside the trajectory")
    delta = (arrays[0] - arrays[1]).reshape(problem.slabs, -1)[slab]
    magnitude = float(np.max(np.abs(delta)))
    if magnitude == 0:
        raise ValueError("The selected temperatures define no perturbation direction")
    direction = np.zeros(len(problem.mesh.nodes))
    direction[problem.free] = delta / magnitude
    return direction


def acceleration_at(problem, state, slab, direction, step_K):
    state = np.asarray(state).reshape(problem.slabs, problem.spatial_size)[slab]
    temperature = (
        problem.temperature_offset
        + problem.temperature_scale * problem.full_temperature(state)
        + step_K * direction
    )
    return problem.acceleration + problem.flow.buoyancy(
        temperature, problem.buoyancy_reference, problem.expansion
    )


def tangent_response(problem, retained, previous, state, slab, direction):
    """Differentiate momentum at fixed preceding velocity and fixed boundaries."""
    flow, free = problem.flow, problem.flow_free
    dt = float(problem.physical_steps[slab])
    matrix = flow.operator(retained.velocity, time_step=dt)
    matrix = (matrix + flow.convection_derivative(retained.velocity))[free][:, free].tocsc()
    load = flow.load(flow.buoyancy(direction, 0.0, problem.expansion))
    rhs = np.r_[load[:, 0], load[:, 1], np.zeros(flow.np)][free]
    scaling = 1 / np.maximum(abs(matrix).max(axis=1).toarray().ravel(), np.finfo(float).tiny)
    scaled = (sparse.diags(scaling) @ matrix).tocsc()
    answer = splu(scaled).solve(scaling * rhs)
    full = np.zeros(flow.size)
    full[free] = answer
    metrics = flow.verify(
        retained,
        acceleration_at(problem, state, slab, direction, 0.0),
        problem.boundary_indices,
        problem.boundary_values,
        previous=previous,
        time_step=dt,
        pressure_gauge=problem.pressure_gauge,
    )
    denominator = max(np.linalg.norm(rhs), np.finfo(float).tiny)
    return full, {
        "retained_residuals": metrics,
        "linear_relative_residual": float(np.linalg.norm(matrix @ answer - rhs) / denominator),
        "row_scaled_linear_relative_residual": float(
            np.linalg.norm(scaled @ answer - scaling * rhs)
            / max(np.linalg.norm(scaling * rhs), np.finfo(float).tiny)
        ),
        "velocity_derivative_norm_m_s_per_K": float(np.linalg.norm(full[: 2 * flow.nv])),
    }


def response_case(
    problem,
    state,
    slab,
    direction,
    retained,
    previous,
    seed,
    step_K,
    *,
    continuation=False,
    budget_seconds=180,
):
    """Keep every terminal state, but compare derivatives only for verified roots."""
    budget_seconds = positive_real(budget_seconds, "Per-case time budget")
    if not np.isfinite(step_K):
        raise ValueError("A finite temperature perturbation is required")
    start = time.perf_counter()
    acceleration = acceleration_at(problem, state, slab, direction, step_K)
    options = dict(
        previous=previous,
        time_step=float(problem.physical_steps[slab]),
        pressure_gauge=problem.pressure_gauge,
    )
    result = solve_momentum(
        problem.flow,
        acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        initial=seed,
        tolerance=problem.flow_tolerance,
        max_iterations=problem.flow_cap,
        continuation=continuation,
        polish_load=continuation,
        stop_requested=lambda: time.perf_counter() - start >= budget_seconds,
        **options,
    )
    checks = problem.flow.verify(
        result, acceleration, problem.boundary_indices, problem.boundary_values, **options
    )
    verified = (
        result.status == "converged"
        and np.isfinite(list(checks.values())).all()
        and max(checks.values()) <= problem.flow_tolerance
    )
    delta = result.velocity - retained.velocity
    return result, {
        "step_K": step_K,
        "continuation": continuation,
        "status": result.status,
        "verified": bool(verified),
        "residuals": checks,
        "history": result.history,
        "velocity_change_norm_m_s": float(np.linalg.norm(delta)),
        "velocity_change_max_m_s": float(np.linalg.norm(delta, axis=1).max()),
        "seconds": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--position", type=int, default=1)
    parser.add_argument("--slab", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=180)
    args = parser.parse_args()
    positive_real(args.budget_seconds, "Per-case time budget")
    source = json.loads(args.record.read_text())
    cfg, env = source["configuration"], environment()
    if source["status"] == "running" or not cfg["transient"]:
        raise ValueError("Use a terminated transient comparison")
    if not 0 < args.position < len(source["cases"]):
        raise ValueError("A preceding target is required")
    if not 0 < args.slab < cfg["slabs"]:
        raise ValueError("Choose a slab with a preceding time level")
    modules = (
        "axisymmetric_flow.py",
        "coupled_flow_solve.py",
        "coupled_optimize.py",
        "coupled_pilot.py",
        "meshes.py",
        "oil_properties.py",
        "assess_transformer.py",
    )
    for name in modules:
        if env["source_sha256"][name] != source["environment"]["source_sha256"][name]:
            raise ValueError(f"The diagnostic changes a frozen momentum input: {name}")
    files = [args.record.parent / f"target-{p:02d}.npz" for p in (args.position - 1, args.position)]
    saved = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            saved.append({k: data[k].copy() for k in ("state", "velocity", "pressure")})
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-local-flow-response-v1",
        "status": "running",
        "environment": env,
        "record_sha256": file_sha256(args.record),
        "fields_sha256": [file_sha256(p) for p in files],
        "numerical_source": source["environment"]["git_head"],
        "position": args.position,
        "target": source["cases"][args.position]["target"],
        "optimization_status": source["cases"][args.position]["status"],
        "slab_zero_based": args.slab,
        "budget_seconds_per_case": args.budget_seconds,
        "scope": "Single-slab momentum responses with the preceding velocity fixed. The direction is the normalized temperature difference from the preceding target, with fixed thermal boundaries. Perturbed temperatures are diagnostic inputs, not constrained optimizer iterates. No optimization or performance claim is made.",
        "cases": [],
    }
    write_report(args.output / "record.json", report)
    with threadpool_limits(cfg["threads"]):
        problem, baseline = load_problem(cfg)
        if baseline["baseline_sha256"] != source["baseline_sha256"]:
            raise ValueError("The physical baseline differs")
        before, current = saved
        state, slab = current["state"], args.slab
        direction = temperature_direction(problem, state, before["state"], slab)
        retained = FlowResult(current["velocity"][slab], current["pressure"][slab], "saved", [])
        previous = current["velocity"][slab - 1]
        tangent, report["tangent"] = tangent_response(
            problem, retained, previous, state, slab, direction
        )
        if max(report["tangent"]["retained_residuals"].values()) > problem.flow_tolerance:
            raise ValueError("The saved reference does not satisfy the momentum equations")
        write_arrays(args.output / "inputs.npz", direction_K=direction, tangent=tangent)
        report["inputs_sha256"] = file_sha256(args.output / "inputs.npz")
        seeds = {
            "retained": retained,
            "preceding_time": FlowResult(previous, current["pressure"][slab - 1], "seed", []),
            "preceding_target": FlowResult(
                before["velocity"][slab], before["pressure"][slab], "seed", []
            ),
        }
        cases = [(name, 0.0, False) for name in seeds]
        cases += [("retained", sign * h, False) for h in (1e-4, 1e-6, 1e-8) for sign in (-1, 1)]
        cases += [("retained", sign * 1e-6, True) for sign in (-1, 1)]
        for index, (name, step, continuation) in enumerate(cases):
            row = {
                "index": index,
                "initial_guess": name,
                "step_K": step,
                "continuation": continuation,
                "status": "running",
            }
            report["cases"].append(row)
            write_report(args.output / "record.json", report)
            try:
                result, metrics = response_case(
                    problem,
                    state,
                    slab,
                    direction,
                    retained,
                    previous,
                    seeds[name],
                    step,
                    continuation=continuation,
                    budget_seconds=args.budget_seconds,
                )
                row.update(metrics)
                path = args.output / f"case-{index:02d}.npz"
                write_arrays(path, velocity=result.velocity, pressure=result.pressure)
                row["field_sha256"] = file_sha256(path)
                if row["verified"] and step != 0:
                    response = (result.velocity - retained.velocity) / step
                    exact = np.column_stack(
                        (tangent[: problem.flow.nv], tangent[problem.flow.nv : 2 * problem.flow.nv])
                    )
                    row["one_sided_velocity_tangent_relative_difference"] = float(
                        np.linalg.norm(response - exact)
                        / max(np.linalg.norm(response), np.linalg.norm(exact), 1e-30)
                    )
            except Exception as error:
                row.update(
                    status="diagnostic_error",
                    error_type=type(error).__name__,
                    error=str(error),
                    verified=False,
                )
            write_report(args.output / "record.json", report)
    report["status"] = (
        "diagnostic_error"
        if any(r["status"] == "diagnostic_error" for r in report["cases"])
        else "complete"
    )
    write_report(args.output / "record.json", report)
    if report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
