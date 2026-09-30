"""Local momentum time refinement at fixed, interpolated temperatures."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest, splu
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_flow_response import temperature_direction
from .coupled_flow_solve import solve_momentum
from .coupled_optimize import load_problem
from .reporting import environment, file_sha256, write_arrays, write_report
from .validation import integer, positive_real


def load_trajectory(path, position):
    path = Path(path)
    record = json.loads(path.read_text())
    if record.get("status") in {None, "running"}:
        raise ValueError("A terminated source trajectory is required")
    if record.get("schema") == "coupled-radius-restart-v1":
        cfg = record["source_configuration"]
        if position != record["position"]:
            raise ValueError("The selected target differs from the restart")
        field = path.parent / "result.npz"
        if file_sha256(field) != record["result_sha256"]:
            raise ValueError("The retained restart fields changed")
    else:
        cfg = record["configuration"]
        if not 0 <= position < len(record["cases"]):
            raise ValueError("Choose a saved source target")
        field = path.parent / f"target-{position:02d}.npz"
    with np.load(field, allow_pickle=False) as archive:
        fields = {k: archive[k].copy() for k in ("state", "velocity", "pressure")}
    if any(not np.isfinite(v).all() for v in fields.values()):
        raise ValueError("Source fields must be finite")
    return cfg, fields, {"record_sha256": file_sha256(path), "field_sha256": file_sha256(field)}


def interval_response(problem, root, previous, acceleration_direction, dt, previous_tangent):
    """Propagate the interval derivative through the computed preceding velocity."""
    flow, free = problem.flow, problem.flow_free
    jacobian = (
        flow.operator(root.velocity, time_step=dt) + flow.convection_derivative(root.velocity)
    )[free][:, free].tocsc()
    scaling = 1 / np.maximum(abs(jacobian).max(axis=1).toarray().ravel(), np.finfo(float).tiny)
    scaled = (sparse.diags(scaling) @ jacobian).tocsc()
    factor = splu(scaled)
    force = flow.load(acceleration_direction) + flow.mass @ previous_tangent / dt
    rhs = np.r_[force[:, 0], force[:, 1], np.zeros(flow.np)][free]
    answer = factor.solve(scaling * rhs)
    full = np.zeros(flow.size)
    full[free] = answer
    inverse = LinearOperator(
        scaled.shape, matvec=factor.solve, rmatvec=lambda v: factor.solve(v, trans="T")
    )
    # onenormest uses random auxiliary probes; seed and restore its legacy RNG.
    rng = np.random.get_state()
    try:
        np.random.seed(20260930)
        estimate = float(sparse.linalg.norm(scaled, 1) * onenormest(inverse))
    finally:
        np.random.set_state(rng)
    return np.column_stack((full[: flow.nv], full[flow.nv : 2 * flow.nv])), {
        "linear_relative_residual": float(
            np.linalg.norm(jacobian @ answer - rhs) / max(np.linalg.norm(rhs), 1e-30)
        ),
        "row_scaled_condition_one_norm_lower_estimate": estimate,
        "velocity_response_nodal_norm_m_s_per_K": float(np.linalg.norm(full[: 2 * flow.nv])),
    }


def refine_interval(
    problem,
    fields,
    direction,
    slab,
    subdivision,
    perturbation_K,
    *,
    budget_seconds=900,
    callback=None,
):
    slab = integer(slab, "Slab", 1)
    subdivision = integer(subdivision, "Subdivision", 1)
    positive_real(budget_seconds, "Interval budget")
    if slab >= problem.slabs or not np.isfinite(perturbation_K):
        raise ValueError("Choose a valid interval and finite endpoint perturbation")
    expected = {
        "state": (problem.size,),
        "velocity": (problem.slabs, problem.flow.nv, 2),
        "pressure": (
            problem.slabs,
            problem.flow.np,
        ),
    }
    if any(
        fields[k].shape != shape or not np.isfinite(fields[k]).all()
        for k, shape in expected.items()
    ):
        raise ValueError("The saved trajectory dimensions differ from the physical problem")
    direction = np.asarray(direction, dtype=float)
    if (
        direction.shape != (len(problem.mesh.nodes),)
        or not np.isfinite(direction).all()
        or np.any(direction[problem.mesh.dirichlet])
    ):
        raise ValueError("The perturbation must match the mesh and vanish on thermal boundaries")
    state = fields["state"].reshape(problem.slabs, problem.spatial_size)
    start_T, end_T = (
        problem.temperature_offset + problem.temperature_scale * problem.full_temperature(state[n])
        for n in (slab - 1, slab)
    )
    previous = fields["velocity"][slab - 1].copy()
    root = FlowResult(fields["velocity"][slab].copy(), fields["pressure"][slab].copy(), "saved", [])
    tangent = np.zeros_like(previous)
    dt = float(problem.physical_steps[slab]) / subdivision
    start, rows = time.perf_counter(), []
    status = "converged"
    for step in range(1, subdivision + 1):
        fraction = step / subdivision
        temperature = start_T + fraction * (end_T - start_T + perturbation_K * direction)
        acceleration = problem.acceleration + problem.flow.buoyancy(
            temperature, problem.buoyancy_reference, problem.expansion
        )
        options = dict(previous=previous, time_step=dt, pressure_gauge=problem.pressure_gauge)
        root = solve_momentum(
            problem.flow,
            acceleration,
            problem.boundary_indices,
            problem.boundary_values,
            initial=root,
            tolerance=problem.flow_tolerance,
            max_iterations=problem.flow_cap,
            continuation=False,
            stop_requested=lambda: time.perf_counter() - start >= budget_seconds,
            **options,
        )
        checks = problem.flow.verify(
            root, acceleration, problem.boundary_indices, problem.boundary_values, **options
        )
        verified = (
            root.status == "converged"
            and np.isfinite(list(checks.values())).all()
            and max(checks.values()) <= problem.flow_tolerance
        )
        row = {
            "substep": step,
            "time_step_s": dt,
            "status": root.status,
            "verified": bool(verified),
            "equations": checks,
            "history": root.history,
        }
        rows.append(row)
        if verified:
            force_direction = problem.flow.buoyancy(fraction * direction, 0.0, problem.expansion)
            tangent, row["tangent"] = interval_response(
                problem, root, previous, force_direction, dt, tangent
            )
            previous = root.velocity.copy()
        else:
            status = root.status if root.status != "converged" else "verification_failed"
        if callback:
            callback(rows)
        if not verified:
            break
    return (
        root,
        tangent,
        {
            "status": status,
            "verified": status == "converged",
            "subdivision": subdivision,
            "perturbation_K": perturbation_K,
            "steps": rows,
            "seconds": time.perf_counter() - start,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--direction-record", type=Path, required=True)
    parser.add_argument("--slab", type=int, default=8)
    parser.add_argument("--subdivision", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--perturbation-K", type=float, choices=(-1e-6, 0, 1e-6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg, fields, inputs = load_trajectory(args.record, args.position)
    direction_cfg, first, _ = load_trajectory(args.direction_record, 0)
    _, second, _ = load_trajectory(args.direction_record, 1)
    if cfg != direction_cfg:
        raise ValueError("Direction and source trajectories must share the physical configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-interval-refinement-v1",
        "status": "running",
        "configuration": cfg,
        "environment": environment(),
        "inputs": inputs,
        "direction_record_sha256": file_sha256(args.direction_record),
        "position": args.position,
        "slab_zero_based": args.slab,
        "budget_seconds": 900,
        "scope": "Local momentum integration over one original interval, with linearly interpolated prescribed temperatures and fixed interval-start velocity. No thermal solve, optimization or temporal-resolution certification.",
    }
    write_report(args.output / "record.json", report)
    try:
        with threadpool_limits(cfg["threads"]):
            problem, baseline = load_problem(cfg)
            original = json.loads(args.direction_record.read_text())
            if baseline["baseline_sha256"] != original["baseline_sha256"]:
                raise ValueError("The physical baseline changed")
            direction = temperature_direction(problem, second["state"], first["state"], args.slab)
            root, tangent, result = refine_interval(
                problem,
                fields,
                direction,
                args.slab,
                args.subdivision,
                args.perturbation_K,
                callback=lambda rows: write_report(args.output / "progress.json", {"steps": rows}),
            )
            report.update(result)
            field = args.output / "result.npz"
            write_arrays(field, velocity=root.velocity, pressure=root.pressure, tangent=tangent)
            report["result_sha256"] = file_sha256(field)
    except Exception as error:
        report.update(status="diagnostic_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_report(args.output / "record.json", report)
    if not report.get("verified"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
