"""Test curvature-based momentum initial guesses at unchanged thermal forcing.

A scalar curvature projection proposes seeds. Only independent convergence
of the full original equations establishes a second numerical solution.
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
from .coupled_flow_response import acceleration_at, response_case
from .coupled_optimize import load_problem
from .reporting import environment, file_sha256, write_arrays, write_report


def curvature_projection(tangent, inverse_curvature, velocity_size):
    """Project J^-1 Q(w,w) onto w using velocity entries alone.

    For F(x + s*w, mu) = F(x,0) + f*(s-mu) + s^2*Q(w,w),
    the one-direction approximation gives mu = s + c*s^2. Its nonzero
    zero-load root s=-1/c provides an initial guess, not a certificate.
    """
    w, q = np.asarray(tangent), np.asarray(inverse_curvature)
    if (
        w.ndim != 1
        or w.shape != q.shape
        or not np.isfinite([w, q]).all()
        or isinstance(velocity_size, bool)
        or not isinstance(velocity_size, int)
        or not 0 < velocity_size <= len(w)
    ):
        raise ValueError("Finite matching vectors and a valid velocity block are required")
    wv, qv = w[:velocity_size], q[:velocity_size]
    norm2 = float(wv @ wv)
    if norm2 <= 0:
        raise ValueError("A nonzero velocity tangent is required")
    coefficient = float(wv @ qv) / norm2
    if not np.isfinite(coefficient) or coefficient == 0:
        raise ValueError("A finite nonzero projected curvature is required")
    return {
        "coefficient_K_inverse": coefficient,
        "projection_relative_remainder": float(
            np.linalg.norm(qv - coefficient * wv) / max(np.linalg.norm(qv), 1e-30)
        ),
        "predicted_turning_step_K": -1 / (4 * coefficient),
        "nonzero_zero_load_seed_parameter_K": -1 / coefficient,
        "scope": "One-direction quadratic approximation used only to select Newton seeds.",
    }


def root_checks(problem, root, previous, state, slab, direction, reference_tangent):
    """Reassemble residual, Newton correction and the signed local response.

    A small Newton correction is a local numerical check, not a rigorous error
    bound for an ill-conditioned nonlinear system.
    """
    flow, free = problem.flow, problem.flow_free
    dt = float(problem.physical_steps[slab])
    acceleration = acceleration_at(problem, state, slab, direction, 0.0)
    load = flow.load(acceleration) + flow.mass @ previous / dt
    rhs = np.r_[load[:, 0], load[:, 1], np.zeros(flow.np)]
    x = np.r_[root.velocity[:, 0], root.velocity[:, 1], root.pressure]
    operator = flow.operator(root.velocity, time_step=dt)
    residual = (operator @ x - rhs)[free]
    jacobian = (operator + flow.convection_derivative(root.velocity))[free][:, free].tocsc()
    scaling = 1 / np.maximum(abs(jacobian).max(axis=1).toarray().ravel(), np.finfo(float).tiny)
    factor = splu((sparse.diags(scaling) @ jacobian).tocsc())
    correction = np.zeros(flow.size)
    correction[free] = factor.solve(-scaling * residual)
    force = flow.load(flow.buoyancy(direction, 0.0, problem.expansion))
    tangent_rhs = np.r_[force[:, 0], force[:, 1], np.zeros(flow.np)][free]
    tangent = np.zeros(flow.size)
    tangent[free] = factor.solve(scaling * tangent_rhs)
    velocity = tangent[: 2 * flow.nv]
    reference = reference_tangent[: 2 * flow.nv]
    cosine = float(velocity @ reference) / max(
        np.linalg.norm(velocity) * np.linalg.norm(reference), 1e-30
    )
    checks = flow.verify(
        root,
        acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        previous=previous,
        time_step=dt,
        pressure_gauge=problem.pressure_gauge,
    )
    return {
        "independent_residuals": checks,
        "newton_velocity_correction_norm_m_s": float(np.linalg.norm(correction[: 2 * flow.nv])),
        "tangent_velocity_norm_m_s_per_K": float(np.linalg.norm(velocity)),
        "tangent_velocity_cosine_with_retained": cosine,
        "tangent_linear_relative_residual": float(
            np.linalg.norm(jacobian @ tangent[free] - tangent_rhs)
            / max(np.linalg.norm(tangent_rhs), 1e-30)
        ),
        "scope": "Fresh local Newton correction and response; no rigorous nonlinear error certificate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.record.read_text())
    response = json.loads(args.response.read_text())
    env, cfg = environment(), source["configuration"]
    if (
        source["status"] == "running"
        or response["status"] != "complete"
        or response["record_sha256"] != file_sha256(args.record)
    ):
        raise ValueError("Use a completed response check of this terminated trajectory")
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
            raise ValueError(f"The frozen momentum implementation differs: {name}")
    input_path = args.response.parent / "inputs.npz"
    if file_sha256(input_path) != response["inputs_sha256"]:
        raise ValueError("The diagnosed temperature direction or tangent changed")
    position, slab = response["position"], response["slab_zero_based"]
    field = args.record.parent / f"target-{position:02d}.npz"
    if file_sha256(field) != response["fields_sha256"][1]:
        raise ValueError("The saved target field changed")
    with np.load(input_path, allow_pickle=False) as data:
        tangent, direction = data["tangent"].copy(), data["direction_K"].copy()
    with np.load(field, allow_pickle=False) as data:
        state, velocities, pressures = (data[k].copy() for k in ("state", "velocity", "pressure"))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-local-flow-branch-v1",
        "status": "running",
        "environment": env,
        "record_sha256": file_sha256(args.record),
        "response_sha256": file_sha256(args.response),
        "field_sha256": file_sha256(field),
        "slab_zero_based": slab,
        "cases": [],
        "seed_multipliers": [-1.0, 1.0, 2.0],
        "budget_seconds_per_case": 180,
        "scope": "Curvature-selected initial guesses at the unchanged temperature, preceding velocity, physical time step and boundary data. An approximate scalar turning point is not a bifurcation certificate. No trajectory optimization or performance comparison is performed.",
    }
    write_report(args.output / "record.json", report)
    start = time.perf_counter()
    try:
        with threadpool_limits(cfg["threads"]):
            problem, baseline = load_problem(cfg)
            if baseline["baseline_sha256"] != source["baseline_sha256"]:
                raise ValueError("The physical baseline differs")
            flow, free = problem.flow, problem.flow_free
            retained = FlowResult(velocities[slab], pressures[slab], "saved", [])
            previous = velocities[slab - 1]
            report["retained_root_checks"] = root_checks(
                problem, retained, previous, state, slab, direction, tangent
            )
            matrix = flow.operator(retained.velocity, time_step=float(problem.physical_steps[slab]))
            matrix = (matrix + flow.convection_derivative(retained.velocity))[free][:, free].tocsc()
            scaling = 1 / np.maximum(
                abs(matrix).max(axis=1).toarray().ravel(), np.finfo(float).tiny
            )
            factor = splu((sparse.diags(scaling) @ matrix).tocsc())
            velocity_tangent = np.column_stack((tangent[: flow.nv], tangent[flow.nv : 2 * flow.nv]))
            quadratic = flow.nonlinear_force(velocity_tangent)
            rhs = np.r_[quadratic[:, 0], quadratic[:, 1], np.zeros(flow.np)][free]
            inverse = np.zeros(flow.size)
            inverse[free] = factor.solve(scaling * rhs)
            report["curvature"] = curvature_projection(tangent, inverse, 2 * flow.nv)
            report["curvature_linear_relative_residual"] = float(
                np.linalg.norm(matrix @ inverse[free] - rhs) / max(np.linalg.norm(rhs), 1e-30)
            )
            write_report(args.output / "record.json", report)
            parameter = report["curvature"]["nonzero_zero_load_seed_parameter_K"]
            base = np.r_[retained.velocity[:, 0], retained.velocity[:, 1], retained.pressure]
            for index, multiplier in enumerate(report["seed_multipliers"]):
                proposal = base + multiplier * parameter * tangent
                seed = FlowResult(
                    np.column_stack((proposal[: flow.nv], proposal[flow.nv : 2 * flow.nv])),
                    proposal[2 * flow.nv :],
                    "seed",
                    [],
                )
                row = {"index": index, "seed_multiplier": multiplier, "status": "running"}
                report["cases"].append(row)
                write_report(args.output / "record.json", report)
                result, metrics = response_case(
                    problem,
                    state,
                    slab,
                    direction,
                    retained,
                    previous,
                    seed,
                    0.0,
                    budget_seconds=180,
                )
                row.update(metrics)
                if row["verified"]:
                    row["root_checks"] = root_checks(
                        problem, result, previous, state, slab, direction, tangent
                    )
                path = args.output / f"case-{index:02d}.npz"
                write_arrays(path, velocity=result.velocity, pressure=result.pressure)
                row["field_sha256"] = file_sha256(path)
                write_report(args.output / "record.json", report)
            report["status"] = "complete"
    except Exception as error:
        report.update(status="diagnostic_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        write_report(args.output / "record.json", report)


if __name__ == "__main__":
    main()
