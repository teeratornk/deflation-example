"""Coupled-optimization derivative and stationary-solution pilots.

Final repeated complete-sequence studies use frozen configurations after the
operating-point and resolution gates. This driver records development pilots.
"""

import hashlib
import json
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy import sparse
from threadpoolctl import threadpool_limits

from .assess_transformer import transformer_boundaries
from .axisymmetric_flow import AxisymmetricFlow, FlowResult
from .coupled_control import CoupledControlProblem, FlowEvaluationError
from .coupled_optimizer import minimize_coupled
from .coupled_pilot import transformer_inputs
from .mesh_control import build_mesh_control
from .mesh_reference import build_mesh_reference
from .mesh_showcases import desired_temperature
from .oil_properties import momentum_reference
from .reporting import environment, write_fields, write_report
from .study_solvers import StudySolver
from .validation import integer, positive_real


def load_problem(config):
    directory = Path(config["baseline_directory"])
    record = json.loads((directory / "record.json").read_text())
    if record["status"] != "converged":
        raise ValueError("Coupled optimization requires a converged isothermal baseline")
    mesh, parameters, k, c, q, hashes = transformer_inputs(record["configuration"]["level"])
    properties = momentum_reference()
    if record["input_sha256"] != hashes or record["properties"] != properties:
        raise ValueError("Baseline input data or momentum properties differ")
    field = directory / "baseline.npz"
    if hashlib.sha256(field.read_bytes()).hexdigest() != record["baseline_sha256"]:
        raise ValueError("Baseline field checksum differs")
    flow = AxisymmetricFlow(
        mesh,
        properties["kinematic_viscosity_m2_s"],
        convection_form=record["configuration"]["convection_form"],
        grad_div=record.get("grad_div_coefficient_m2_s", 0.0),
    )
    fixed, values = transformer_boundaries(flow, record["inlet_velocity_m_s"])
    with np.load(field, allow_pickle=False) as data:
        baseline = FlowResult(data["velocity"].copy(), data["pressure"].copy(), "converged", [])
    checks = flow.verify(baseline, np.zeros_like(flow.quadrature_points), fixed, values)
    flux = flow.boundary_flux(baseline.velocity)
    if (
        not np.isfinite(list(checks.values())).all()
        or max(checks.values()) > 1e-8
        or abs(flux.sum()) / max(np.abs(flux).sum() / 2, 1e-30) > 1e-6
    ):
        raise ValueError("The baseline fails independent momentum or mass verification")
    p = parameters["physical"]
    steps = None
    if config["transient"]:
        count = integer(config["slabs"], "Number of time slabs", 1)
        steps = np.full(count, positive_real(config["horizon_s"], "Physical horizon") / count)
    problem = CoupledControlProblem(
        flow,
        k,
        c,
        q,
        fixed,
        values,
        baseline,
        alpha=config["alpha"],
        expansion=properties["expansion_coefficient_K_inverse"],
        temperature_scale=p["temperature_scale_K"],
        temperature_offset=p["inlet_temperature_K"],
        buoyancy_reference=p["inlet_temperature_K"],
        velocity_scale=p["time_scale_s"] / p["length_scale_m"],
        time_scale=p["time_scale_s"],
        physical_steps=steps,
        flow_tolerance=config["flow_tolerance"],
        flow_cap=config["flow_cap"],
        flow_continuation=config["flow_continuation"],
    )
    return problem, record


def derivative_report(problem, desired):
    """Fixed, target-independent random direction at a declared smooth trial."""
    x = problem.mesh.nodes[problem.free]
    center, extent = x.mean(axis=0), np.ptp(x, axis=0)
    smooth = 0.02 * np.exp(-np.sum(((x - center) / extent) ** 2, axis=1))
    state = np.tile(smooth, problem.slabs)
    evaluation = problem.evaluate(state)
    value, gradient = problem.objective_gradient(evaluation, desired)
    rng = np.random.default_rng(7281)
    direction, dual = rng.normal(size=(2, problem.size))
    direction /= np.linalg.norm(direction)
    dual /= np.linalg.norm(dual)
    J = evaluation.jacobian
    tangent = J @ direction
    a, b = float(dual @ (J @ direction)), float(direction @ (J.T @ dual))
    steps = [0.01, 0.005, 0.0025, 0.00125]
    rows = []
    for step in steps:
        trial = problem.evaluate(state + step * direction, initial=evaluation)
        trial_value, _ = problem.objective_gradient(trial, desired)
        rows.append(
            {
                "step": step,
                "objective": trial_value,
                "taylor_remainder": abs(trial_value - value - step * (gradient @ direction)),
                "control_linearization_error": float(
                    np.linalg.norm(trial.control - evaluation.control - step * tangent)
                ),
                "relative_control_linearization_error": float(
                    np.linalg.norm(trial.control - evaluation.control - step * tangent)
                    / max(step * np.linalg.norm(tangent), np.finfo(float).tiny)
                ),
            }
        )
    remainders = np.array([r["taylor_remainder"] for r in rows])
    return {
        "trial_temperature": "smooth Gaussian, amplitude 0.02 nondimensional",
        "dot_product_left": a,
        "dot_product_right": b,
        "relative_dot_product_error": abs(a - b) / max(abs(a), abs(b), 1e-30),
        "taylor": rows,
        "taylor_orders": np.log2(remainders[:-1] / remainders[1:]).tolist(),
        "equations": problem.verify(evaluation),
    }


def equations_verified(rows):
    """Common independent equation and conservation criteria for every mode."""
    return bool(rows) and all(
        np.isfinite(
            [
                r["momentum_relative_residual"],
                r["continuity_relative_residual"],
                r["thermal_relative_residual"],
                r["mass_relative_imbalance"],
                r["energy"]["relative_defect"],
            ]
        ).all()
        and max(
            r["momentum_relative_residual"],
            r["continuity_relative_residual"],
            r["thermal_relative_residual"],
        )
        <= 1e-8
        and r["mass_relative_imbalance"] <= 1e-6
        and r["energy"]["relative_defect"] <= 1e-6
        for r in rows
    )


def observe_linear_solves(solver, destination):
    """Optional pilot-only progress; file writes are outside inner timer components."""
    solve = solver.solve
    calls = 0

    def observed(B, b, indices, initial=None):
        nonlocal calls
        calls += 1
        metadata = {"call": calls, "inactive_dofs": len(b)}
        write_report(destination, {**metadata, "status": "running"})
        result, metrics = solve(B, b, indices, initial=initial)
        write_report(
            destination,
            {
                **metadata,
                "status": result.status,
                "iterations": result.iterations,
                "residual": result.residual,
                "rank": result.rank,
                "coarse_condition": result.coarse_condition,
                "fallback": result.fallback_reason,
                "timing": metrics,
            },
        )
        return result, metrics

    solver.solve = observed


def run(config):
    cfg = OmegaConf.to_container(config, resolve=True)
    if not cfg["baseline_directory"]:
        raise ValueError("A verified baseline_directory is required")
    if cfg["mode"] not in {"derivatives", "optimize"}:
        raise ValueError("Choose derivatives or optimize")
    if cfg["device"] not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda")
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(integer(cfg["threads"], "Threads", 1)):
        start = time.perf_counter()
        problem, baseline_record = load_problem(cfg)
        if cfg["evaluation_progress"]:
            problem.evaluation_callback = lambda row: write_report(
                output / "evaluation-progress.json", row
            )
        desired = desired_temperature(problem, cfg["query"], cfg["target_count"])
        metadata = {
            "schema": "coupled-optimization-pilot-v1",
            "environment": environment(),
            "configuration": {
                k: v for k, v in cfg.items() if k not in {"output", "baseline_directory"}
            },
            "baseline_sha256": baseline_record["baseline_sha256"],
            "baseline_configuration": baseline_record["configuration"],
            "input_sha256": baseline_record["input_sha256"],
            "state_dofs": problem.size,
            "thermal_offset_K": problem.temperature_offset,
            "thermal_scale_K": problem.temperature_scale,
            "preconditioner": "positive diagonal of the frozen-velocity thermal normal operator",
        }
        write_report(output / "record.json", {**metadata, "status": "running"})
        try:
            if cfg["mode"] == "derivatives":
                report = derivative_report(problem, desired)
                passed = (
                    report["relative_dot_product_error"] <= 1e-9
                    and min(report["taylor_orders"]) > 1.9
                    and np.isfinite(report["taylor_orders"]).all()
                    and equations_verified(report["equations"])
                )
                write_report(
                    output / "record.json",
                    {
                        **metadata,
                        "status": "verified" if passed else "derivative_check_failed",
                        "derivatives": report,
                        "seconds": time.perf_counter() - start,
                    },
                )
                return
            lower = (cfg["lower_K"] - problem.temperature_offset) / problem.temperature_scale
            upper = (cfg["upper_K"] - problem.temperature_offset) / problem.temperature_scale
            # This pilot uses the current mesh as its reference mesh. Refinement
            # studies must declare and charge a nested coarse construction.
            frozen = build_mesh_control(
                problem.assembly,
                alpha=problem.alpha,
                time_steps=problem.steps if len(problem.steps) else None,
            )
            results = []
            for method in cfg["methods"]:
                tick = time.perf_counter()
                reference = None
                if method == "reference":
                    reference = build_mesh_reference(
                        frozen,
                        problem.assembly,
                        sparse.eye(problem.spatial_size, format="csr"),
                        cfg["rank"],
                        spatial_policy="scaled_schur",
                        temporal_metric="jacobi",
                    )
                setup = time.perf_counter() - tick
                solver_class = StudySolver
                if cfg["device"] == "cuda":
                    from .coupled_cuda_solver import CudaCoupledSolver

                    solver_class = CudaCoupledSolver
                solver = solver_class(
                    method,
                    rank=cfg["rank"],
                    window=cfg["recycle_window"],
                    reference=reference,
                    rtol=cfg["inner_tolerance"],
                    maxiter=cfg["inner_cap"],
                    cg_factor=0.1,
                    residual_policy="refine",
                )
                if cfg["linear_progress"]:
                    observe_linear_solves(solver, output / (method + "-linear-progress.json"))
                try:
                    result = minimize_coupled(
                        problem,
                        desired,
                        lower,
                        upper,
                        solver,
                        tolerance=cfg["nonlinear_tolerance"],
                        max_iterations=cfg["nonlinear_cap"],
                        backtracking=cfg["backtracking"],
                        secant_memory=cfg["secant_memory"],
                        qp_tolerance=cfg["qp_tolerance"],
                        qp_cap=cfg["qp_cap"],
                        callback=lambda row, ev: write_report(
                            output / (method + "-progress.json"), row
                        ),
                    )
                    checks = problem.verify(result.evaluation)
                    adjoint = problem.verify_adjoint(result.evaluation, desired)
                    equations_pass = equations_verified(checks)
                    adjoint_pass = (
                        np.isfinite(adjoint["maximum_momentum_adjoint_relative_residual"])
                        and adjoint["maximum_momentum_adjoint_relative_residual"] <= 1e-8
                    )
                    row = {
                        "method": method,
                        "optimizer_status": result.status,
                        "status": result.status
                        if equations_pass and adjoint_pass
                        else "adjoint_verification_failed"
                        if equations_pass
                        else "equation_verification_failed",
                        "reference_seconds": setup,
                        "optimization_seconds": result.seconds,
                        "objective": result.objective * problem.objective_scale,
                        "kkt": result.kkt,
                        "equations": checks,
                        "adjoint": adjoint,
                        "history": result.history,
                    }
                    results.append(row)
                    write_fields(
                        output / (method + "-fields.npz"),
                        state=result.evaluation.state,
                        control=result.evaluation.control,
                        desired=desired,
                        velocity=np.stack([f.velocity for f in result.evaluation.flows]),
                        pressure=np.stack([f.pressure for f in result.evaluation.flows]),
                    )
                finally:
                    solver.close()
                write_report(
                    output / "record.json", {**metadata, "status": "running", "results": results}
                )
            write_report(
                output / "record.json",
                {
                    **metadata,
                    "status": "complete",
                    "results": results,
                    "seconds": time.perf_counter() - start,
                },
            )
        except FlowEvaluationError as failure:
            failure_record = {
                "status": "flow_" + failure.result.status,
                "slab": failure.slab,
                "metrics": failure.metrics,
                "history": failure.result.history,
                "seconds": time.perf_counter() - start,
            }
            write_report(
                output / "failure.json",
                failure_record,
            )
            write_fields(
                output / "failed-flow.npz",
                velocity=failure.result.velocity,
                pressure=failure.result.pressure,
                temperature=np.empty(0) if failure.temperature is None else failure.temperature,
            )
            write_report(output / "record.json", {**metadata, **failure_record})
            raise


@hydra.main(version_base="1.3", config_path="conf", config_name="coupled_optimize"