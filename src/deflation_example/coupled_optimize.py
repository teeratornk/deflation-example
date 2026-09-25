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
from threadpoolctl import threadpool_limits

from .assess_transformer import transformer_boundaries
from .axisymmetric_flow import AxisymmetricFlow, FlowResult
from .coupled_control import CoupledControlProblem, FlowEvaluationError
from .coupled_bounds import temperature_bounds
from .coupled_optimizer import NUMERICAL_POLICY, minimize_coupled
from .coupled_pilot import transformer_inputs
from .coupled_reference import configured_reference
from .coupled_saved import load_saved_solution, require_matching_baseline
from .coupled_targets import desired_temperature
from .oil_properties import momentum_reference
from .reporting import environment, write_fields, write_report
from .study_solvers import StudySolver
from .validation import integer, positive_real


def load_problem(config):
    policy = config.get("momentum_factor_policy", "retained")
    if policy not in {"retained", "recompute"}:
        raise ValueError("Choose retained or recompute momentum-factor storage")
    if policy == "recompute" and config.get("device", "cpu") != "cpu":
        raise ValueError("Recomputed momentum factors currently require the CPU backend")
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
    feedback = float(config.get("feedback_multiplier", 1.0))
    if not np.isfinite(feedback) or not 0 <= feedback <= 1:
        raise ValueError("Diagnostic feedback multiplier must lie in [0, 1]")
    problem = CoupledControlProblem(
        flow,
        k,
        c,
        q,
        fixed,
        values,
        baseline,
        alpha=config["alpha"],
        expansion=properties["expansion_coefficient_K_inverse"] * feedback,
        temperature_scale=p["temperature_scale_K"],
        temperature_offset=p["inlet_temperature_K"],
        buoyancy_reference=p["inlet_temperature_K"],
        velocity_scale=p["time_scale_s"] / p["length_scale_m"],
        time_scale=p["time_scale_s"],
        physical_steps=steps,
        flow_tolerance=config["flow_tolerance"],
        flow_cap=config["flow_cap"],
        flow_continuation=config["flow_continuation"],
        momentum_factor_policy=policy,
        transport_form=config.get("transport_form", "advective"),
        consistent_stabilization=config.get("consistent_stabilization", False),
        reference_stabilization=config.get("reference_stabilization", "shipped"),
        streamline_rule=config.get("streamline_rule", "hard_min"),
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


def equations_verified(rows, *, equation_tolerance=1e-8, conservation_tolerance=1e-6):
    """Common independent equation and conservation criteria for every mode."""
    equation_tolerance = positive_real(equation_tolerance, "Equation acceptance tolerance")
    conservation_tolerance = positive_real(conservation_tolerance, "Conservation tolerance")
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
        and min(
            r["momentum_relative_residual"],
            r["continuity_relative_residual"],
            r["thermal_relative_residual"],
            r["mass_relative_imbalance"],
            r["energy"]["relative_defect"],
        )
        >= 0
        and max(
            r["momentum_relative_residual"],
            r["continuity_relative_residual"],
            r["thermal_relative_residual"],
        )
        <= equation_tolerance
        and r["mass_relative_imbalance"] <= conservation_tolerance
        and r["energy"]["relative_defect"] <= conservation_tolerance
        for r in rows
    )


def equation_acceptance(rows, config):
    """Use the declared final criteria; retain legacy defaults for saved inputs."""
    return equations_verified(
        rows,
        equation_tolerance=config.get("equation_acceptance_tolerance", 1e-8),
        conservation_tolerance=config.get("conservation_tolerance", 1e-6),
    )


def adjoint_acceptance(report):
    """Require source, momentum and outer-normalized gradient checks at 1e-8.

    This gate applies to newly evaluated states; it does not relabel archived
    records containing the earlier momentum-only verification.
    """
    names = (
        "maximum_momentum_adjoint_relative_residual",
        "maximum_source_adjoint_relative_residual",
        "gradient_weight_normalized_difference",
    )
    values = np.asarray([report.get(name, np.nan) for name in names], dtype=float)
    return bool(np.isfinite(values).all() and np.all(values >= 0) and np.all(values <= 1e-8))


def observe_linear_solves(solver, destination, *, heartbeat_seconds=None):
    """Report each solve and optionally its CPU kernels.

    Enclosing records are outside kernel timers; heartbeat I/O is included in
    the measured kernel interval. Kernel residuals refer to that kernel's right-
    hand side, which may be an error equation during residual correction.
    """
    if heartbeat_seconds is not None:
        heartbeat_seconds = positive_real(heartbeat_seconds, "Heartbeat interval")
        if solver.device != "cpu":
            raise ValueError("Iteration heartbeats require the CPU kernel")
    solve = solver.solve
    calls = 0

    def observed(B, b, indices, initial=None):
        nonlocal calls
        calls += 1
        metadata = {"call": calls, "inactive_dofs": len(b)}
        write_report(destination, {**metadata, "status": "running"})
        previous_progress = solver.progress_callback
        if heartbeat_seconds is not None:
            from .linear_progress import LinearProgressWriter

            writer = LinearProgressWriter(destination, metadata, heartbeat_seconds)

            def progress(event):
                writer(event)
                if previous_progress is not None:
                    previous_progress(event)

            solver.progress_callback = progress
        try:
            result, metrics = solve(B, b, indices, initial=initial)
        finally:
            solver.progress_callback = previous_progress
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


def prolonged_initial_state(cfg, problem, baseline_record):
    """A saved optimum on a coarser time grid, repeated per original slab.

    The saved trajectory is piecewise constant in time on its own slabs; each
    slab value is repeated over the finer slabs it covers. Flows are recomputed
    from the isothermal baseline, so this is only an initial iterate. The saved
    problem must be the same declared target, bounds, horizon and startup.
    """
    repeat = integer(cfg.get("initial_control_repeat", 1), "Temporal repetition", 1)
    position = cfg.get("initial_control_position")
    record, saved_cfg, arrays, digest = load_saved_solution(
        cfg["initial_control_directory"], cfg["initial_control_method"], position
    )
    require_matching_baseline(record, baseline_record)
    saved_slabs = integer(saved_cfg.get("slabs", 1), "Saved slabs", 1)
    if saved_cfg.get("transient") != cfg.get("transient"):
        raise ValueError("A saved initial control must share the transient setting")
    if saved_slabs * repeat != integer(cfg["slabs"], "Slabs", 1):
        raise ValueError("The temporal repetition must map the saved slabs onto the declared slabs")
    for key in ("query", "upper_K", "lower_K", "horizon_s", "target_count"):
        if key in saved_cfg and saved_cfg[key] != cfg[key]:
            raise ValueError(f"A saved initial control must share the declared {key}")
    if saved_cfg.get("target_startup_s", 0.0) != cfg.get("target_startup_s", 0.0):
        raise ValueError("A saved initial control must share the declared target startup")
    state = np.asarray(arrays["state"], dtype=float).reshape(saved_slabs, -1)
    prolonged = np.repeat(state, repeat, axis=0).ravel()
    if prolonged.shape != (problem.size,):
        raise ValueError("The prolonged initial control does not match the problem size")
    return prolonged, {
        "directory": str(cfg["initial_control_directory"]),
        "method": cfg["initial_control_method"],
        "position": position,
        "field_sha256": digest,
        "saved_slabs": saved_slabs,
        "temporal_repetition": repeat,
        "scope": "Initial iterate only; flows are recomputed from the isothermal baseline and every optimality check applies to the new optimum.",
    }


def preconditioner_description(cfg):
    """What the inner solver actually divides by, for the run's own record."""
    if cfg.get("inner_preconditioner", "jacobi") == "frozen":
        sweeps = cfg.get("frozen_sweeps", 3)
        return (
            f"{sweeps} block-Jacobi sweeps over the slab-diagonal blocks of the "
            "frozen-velocity thermal normal operator"
        )
    return "positive diagonal of the frozen-velocity thermal normal operator"


def solver_options(cfg, solver_class):
    """Runner-level inner-solver settings that every method shares.

    The residual-refresh interval applies to all methods. Hybrid-only settings
    reach only the hybrid backend, so CPU and CUDA-resident solvers are unchanged.
    """
    options = {"refresh": integer(cfg.get("inner_refresh", 1000), "Residual refresh interval", 1)}
    if solver_class.__name__ == "HybridCoupledSolver":
        options["block_min_columns"] = integer(
            cfg.get("hybrid_block_min_columns", 20), "CUDA block threshold", 2
        )
        options["block_max_columns"] = integer(
            cfg.get("hybrid_block_max_columns", 100), "CUDA block chunk width", 1
        )
        options["coarse_device"] = cfg.get("hybrid_coarse_device", "cpu")
    return options


def run(config):
    cfg = OmegaConf.to_container(config, resolve=True)
    if not cfg["baseline_directory"]:
        raise ValueError("A verified baseline_directory is required")
    if cfg["mode"] not in {"derivatives", "optimize"}:
        raise ValueError("Choose derivatives or optimize")
    if cfg["device"] not in {"cpu", "cuda", "hybrid"}:
        raise ValueError("Choose cpu, cuda or hybrid")
    if cfg.get("inner_preconditioner", "jacobi") not in {"jacobi", "frozen"}:
        raise ValueError("Choose the Jacobi diagonal or the velocity-frozen preconditioner")
    if cfg.get("inner_preconditioner", "jacobi") == "frozen" and cfg["device"] == "cuda":
        raise ValueError("The frozen preconditioner runs on the host kernel, so cpu or hybrid")
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(integer(cfg["threads"], "Threads", 1)):
        start = time.perf_counter()
        problem, baseline_record = load_problem(cfg)
        if cfg["evaluation_progress"]:
            problem.evaluation_callback = lambda row: write_report(
                output / "evaluation-progress.json", row
            )
        desired = desired_temperature(
            problem, cfg["query"], cfg["target_count"], cfg.get("target_startup_s", 0.0)
        )
        metadata = {
            "schema": "coupled-optimization-pilot-v1",
            "numerical_policy": NUMERICAL_POLICY,
            "environment": environment(),
            "configuration": {
                k: v
                for k, v in cfg.items()
                if k not in {"output", "baseline_directory", "reference_baseline_directory"}
            },
            "baseline_sha256": baseline_record["baseline_sha256"],
            "baseline_configuration": baseline_record["configuration"],
            "input_sha256": baseline_record["input_sha256"],
            "state_dofs": problem.size,
            "thermal_offset_K": problem.temperature_offset,
            "thermal_scale_K": problem.temperature_scale,
            "preconditioner": preconditioner_description(cfg),
        }
        write_report(output / "record.json", {**metadata, "status": "running"})
        try:
            if cfg["mode"] == "derivatives":
                report = derivative_report(problem, desired)
                passed = (
                    report["relative_dot_product_error"] <= 1e-9
                    and min(report["taylor_orders"]) > 1.9
                    and np.isfinite(report["taylor_orders"]).all()
                    and equation_acceptance(report["equations"], cfg)
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
            bounds = temperature_bounds(cfg)
            metadata["temperature_bounds"] = bounds
            lower = (
                bounds["optimization_lower_K"] - problem.temperature_offset
            ) / problem.temperature_scale
            upper = (
                bounds["optimization_upper_K"] - problem.temperature_offset
            ) / problem.temperature_scale
            initial_state = None
            if cfg.get("initial_control_directory"):
                initial_state, metadata["initial_control"] = prolonged_initial_state(
                    cfg, problem, baseline_record
                )
                write_report(output / "record.json", {**metadata, "status": "running"})
            results = []
            for method in cfg["methods"]:
                tick = time.perf_counter()
                reference = None
                if method == "reference":
                    reference = configured_reference(problem, cfg, baseline_record)
                setup = time.perf_counter() - tick
                solver_class = StudySolver
                if cfg["device"] == "cuda":
                    from .coupled_cuda_solver import CudaCoupledSolver

                    solver_class = CudaCoupledSolver
                elif cfg["device"] == "hybrid":
                    from .coupled_hybrid_solver import HybridCoupledSolver

                    solver_class = HybridCoupledSolver
                solver = solver_class(
                    method,
                    rank=cfg["rank"],
                    window=cfg["recycle_window"],
                    reference=reference,
                    rtol=cfg["inner_tolerance"],
                    maxiter=cfg["inner_cap"],
                    cg_factor=0.1,
                    residual_policy="refine",
                    **solver_options(cfg, solver_class),
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
                        initial=initial_state,
                        tolerance=cfg["nonlinear_tolerance"],
                        max_iterations=cfg["nonlinear_cap"],
                        backtracking=cfg["backtracking"],
                        secant_memory=cfg["secant_memory"],
                        inner_preconditioner=cfg.get("inner_preconditioner", "jacobi"),
                        frozen_sweeps=cfg.get("frozen_sweeps", 3),
                        qp_tolerance=cfg["qp_tolerance"],
                        qp_cap=cfg["qp_cap"],
                        callback=lambda row, ev: write_report(
                            output / (method + "-progress.json"), row
                        ),
                    )
                    checks = problem.verify(
                        result.evaluation, local_mass=cfg.get("local_mass_diagnostics", False)
                    )
                    adjoint = problem.verify_adjoint(result.evaluation, desired)
                    equations_pass = equation_acceptance(checks, cfg)
                    adjoint_pass = adjoint_acceptance(adjoint)
                    row = {
                        "method": method,
                        "optimizer_status": result.status,
                        "status": result.status
                        if equations_pass and adjoint_pass
                        else "adjoint_verification_failed"
                        if equations_pass
                        else "equation_verification_failed",
                        "reference_seconds": setup,
                        "reference_description": None
                        if reference is None
                        else reference.description,
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


@hydra.main(version_base="1.3", config_path="conf", config_name="coupled_optimize")
def main(config):
    run(config)


if __name__ == "__main__":
    main()
