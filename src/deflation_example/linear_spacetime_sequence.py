"""Complete fixed-flow trajectory optimization matched to the nonlinear study.

The thermal equations retain consistent source and storage operators. The
isothermal computed velocity stays fixed at every time level; each target is
one coupled-in-time quadratic optimization, solved directly by PDAS.
"""

from dataclasses import replace
import hashlib
import time

import numpy as np
from scipy.sparse.linalg import splu

from .axisymmetric_flow import FlowResult
from .coupled_bounds import temperature_bounds
from .coupled_optimize import adjoint_acceptance, equation_acceptance, load_problem
from .coupled_targets import desired_temperature
from .linear_spacetime import FixedFlowProblem, solve_trajectory
from .validation import integer, positive_real

NUMERICAL_POLICY = "fixed-flow-exact-affine-spacetime-pdas-v1"


class IsothermalRefinementError(ValueError):
    """Preserve the actual residuals and field change when preparation fails."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("Isothermal refinement fails accuracy or the fixed-baseline change limit")


def isothermal_newton_correction(problem, result):
    """Accumulate the small residual accurately; keep the solve in float64.

    This corrects the already verified steady isothermal baseline. The final
    test still uses the original float64 time-discrete residual evaluator.
    """
    flow = problem.flow
    x = np.r_[result.velocity[:, 0], result.velocity[:, 1], result.pressure]
    operator = flow.operator(result.velocity)
    force = flow.load(problem.acceleration)
    rhs = np.r_[force[:, 0], force[:, 1], np.zeros(flow.np)]
    defect = operator.astype(np.longdouble) @ x.astype(np.longdouble) - rhs.astype(np.longdouble)
    free = problem.flow_free
    jacobian = (operator + flow.convection_derivative(result.velocity))[free][:, free].tocsc()
    correction_rhs = -np.asarray(defect[free], dtype=np.float64)
    step = splu(jacobian).solve(correction_rhs)
    if not np.isfinite(step).all():
        raise RuntimeError("The isothermal correction is nonfinite")
    x[free] += step
    residual = jacobian @ step - correction_rhs
    norm = np.linalg.norm(correction_rhs)
    history = [
        {
            "residual_accumulation_epsilon": float(np.finfo(np.longdouble).eps),
            "linear_relative_residual": float(
                np.linalg.norm(residual) / norm if norm else np.linalg.norm(residual)
            ),
            "newton_step_norm": float(np.linalg.norm(step)),
            "linear_solve_dtype": "float64",
        }
    ]
    return FlowResult(
        np.column_stack((x[: flow.nv], x[flow.nv : 2 * flow.nv])),
        x[2 * flow.nv :].copy(),
        "newton_correction",
        history,
    )


def refine_isothermal(problem, cfg):
    """Verify the saved flow and, if needed, refine the same steady equations.

    This once-per-sequence work is charged as model preparation. Refinement
    never uses a desired temperature, optimized control or buoyancy feedback.
    """
    tolerance = positive_real(cfg["equation_acceptance_tolerance"], "Equation tolerance")
    cap = integer(cfg["flow_cap"], "Isothermal correction cap", 1)
    original = problem.initial_flow
    flow = problem.flow
    start = time.perf_counter()

    def verify(result, dt=None):
        return flow.verify(
            result,
            problem.acceleration,
            problem.boundary_indices,
            problem.boundary_values,
            pressure_gauge=problem.pressure_gauge,
            previous=result.velocity if dt is not None else None,
            time_step=dt,
        )

    # The constant velocity cancels transient storage analytically. Evaluate
    # the actual time-discrete equations and their unchanged normalization.
    steps = np.unique(problem.physical_steps).tolist() if len(problem.physical_steps) else [None]

    def trajectory_checks(result):
        rows = [verify(result, dt) for dt in steps]
        return {key: max(row[key] for row in rows) for key in rows[0]}

    def score(checks):
        values = np.asarray(list(checks.values()))
        return float(values.max()) if np.isfinite(values).all() and np.all(values >= 0) else np.inf

    steady_before = verify(original)
    before = trajectory_checks(original)
    refined, after, history = original, before, []
    status = "correction_cap"
    for attempt in range(cap):
        if score(after) <= tolerance:
            status = "verified"
            break
        if not np.isfinite(score(after)):
            status = "invalid_residual"
            break
        candidate = isothermal_newton_correction(problem, refined)
        candidate_checks = trajectory_checks(candidate)
        retained = candidate.status == "newton_correction" and score(candidate_checks) < score(
            after
        )
        history.append(
            {
                "attempt": attempt,
                "newton_step_budget": 1,
                "kernel_status": candidate.status,
                "newton_history": candidate.history,
                "trajectory_checks": candidate_checks,
                "candidate_retained": retained,
            }
        )
        if not retained:
            status = "stagnation_or_kernel_failure"
            break
        refined, after = candidate, candidate_checks
    after = trajectory_checks(refined)
    if score(after) <= tolerance:
        status = "verified"
    steady_after = verify(refined)
    difference = np.linalg.norm(refined.velocity - original.velocity)
    norm = np.linalg.norm(original.velocity)
    relative = difference / norm if norm else difference
    preparation = {
        "policy": "Newton corrections of the same steady isothermal equations; independently verify the actual time-discrete residual and retain improving candidates",
        "correction_residual_accumulation": "numpy.longdouble; correction factors and solves use float64",
        "final_residual_accumulation": "Unchanged float64 time-discrete evaluator",
        "final_equation_tolerance": tolerance,
        "residual_time_steps_s": steps,
        "steady_before": steady_before,
        "steady_after": steady_after,
        "steady_baseline_tolerance": 1e-8,
        "maximum_relative_velocity_change": 1e-6,
        "before": before,
        "after": after,
        "velocity_relative_change": float(relative),
        "velocity_maximum_absolute_change_m_s": float(
            np.max(np.abs(refined.velocity - original.velocity))
        ),
        "pressure_maximum_absolute_change_m2_s2": float(
            np.max(np.abs(refined.pressure - original.pressure))
        ),
        "history": history,
        "status": status,
        "velocity_sha256": hashlib.sha256(
            np.ascontiguousarray(refined.velocity).tobytes()
        ).hexdigest(),
        "seconds": time.perf_counter() - start,
    }
    if (
        status != "verified"
        or score(after) > tolerance
        or score(steady_after) > 1e-8
        or not np.isfinite(relative)
        or relative > 1e-6
    ):
        raise IsothermalRefinementError(preparation)
    problem.initial_flow = refined
    problem.assembly = problem.assemble(refined.velocity)
    if not problem.consistent_stabilization:
        problem.reference_assembly = problem.assembly
    elif problem.reference_stabilization == "shipped":
        problem.reference_assembly = replace(
            problem.assembly, stabilized_storage=None, stabilized_source=None
        )
    else:
        problem.reference_assembly = problem.assemble(
            refined.velocity, consistent=False, bound_streamline=False
        )
    return preparation


def load_fixed_flow(cfg):
    if cfg.get("physics") != "prescribed_flow" or cfg.get("feedback_multiplier") != 0:
        raise ValueError("Declare fixed isothermal flow with zero thermal feedback")
    problem, baseline = load_problem(cfg)
    preparation = refine_isothermal(problem, cfg)
    fixed = FixedFlowProblem(problem)
    fixed.fixed_flow_preparation = preparation
    return fixed, baseline


def optimize_targets(
    problem,
    solver,
    cfg,
    callback=None,
    *,
    positions=None,
    previous=None,
    checkpoint=None,
    resume=None,
    initial_state=None,
    initial_guess=None,
    initial_guess_kind="snapshot",
    observer=None,
):
    """Keep all declared outcomes and warm-start only from verified solutions."""
    if any(value is not None for value in (positions, previous, checkpoint, resume, observer)):
        raise ValueError("The matched linear control uses fresh complete sequences")
    if initial_state is not None or initial_guess_kind != "snapshot":
        raise ValueError("Use only the common initial temperature snapshot")
    cases, fields = [], []
    solver.reset_history()
    for position, query in enumerate(cfg["queries"]):
        start = time.perf_counter()
        desired = desired_temperature(
            problem, query["target"], cfg["target_count"], cfg.get("target_startup_s", 0.0)
        )
        bounds = temperature_bounds(cfg, upper_K=query["upper_K"])
        lower = (
            bounds["optimization_lower_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        upper = (
            bounds["optimization_upper_K"] - problem.temperature_offset
        ) / problem.temperature_scale
        initial = (
            previous.state
            if previous is not None
            else initial_guess.state
            if position == 0 and initial_guess is not None
            else np.clip(np.zeros(problem.size), lower, upper)
        )
        row = {
            "position": position,
            **query,
            "temperature_bounds": bounds,
            "warm_start_used": previous is not None,
            "initial_state_snapshot_used": position == 0 and initial_guess is not None,
        }
        try:
            final, result, objective, kkt = solve_trajectory(
                problem, desired, lower, upper, solver, cfg, initial
            )
            checks = problem.verify(final, local_mass=cfg.get("local_mass_diagnostics", False))
            adjoint = problem.verify_adjoint(final, desired)
            equations_pass = equation_acceptance(checks, cfg)
            adjoint_pass = adjoint_acceptance(adjoint)
            kkt_pass = (
                bool(np.isfinite(list(kkt.values())).all())
                and max(kkt.values()) <= cfg["nonlinear_tolerance"]
            )
            status = (
                "equation_verification_failed"
                if not equations_pass
                else "adjoint_verification_failed"
                if not adjoint_pass
                else "kkt_verification_failed"
                if result.status == "converged" and not kkt_pass
                else result.status
            )
            row.update(
                status=status,
                verified=status == "converged" and kkt_pass,
                kkt=kkt,
                equations=checks,
                adjoint=adjoint,
                objective=objective * problem.objective_scale,
                pdas_steps=len(result.history),
                pdas_history=result.history,
                inner_iterations=sum(step.get("linear_iterations", 0) for step in result.history),
            )
            fields.append(
                {
                    "state": final.state.copy(),
                    "control": final.control.copy(),
                    "desired": desired.copy(),
                    "velocity": np.stack([f.velocity for f in final.flows]),
                    "pressure": np.stack([f.pressure for f in final.flows]),
                }
            )
            previous = final if cfg["warm_start"] and row["verified"] else None
            if callback is not None:
                callback(
                    position,
                    {"status": status, "pdas_steps": len(result.history), "kkt": kkt},
                    final,
                )
        except (MemoryError, RuntimeError, np.linalg.LinAlgError) as failure:
            row.update(
                status="memory_limited" if isinstance(failure, MemoryError) else "numerical_error",
                verified=False,
                error_type=type(failure).__name__,
            )
            row["seconds"] = time.perf_counter() - start
            cases.append(row)
            fields.append({"state": np.empty(0)})
            for next_position in range(position + 1, len(cfg["queries"])):
                cases.append(
                    {
                        "position": next_position,
                        **cfg["queries"][next_position],
                        "status": "not_run_after_numerical_error",
                        "verified": False,
                        "warm_start_used": False,
                        "seconds": 0.0,
                    }
                )
                fields.append({"state": np.empty(0)})
            break
        if not row["verified"]:
            solver.reset_history()
        row["seconds"] = time.perf_counter() - start
        cases.append(row)
    return cases, fields


def run(config):
    from .coupled_sequence import run as run_sequence

    if any(
        config.get(key)
        for key in (
            "stage",
            "capture_linear_systems",
            "initial_control_directory",
            "initial_trajectory_directory",
        )
    ):
        raise ValueError("The linear comparison requires a fresh complete sequence")
    return run_sequence(
        config,
        problem_loader=load_fixed_flow,
        target_optimizer=optimize_targets,
        numerical_policy=NUMERICAL_POLICY,
    )
