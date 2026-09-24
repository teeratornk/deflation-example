"""Complete fixed-flow trajectory optimization matched to the nonlinear study.

The thermal equations retain consistent source and storage operators. The
isothermal computed velocity stays fixed at every time level; each target is
one coupled-in-time quadratic optimization, solved directly by PDAS.
"""

import time

import numpy as np

from .coupled_bounds import temperature_bounds
from .coupled_optimize import adjoint_acceptance, equation_acceptance, load_problem
from .coupled_targets import desired_temperature
from .linear_spacetime import FixedFlowProblem, solve_trajectory

NUMERICAL_POLICY = "fixed-flow-exact-affine-spacetime-pdas-v1"


def load_fixed_flow(cfg):
    if cfg.get("physics") != "prescribed_flow" or cfg.get("feedback_multiplier") != 0:
        raise ValueError("Declare fixed isothermal flow with zero thermal feedback")
    problem, baseline = load_problem(cfg)
    return FixedFlowProblem(problem), baseline


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
