"""Monolithic Newton verification of a fixed-source transient trajectory.

This separate forward procedure changes neither the saved control nor the
optimization algorithm. Sparse LU solves the indefinite coupled Jacobian.
Each return retains one temperature/velocity pair and its original equations.
"""

import argparse
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_forward import CoupledResult
from .coupled_optimize import load_problem
from .coupled_resolution import forward_model
from .coupled_saved import load_saved_solution, require_matching_baseline
from .coupled_step_spectrum import step_linearization
from .coupled_targets import desired_temperature
from .coupled_time_integration import effective_step
from .reporting import environment, write_fields, write_report
from .validation import integer, positive_real, real_array


def step_equations(problem, model, state, flow, source, previous_state, previous_flow, slab):
    """Evaluate both original equations at the same candidate fields."""
    dt = float(problem.physical_steps[slab])
    acceleration = model._force(state, True)
    load = problem.flow.load(acceleration) + problem.flow.mass @ previous_flow.velocity / dt
    rhs = np.r_[load[:, 0], load[:, 1], np.zeros(problem.flow.np)]
    x = np.r_[flow.velocity[:, 0], flow.velocity[:, 1], flow.pressure]
    momentum = problem.flow.operator(flow.velocity, time_step=dt) @ x - rhs
    assembly, K, thermal_rhs = model._thermal_system(flow.velocity, source, previous_state, dt)
    thermal = K @ state - thermal_rhs
    residual = np.r_[momentum[problem.flow_free], thermal[problem.free]]
    thermal_scale = np.linalg.norm(thermal_rhs[problem.free]) + np.linalg.norm(
        K[problem.free][:, problem.mesh.dirichlet] @ state[problem.mesh.dirichlet]
    )
    metrics = problem.flow.verify(
        flow,
        acceleration,
        problem.boundary_indices,
        problem.boundary_values,
        previous=previous_flow.velocity,
        time_step=dt,
        pressure_gauge=problem.pressure_gauge,
    )
    metrics["thermal_relative_residual"] = float(
        np.linalg.norm(thermal[problem.free]) / max(thermal_scale, 1e-30)
    )
    flux = problem.flow.boundary_flux(flow.velocity)
    metrics["mass_relative_imbalance"] = float(abs(flux.sum()) / max(np.abs(flux).sum() / 2, 1e-30))
    metrics["energy_relative_defect"] = model.energy_balance(
        assembly, state, flow.velocity, source, previous_state, dt
    )["relative_defect"]
    if not np.isfinite(residual).all() or not np.isfinite(list(metrics.values())).all():
        raise ValueError("The coupled equations contain nonfinite values")
    return residual, metrics


def equation_merit(metrics):
    return max(
        metrics[key]
        for key in (
            "momentum_relative_residual",
            "continuity_relative_residual",
            "thermal_relative_residual",
        )
    )


def criteria_met(metrics, tolerance):
    return (
        equation_merit(metrics) <= tolerance
        and metrics["mass_relative_imbalance"] <= 1e-6
        and metrics["energy_relative_defect"] <= 1e-6
    )


def newton_step(
    problem,
    source,
    previous_state,
    previous_flow,
    slab,
    *,
    tolerance=1e-12,
    max_iterations=30,
    initial_state=None,
    initial_flow=None,
    line_search="equation_max",
    backtrack_cap=21,
):
    """Solve one unchanged transient step, starting from past fields by default."""
    slab = integer(slab, "Slab index", 0)
    if not len(problem.physical_steps) or slab >= problem.slabs:
        raise ValueError("Select an existing transient time slab")
    tolerance = positive_real(tolerance, "Coupled tolerance")
    if tolerance > 1e-8:
        raise ValueError("The forward tolerance may only tighten the original target")
    max_iterations = integer(max_iterations, "Newton iteration cap", 0)
    if line_search not in {"equation_max", "fixed_scaled"}:
        raise ValueError("Choose equation_max or fixed_scaled Newton line search")
    backtrack_cap = integer(backtrack_cap, "Backtracking trial cap", 1)
    source = real_array(source, "Fixed source").copy()
    previous_state = real_array(previous_state, "Previous temperature").copy()
    state = real_array(
        previous_state if initial_state is None else initial_state, "Initial temperature"
    ).copy()
    if any(
        a.shape != (len(problem.mesh.nodes),) or not np.isfinite(a).all()
        for a in (source, previous_state, state)
    ):
        raise ValueError("Source and temperatures must match all mesh nodes")
    source.flags.writeable = False
    model = forward_model(problem)
    state[problem.mesh.dirichlet] = problem.thermal_boundary
    initial = previous_flow if initial_flow is None else initial_flow
    flow = FlowResult(initial.velocity.copy(), initial.pressure.copy(), "initial", [])
    flow.velocity[problem.boundary_indices] = problem.boundary_values
    if problem.pressure_gauge is not None:
        flow.pressure[problem.pressure_gauge[0]] = problem.pressure_gauge[1]
    start = time.perf_counter()
    residual, metrics = step_equations(
        problem, model, state, flow, source, previous_state, previous_flow, slab
    )
    history = [{"iteration": 0, **metrics}]
    status = "newton_iteration_cap"
    for iteration in range(max_iterations + 1):
        if criteria_met(metrics, tolerance):
            status = "converged"
            break
        if iteration == max_iterations:
            break
        H = step_linearization(problem, state[problem.free], flow.velocity, slab)[0]
        scaling = 1 / np.maximum(abs(H).max(axis=1).toarray().ravel(), np.finfo(float).tiny)
        try:
            factor = splu((sparse.diags(scaling) @ H).tocsc())
            update = factor.solve(-scaling * residual)
            denominator = max(np.linalg.norm(scaling * residual), np.finfo(float).tiny)
            for correction in range(3):
                defect = residual + H @ update
                linear_residual = np.linalg.norm(scaling * defect) / denominator
                if linear_residual <= 1e-10 or correction == 2:
                    break
                update += factor.solve(-scaling * defect)
        except (RuntimeError, np.linalg.LinAlgError):
            status = "newton_linear_failure"
            break
        if (
            not np.isfinite(update).all()
            or not np.isfinite(linear_residual)
            or linear_residual > 1e-8
        ):
            status = "newton_linear_residual_failed"
            history.append(
                {
                    "iteration": iteration + 1,
                    "status": status,
                    "linear_relative_residual": float(linear_residual),
                    **metrics,
                }
            )
            break
        dx = np.zeros(problem.flow.size)
        dx[problem.flow_free] = update[: len(problem.flow_free)]
        velocity_update = np.column_stack(
            (dx[: problem.flow.nv], dx[problem.flow.nv : 2 * problem.flow.nv])
        )
        trials, retained = [], False
        for backtrack in range(backtrack_cap):
            step = 0.5**backtrack
            candidate = state.copy()
            candidate[problem.free] += step * update[len(problem.flow_free) :]
            candidate_flow = FlowResult(
                flow.velocity + step * velocity_update,
                flow.pressure + step * dx[2 * problem.flow.nv :],
                "newton",
                [],
            )
            candidate_residual, candidate_metrics = step_equations(
                problem,
                model,
                candidate,
                candidate_flow,
                source,
                previous_state,
                previous_flow,
                slab,
            )
            # Hold the row scaling fixed throughout this Newton line search.
            # The independent final equations retain their original criteria.
            scaled_ratio = float(np.linalg.norm(scaling * candidate_residual) / denominator)
            trials.append(
                {"step": step, "fixed_scaled_residual_ratio": scaled_ratio, **candidate_metrics}
            )
            decrease = (
                scaled_ratio < 1 - 1e-4 * step
                if line_search == "fixed_scaled"
                else equation_merit(candidate_metrics) < (1 - 1e-4 * step) * equation_merit(metrics)
            )
            if decrease or criteria_met(candidate_metrics, tolerance):
                state, flow, residual, metrics = (
                    candidate,
                    candidate_flow,
                    candidate_residual,
                    candidate_metrics,
                )
                retained = True
                break
        history.append(
            {
                "iteration": iteration + 1,
                "linear_relative_residual": float(linear_residual),
                "linear_corrections": correction,
                "candidate_retained": retained,
                "line_search": line_search,
                "trials": trials,
                **metrics,
            }
        )
        if not retained:
            status = "newton_line_search_stagnation"
            break
    return CoupledResult(state, flow, status, history, time.perf_counter() - start)


def newton_trajectory(
    problem,
    controls,
    *,
    tolerance=1e-12,
    cap=30,
    callback=None,
    time_scheme="backward_euler",
    restart_interval=None,
    line_search="equation_max",
    backtrack_cap=21,
):
    """Replay a signed source history using the problem's unchanged time grid."""
    values = real_array(controls, "Fixed source history").copy()
    if values.size != problem.size or not np.isfinite(values).all():
        raise ValueError("The source history must match the complete trajectory")
    values = values.reshape(problem.slabs, problem.spatial_size)
    values.flags.writeable = False
    state, flow = problem.full_temperature(problem.initial), problem.initial_flow
    older_state, older_flow = None, None
    if restart_interval is not None:
        restart_interval = integer(restart_interval, "Source interval in time steps", 1)
    states, velocities, pressures, rows = [], [], [], []
    start = time.perf_counter()
    for n, current in enumerate(values):
        source = np.zeros(len(problem.mesh.nodes))
        source[problem.free] = current
        effective, history_state, history_flow, coefficients = effective_step(
            problem,
            n,
            state,
            flow,
            older_state,
            older_flow,
            time_scheme,
            restart=restart_interval is not None and n % restart_interval == 0,
        )
        result = newton_step(
            effective,
            source,
            history_state,
            history_flow,
            n,
            tolerance=tolerance,
            max_iterations=cap,
            initial_state=state,
            initial_flow=flow,
            line_search=line_search,
            backtrack_cap=backtrack_cap,
        )
        states.append(result.state[problem.free].copy())
        velocities.append(result.flow.velocity.copy())
        pressures.append(result.flow.pressure.copy())
        row = {
            "slab_zero_based": n,
            "time_s": float(problem.physical_steps[: n + 1].sum()),
            "status": result.status,
            "seconds": result.seconds,
            "history": result.history,
            "storage_derivative_coefficients_s_inverse": coefficients.tolist(),
        }
        rows.append(row)
        if callback is not None:
            callback(row.copy())
        if result.status != "converged":
            break
        older_state, older_flow = state, flow
        state, flow = result.state, result.flow
    return {
        "status": result.status,
        "steps": rows,
        "states": np.stack(states),
        "velocities": np.stack(velocities),
        "pressures": np.stack(pressures),
        "seconds": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("reference", "jacobi", "recycling"), default="reference"
    )
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--cap", type=int, default=30)
    parser.add_argument("--subdivision", type=int, default=1)
    parser.add_argument(
        "--time-scheme", choices=("backward_euler", "bdf2"), default="backward_euler"
    )
    parser.add_argument(
        "--line-search", choices=("equation_max", "fixed_scaled"), default="equation_max"
    )
    parser.add_argument("--backtrack-cap", type=int, default=21)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subdivision = integer(args.subdivision, "Time subdivision", 1)
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        record, cfg, fields, digest = load_saved_solution(
            args.optimization, args.method, args.target_position
        )
        original_slabs = integer(cfg["slabs"], "Original slabs", 1)
        if not cfg["transient"]:
            raise ValueError("Temporal replay requires a saved transient optimization")
        problem, baseline = load_problem(
            {**cfg, "baseline_directory": str(args.baseline), "slabs": original_slabs * subdivision}
        )
        require_matching_baseline(record, baseline)
        controls = fields["control"].reshape(original_slabs, problem.spatial_size)
        original = fields["state"].reshape(controls.shape)
        starts = np.vstack((problem.initial[None, :], original[:-1]))
        fraction = np.arange(1, subdivision + 1) / subdivision
        expected = (
            starts[:, None, :] + fraction[None, :, None] * (original - starts)[:, None, :]
        ).reshape(problem.slabs, problem.spatial_size)
        expected[subdivision - 1 :: subdivision] = original
        controls = np.repeat(controls, subdivision, axis=0)
        args.output.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema": "coupled-fixed-source-newton-replay-v1",
            "environment": environment(),
            "optimization_field_sha256": digest,
            "baseline_sha256": baseline["baseline_sha256"],
            "tolerance": args.tolerance,
            "newton_cap": args.cap,
            "subdivision": subdivision,
            "forward_slabs": problem.slabs,
            "forward_solver": {
                "procedure": "monolithic_newton",
                "tolerance": args.tolerance,
                "newton_cap": args.cap,
                "linear_internal_target": 1e-10,
                "linear_acceptance_target": 1e-8,
                "linear_correction_cap": 2,
                "linear_residual": "Row-equilibrated original Jacobian equation",
                "mass_tolerance": 1e-6,
                "energy_tolerance": 1e-6,
                "line_search": args.line_search,
                "backtrack_cap": args.backtrack_cap,
                "time_scheme": args.time_scheme,
                "time_integrator_restart": "Backward Euler on the first substep of every original piecewise-constant source interval",
            },
            "target_position": args.target_position,
            "configuration": {
                k: v
                for k, v in cfg.items()
                if k not in {"baseline_directory", "reference_baseline_directory", "output"}
            },
            "control_representation": "Piecewise constant on each original time interval; each saved value is copied unchanged into its subdivisions.",
            "trajectory_comparison": "Forward states versus piecewise-linear interpolation of the optimized states, including the initial condition.",
            "scope": "Forward verification of the fixed saved source, with the declared temporal subdivision; no reoptimization or clipping.",
        }
        write_report(args.output / "record.json", {**metadata, "status": "running", "steps": []})
        state, flow = problem.full_temperature(problem.initial), problem.initial_flow
        older_state, older_flow = None, None
        states, velocities, pressures, rows = [], [], [], []
        start = time.perf_counter()
        for n, values in enumerate(controls):
            source = np.zeros(len(problem.mesh.nodes))
            source[problem.free] = values
            effective, history_state, history_flow, coefficients = effective_step(
                problem,
                n,
                state,
                flow,
                older_state,
                older_flow,
                args.time_scheme,
                restart=n % subdivision == 0,
            )
            result = newton_step(
                effective,
                source,
                history_state,
                history_flow,
                n,
                tolerance=args.tolerance,
                max_iterations=args.cap,
                line_search=args.line_search,
                backtrack_cap=args.backtrack_cap,
                initial_state=state,
                initial_flow=flow,
            )
            states.append(result.state[problem.free].copy())
            velocities.append(result.flow.velocity.copy())
            pressures.append(result.flow.pressure.copy())
            rows.append(
                {
                    "slab_zero_based": n,
                    "time_s": float(problem.physical_steps[: n + 1].sum()),
                    "status": result.status,
                    "seconds": result.seconds,
                    "history": result.history,
                    "storage_derivative_coefficients_s_inverse": coefficients.tolist(),
                    "maximum_temperature_difference_K": float(
                        np.max(np.abs(states[-1] - expected[n])) * problem.temperature_scale
                    ),
                }
            )
            write_report(
                args.output / "record.json", {**metadata, "status": "running", "steps": rows}
            )
            if result.status != "converged":
                break
            older_state, older_flow = state, flow
            state, flow = result.state, result.flow
        elapsed = time.perf_counter() - start
        write_fields(
            args.output / "states.npz",
            state=np.stack(states),
            velocity=np.stack(velocities),
            pressure=np.stack(pressures),
            times_s=np.array([row["time_s"] for row in rows]),
        )
        complete = len(rows) == problem.slabs and all(row["status"] == "converged" for row in rows)
        temperatures = problem.temperature_offset + problem.temperature_scale * np.stack(states)
        resolution = {}
        if complete:
            computed = np.stack(states)
            mass = problem.assembly.mass[problem.free]
            original_steps = problem.physical_steps.reshape(original_slabs, subdivision).sum(axis=1)
            original_tracking = float(
                np.sum(
                    original_steps[:, None]
                    * mass
                    * (
                        problem.temperature_scale
                        * (original - fields["desired"].reshape(original.shape))
                    )
                    ** 2
                )
            )
            desired = desired_temperature(
                problem, cfg["query"], cfg["target_count"], cfg.get("target_startup_s", 0.0)
            ).reshape(computed.shape)
            refined_tracking = float(
                np.sum(
                    problem.physical_steps[:, None]
                    * mass
                    * (problem.temperature_scale * (computed - desired)) ** 2
                )
            )
            tracking_change = abs(refined_tracking - original_tracking) / max(
                original_tracking, np.finfo(float).tiny
            )
            trajectory_error = float(
                np.max(np.abs(computed - expected)) * problem.temperature_scale
            )
            resolution = {
                "maximum_endpoint_difference_K": float(
                    np.max(np.abs(computed[subdivision - 1 :: subdivision] - original))
                    * problem.temperature_scale
                ),
                "maximum_trajectory_difference_K": trajectory_error,
                "tracking_integral_original_K2_m3_s": original_tracking,
                "tracking_integral_refined_K2_m3_s": refined_tracking,
                "tracking_relative_change": tracking_change,
                "resolution_thresholds_met": None
                if subdivision == 1
                else trajectory_error <= 0.05 and tracking_change <= 0.01,
            }
        write_report(
            args.output / "record.json",
            {
                **metadata,
                "status": "converged" if complete else result.status,
                "seconds": elapsed,
                "steps": rows,
                "maximum_recorded_difference_K": max(
                    row["maximum_temperature_difference_K"] for row in rows
                ),
                "maximum_endpoint_difference_K": max(
                    row["maximum_temperature_difference_K"] for row in rows
                )
                if complete
                else None,
                "maximum_recorded_upper_violation_K": float(
                    max(0, temperatures.max() - cfg["upper_K"])
                ),
                "maximum_recorded_lower_violation_K": float(
                    max(0, cfg["lower_K"] - temperatures.min())
                ),
                **resolution,
            },
        )


if __name__ == "__main__":
    main()
