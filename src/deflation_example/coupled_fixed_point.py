"""Safeguarded fixed-point policies with independently evaluated equations.

Forward coupling accelerates temperature only. Momentum acceleration selects
coefficients in velocity coordinates and applies that affine combination to
velocity and kinematic pressure. Neither policy changes the PDE or its Jacobian.
"""

import time
import warnings

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .anderson import Anderson
from .axisymmetric_flow import FlowResult
from .coupled_forward import CoupledResult
from .validation import integer, positive_real


def merit(checks):
    values = [
        checks[k]
        for k in (
            "momentum_relative_residual",
            "continuity_relative_residual",
            "thermal_relative_residual",
        )
        if k in checks
    ]
    return max(values) if values and np.isfinite(values).all() else float("inf")


def solve_forward(
    model,
    control,
    initial_state,
    initial_flow,
    *,
    previous_state=None,
    previous_velocity=None,
    time_step=None,
    feedback=True,
    policy="anderson",
    relaxation=0.5,
    depth=3,
    tolerance=1e-12,
    max_iterations=300,
    flow_cap=100,
    callback=None,
):
    start = time.perf_counter()
    tolerance = positive_real(tolerance, "Coupled tolerance")
    max_iterations = integer(max_iterations, "Coupling cap", 1)
    flow_cap = integer(flow_cap, "Flow cap", 1)
    if policy not in {"relaxed", "anderson"}:
        raise ValueError("Unknown forward fixed-point policy")
    relaxation = positive_real(relaxation, "Relaxation")
    if relaxation > 1:
        raise ValueError("Relaxation must not exceed one")
    state = np.asarray(initial_state, dtype=float).copy()
    source = np.asarray(control, dtype=float).copy()
    n = len(model.flow.mesh.nodes)
    if state.shape != (n,) or source.shape != (n,) or not np.isfinite([state, source]).all():
        raise ValueError("Finite full-node state and source are required")
    source.flags.writeable = False
    state[model.J] = model.boundary
    if (time_step is None) != (previous_state is None) or (time_step is None) != (
        previous_velocity is None
    ):
        raise ValueError("Transient solves require both preceding fields and the time step")
    if time_step is not None:
        time_step = positive_real(time_step, "Time step")
        previous_state = np.asarray(previous_state, dtype=float).copy()
        if previous_state.shape != state.shape or not np.isfinite(previous_state).all():
            raise ValueError("Invalid previous temperature")
    flow = FlowResult(initial_flow.velocity.copy(), initial_flow.pressure.copy(), "initial", [])
    flow.velocity[model.velocity_boundary] = model.velocity_values
    if model.pressure_gauge is not None:
        flow.pressure[model.pressure_gauge[0]] = model.pressure_gauge[1]

    def checks(y, v, system=None):
        assembly, matrix, rhs = system or model._thermal_system(
            v.velocity, source, previous_state, time_step
        )
        scale = np.linalg.norm(rhs[model.I]) + np.linalg.norm(
            matrix[model.I][:, model.J] @ model.boundary
        )
        metrics = model.flow.verify(
            v,
            model._force(y, feedback),
            model.velocity_boundary,
            model.velocity_values,
            previous=previous_velocity,
            time_step=time_step,
            pressure_gauge=model.pressure_gauge,
        )
        flux = model.flow.boundary_flux(v.velocity)
        return {
            **metrics,
            "thermal_relative_residual": float(
                np.linalg.norm((matrix @ y - rhs)[model.I]) / max(scale, 1e-30)
            ),
            "mass_relative_imbalance": float(abs(flux.sum()) / max(np.abs(flux).sum() / 2, 1e-30)),
            "energy": model.energy_balance(
                assembly, y, v.velocity, source, previous_state, time_step
            ),
        }

    def converged(row):
        return (
            merit(row) <= tolerance
            and row["mass_relative_imbalance"] <= 1e-6
            and row["energy"]["relative_defect"] <= 1e-6
        )

    current = checks(state, flow)
    history = [{"coupling_iteration": 0, "procedure": "initial_verification", **current}]
    best = (merit(current), state.copy(), flow, current)
    if converged(current):
        return CoupledResult(state, flow, "converged", history, time.perf_counter() - start)
    mixer = None
    status = "coupling_iteration_cap"
    for iteration in range(1, max_iterations + 1):
        candidate_flow = model.flow.solve(
            model._force(state, feedback),
            model.velocity_boundary,
            model.velocity_values,
            initial=flow,
            previous=previous_velocity,
            time_step=time_step,
            pressure_gauge=model.pressure_gauge,
            tolerance=tolerance * 0.1,
            max_iterations=flow_cap,
            method="newton",
        )
        if candidate_flow.status != "converged":
            history.append(
                {
                    "coupling_iteration": iteration,
                    "flow_status": candidate_flow.status,
                    "flow_history": candidate_flow.history,
                }
            )
            status = "flow_" + candidate_flow.status
            break
        system = model._thermal_system(candidate_flow.velocity, source, previous_state, time_step)
        assembly, matrix, rhs = system
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                mapped = spsolve(
                    matrix[model.I][:, model.I],
                    rhs[model.I] - matrix[model.I][:, model.J] @ model.boundary,
                )
        except (MatrixRankWarning, RuntimeError):
            status = "thermal_linear_failure"
            break
        if not np.isfinite(mapped).all():
            status = "thermal_nonfinite_iterate"
            break
        factor = relaxation if feedback and model.expansion else 1.0
        plain = state.copy()
        plain[model.I] += factor * (mapped - state[model.I])
        proposal, row = plain, {"proposal": "relaxed", "history_rank": 0}
        if policy == "anderson":
            if mixer is None:
                mixer = Anderson(assembly.mass[model.I], depth=depth, damping=factor)
            mixed, row = mixer.propose(state[model.I], mapped)
            proposal = state.copy()
            proposal[model.I] = mixed
        metrics = checks(proposal, candidate_flow, system)
        if row["proposal"] == "anderson" and merit(metrics) > merit(current):
            row["rejected_merit"] = merit(metrics)
            proposal = plain
            metrics = checks(proposal, candidate_flow, system)
            row["proposal"] = "residual_safeguard"
            mixer.reset()
        stagnant = np.array_equal(proposal, state) and np.array_equal(
            candidate_flow.velocity, flow.velocity
        )
        state, flow, current = proposal, candidate_flow, metrics
        if merit(metrics) < best[0]:
            best = (merit(metrics), state.copy(), flow, metrics)
        entry = {
            "coupling_iteration": iteration,
            **row,
            **metrics,
            "flow_status": flow.status,
            "flow_history": flow.history,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(entry)
        if callback is not None:
            callback(entry)
        if converged(metrics):
            return CoupledResult(state, flow, "converged", history, time.perf_counter() - start)
        if stagnant:
            status = "coupling_stagnation"
            break
    # Preserve state/residual correspondence even after a deteriorating step.
    history.append({"procedure": "returned_state_verification", **checks(best[1], best[2])})
    return CoupledResult(best[1], best[2], status, history, time.perf_counter() - start)


def solve_picard_momentum(
    flow,
    acceleration,
    boundary_indices,
    boundary_values,
    *,
    initial,
    previous=None,
    time_step=None,
    pressure_gauge=None,
    tolerance=1e-12,
    max_iterations=100,
    method="anderson",
    relaxation=0.5,
    depth=3,
):
    """Verified Oseen fixed points; all derivative factors remain exact Jacobians."""
    tolerance = positive_real(tolerance, "Momentum tolerance")
    max_iterations = integer(max_iterations, "Momentum cap", 1)
    if method not in {"picard", "anderson"}:
        raise ValueError("Unknown fixed-point momentum policy")
    weights = np.r_[flow.mass.diagonal(), flow.mass.diagonal(), np.zeros(flow.np)]
    mixer = Anderson(weights, depth, relaxation)
    args = dict(previous=previous, time_step=time_step, pressure_gauge=pressure_gauge)

    def pack(v):
        return np.r_[v.velocity[:, 0], v.velocity[:, 1], v.pressure]

    def unpack(x):
        result = FlowResult(
            np.column_stack((x[: flow.nv], x[flow.nv : 2 * flow.nv])),
            x[2 * flow.nv :].copy(),
            "iterate",
            [],
        )
        result.velocity[boundary_indices] = boundary_values
        if pressure_gauge is not None:
            result.pressure[pressure_gauge[0]] = pressure_gauge[1]
        return result

    def check(v):
        return flow.verify(v, acceleration, boundary_indices, boundary_values, **args)

    current = unpack(pack(initial))
    checks = check(current)
    history = [{"iteration": 0, **checks}]
    best = (merit(checks), current)
    if merit(checks) <= tolerance:
        current.status, current.history = "converged", history
        return current
    status = "iteration_cap"
    for k in range(1, max_iterations + 1):
        mapped = flow.solve(
            acceleration,
            boundary_indices,
            boundary_values,
            initial=current,
            tolerance=tolerance,
            max_iterations=1,
            method="picard",
            **args,
        )
        if mapped.status not in {"converged", "iteration_cap"}:
            history.append(
                {"iteration": k, "map_status": mapped.status, "map_history": mapped.history}
            )
            status = mapped.status
            break
        x, g = pack(current), pack(mapped)
        plain = x + relaxation * (g - x)
        trial, info = (
            mixer.propose(x, g) if method == "anderson" else (plain, {"proposal": "relaxed"})
        )
        candidate = unpack(trial)
        verified = check(candidate)
        if info["proposal"] == "anderson" and merit(verified) > merit(checks):
            info["rejected_merit"] = merit(verified)
            candidate, info["proposal"] = unpack(plain), "residual_safeguard"
            verified = check(candidate)
            mixer.reset()
        stagnant = np.array_equal(pack(candidate), x)
        current, checks = candidate, verified
        if merit(checks) < best[0]:
            best = (merit(checks), current)
        history.append({"iteration": k, **info, **checks, "map_history": mapped.history})
        if merit(checks) <= tolerance:
            current.status, current.history = "converged", history
            return current
        if stagnant:
            status = "stagnation"
            break
    result = best[1]
    history.append({"procedure": "returned_state_verification", **check(result)})
    result.status, result.history = status, history
    return result
