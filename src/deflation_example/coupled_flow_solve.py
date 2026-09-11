"""Residual-load continuation for the unchanged coupled momentum equations."""

import numpy as np
from scipy.sparse.linalg import splu

from .axisymmetric_flow import FlowResult
from .validation import integer, positive_real


def solve_momentum(
    flow,
    acceleration,
    boundary_indices,
    boundary_values,
    *,
    initial,
    previous=None,
    time_step=None,
    pressure_gauge=None,
    tolerance=1e-9,
    max_iterations=100,
    continuation=False,
    max_stages=80,
):
    """Try Newton, then optionally continue an artificial residual load to zero.

    A P2 acceleration represents the momentum residual of the initial flow.
    At continuation fraction zero this load makes the initial velocity and
    pressure an equilibrium. At fraction one the load vanishes. Only the
    original target equations can accept a final result. Every failed stage
    remains in the returned history. Viscosity and boundary data stay fixed.
    """
    tolerance = positive_real(tolerance, "Flow tolerance")
    max_stages = integer(max_stages, "Load continuation stage cap", 1)
    arguments = dict(
        previous=previous,
        time_step=time_step,
        pressure_gauge=pressure_gauge,
        tolerance=tolerance,
        max_iterations=max_iterations,
        method="newton",
    )
    direct = flow.solve(
        acceleration, boundary_indices, boundary_values, initial=initial, **arguments
    )
    if direct.status == "converged" or not continuation:
        return direct
    original = np.r_[initial.velocity[:, 0], initial.velocity[:, 1], initial.pressure]
    force = flow.load(acceleration)
    if time_step is not None:
        force += flow.mass @ previous / time_step
    rhs = np.r_[force[:, 0], force[:, 1], np.zeros(flow.np)]
    residual = flow.operator(initial.velocity, time_step=time_step) @ original - rhs
    coefficients = splu(flow.mass.tocsc()).solve(
        np.column_stack((residual[: flow.nv], residual[flow.nv : 2 * flow.nv]))
    )
    lift = flow.sampled_velocity(coefficients)
    base = flow.verify(
        initial,
        acceleration + lift,
        boundary_indices,
        boundary_values,
        previous=previous,
        time_step=time_step,
        pressure_gauge=pressure_gauge,
    )
    history = [
        {"load_fraction": 1.0, "status": direct.status, "history": direct.history},
        {"load_fraction": 0.0, "status": "initial_load_check", "metrics": base},
    ]
    if not np.isfinite(list(base.values())).all() or max(base.values()) > tolerance:
        return FlowResult(
            initial.velocity.copy(), initial.pressure.copy(), "load_initialization_failed", history
        )
    current, fraction, increment = initial, 0.0, 0.25
    status = "load_continuation_cap"
    for _ in range(max_stages):
        target = min(1.0, fraction + increment)
        candidate = flow.solve(
            acceleration + (1 - target) * lift,
            boundary_indices,
            boundary_values,
            initial=current,
            **arguments,
        )
        history.append(
            {"load_fraction": target, "status": candidate.status, "history": candidate.history}
        )
        if candidate.status == "converged":
            current, fraction = candidate, target
            if target == 1.0:
                checks = flow.verify(
                    current,
                    acceleration,
                    boundary_indices,
                    boundary_values,
                    previous=previous,
                    time_step=time_step,
                    pressure_gauge=pressure_gauge,
                )
                status = (
                    "converged" if max(checks.values()) <= tolerance else "final_residual_failed"
                )
                break
            increment = min(0.25, increment * 1.5)
        else:
            increment *= 0.5
            if increment < 1e-5:
                status = "load_continuation_stagnation"
                break
    return FlowResult(current.velocity.copy(), current.pressure.copy(), status, history)
