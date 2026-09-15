"""Backward differentiation for an independently replayed signed control.

The second-order formula is written as an effective backward-Euler step.
Momentum and thermal storage use the same coefficients. Pressure has no time
derivative. The original problem's physical time grid is never modified.
"""

from copy import copy

import numpy as np

from .axisymmetric_flow import FlowResult
from .validation import integer, real_array


def replay_time_scheme(record):
    """Identify the supported saved scheme without silently changing its restart policy."""
    protocol = record.get("forward_solver", {})
    scheme = protocol.get("time_scheme", "backward_euler")
    if scheme not in {"backward_euler", "bdf2"}:
        raise ValueError("Saved replay uses an unsupported time scheme")
    if scheme == "bdf2" and protocol.get("time_integrator_restart") != (
        "Backward Euler on the first substep of every original piecewise-constant source interval"
    ):
        raise ValueError("Saved BDF2 replay must identify its source-interval restarts")
    return scheme


def derivative_coefficients(steps, slab, scheme="backward_euler", *, restart=False):
    steps = real_array(steps, "Physical time steps")
    slab = integer(slab, "Slab index", 0)
    if steps.ndim != 1 or slab >= len(steps) or not np.isfinite(steps).all() or np.any(steps <= 0):
        raise ValueError("Select an existing positive physical time step")
    if scheme not in {"backward_euler", "bdf2"}:
        raise ValueError("Choose backward_euler or bdf2 time integration")
    if not isinstance(restart, bool):
        raise ValueError("The time-integrator restart flag must be Boolean")
    if scheme == "backward_euler" or slab == 0 or restart:
        return np.array([1.0, -1.0, 0.0]) / steps[slab]
    ratio = steps[slab] / steps[slab - 1]
    if ratio >= 1 + np.sqrt(2):
        raise ValueError(
            "This BDF2 implementation requires successive step ratios below 1 + sqrt(2)"
        )
    return (
        np.array([(1 + 2 * ratio) / (1 + ratio), -(1 + ratio), ratio**2 / (1 + ratio)])
        / steps[slab]
    )


def effective_step(problem, slab, state, flow, older_state, older_flow, scheme, *, restart=False):
    coefficients = derivative_coefficients(problem.physical_steps, slab, scheme, restart=restart)
    if coefficients[2] == 0:
        return problem, state, flow, coefficients
    if older_state is None or older_flow is None:
        raise ValueError("A BDF2 step requires both preceding temperature and velocity fields")
    a0, a1, a2 = coefficients
    # A private view changes only the current storage coefficients; all shared
    # spatial operators are read-only during the forward Newton solve.
    effective = copy(problem)
    effective.physical_steps = problem.physical_steps.copy()
    effective.physical_steps[slab] = 1 / a0
    effective.steps = effective.physical_steps / problem.time_scale
    history = -(a1 * state + a2 * older_state) / a0
    history_flow = FlowResult(
        -(a1 * flow.velocity + a2 * older_flow.velocity) / a0,
        flow.pressure.copy(),
        "bdf2_history",
        [],
    )
    return effective, history, history_flow, coefficients


def saved_history(problem, states, velocities, pressures, slab, scheme, *, restart=False):
    """Recover both storage histories for a diagnostic at an unchanged saved step."""
    slab = integer(slab, "Saved slab", 0)
    states = real_array(states, "Saved temperatures")
    velocities = real_array(velocities, "Saved velocities")
    pressures = real_array(pressures, "Saved pressures")
    count = len(states)
    if (
        states.shape != (count, problem.spatial_size)
        or velocities.shape != (count, problem.flow.nv, 2)
        or pressures.shape != (count, problem.flow.np)
        or slab >= count
        or count > problem.slabs
        or not all(np.isfinite(a).all() for a in (states, velocities, pressures))
    ):
        raise ValueError("Saved temperature and flow histories must match the declared trajectory")

    def preceding(index):
        if index < 0:
            return problem.full_temperature(problem.initial), problem.initial_flow
        return problem.full_temperature(states[index]), FlowResult(
            velocities[index].copy(), pressures[index].copy(), "saved", []
        )

    previous_state, previous_flow = preceding(slab - 1)
    older_state, older_flow = preceding(slab - 2)
    return effective_step(
        problem,
        slab,
        previous_state,
        previous_flow,
        older_state,
        older_flow,
        scheme,
        restart=restart,
    )
