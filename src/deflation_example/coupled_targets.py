"""Desired trajectories with a startup defined in physical seconds."""

from types import SimpleNamespace

import numpy as np

from .mesh_showcases import desired_temperature as original_desired_temperature
from .validation import finite_real, integer, real_array


def smooth_startup(times_s, duration_s):
    """Cubic smoothstep, with zero endpoint slopes and exact unit weight thereafter."""
    duration = finite_real(duration_s, "Target startup duration")
    times = real_array(times_s, "Physical target times")
    if duration < 0 or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("Target times and startup duration must be finite and nonnegative")
    if duration == 0:
        return np.ones_like(times)
    fraction = np.clip(times / duration, 0, 1)
    return fraction**2 * (3 - 2 * fraction)


def desired_temperature(problem, query, count, startup_s=0.0):
    """Blend the initial field into the original target during the startup interval.

    Temperatures use the problem's nondimensional units; startup time uses
    physical seconds. Zero startup preserves earlier targets exactly. A steady
    calculation requires zero startup because it has no initial-time interval.
    """
    duration = finite_real(startup_s, "Target startup duration")
    if duration < 0:
        raise ValueError("Target startup duration must be nonnegative")
    original = original_desired_temperature(problem, query, count)
    if duration == 0:
        return original
    steps = real_array(problem.physical_steps, "Physical time steps")
    if steps.ndim != 1 or not len(steps) or not np.isfinite(steps).all() or np.any(steps <= 0):
        raise ValueError("A positive startup requires positive physical time steps")
    initial = real_array(problem.initial, "Initial temperature")
    if initial.shape != (len(problem.free),) or not np.isfinite(initial).all():
        raise ValueError("Initial temperature must match the spatial unknowns")
    trajectory = original.reshape(len(steps), len(problem.free)).copy()
    times = np.cumsum(steps)
    early = times < duration
    weights = smooth_startup(times[early], duration)[:, None]
    trajectory[early] = initial + weights * (trajectory[early] - initial)
    return trajectory.ravel()


def refined_desired_temperature(problem, query, count, subdivision, startup_s=0.0):
    """Evaluate the same physical target at subdivided forward-replay time levels."""
    subdivision = integer(subdivision, "Time subdivision", 1)
    steps = np.repeat(problem.physical_steps / subdivision, subdivision)
    view = SimpleNamespace(
        assembly=problem.assembly,
        free=problem.free,
        steps=steps / problem.time_scale,
        physical_steps=steps,
        initial=problem.initial,
    )
    return desired_temperature(view, query, count, startup_s)
