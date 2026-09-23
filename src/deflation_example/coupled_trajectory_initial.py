"""Temporal interpolation of a saved optimum for fresh, finer-grid optimization."""

from pathlib import Path

import numpy as np

from .coupled_bounds import temperature_bounds
from .coupled_saved import load_saved_solution, require_matching_baseline, file_digest
from .validation import integer, positive_real


def interpolate_trajectory(values, initial, times, horizon):
    """Linear physical-time interpolation, including the fixed initial condition."""
    values, initial, times = map(np.asarray, (values, initial, times))
    horizon = positive_real(horizon, "Physical horizon")
    if (
        values.ndim < 2
        or len(values) < 1
        or values.shape[1:] != initial.shape
        or times.ndim != 1
        or not times.size
        or not all(np.isfinite(a).all() for a in (values, initial, times))
        or np.any(np.diff(times) <= 0)
        or times[0] <= 0
        or times[-1] > horizon + 1e-10
    ):
        raise ValueError("Finite trajectories must share the initial shape and physical horizon")
    knots = np.linspace(0, horizon, len(values) + 1)
    right = np.searchsorted(knots, np.minimum(times, horizon), side="left")
    right = np.clip(right, 1, len(values))
    fraction = (times - knots[right - 1]) / (knots[right] - knots[right - 1])
    extended = np.concatenate((initial[None], values))
    shape = (len(times),) + (1,) * initial.ndim
    return extended[right - 1] + fraction.reshape(shape) * (extended[right] - extended[right - 1])


def trajectory_initial_guess(cfg, problem, baseline, position):
    """Reuse guesses only; all flow equations, controls and gradients are reevaluated."""
    from .coupled_sequence import RestoredEvaluation

    directory = Path(cfg["initial_trajectory_directory"])
    method = cfg.get("initial_trajectory_method", "reference")
    source_position = integer(cfg.get("initial_trajectory_position", 0), "Source position", 0)
    record, original, fields, digest = load_saved_solution(directory, method, source_position)
    require_matching_baseline(record, baseline)
    if not original["transient"] or not cfg["transient"]:
        raise ValueError("Trajectory initialization requires transient optimization")
    for key in ("horizon_s", "alpha", "target_count", "lower_K"):
        if original[key] != cfg[key]:
            raise ValueError(f"Initial trajectory differs in {key}")
    for key, default in (
        ("target_startup_s", 0.0),
        ("temperature_margin_K", 0.0),
        ("transport_form", "advective"),
        ("consistent_stabilization", False),
        ("streamline_rule", "hard_min"),
    ):
        if original.get(key, default) != cfg.get(key, default):
            raise ValueError(f"Initial trajectory differs in {key}")
    query = cfg["queries"][position]
    if original["query"] != query["target"] or original["upper_K"] != query["upper_K"]:
        raise ValueError("Initial trajectory must use the same target and physical bound")
    count = integer(original["slabs"], "Source slabs", 1)
    if problem.slabs < count or problem.slabs % count or cfg["slabs"] != problem.slabs:
        raise ValueError("Initialize a nested temporal grid with the declared slab count")
    times = np.cumsum(problem.physical_steps)
    if not np.allclose(
        times, np.linspace(0, cfg["horizon_s"], problem.slabs + 1)[1:], rtol=0, atol=1e-10
    ):
        raise ValueError("Declared initialization currently requires uniform physical time steps")
    filename = directory / f"target-{source_position:02d}.npz"
    with np.load(filename, allow_pickle=False) as data:
        velocity, pressure = data["velocity"].copy(), data["pressure"].copy()
    if file_digest(filename) != digest:
        raise ValueError("Initial-trajectory fields changed during loading")
    if (
        fields["state"].shape != (count * problem.spatial_size,)
        or velocity.shape != (count, problem.flow.nv, 2)
        or pressure.shape != (count, problem.flow.np)
    ):
        raise ValueError("Initial trajectory dimensions differ from the spatial model")
    state = interpolate_trajectory(
        fields["state"].reshape(count, -1), problem.initial, times, cfg["horizon_s"]
    ).ravel()
    velocity = interpolate_trajectory(
        velocity, problem.initial_flow.velocity, times, cfg["horizon_s"]
    )
    pressure = interpolate_trajectory(
        pressure, problem.initial_flow.pressure, times, cfg["horizon_s"]
    )
    bounds = temperature_bounds(cfg, upper_K=query["upper_K"])
    lower = (
        bounds["optimization_lower_K"] - problem.temperature_offset
    ) / problem.temperature_scale
    upper = (
        bounds["optimization_upper_K"] - problem.temperature_offset
    ) / problem.temperature_scale
    if np.any(state < lower) or np.any(state > upper):
        raise ValueError("Interpolated initial trajectory violates bounds; no clipping is applied")
    return RestoredEvaluation(state, velocity, pressure), {
        "policy": "linear_physical_time_interpolation_for_fresh_optimization",
        "source_record_sha256": file_digest(directory / "record.json"),
        "source_fields_sha256": digest,
        "baseline_sha256": baseline["baseline_sha256"],
        "source_slabs": count,
        "new_slabs": problem.slabs,
        "initial_condition_included": True,
        "retained_secant_pairs": 0,
        "retained_recycling_directions": 0,
        "scope": "Temperature and flow guesses only. Every time level is reevaluated; controls and gradients are recovered on the new grid. Previous optimization cost is excluded. No clipping or old solver history is used.",
    }
