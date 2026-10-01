"""Time and forcing-history diagnostics for captured temperature trajectories.

Temperature is prescribed in these calculations. The routines solve momentum
and continuity; they do not assess fixed source controls or optimize temperature.
"""

import time

import numpy as np

from .axisymmetric_flow import FlowResult
from .coupled_flow_solve import solve_momentum
from .coupled_interval_refinement import interval_response
from .validation import integer, positive_real


def verified(result, checks, tolerance):
    return bool(
        result.status == "converged"
        and np.isfinite(list(checks.values())).all()
        and max(checks.values()) <= tolerance
    )


def volume_rms(flow, velocity):
    """Axisymmetric volume-weighted root-mean-square velocity magnitude."""
    return float(np.sqrt(max(0.0, np.sum(velocity * (flow.mass @ velocity))) / flow.measure.sum()))


def validate_fields(problem, state, base):
    expected = {
        "state": (problem.size,),
        "velocity": (problem.slabs, problem.flow.nv, 2),
        "pressure": (problem.slabs, problem.flow.np),
    }
    if len(problem.physical_steps) != problem.slabs:
        raise ValueError("A complete transient problem is required")
    state = np.asarray(state, dtype=float)
    if state.shape != expected["state"] or not np.isfinite(state).all():
        raise ValueError("Temperature must cover every state degree of freedom")
    for name, shape in expected.items():
        if name not in base or base[name].shape != shape or not np.isfinite(base[name]).all():
            raise ValueError("Captured base fields have invalid dimensions or values")
    return state.reshape(problem.slabs, problem.spatial_size)


def acceleration(problem, state):
    temperature = problem.temperature_offset + problem.temperature_scale * problem.full_temperature(
        state
    )
    return problem.acceleration + problem.flow.buoyancy(
        temperature, problem.buoyancy_reference, problem.expansion
    )


def integrate(
    problem,
    state,
    base,
    subdivision=1,
    *,
    start_slab=0,
    stop_slab=None,
    previous=None,
    budget_seconds=3600,
    callback=None,
):
    """Integrate selected original intervals, propagating every computed substep.

    Each original interval starts Newton from its saved retained endpoint flow.
    Later substeps use the preceding verified result. The temperature path is
    piecewise linear, including the physical initial temperature. Only a complete
    verified prefix can supply the predecessor of a subsequent interval.
    """
    subdivision = integer(subdivision, "Subdivision", 1)
    budget_seconds = positive_real(budget_seconds, "Momentum integration budget")
    start_slab = integer(start_slab, "First slab", 0)
    stop_slab = problem.slabs if stop_slab is None else integer(stop_slab, "Stop slab", 1)
    if not 0 <= start_slab < stop_slab <= problem.slabs:
        raise ValueError("Choose a nonempty interval within the trajectory")
    Y = validate_fields(problem, state, base)
    if start_slab == 0:
        if previous is not None:
            raise ValueError("Full-horizon integration uses the declared physical initial flow")
        previous = problem.initial_flow
    elif previous is None:
        raise ValueError("A local interval requires its preceding flow explicitly")
    if previous.velocity.shape != (problem.flow.nv, 2) or not np.isfinite(previous.velocity).all():
        raise ValueError("The preceding velocity must match the flow discretization")
    start = time.perf_counter()
    rows, flows, predecessors, times = [], [], [], []
    report = {
        "status": "running",
        "verified": False,
        "subdivision": subdivision,
        "start_slab": start_slab,
        "stop_slab": stop_slab,
        "steps": rows,
        "original_momentum_tolerance": problem.flow_tolerance,
        "scope": "Prescribed-temperature momentum integration; no thermal solve, optimization, or complete-solver speedup.",
    }
    endpoints = np.cumsum(problem.physical_steps)
    for slab in range(start_slab, stop_slab):
        first = problem.initial if slab == 0 else Y[slab - 1]
        interval_start = 0.0 if slab == 0 else float(endpoints[slab - 1])
        seed = FlowResult(base["velocity"][slab], base["pressure"][slab], "saved", [])
        dt = float(problem.physical_steps[slab]) / subdivision
        for j in range(1, subdivision + 1):
            # Preserve the exact recorded endpoint at each original time level.
            temperature_state = (
                Y[slab] if j == subdivision else first + (j / subdivision) * (Y[slab] - first)
            )
            force = acceleration(problem, temperature_state)
            options = dict(
                previous=previous.velocity, time_step=dt, pressure_gauge=problem.pressure_gauge
            )
            result = solve_momentum(
                problem.flow,
                force,
                problem.boundary_indices,
                problem.boundary_values,
                initial=seed,
                tolerance=problem.flow_tolerance,
                max_iterations=problem.flow_cap,
                continuation=False,
                stop_requested=lambda: time.perf_counter() - start >= budget_seconds,
                **options,
            )
            checks = problem.flow.verify(
                result, force, problem.boundary_indices, problem.boundary_values, **options
            )
            passed = verified(result, checks, problem.flow_tolerance)
            flux = problem.flow.boundary_flux(result.velocity)
            row = {
                "slab_zero_based": slab,
                "substep": j,
                "time_step_s": dt,
                "time_s": interval_start + j * dt,
                "status": result.status,
                "verified": passed,
                "equations": checks,
                "history": result.history,
                "global_mass_relative_imbalance": float(
                    abs(flux.sum()) / max(np.abs(flux).sum() / 2, 1e-30)
                ),
            }
            rows.append(row)
            flows.append(result)
            predecessors.append(previous.velocity.copy())
            times.append(row["time_s"])
            if callback is not None:
                callback(rows)
            if not passed:
                report["status"] = (
                    result.status if result.status != "converged" else "verification_failed"
                )
                break
            previous, seed = result, result
        if report["status"] != "running":
            break
    report.update(
        seconds=time.perf_counter() - start,
        completed_steps=len(rows),
        expected_steps=(stop_slab - start_slab) * subdivision,
    )
    if report["status"] == "running":
        report.update(status="converged", verified=True)
    arrays = {
        "velocity": np.stack([r.velocity for r in flows]),
        "pressure": np.stack([r.pressure for r in flows]),
        "previous_velocity": np.stack(predecessors),
        "time_s": np.asarray(times),
    }
    return report, arrays


def history_path(
    problem, base, candidate, slab, updated_previous, fractions, *, budget_seconds=90, callback=None
):
    """Continue a single implicit equation along a temperature/history path.

    Intermediate predecessors are algebraic inputs, not a solved trajectory.
    Every unsuccessful solve is retained and excluded from later initial guesses.
    """
    budget_seconds = positive_real(budget_seconds, "Per-fraction budget")
    Y = validate_fields(problem, candidate, base)
    slab = integer(slab, "Slab", 1)
    if slab >= problem.slabs:
        raise ValueError("The local slab must have a preceding time level")
    fractions = np.asarray(fractions, dtype=float)
    if (
        fractions.ndim != 1
        or not len(fractions)
        or not np.isfinite(fractions).all()
        or fractions[0] != 0
        or np.any(np.diff(fractions) <= 0)
        or fractions[-1] > 1
    ):
        raise ValueError("Fractions must increase from zero and remain in [0, 1]")
    updated_previous = np.asarray(updated_previous)
    if (
        updated_previous.shape != base["velocity"][slab - 1].shape
        or not np.isfinite(updated_previous).all()
    ):
        raise ValueError("The updated predecessor has invalid dimensions or values")
    old = base["state"].reshape(problem.slabs, -1)[slab]
    old_prev = base["velocity"][slab - 1]
    delta_temperature = np.zeros(len(problem.mesh.nodes))
    delta_temperature[problem.free] = problem.temperature_scale * (Y[slab] - old)
    force_direction = problem.flow.buoyancy(delta_temperature, 0.0, problem.expansion)
    current = FlowResult(base["velocity"][slab], base["pressure"][slab], "saved", [])
    last, rows, flows, predecessors = None, [], [], []
    for fraction in fractions:
        state = Y[slab] if fraction == 1 else old + fraction * (Y[slab] - old)
        previous = (
            updated_previous
            if fraction == 1
            else old_prev + fraction * (updated_previous - old_prev)
        )
        force = acceleration(problem, state)
        dt = float(problem.physical_steps[slab])
        start = time.perf_counter()
        options = dict(previous=previous, time_step=dt, pressure_gauge=problem.pressure_gauge)
        result = solve_momentum(
            problem.flow,
            force,
            problem.boundary_indices,
            problem.boundary_values,
            initial=current,
            tolerance=problem.flow_tolerance,
            max_iterations=problem.flow_cap,
            continuation=False,
            stop_requested=lambda start=start: time.perf_counter() - start >= budget_seconds,
            **options,
        )
        checks = problem.flow.verify(
            result, force, problem.boundary_indices, problem.boundary_values, **options
        )
        passed = verified(result, checks, problem.flow_tolerance)
        row = {
            "fraction": float(fraction),
            "seed_fraction": last,
            "status": result.status,
            "verified": passed,
            "equations": checks,
            "history": result.history,
        }
        if passed:
            tangent, metrics = interval_response(
                problem, result, previous, force_direction, dt, updated_previous - old_prev
            )
            metrics["velocity_response_nodal_norm_m_s_per_fraction"] = metrics.pop(
                "velocity_response_nodal_norm_m_s_per_K"
            )
            row.update(
                tangent=metrics,
                tangent_volume_rms_m_s_per_fraction=volume_rms(problem.flow, tangent),
            )
            current, last = result, float(fraction)
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
        flows.append(result)
        predecessors.append(previous.copy())
        if callback is not None:
            callback(rows)
    return {
        "status": "complete",
        "cases": rows,
        "verified_count": sum(r["verified"] for r in rows),
        "scope": "Algebraic continuation of one momentum equation. Failed solves do not establish nonexistence of a root or a physical bifurcation.",
    }, {
        "velocity": np.stack([r.velocity for r in flows]),
        "pressure": np.stack([r.pressure for r in flows]),
        "previous_velocity": np.stack(predecessors),
        "fraction": fractions,
    }
