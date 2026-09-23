"""Matched fixed-source forward repair and complete-equation derivative checks."""

import numpy as np

from .axisymmetric_flow import FlowResult
from .coupled_hybrid_forward import hybrid_step
from .coupled_newton_replay import newton_step, step_equations, criteria_met
from .coupled_resolution import forward_model
from .coupled_step_spectrum import step_linearization


POLICIES = (
    ("equation_max", "equation_max", 21, None, False),
    ("fixed_scaled", "fixed_scaled", 40, None, False),
    ("fixed_scaled_trust", "fixed_scaled", 40, 0.05, False),
    ("newton_anderson", "fixed_scaled", 40, None, True),
)


def derivative_check(problem, source, previous, previous_flow, state, flow, slab):
    """Compare the full current-state Jacobian with centered original equations."""
    H = step_linearization(
        problem, state[problem.free], flow.velocity, slab, control=source, previous=previous
    )[0]
    rng = np.random.default_rng(7241)
    full = np.zeros(problem.flow.size)
    full[problem.flow_free] = rng.uniform(-1, 1, len(problem.flow_free))
    full[: 2 * problem.flow.nv] *= max(float(np.abs(flow.velocity).max()), 1e-3)
    full[2 * problem.flow.nv :] *= max(float(np.abs(flow.pressure).max()), 1e-3)
    dv = np.column_stack((full[: problem.flow.nv], full[problem.flow.nv : 2 * problem.flow.nv]))
    dy = np.zeros_like(state)
    dy[problem.free] = rng.uniform(-1, 1, problem.spatial_size)
    direction = np.r_[full[problem.flow_free], dy[problem.free]]
    exact = H @ direction
    model = forward_model(problem)
    nv = len(problem.flow_free)
    rows = []
    for h in (1e-4, 1e-5, 1e-6):
        residuals = []
        for sign in (-1, 1):
            candidate_flow = FlowResult(
                flow.velocity + sign * h * dv,
                flow.pressure + sign * h * full[2 * problem.flow.nv :],
                "difference",
                [],
            )
            residuals.append(
                step_equations(
                    problem,
                    model,
                    state + sign * h * dy,
                    candidate_flow,
                    source,
                    previous,
                    previous_flow,
                    slab,
                )[0]
            )
        finite = (residuals[1] - residuals[0]) / (2 * h)
        errors = {}
        for label, part in (("momentum_continuity", slice(0, nv)), ("thermal", slice(nv, None))):
            errors[label] = float(
                np.linalg.norm(finite[part] - exact[part])
                / max(np.linalg.norm(finite[part]), np.linalg.norm(exact[part]), 1e-30)
            )
        rows.append({"step": h, "relative_errors": errors})
    return {
        "rows": rows,
        "relative_threshold": 1e-5,
        "passed": all(
            min(r["relative_errors"][key] for r in rows) <= 1e-5
            for key in ("momentum_continuity", "thermal")
        ),
        "scope": "Directional derivative of both coupled equations, including fixed source and temporal history; no optimization derivative claim.",
    }


def compare_repairs(
    problem,
    source,
    previous,
    previous_flow,
    state,
    flow,
    slab,
    *,
    initialization,
    cap=100,
    callback=None,
    continuation=False,
):
    """Every policy receives copies of the same initial and previous fields."""
    rows, fields = [], {}
    policies = (
        (("time_continuation", "fixed_scaled", 40, None, False),) if continuation else POLICIES
    )
    for name, merit, backtracks, trust, hybrid in policies:
        solve = hybrid_step if hybrid else newton_step
        extra = {"anderson_cap": 300, "flow_cap": 100} if hybrid else {}
        if continuation:
            from .coupled_time_continuation import continued_step

            solve = continued_step
        else:
            extra.update(line_search=merit, backtrack_cap=backtracks, trust_region=trust)
        result = solve(
            problem,
            source.copy(),
            previous.copy(),
            FlowResult(
                previous_flow.velocity.copy(), previous_flow.pressure.copy(), "previous", []
            ),
            slab,
            initial_state=state.copy(),
            initial_flow=FlowResult(flow.velocity.copy(), flow.pressure.copy(), "initial", []),
            tolerance=1e-12,
            max_iterations=cap,
            **extra,
        )
        _, checks = step_equations(
            problem,
            forward_model(problem),
            result.state,
            result.flow,
            source,
            previous,
            previous_flow,
            slab,
        )
        key = f"{initialization}_{name}"
        rows.append(
            {
                "policy": name,
                "initialization": initialization,
                "status": result.status,
                "seconds": result.seconds,
                "newton_cap": cap,
                "backtrack_cap": backtracks,
                "trust_region": trust,
                "anderson_cap": 300 if hybrid else None,
                "continuation_stage_cap": 64 if continuation else None,
                "minimum_continuation_increment": 1 / 1024 if continuation else None,
                "independent_checks": checks,
                "independent_criteria_met": bool(criteria_met(checks, 1e-12)),
                "history": result.history,
            }
        )
        fields.update(
            {
                f"{key}_state": result.state,
                f"{key}_velocity": result.flow.velocity,
                f"{key}_pressure": result.flow.pressure,
            }
        )
        if callback is not None:
            callback(rows.copy(), fields.copy())
    return rows, fields


def paired_selection(rows, fields, temperature_scale, *, continuation=False):
    """Select the first predeclared policy verified from both initializations."""
    selections = []
    names = ("time_continuation",) if continuation else tuple(p[0] for p in POLICIES)
    for name in names:
        pair = [r for r in rows if r["policy"] == name]
        good = (
            len(pair) == 2
            and {r["initialization"] for r in pair} == {"saved", "previous"}
            and all(r["independent_criteria_met"] for r in pair)
        )
        difference = None
        flow_differences = {}
        if good:
            difference = float(
                temperature_scale
                * np.max(np.abs(fields[f"saved_{name}_state"] - fields[f"previous_{name}_state"]))
            )
            for field in ("velocity", "pressure"):
                a, b = fields[f"saved_{name}_{field}"], fields[f"previous_{name}_{field}"]
                absolute = float(np.max(np.abs(a - b)))
                scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-30)
                flow_differences[field] = {
                    "maximum_absolute_difference": absolute,
                    "relative_maximum_difference": absolute / scale,
                    "agrees": absolute <= 1e-10 + 1e-5 * scale,
                }
        selections.append(
            {
                "policy": name,
                "both_verified": good,
                "maximum_temperature_difference_K": difference,
                "flow_differences": flow_differences,
                "same_root_within_thresholds": good
                and difference <= 1e-4
                and all(v["agrees"] for v in flow_differences.values()),
            }
        )
    selected = next((r["policy"] for r in selections if r["same_root_within_thresholds"]), None)
    return {
        "selected_policy": selected,
        "comparisons": selections,
        "temperature_agreement_K": 1e-4,
        "flow_absolute_threshold": 1e-10,
        "flow_relative_threshold": 1e-5,
        "scope": "Same-step agreement from two initializations; a complete replay remains required.",
    }
