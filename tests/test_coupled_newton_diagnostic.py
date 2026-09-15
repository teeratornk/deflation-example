"""The forward diagnostic checks the original derivative without mutation."""

import numpy as np

from deflation_example.coupled_newton_diagnostic import inspect_step
from test_coupled_newton_replay import data


def test_saved_step_diagnostic_checks_derivative_and_preserves_inputs():
    problem, _, _, source = data()
    state = problem.full_temperature(problem.initial)
    flow = problem.initial_flow
    originals = [a.copy() for a in (state, source, flow.velocity, flow.pressure)]
    result = inspect_step(problem, source, state, flow, state, flow, 0)
    assert result["status"] == "complete"
    assert result["linear_relative_residual"] < 1e-10
    for name in ("random", "newton"):
        rows = [r for r in result["derivative_checks"] if r["direction"] == name]
        assert min(r["row_scaled_relative_error"] for r in rows) < 1e-6
    assert result["line_search_checks"][-1]["fixed_row_scaled_residual_ratio"] < 1
    for actual, original in zip(
        (state, source, flow.velocity, flow.pressure), originals, strict=True
    ):
        np.testing.assert_array_equal(actual, original)
