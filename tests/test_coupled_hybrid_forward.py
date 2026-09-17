"""Forward fallback preserves the equations, retained fields, and attempt costs."""

import numpy as np
import pytest

from deflation_example import coupled_hybrid_forward as hybrid
from deflation_example import coupled_newton_replay as newton
from deflation_example.coupled_derivatives import StabilizationBranchError
from deflation_example.coupled_forward import CoupledResult
from test_coupled_derivatives import small_coupled_problem


def step_data():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True, consistent=True)
    state = np.linspace(0.03, 0.07, problem.size)
    evaluation = problem.evaluate(state)
    temperatures = state.reshape(problem.slabs, -1)
    source = np.zeros(len(problem.mesh.nodes))
    source[problem.free] = evaluation.control.reshape(problem.slabs, -1)[1]
    previous = problem.full_temperature(temperatures[0])
    exact = problem.full_temperature(temperatures[1])
    return problem, source, previous, evaluation.flows[0], exact, evaluation.flows[1]


def test_converged_initial_fields_skip_both_iterative_procedures(monkeypatch):
    problem, source, previous, prior_flow, exact, flow = step_data()
    monkeypatch.setattr(
        newton, "step_linearization", lambda *a, **k: pytest.fail("Jacobian unused")
    )
    monkeypatch.setattr(hybrid, "solve_forward", lambda *a, **k: pytest.fail("Fallback unused"))
    result = hybrid.hybrid_step(
        problem, source, previous, prior_flow, 1, initial_state=exact, initial_flow=flow
    )
    assert result.status == "converged"
    np.testing.assert_array_equal(result.state, exact)
    assert len(result.history[0]["attempts"]) == 1


def test_nonsmooth_newton_returns_its_fields_for_the_fallback(monkeypatch):
    problem, source, previous, prior_flow, exact, _ = step_data()

    def switch(*args, **kwargs):
        raise StabilizationBranchError("controlled branch test")

    monkeypatch.setattr(newton, "step_linearization", switch)
    before = source.copy()
    result = hybrid.hybrid_step(problem, source, previous, prior_flow, 1)
    assert result.status == "converged", result.history[-1]
    np.testing.assert_allclose(result.state, exact, rtol=0, atol=2e-10)
    np.testing.assert_array_equal(source, before)
    attempts = result.history[0]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["status"] == "newton_stabilization_switch"
    assert attempts[1]["procedure"] == "complete_field_anderson"
    assert result.seconds >= sum(row["seconds"] for row in attempts)
    assert newton.criteria_met(result.history[-1], 1e-12)


def test_worse_fallback_cannot_replace_retained_fields_or_claim_convergence(monkeypatch):
    problem, source, previous, prior_flow, _, _ = step_data()
    saved = {}

    def worse(model, control, state, flow, **kwargs):
        saved["state"] = state.copy()
        bad = state.copy()
        bad[problem.free] += 1000
        return CoupledResult(bad, flow, "converged", [], 0.0)

    monkeypatch.setattr(hybrid, "solve_forward", worse)
    result = hybrid.hybrid_step(problem, source, previous, prior_flow, 1, max_iterations=0)
    assert result.status == "hybrid_no_convergence"
    np.testing.assert_array_equal(result.state, saved["state"])
    assert result.history[-1]["selected"] == "newton"
    _, metrics = newton.step_equations(
        problem,
        hybrid.forward_model(problem),
        result.state,
        result.flow,
        source,
        previous,
        prior_flow,
        1,
    )
    for key, value in metrics.items():
        assert result.history[-1][key] == value


def test_invalid_fallback_budget_is_rejected_before_solving():
    with pytest.raises(ValueError, match="Anderson iteration cap"):
        hybrid.hybrid_step(None, None, None, None, 0, anderson_cap=0)
