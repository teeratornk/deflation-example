"""A substep predictor never replaces verification of the original step."""

from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_late_step import attempt_step
from test_coupled_derivatives import small_coupled_problem


def test_predictor_corrector_keeps_original_equations_and_charges_every_attempt(monkeypatch):
    import deflation_example.coupled_late_step as module

    p = SimpleNamespace(physical_steps=np.array([2.0, 3.0]), time_scale=4.0)
    previous, source = np.array([1.0]), np.array([7.0])
    original_flow = object()
    calls, attempts = [], []

    def solve(problem, control, state, flow, slab, **kwargs):
        calls.append((problem.physical_steps.copy(), control.copy(), state.copy(), flow, kwargs))
        return SimpleNamespace(
            state=state + 1, flow=object(), status="converged", seconds=1.0, history=[]
        )

    def check(problem, control, state, flow, slab, result):
        return {
            "momentum_relative_residual": 0.0,
            "continuity_relative_residual": 0.0,
            "thermal_relative_residual": 0.0,
            "mass_relative_imbalance": 0.0,
            "energy_relative_defect": 0.0,
        }

    monkeypatch.setattr(module, "newton_step", solve)
    monkeypatch.setattr(module, "verify", check)
    _, summary = attempt_step(
        p, source, previous, original_flow, 1, "half_predictor", lambda r, d: attempts.append(d)
    )
    assert summary["full_step_verified"]
    assert len(attempts) == len(calls) == 3
    np.testing.assert_array_equal(p.physical_steps, [2, 3])
    assert [c[0][1] for c in calls] == [1.5, 1.5, 3.0]
    np.testing.assert_array_equal(calls[-1][2], previous)
    assert calls[-1][3] is original_flow
    np.testing.assert_array_equal(calls[-1][4]["initial_state"], [3.0])
    assert all(np.array_equal(c[1], source) for c in calls)


@pytest.mark.parametrize("policy", ["half_predictor", "anderson_previous"])
def test_small_complete_equations_verified_after_each_policy(policy):
    p = small_coupled_problem([0.2, 0.35], uniform_capacity=True, consistent=True)
    source = np.zeros(len(p.mesh.nodes))
    source[p.free] = 0.05
    attempts = []
    _, report = attempt_step(
        p,
        source,
        p.full_temperature(p.initial),
        p.initial_flow,
        0,
        policy,
        lambda r, d: attempts.append(d),
    )
    assert report["status"] == "converged"
    assert report["full_step_verified"]
    assert len(attempts) == (3 if policy == "half_predictor" else 1)
    assert all(d["independent_criteria_met"] for d in attempts)
