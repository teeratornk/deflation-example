"""Original-equation acceptance, state retention and acceleration contracts."""

import numpy as np
import pytest

from deflation_example.anderson import Anderson
from deflation_example.coupled_fixed_point import solve_forward, solve_picard_momentum
from test_coupled_forward import manufactured


def test_anderson_linear_map_and_dependent_history():
    mixer = Anderson(np.ones(4), depth=5)
    x = np.zeros(4)
    for _ in range(5):
        x, info = mixer.propose(x, 0.8 * x + 1)
    np.testing.assert_allclose(x, 5, atol=1e-11)
    assert info["history_rank"] <= 1
    mixer.reset()
    assert not mixer.xs


def test_anderson_validation_and_zero_weight_auxiliary():
    with pytest.raises(ValueError):
        Anderson(np.zeros(3))
    mixer = Anderson([1, 0])
    with pytest.raises(ValueError):
        mixer.propose([0, 0], [np.nan, 1])
    x = np.zeros(2)
    for _ in range(5):
        x, _ = mixer.propose(x, 0.8 * x + [1, 2])
    np.testing.assert_allclose(x, [5, 10], atol=1e-10)


@pytest.mark.parametrize("policy", ["relaxed", "anderson"])
@pytest.mark.parametrize("transient", [False, True])
def test_forward_manufactured_and_initial_guard(policy, transient, monkeypatch):
    model, temperature, velocity, initial, args = manufactured(transient)
    control = np.full_like(temperature, 2)
    source = control.copy()
    result = solve_forward(
        model, control, np.zeros_like(temperature), initial, policy=policy, tolerance=1e-10, **args
    )
    assert result.status == "converged", result.history[-1]
    np.testing.assert_allclose(result.state, temperature, atol=2e-9)
    np.testing.assert_allclose(result.flow.velocity, velocity, atol=2e-10)
    np.testing.assert_array_equal(control, source)
    monkeypatch.setattr(
        model.flow, "solve", lambda *a, **k: pytest.fail("Converged initial guess must not solve")
    )
    checked = solve_forward(
        model, control, result.state, result.flow, policy=policy, tolerance=1e-10, **args
    )
    assert checked.status == "converged"
    assert len(checked.history) == 1


def test_momentum_zero_initial_guard():
    model, _, _, initial, _ = manufactured(feedback=False)
    # A separate zero-boundary, unforced problem has an exactly zero solution.
    result = solve_picard_momentum(
        model.flow,
        np.zeros_like(model.flow.quadrature_points),
        model.flow.boundary,
        np.zeros((len(model.flow.boundary), 2)),
        initial=initial,
        pressure_gauge=(0, 0),
    )
    assert result.status == "converged"
    assert len(result.history) == 1


def test_forward_native_target_matches_final_momentum_criterion(monkeypatch):
    model, temperature, _, initial, args = manufactured(True)
    original = model.flow.solve
    calls = []

    def checked(*a, **kw):
        calls.append(kw["tolerance"])
        assert kw["tolerance"] == 1e-10
        return original(*a, **kw)

    monkeypatch.setattr(model.flow, "solve", checked)
    result = solve_forward(
        model,
        np.full_like(temperature, 2),
        np.zeros_like(temperature),
        initial,
        tolerance=1e-10,
        **args,
    )
    assert calls and result.status == "converged"
    assert result.history[-1]["momentum_relative_residual"] <= 1e-10


def test_rejected_acceleration_uses_relaxed_step(monkeypatch):
    model, temperature, _, initial, args = manufactured(True)
    original = Anderson.propose

    def bad(self, x, mapped):
        value, info = original(self, x, mapped)
        if info["proposal"] == "anderson":
            return x + 1e5, info
        return value, info

    monkeypatch.setattr(Anderson, "propose", bad)
    result = solve_forward(
        model,
        np.full_like(temperature, 2),
        np.zeros_like(temperature),
        initial,
        tolerance=1e-10,
        **args,
    )
    assert result.status == "converged"
    assert any(row.get("proposal") == "residual_safeguard" for row in result.history)
    np.testing.assert_allclose(result.state, temperature, atol=2e-9)


@pytest.mark.parametrize("bad_iteration", [1, 2])
def test_deteriorating_flow_returns_best_consistent_pair(monkeypatch, bad_iteration):
    model, temperature, _, initial, args = manufactured(True)
    original = model.flow.solve
    calls = 0

    def broken(*a, **kw):
        nonlocal calls
        calls += 1
        result = original(*a, **kw)
        if calls == bad_iteration:
            result.velocity *= 1000
            result.velocity[model.velocity_boundary] = model.velocity_values
        return result

    monkeypatch.setattr(model.flow, "solve", broken)
    result = solve_forward(
        model,
        np.full_like(temperature, 2),
        np.zeros_like(temperature),
        initial,
        tolerance=1e-10,
        max_iterations=bad_iteration,
        policy="relaxed",
        **args,
    )
    last = result.history[-1]
    assert last["procedure"] == "returned_state_verification"
    checks = model.flow.verify(
        result.flow,
        model._force(result.state, True),
        model.velocity_boundary,
        model.velocity_values,
        previous=args["previous_velocity"],
        time_step=args["time_step"],
        pressure_gauge=model.pressure_gauge,
    )
    assert last["momentum_relative_residual"] == checks["momentum_relative_residual"]
    assert last["momentum_relative_residual"] < result.history[-2]["momentum_relative_residual"]
