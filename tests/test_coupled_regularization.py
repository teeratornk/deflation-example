"""Regularization changes the objective, gradient, Hessian and active-set solve."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize

from deflation_example.coupled_regularization import (
    ALPHAS,
    RANKS,
    TARGETS,
    checked_alpha,
    configuration,
    quadratic_data,
    solve_quadratic,
)
from deflation_example.study_solvers import StudySolver
from test_coupled_derivatives import small_coupled_problem


def protocol():
    path = Path(__file__).resolve().parents[1] / "examples/coupled_regularization/protocol.json"
    return json.loads(path.read_text())


def test_declared_grid_and_unchanged_configuration():
    p = protocol()
    assert tuple(p["alphas"]) == ALPHAS
    assert tuple(p["ranks"]) == RANKS
    assert tuple(p["targets"]) == TARGETS
    cfg = {**p, "alpha": 1e-14, "queries": [{"target": 7, "upper_K": 357.3}]}
    manifest = {"configuration": cfg}
    for alpha in ALPHAS:
        actual = configuration(manifest, alpha)
        assert actual["alpha"] == alpha
        assert {k: v for k, v in actual.items() if k not in {"alpha", "device", "threads"}} == {
            k: v for k, v in cfg.items() if k not in {"alpha", "device", "threads"}
        }
    assert manifest["configuration"]["alpha"] == 1e-14


@pytest.mark.parametrize("value", [True, 0, -1e-14, np.nan, np.inf, 1e-10, 5e-14])
def test_out_of_protocol_regularization_is_rejected(value):
    with pytest.raises(ValueError):
        checked_alpha(value)


def test_changed_physics_or_accuracy_is_rejected():
    p = protocol()
    for key in ("slabs", "lower_K", "inner_tolerance", "conservation_tolerance"):
        with pytest.raises(ValueError, match="fixed study protocol"):
            configuration({"configuration": {**p, key: p[key] * 2}}, ALPHAS[0])


def test_reweighted_gradient_and_hessian_match_the_new_objective():
    problem = small_coupled_problem([0.2, 0.35], consistent=True, streamline_rule="smooth_p8")
    state = np.linspace(0.05, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    desired = np.full(problem.size, 0.12)
    probe = np.random.default_rng(104).normal(size=problem.size)
    old = problem.alpha
    H0, g0, d0, lo, hi = quadratic_data(problem, evaluation, desired, 0.0, 0.15)
    problem.alpha = 10 * old
    H1, g1, d1, _, _ = quadratic_data(problem, evaluation, desired, 0.0, 0.15)
    J, W = evaluation.jacobian, problem.weights
    np.testing.assert_allclose(
        (H1 @ probe - H0 @ probe) / (9 * old), J.T @ (W * (J @ probe)), rtol=1e-10
    )
    np.testing.assert_allclose((g1 - g0) / (9 * old), J.T @ (W * evaluation.control), rtol=1e-10)
    assert not np.array_equal(d0, d1)
    np.testing.assert_array_equal(lo, -state)
    np.testing.assert_array_equal(hi, 0.15 - state)
    np.testing.assert_array_equal(evaluation.state, state)


def test_constrained_quadratic_matches_an_independent_dense_optimizer():
    problem = small_coupled_problem([0.2, 0.35], consistent=True, streamline_rule="smooth_p8")
    evaluation = problem.evaluate(np.linspace(0.04, 0.1, problem.size))
    desired = np.full(problem.size, 0.2)
    cfg = {"inner_tolerance": 1e-10, "qp_tolerance": 1e-10, "qp_cap": 100}
    solver = StudySolver("jacobi", rank=0, rtol=1e-10, cg_factor=0.1, residual_policy="refine")
    x, row = solve_quadratic(problem, evaluation, desired, 0.0, 0.11, solver, cfg)
    H, g, _, lo, hi = quadratic_data(problem, evaluation, desired, 0.0, 0.11)
    dense = H @ np.eye(problem.size)
    oracle = optimize.minimize(
        lambda z: (0.5 * z @ dense @ z + g @ z, dense @ z + g),
        np.zeros(problem.size),
        jac=True,
        bounds=list(zip(lo, hi, strict=True)),
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000},
    )
    assert row["verified"]
    assert row["maximum_original_relative_residual"] <= 1e-10
    assert max(row["kkt"].values()) <= 1e-10
    np.testing.assert_allclose(x, oracle.x, atol=2e-7, rtol=0)
    assert row["quadratic_objective"] == pytest.approx(oracle.fun, abs=1e-10)
    solver.close()
