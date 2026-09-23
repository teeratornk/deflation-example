"""Independent error metrics and exact reconstruction of captured operators."""

import numpy as np
import pytest
from scipy.sparse.linalg import aslinearoperator

from deflation_example.coupled_derivatives import GaussNewtonOperator
from deflation_example.coupled_retention_replay import (
    ReplayPreconditioner,
    error_diagnostic,
    load_reference,
    rebuild,
    save_reference,
)
from deflation_example.study_solvers import ArrayReference
from deflation_example.spacetime_reference import SpaceTimeReference
from test_coupled_derivatives import small_coupled_problem


def test_energy_removal_uses_actual_initial_guess_and_has_correct_spectrum():
    H = np.diag([0.1, 1.0, 5.0, 20.0])
    B = aslinearoperator(H)
    ref = ArrayReference(np.eye(4)[:, :1], {})
    rhs, initial = np.ones(4), np.array([9.0, 0.0, 0.0, 0.0])
    record = error_diagnostic(B, rhs, initial, ref, np.arange(4), np.ones(4))
    error = np.linalg.solve(H, rhs) - initial
    expected = 0.1 * error[0] ** 2 / (error @ H @ error)
    assert record["energy_fraction_removed"] == pytest.approx(expected)
    assert record["spectrum"]["original_condition"] == pytest.approx(200)
    assert record["spectrum"]["deflated_nonzero_condition"] == pytest.approx(20)
    assert record["original_residual"] < 1e-13


def test_zero_load_and_converged_initial_guess():
    B = aslinearoperator(np.eye(4))
    for rhs in (np.zeros(4), np.arange(4.0)):
        report = error_diagnostic(B, rhs, rhs.copy(), None, np.arange(4), np.ones(4))
        assert report["status"] == "initially_converged"
        assert report["energy_fraction_removed"] is None


@pytest.mark.parametrize("compact", [False, True])
def test_reference_bank_serialization_preserves_all_entries(tmp_path, compact):
    rng = np.random.default_rng(8)
    ref = (
        SpaceTimeReference(
            rng.normal(size=(7, 3)), rng.normal(size=(4, 5)), [0, 1, 2, 0, 2], {"test": "compact"}
        )
        if compact
        else ArrayReference(rng.normal(size=(28, 5)), {"test": "dense"})
    )
    stored = save_reference(tmp_path, "ref", ref)
    actual = load_reference(tmp_path, stored)
    np.testing.assert_array_equal(ref.restrict(np.arange(28)), actual.restrict(np.arange(28)))
    stored["sha256"] = "incorrect"
    with pytest.raises(ValueError, match="checksum"):
        load_reference(tmp_path, stored)


def test_rebuilt_coupled_quadratic_matches_operator_and_gradient():
    problem = small_coupled_problem([0.2, 0.35], consistent=True, streamline_rule="smooth_p8")
    state = np.linspace(0.05, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    desired = np.full(problem.size, 0.12)
    _, gradient = problem.objective_gradient(evaluation, desired)
    arrays = dict(
        state=state,
        desired=desired,
        gradient=gradient,
        velocity=np.stack([f.velocity for f in evaluation.flows]),
        pressure=np.stack([f.pressure for f in evaluation.flows]),
        secant_steps=np.empty((0, problem.size)),
        secant_gradients=np.empty((0, problem.size)),
    )
    _, H, d, g = rebuild(problem, arrays, {"damping": 0.001})
    expected = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha, 0.001)
    probe = np.random.default_rng(67).normal(size=problem.size)
    np.testing.assert_allclose(H @ probe, expected @ probe, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(g, gradient, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(d, problem.preconditioning_diagonal(evaluation, 0.001))


def test_feedback_changes_force_and_full_derivative_consistently():
    problems = [
        small_coupled_problem([0.2, 0.35], feedback=b, consistent=True, streamline_rule="smooth_p8")
        for b in (0, 0.0015, 0.003)
    ]
    state = np.linspace(0.05, 0.1, problems[0].size)
    probe = np.random.default_rng(45).normal(size=len(state))
    probe /= np.linalg.norm(probe)
    for problem in problems:
        ev = problem.evaluate(state)
        eps = 1e-5
        plus = problem.evaluate(state + eps * probe, initial=ev)
        minus = problem.evaluate(state - eps * probe, initial=ev)
        np.testing.assert_allclose(
            (plus.control - minus.control) / (2 * eps), ev.jacobian @ probe, rtol=1e-6, atol=1e-7
        )


def test_replay_preconditioner_reuses_only_an_unchanged_mask():
    problem = small_coupled_problem([0.2, 0.35], consistent=True)
    evaluation = problem.evaluate(np.linspace(0.05, 0.1, problem.size))
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    preconditioner = ReplayPreconditioner(problem, evaluation, 0.0, "frozen", 3)
    indices = np.arange(problem.size)
    B = H.restrict(indices)
    preconditioner.attach(B, indices)
    preconditioner.attach(B, indices.copy())
    assert preconditioner.builds == 1
    r = np.ones(len(indices))
    z = B.preconditioner(r)
    assert r @ z > 0
    assert preconditioner.applications == 1
    preconditioner.attach(H.restrict(indices[::2]), indices[::2])
    assert preconditioner.builds == 2
    preconditioner.attach(B, indices)
    assert preconditioner.builds == 3


def test_rank_zero_jacobi_replay_has_no_preconditioner_setup():
    preconditioner = ReplayPreconditioner(None, None, 0.0)
    B = aslinearoperator(np.eye(3))
    preconditioner.attach(B, np.arange(3))
    assert not hasattr(B, "preconditioner")
    assert preconditioner.report()["builds"] == 0


def test_new_replay_refuses_even_sweeps():
    with pytest.raises(ValueError, match="odd sweeps"):
        ReplayPreconditioner(None, None, 0.0, "frozen", 2)
