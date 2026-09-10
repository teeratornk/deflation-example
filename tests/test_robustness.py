"""Regressions for numerical scale, input contracts and unsuccessful solves."""

import numpy as np
import pytest
from scipy import sparse
from deflation_example.problems import build_problem, laplacian, sine_modes
from deflation_example.solvers import (
    LinearResult,
    deflated_cg,
    independent_residual,
    orthonormalize,
    pdas,
    restricted_normal_operator,
)


@pytest.mark.parametrize("problem", ["diffusion", "thermal", "cht"])
@pytest.mark.parametrize(
    "angle", [0.1j, 0.3 + 0j, True, "0.3", [0.3], np.array([0.3]), np.nan, np.inf]
)
def test_target_rejects_nonreal_or_nonscalar_angles(problem, angle):
    with pytest.raises(ValueError, match="finite real scalar"):
        build_problem(problem, 4).target(angle)


@pytest.mark.parametrize("angle", [0, -0.3, np.float32(0.3), np.float64(0.3)])
def test_target_accepts_finite_real_scalars(angle):
    target = build_problem("cht", 4).target(angle)
    assert target.shape == (64,)
    assert target.dtype == np.float64 and np.isfinite(target).all()


@pytest.mark.parametrize("dim", [True, 2.0, 2.5, "2", np.array([2]), 1, 4])
def test_cartesian_dimension_contract(dim):
    with pytest.raises(ValueError):
        laplacian(4, dim)
    with pytest.raises(ValueError):
        sine_modes(4, dim, 2)


@pytest.mark.parametrize("bound", [1, [1, 1], np.array([1, 1], dtype=np.int32)])
def test_integer_bounds_do_not_truncate_states(bound):
    result = pdas(np.eye(2), [0.5, -0.5], bound)
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["y"], [0.5, -0.5])


@pytest.mark.parametrize("scale", [1e-200, 1e200, 1e308])
def test_extreme_rhs_cannot_falsely_converge(scale):
    b = np.ones(4) * scale
    assert independent_residual(np.eye(4), np.zeros(4), b) == pytest.approx(1.0)
    with np.errstate(all="ignore"):
        result = deflated_cg(np.eye(4), b)
    if result.status == "converged":
        np.testing.assert_allclose(result.x / scale, np.ones(4), rtol=1e-10)
    else:
        assert result.residual > 1e-10


def test_failed_inner_solve_has_current_face_diagnostics():
    H = np.array([[2.0, 1.0], [1.0, 2.0]])
    result = pdas(
        H,
        [0.0, 0.0],
        1,
        initial_active=[True, False],
        linear_solver=lambda *args: LinearResult(np.array([0.25]), 3, 123.0, "maxiter"),
    )
    assert result["status"] == "inner_maxiter"
    np.testing.assert_allclose(result["multiplier"], [-2.25, 0.0])
    assert result["history"][-1]["linear_iterations"] == 3
    assert result["history"][-1]["linear_residual"] != 123.0


def test_empty_basis_dimensions():
    assert orthonormalize(np.empty((0, 3))).shape == (0, 0)
    with pytest.raises(ValueError):
        deflated_cg(np.empty((0, 0)), [])


@pytest.mark.parametrize(
    "options",
    [
        {"b": [1j, 1]},
        {"basis": np.ones((2, 1), dtype=complex)},
        {"x0": [0j, 0]},
        {"maxiter": True},
        {"refresh": False},
        {"rtol": True},
        {"b": [1, 1], "A": np.array([[1, np.nan], [0, 1]])},
    ],
)
def test_invalid_linear_contract(options):
    args = {"A": np.eye(2), "b": np.ones(2), **options}
    with pytest.raises(ValueError):
        deflated_cg(**args)


@pytest.mark.parametrize("indices", [[-1], [0, 0], [0.5], [4], [True, False]])
def test_invalid_restriction_indices(indices):
    with pytest.raises(ValueError):
        restricted_normal_operator(sparse.eye(4), 0.1, indices)


def test_pdas_rejects_nonfinite_inner_solution_and_invalid_mask():
    with pytest.raises(ValueError):
        pdas(np.eye(2), [1, 1], 0.5, initial_active=[np.nan, 0])
    with pytest.raises(ValueError):
        pdas(np.eye(2), [1, 1], [1j, 1])
    result = pdas(
        np.eye(2),
        [1, 1],
        0.5,
        linear_solver=lambda *args: LinearResult(np.array([np.nan, 0]), 1, 0, "converged"),
    )
    assert result["status"] != "converged"
