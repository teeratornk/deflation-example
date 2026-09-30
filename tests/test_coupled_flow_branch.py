"""A quadratic scalar fold checks the seed sign and parameter units."""

import numpy as np
import pytest

from deflation_example.coupled_flow_branch import curvature_projection


@pytest.mark.parametrize("root", [0.2, -0.2, 1e-4])
def test_exact_scalar_quadratic_recovers_other_root(root):
    # F(x, mu) = x*x - root*root - mu; J=2*root and f=1.
    w = np.array([1 / (2 * root)])
    inverse_q = w**2 / (2 * root)
    result = curvature_projection(w, inverse_q, 1)
    assert result["predicted_turning_step_K"] == pytest.approx(-(root**2))
    parameter = result["nonzero_zero_load_seed_parameter_K"]
    assert root + parameter * w[0] == pytest.approx(-root)
    assert result["projection_relative_remainder"] < 1e-15


def test_pressure_scaling_cannot_change_velocity_projection():
    first = curvature_projection(np.array([1.0, 3e8]), np.array([2.0, -7e-9]), 1)
    second = curvature_projection(np.array([1.0, 0]), np.array([2.0, 0]), 1)
    assert first == second


@pytest.mark.parametrize(
    "w,q,n",
    [
        ([0.0], [1.0], 1),
        ([1.0], [0.0], 1),
        ([1.0], [float("nan")], 1),
        ([1.0], [2.0], 0),
        ([1.0], [2.0], True),
    ],
)
def test_invalid_curvature_projections_fail_explicitly(w, q, n):
    with pytest.raises(ValueError):
        curvature_projection(np.array(w), np.array(q), n)
