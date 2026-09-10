"""Formula-based checks distinguish alignment from convergence certificates."""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from deflation_example.benchmark_cht_conditioning import controlled_cht
from deflation_example.benchmark_cht_support import verify_conditioning


def test_controlled_cht_formula_and_transition():
    with threadpool_limits(limits=1):
        result = controlled_cht(6, [1, 48, 64, 120])
    assert [r["certifies_conditioning_improvement"] for r in result["rows"]] == [
        True,
        True,
        False,
        False,
    ]
    assert result["rows"][-1]["sufficient_bound_status"] == "trivial_one"
    for row in result["rows"]:
        eta, sigma = row["separation"], row["minimum_restricted_singular_value"]
        expected = min(
            1.0,
            (row["restriction_coupling_frobenius"] + row["restricted_reference_residual_frobenius"])
            / (sigma * eta),
        )
        assert row["sufficient_angle_bound"] == pytest.approx(expected, abs=1e-14)
        assert row["actual_sine"] <= expected + 1e-12
        assert row["measured_effective_condition"] <= row["deflated_condition_upper_bound"]
        assert np.isfinite(row["deflated_condition_upper_bound"])
        assert row["restriction_identity_error"] < 1e-12
    verify_conditioning(result)
    result["rows"][0]["deflated_condition_upper_bound"] *= 1.01
    with pytest.raises(ValueError, match="deflated_condition_upper_bound"):
        verify_conditioning(result)
