"""A checkpoint reconstruction must reject ambiguous or clipped displacements."""

from copy import deepcopy

import numpy as np
import pytest

from deflation_example.coupled_review_direction import reconstruct


def example():
    trial = dict(status="decrease", step=0.25, objective=3.0)
    attempt = dict(qp_status="converged", qp_kkt={}, qp_history=[], trials=[trial])
    meta = dict(iteration=0, position=0, objective=3.0, history=[dict(attempts=[attempt])])
    return meta, dict(state=np.array([0.1, -0.05]), secant_steps=np.array([[0.1, -0.05]]))


def test_first_direction_recovers_original_unclipped_trial():
    meta, arrays = example()
    initial, direction, lineage = reconstruct(meta, arrays, -1, 1)
    np.testing.assert_array_equal(initial, [0, 0])
    np.testing.assert_allclose(direction, [0.4, -0.2])
    np.testing.assert_allclose(initial + 0.25 * direction, arrays["state"])
    assert not lineage["new_qp_solved"]
    assert lineage["initial_displacement_defect"] == 0


@pytest.mark.parametrize(
    "failure", ["later", "clipped", "missing", "initial", "qp", "objective", "nan", "duplicate"]
)
def test_ambiguous_reconstructions_are_rejected(failure):
    meta, arrays = example()
    if failure == "later":
        meta["iteration"] = 1
    elif failure == "clipped":
        arrays["state"][0] = arrays["secant_steps"][0, 0] = 0.5
    elif failure == "missing":
        arrays["secant_steps"] = np.empty((0, 2))
    elif failure == "initial":
        arrays["state"][0] += 0.01
    elif failure == "qp":
        meta["history"][0]["attempts"][0]["qp_status"] = "failed"
    elif failure == "objective":
        meta["objective"] = 4.0
    elif failure == "nan":
        arrays["state"][0] = np.nan
    else:
        meta["history"][0]["attempts"].append(deepcopy(meta["history"][0]["attempts"][0]))
    with pytest.raises(ValueError):
        reconstruct(meta, arrays, -1, 1)
