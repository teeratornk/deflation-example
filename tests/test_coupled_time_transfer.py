"""Explicit temporal transfer preserves source checks and never imports controls."""

from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_initial_state import temporal_indices


@pytest.mark.parametrize("slabs", [16, 32, 64])
def test_nested_endpoints_are_coincident_and_include_final_time(slabs):
    original = {"slabs": 64, "horizon_s": 600.0, "transient": True}
    cfg = {**original, "slabs": slabs, "initial_state_time_policy": "nested_endpoints"}
    problem = SimpleNamespace(slabs=slabs, physical_steps=np.full(slabs, 600 / slabs))
    indices, metadata = temporal_indices(original, cfg, problem)
    np.testing.assert_array_equal(indices, np.arange(64 // slabs - 1, 64, 64 // slabs))
    np.testing.assert_allclose(metadata["target_times_s"], np.cumsum(problem.physical_steps))
    assert indices[-1] == 63


@pytest.mark.parametrize("defect", ["implicit", "nonnested", "horizon", "nonuniform", "steady"])
def test_undeclared_or_incompatible_transfer_is_rejected(defect):
    original = {"slabs": 64, "horizon_s": 600.0, "transient": True}
    cfg = {**original, "slabs": 16, "initial_state_time_policy": "nested_endpoints"}
    problem = SimpleNamespace(slabs=16, physical_steps=np.full(16, 37.5))
    if defect == "implicit":
        cfg.pop("initial_state_time_policy")
    elif defect == "nonnested":
        cfg["slabs"] = 24
    elif defect == "horizon":
        cfg["horizon_s"] = 1200.0
    elif defect == "nonuniform":
        problem.physical_steps[:2] += [-1, 1]
    else:
        cfg["transient"] = False
    with pytest.raises(ValueError):
        temporal_indices(original, cfg, problem)
