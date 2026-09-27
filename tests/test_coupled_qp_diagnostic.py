"""A precise correction can change the active set and increase a different KKT component."""

import numpy as np
import pytest
from scipy import sparse

from deflation_example.coupled_optimizer import box_kkt
from deflation_example.coupled_qp_diagnostic import inspect_correction, read_previous
from deflation_example.coupled_recovery import RecoveryStore
from test_coupled_optimizer import solver


def test_correction_can_require_activation_despite_increased_primal_residual():
    H = sparse.diags([1e-3], format="csr")
    g, d = np.array([-0.0011]), np.array([0.01])
    qp = {"x": np.array([0.9]), "partition": np.array([0], dtype=np.int8)}
    report, arrays = inspect_correction(
        H,
        g,
        d,
        np.array([-1.0]),
        np.array([1.0]),
        solver(),
        qp,
        lambda x, gradient: box_kkt(x, gradient, -1, 1),
    )
    assert report["linear_checks_pass"]
    assert report["existing_stagnation_test_rejects"]
    assert report["mask_changes"] == report["newly_active"] == 1
    assert report["before_kkt"]["stationarity"] == pytest.approx(2e-4)
    assert report["candidate_kkt"]["primal_absolute"] == pytest.approx(0.1)
    np.testing.assert_allclose(arrays["candidate"], [1.1])
    np.testing.assert_array_equal(qp["x"], [0.9])


def test_changed_initial_mask_is_not_a_fixed_mask_replay():
    with pytest.raises(ValueError, match="does not repeat"):
        inspect_correction(
            sparse.eye(1),
            np.array([-2.0]),
            np.ones(1),
            np.array([-1.0]),
            np.array([1.0]),
            solver(),
            {"x": np.zeros(1), "partition": np.zeros(1, dtype=np.int8)},
            lambda x, g: box_kkt(x, g, -1, 1),
        )


def test_previous_slot_requires_same_manifest_bound_nonlinear_fields(tmp_path):
    store = RecoveryStore(tmp_path, "source-bound")
    opt = {"state": np.zeros(2), "velocity": np.ones((2, 2)), "pressure": np.zeros(2)}
    store.save({"baseline_sha256": "baseline", "optimizer": {**opt, "qp": {"x": np.ones(2)}}})
    store.save({"baseline_sha256": "baseline", "optimizer": {**opt, "qp": None}})
    previous, evidence = read_previous(tmp_path, "source-bound")
    np.testing.assert_array_equal(previous["optimizer"]["qp"]["x"], [1, 1])
    assert evidence["manifest_bound"] is False and len(evidence["sha256"]) == 64
    store.save({"baseline_sha256": "baseline", "optimizer": {**opt, "qp": {"x": np.ones(2)}}})
    store.save(
        {"baseline_sha256": "baseline", "optimizer": {**opt, "state": np.ones(2), "qp": None}}
    )
    with pytest.raises(ValueError, match="nonlinear state differs"):
        read_previous(tmp_path, "source-bound")
