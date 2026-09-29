"""Frozen study axes and verification gates prevent unmatched comparisons."""

import numpy as np
import pytest

from deflation_example.coupled_flow_replay import reconstructed_trial
from deflation_example.coupled_small_study import (
    configuration,
    require_verification,
    verification_identity,
)


def settings(**updates):
    original = dict(
        transient=True,
        horizon_s=600,
        target_startup_s=60,
        target_count=16,
        queries=[{"target": 7, "upper_K": 357.3}],
        lower_K=337.3,
    )
    return configuration(
        original,
        **dict(
            slabs=16, alpha=1e-11, rank=0, sequence="development", device="cpu", continuation=False
        )
        | updates,
    )


def test_matched_arms_keep_physics_accuracy_and_initial_verification():
    baseline = settings()
    for rank in (8, 16, 32):
        reference = settings(rank=rank, device="hybrid", sequence="nearby")
        assert verification_identity(reference) == verification_identity(baseline)
        assert reference["inner_tolerance"] == baseline["inner_tolerance"] == 1e-10
        assert reference["nonlinear_tolerance"] == baseline["nonlinear_tolerance"] == 1e-8
        assert reference["reference_krylov_steps"] == 48
    assert baseline["method"] == "jacobi" and baseline["inner_preconditioner"] == "frozen"
    assert [q["target"] for q in settings(sequence="stress")["queries"]] == [7, 15, 14]


@pytest.mark.parametrize("variant", ["standard", "sequential", "recycling", "lowest"])
def test_policy_ablations_change_only_declared_solver_settings(variant):
    base = settings(rank=8)
    cfg = settings(rank=8, variant=variant, repetition=2)
    assert verification_identity(cfg) == verification_identity(base)
    allowed = {
        "study_variant",
        "repetition",
        "method",
        "reference_transfer",
        "reference_selection",
        "reference_krylov_selection",
    }
    assert {k for k in cfg if cfg[k] != base[k]} <= allowed
    assert cfg["method"] == ("recycling" if variant == "recycling" else "reference")
    assert cfg["reference_transfer"] == ("sequential" if variant == "sequential" else "full")
    assert cfg["reference_krylov_selection"] == (
        "lowest" if variant == "lowest" else "alternating_low_high"
    )
    assert cfg["recycle_window"] == 16


@pytest.mark.parametrize(
    "changes",
    [
        {"variant": "recycling"},
        {"variant": "sequential"},
        {"variant": "lowest"},
        {"variant": "unknown"},
        {"repetition": -1},
        {"repetition": True},
        {"repetition": 1.5},
        {"capture_trace": 1},
    ],
)
def test_invalid_ablation_arguments_fail_before_running(changes):
    with pytest.raises(ValueError):
        settings(**changes)


def test_trace_capture_has_same_equation_gate_but_explicit_cost_scope():
    assert settings(capture_trace=True)["capture_linear_systems"] is True
    assert verification_identity(settings(capture_trace=True)) == verification_identity(settings())


@pytest.mark.parametrize("updates", [{"slabs": 8}, {"alpha": 1e-8}, {"rank": 100}])
def test_undeclared_study_axis_is_rejected(updates):
    with pytest.raises(ValueError):
        settings(**updates)


@pytest.mark.parametrize("change", ["status", "source", "slabs", "alpha", "continuation"])
def test_verification_cannot_be_transferred_to_unchecked_physics_or_source(change):
    cfg = settings()
    gate = {
        "schema": "coupled-small-study-verification-v1",
        "status": "verified",
        "configuration_identity": verification_identity(cfg),
        "environment": {"source_sha256": {"source": "a"}},
    }
    require_verification(gate, cfg, {"source": "a"})
    if change == "status":
        gate["status"] = "verification_failed"
    elif change == "source":
        gate["environment"]["source_sha256"] = {"source": "b"}
    else:
        cfg = settings(**{change: {"slabs": 32, "alpha": 1e-12, "continuation": True}[change]})
    with pytest.raises(ValueError):
        require_verification(gate, cfg, {"source": "a"})


def test_reconstruction_matches_recorded_trial_and_rejects_clipped_directions():
    base, direction = np.array([0.1, 0.3]), np.array([0.2, -0.1])
    retained = base + direction / 16
    state, metadata = reconstructed_trial(base, retained, 1 / 16, 0.5, 0, 1, 10, 1)
    np.testing.assert_allclose(state, base + direction / 2)
    assert metadata["increment_K"] == pytest.approx(1)
    with pytest.raises(ValueError, match="recorded"):
        reconstructed_trial(base, retained, 1 / 16, 0.5, 0, 1, 10, 2)
    with pytest.raises(ValueError, match="physical box"):
        reconstructed_trial(base, base + 0.2, 1 / 16, 1, 0, 1, 10, 32)
