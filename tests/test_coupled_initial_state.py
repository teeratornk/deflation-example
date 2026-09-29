"""Fresh model initialization cannot import old optimization history or accuracy."""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest


from deflation_example import coupled_sequence as sequence
from deflation_example.coupled_initial_state import snapshot_initial_guess
from deflation_example.coupled_reference import configured_reference
from deflation_example.coupled_review import snapshot
from deflation_example.reporting import file_sha256, write_report
from test_coupled_derivatives import small_coupled_problem
from test_coupled_optimizer import solver
from test_coupled_review import source_files
from test_coupled_sequence import configuration

pytestmark = pytest.mark.cupy


@pytest.fixture
def assessed(tmp_path):
    old = small_coupled_problem([0.2, 0.35], uniform_capacity=True, consistent=True, inlet=0.2)
    new = small_coupled_problem(
        [0.2, 0.35],
        uniform_capacity=True,
        consistent=True,
        inlet=0.2,
        reference_stabilization="matched",
        streamline_rule="smooth_p8",
    )
    state = np.full(old.size, 0.06)
    evaluation = old.evaluate(state)
    cfg = configuration(new)
    cfg.update(
        transient=True,
        slabs=2,
        horizon_s=0.55,
        alpha=old.alpha,
        consistent_stabilization=True,
        streamline_rule="hard_min",
        method="reference",
    )
    arrays = {
        "state": state,
        "velocity": np.stack([f.velocity for f in evaluation.flows]),
        "pressure": np.stack([f.pressure for f in evaluation.flows]),
        "secant_steps": np.ones((1, old.size)),
        "secant_gradients": np.full((1, old.size), 99.0),
    }
    source_files(tmp_path / "source", cfg, arrays)
    manifest = snapshot(tmp_path / "source", tmp_path / "snapshot")
    assessment = {
        "schema": "coupled-frozen-prefix-review-v1",
        "status": "review_complete",
        "snapshot": manifest,
        "baseline_sha256": "test-baseline",
        "evaluated_streamline_rule": "smooth_p8",
        "centered_checks": {"rows": [{"status": "evaluated"} for _ in range(4)]},
    }
    write_report(tmp_path / "assessment.json", assessment)
    cfg.update(
        streamline_rule="smooth_p8",
        initial_state_snapshot=str(tmp_path / "snapshot"),
        initial_state_assessment=str(tmp_path / "assessment.json"),
    )
    return cfg, new, {"baseline_sha256": "test-baseline"}, arrays


def test_only_temperature_and_flow_guesses_are_loaded(assessed):
    cfg, problem, baseline, arrays = assessed
    before = file_sha256(cfg["initial_state_assessment"])
    guess, metadata = snapshot_initial_guess(cfg, problem, baseline, 0)
    np.testing.assert_array_equal(guess.state, arrays["state"])
    assert set(vars(guess)) == {"state", "flows"}
    assert metadata["retained_secant_pairs"] == metadata["retained_recycling_directions"] == 0
    assert metadata["source_streamline_rule"] == "hard_min"
    assert metadata["evaluated_streamline_rule"] == "smooth_p8"
    actual = problem.evaluate(guess.state, initial=guess)
    assert actual is not guess
    assert np.isfinite(actual.control).all()
    assert before == file_sha256(cfg["initial_state_assessment"])
    guess.state[:] = 0
    again, _ = snapshot_initial_guess(cfg, problem, baseline, 0)
    np.testing.assert_array_equal(again.state, arrays["state"])


def test_explicit_nested_transfer_selects_only_coincident_endpoints(assessed, monkeypatch):
    from deflation_example import coupled_review

    cfg, problem, baseline, _ = assessed
    record, meta, arrays, manifest = coupled_review.read_snapshot(cfg["initial_state_snapshot"])
    record["configuration"]["slabs"] = 4
    original_state = np.arange(4)[:, None] * 0.01 + np.full((4, problem.spatial_size), 0.02)
    arrays["state"] = original_state.ravel()
    arrays["velocity"] = np.repeat(arrays["velocity"][:1], 4, axis=0)
    arrays["pressure"] = np.repeat(arrays["pressure"][:1], 4, axis=0)
    problem.physical_steps[:] = 0.275
    monkeypatch.setattr(
        coupled_review, "read_snapshot", lambda path: (record, meta, arrays, manifest)
    )
    cfg["initial_state_time_policy"] = "nested_endpoints"
    guess, metadata = snapshot_initial_guess(cfg, problem, baseline, 0)
    np.testing.assert_array_equal(guess.state, original_state[[1, 3]].ravel())
    assert metadata["time_transfer"]["source_endpoint_indices"] == [1, 3]
    assert set(vars(guess)) == {"state", "flows"}


@pytest.mark.parametrize(
    "key,value",
    [
        ("alpha", 0.2),
        ("slabs", 4),
        ("horizon_s", 3.0),
        ("target_startup_s", 0.1),
        ("lower_K", -1.0),
        ("consistent_stabilization", False),
        ("temperature_margin_K", 0.01),
    ],
)
def test_initialization_rejects_changed_declared_problem(assessed, key, value):
    cfg, problem, baseline, _ = assessed
    with pytest.raises(ValueError, match=key):
        snapshot_initial_guess({**cfg, key: value}, problem, baseline, 0)


def test_explicit_shared_temperature_recomputes_the_new_objective(assessed):
    cfg, problem, baseline, arrays = assessed
    old_alpha = problem.alpha
    old = problem.evaluate(arrays["state"])
    desired = np.full(problem.size, 0.12)
    old_objective, old_gradient = problem.objective_gradient(old, desired)
    problem.alpha *= 10
    cfg.update(alpha=problem.alpha, initial_state_alpha_policy="shared_temperature")
    guess, metadata = snapshot_initial_guess(cfg, problem, baseline, 0)
    np.testing.assert_array_equal(guess.state, arrays["state"])
    assert set(vars(guess)) == {"state", "flows"}
    fresh = problem.evaluate(guess.state, initial=guess)
    objective, gradient = problem.objective_gradient(fresh, desired)
    assert objective > old_objective
    assert not np.allclose(gradient, old_gradient, rtol=1e-8, atol=1e-14)
    assert metadata["source_alpha"] == old_alpha
    assert metadata["optimization_alpha"] == problem.alpha
    assert metadata["alpha_policy"] == "shared_temperature"
    assert metadata["retained_secant_pairs"] == metadata["retained_recycling_directions"] == 0


@pytest.mark.parametrize("key,value", [("slabs", 4), ("lower_K", -1), ("horizon_s", 3)])
def test_shared_temperature_preserves_physical_guards(assessed, key, value):
    cfg, problem, baseline, _ = assessed
    cfg.update(initial_state_alpha_policy="shared_temperature")
    with pytest.raises(ValueError, match=key):
        snapshot_initial_guess({**cfg, key: value}, problem, baseline, 0)


def test_unknown_policy_and_inconsistent_loaded_alpha_are_rejected(assessed):
    cfg, problem, baseline, _ = assessed
    with pytest.raises(ValueError, match="policy"):
        snapshot_initial_guess(
            {**cfg, "initial_state_alpha_policy": "ignore"}, problem, baseline, 0
        )
    with pytest.raises(ValueError, match="Loaded problem"):
        snapshot_initial_guess(
            {**cfg, "initial_state_alpha_policy": "shared_temperature", "alpha": 0.2},
            problem,
            baseline,
            0,
        )


@pytest.mark.parametrize("change", ["baseline", "rule", "snapshot", "failed_check", "target"])
def test_initialization_requires_matching_assessment_and_target(assessed, change):
    cfg, problem, baseline, _ = assessed
    assessment = json.loads(Path(cfg["initial_state_assessment"]).read_text())
    if change == "baseline":
        baseline = {"baseline_sha256": "other"}
    elif change == "rule":
        cfg = {**cfg, "streamline_rule": "hard_min"}
    elif change == "snapshot":
        assessment["snapshot"]["iteration"] += 1
    elif change == "failed_check":
        assessment["centered_checks"]["rows"][0]["status"] = "flow_failed"
    else:
        cfg = {**cfg, "queries": [{"target": 9, "upper_K": 0.3}]}
    write_report(cfg["initial_state_assessment"], assessment)
    with pytest.raises(ValueError):
        snapshot_initial_guess(cfg, problem, baseline, 0)


@pytest.mark.parametrize("defect", ["size", "velocity", "pressure", "nonfinite", "bounds"])
def test_invalid_initial_arrays_are_rejected_without_clipping(assessed, monkeypatch, defect):
    from deflation_example import coupled_review

    cfg, problem, baseline, _ = assessed
    record, meta, arrays, manifest = coupled_review.read_snapshot(cfg["initial_state_snapshot"])
    if defect == "size":
        arrays["state"] = arrays["state"][:-1]
    elif defect == "velocity":
        arrays["velocity"] = arrays["velocity"][:-1]
    elif defect == "pressure":
        arrays["pressure"] = arrays["pressure"][:, :-1]
    elif defect == "nonfinite":
        arrays["state"][0] = np.nan
    else:
        arrays["state"][0] = 999.0
    monkeypatch.setattr(
        coupled_review, "read_snapshot", lambda path: (record, meta, arrays, manifest)
    )
    with pytest.raises(ValueError, match="Initial"):
        snapshot_initial_guess(cfg, problem, baseline, 0)
    if defect == "bounds":
        assert arrays["state"][0] == 999.0


@pytest.mark.parametrize("defect", ["missing_assessment", "missing_snapshot", "restore", "control"])
def test_runner_rejects_ambiguous_initialization_before_starting(assessed, tmp_path, defect):
    cfg, _, _, _ = assessed
    cfg.update(repetition=0, output=str(tmp_path / "rejected"), stage={"positions": [0]})
    if defect == "missing_assessment":
        cfg["initial_state_assessment"] = None
    elif defect == "missing_snapshot":
        cfg["initial_state_snapshot"] = None
    elif defect == "restore":
        cfg["stage"] = {"positions": [1], "restore": "previous"}
    else:
        cfg["initial_control_directory"] = "previous"
    with pytest.raises(ValueError):
        sequence.run(OmegaConf.create(cfg))
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_fresh_sequence_reevaluates_and_starts_without_history(assessed, monkeypatch, method):
    cfg, problem, baseline, arrays = assessed
    guess, _ = snapshot_initial_guess(cfg, problem, baseline, 0)
    reference = (
        configured_reference(problem, {"rank": 2}, baseline) if method == "reference" else None
    )
    policy = solver(method, reference, rank=2)
    policy.previous = np.arange(problem.size)
    original = sequence.minimize_coupled
    calls = []

    def optimize(*args, **kwargs):
        assert kwargs["resume"] is None
        assert policy.previous is None
        assert not hasattr(kwargs["initial_evaluation"], "control")
        calls.append(kwargs["initial"].copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(sequence, "minimize_coupled", optimize)
    rows, fields = sequence.optimize_targets(
        problem, policy, cfg, positions=[0], initial_guess=guess
    )
    assert rows[0]["verified"], rows[0]
    assert rows[0]["initial_state_snapshot_used"]
    assert not rows[0]["warm_start_used"] and not rows[0]["initial_control_used"]
    np.testing.assert_array_equal(calls[0], arrays["state"])
    np.testing.assert_array_equal(guess.state, arrays["state"])
    assert np.isfinite(fields[0]["control"]).all()
    with pytest.raises(ValueError, match="another initialization"):
        sequence.optimize_targets(problem, policy, cfg, previous=guess, initial_guess=guess)
    policy.close()


@pytest.mark.parametrize("method", ["jacobi", "reference"])
def test_complete_runner_records_fresh_initialization_and_cost(
    assessed, monkeypatch, tmp_path, method
):
    cfg, problem, baseline, _ = assessed
    cfg.update(
        method=method,
        device="cpu",
        memory_interval=0.01,
        repetition=0,
        rank=2,
        recycle_window=4,
        inner_tolerance=1e-11,
        inner_cap=1000,
        threads=1,
        output=str(tmp_path / "run"),
        stage={"positions": [0], "restore": None, "resume": None},
    )
    baseline.update(configuration={}, input_sha256={}, seconds=0.25)
    monkeypatch.setattr(sequence, "load_problem", lambda cfg: (problem, baseline))
    optimize = sequence.optimize_targets

    def check_persisted_inputs(*args, **kwargs):
        progress = json.loads((tmp_path / "run/record.json").read_text())
        assert progress["status"] == "running"
        assert progress["baseline_sha256"] == baseline["baseline_sha256"]
        assert progress["initial_state"]["retained_secant_pairs"] == 0
        assert "reference_construction" in progress["components_seconds"]
        return optimize(*args, **kwargs)

    monkeypatch.setattr(sequence, "optimize_targets", check_persisted_inputs)
    report = sequence.run(OmegaConf.create(cfg))
    assert report["all_problems_verified"], report
    assert report["initial_state"]["retained_secant_pairs"] == 0
    assert report["stage"]["resume"] is None
    assert report["stage"]["checkpoints"]["count"] > 0
    assert report["components_seconds"]["reference_construction"] >= 0
    assert sum(report["components_seconds"].values()) == pytest.approx(report["sequence_seconds"])
    assert report["cases"][0]["initial_state_snapshot_used"]
    meta = json.loads((tmp_path / "run/checkpoint-latest.json").read_text())
    assert meta["iteration"] < cfg["nonlinear_cap"]


def test_full_sequence_builds_one_reference_and_uses_snapshot_only_first(
    assessed, monkeypatch, tmp_path
):
    from deflation_example import coupled_selected_reference as selected

    cfg, problem, baseline, _ = assessed
    cfg.update(
        method="reference",
        device="cpu",
        memory_interval=0.01,
        repetition=0,
        rank=2,
        recycle_window=4,
        inner_tolerance=1e-11,
        inner_cap=1000,
        threads=1,
        output=str(tmp_path / "full"),
        stage=None,
        reference_selection="preconditioned_coupled",
        reference_candidates=4,
        inner_preconditioner="frozen",
        frozen_sweeps=3,
    )
    cfg["queries"][1]["upper_K"] = cfg["queries"][0]["upper_K"]
    baseline.update(configuration={}, input_sha256={}, seconds=0.25)
    monkeypatch.setattr(sequence, "load_problem", lambda cfg: (problem, baseline))
    original = selected.configured_selected_reference
    calls = []

    def construct(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(selected, "configured_selected_reference", construct)
    report = sequence.run(OmegaConf.create(cfg))
    assert report["all_problems_verified"], report
    assert len(calls) == 1
    assert report["schema"] == sequence.COMPLETE_SCHEMA
    assert report["cases"][0]["initial_state_snapshot_used"]
    assert not report["cases"][1]["initial_state_snapshot_used"]
    assert report["cases"][1]["warm_start_used"]


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["jacobi", "reference"])
def test_smooth_full_optimization_with_hybrid_backend(assessed, method):
    pytest.importorskip("cupy")
    from deflation_example.coupled_hybrid_solver import HybridCoupledSolver

    cfg, problem, baseline, _ = assessed
    guess, _ = snapshot_initial_guess(cfg, problem, baseline, 0)
    reference = (
        configured_reference(problem, {"rank": 2}, baseline) if method == "reference" else None
    )
    policy = HybridCoupledSolver(
        method,
        reference=reference,
        rank=2,
        window=4,
        block_min_columns=2,
        block_max_columns=2,
        coarse_device="cuda",
        rtol=1e-10,
        maxiter=1000,
        cg_factor=0.1,
        residual_policy="refine",
    )
    try:
        rows, _ = sequence.optimize_targets(
            problem, policy, cfg, positions=[0], initial_guess=guess
        )
        assert rows[0]["verified"], rows[0]
        assert max(rows[0]["kkt"].values()) <= cfg["nonlinear_tolerance"]
    finally:
        policy.close()
