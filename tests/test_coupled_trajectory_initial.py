"""Finer-grid initialization preserves physics and reevaluates the complete problem."""

import numpy as np
from omegaconf import OmegaConf
import pytest


from deflation_example import coupled_sequence as sequence
from deflation_example.coupled_reference import configured_reference
from deflation_example.coupled_trajectory_initial import (
    interpolate_trajectory,
    trajectory_initial_guess,
)
from deflation_example.reporting import write_fields, write_report
from test_coupled_derivatives import small_coupled_problem
from test_coupled_sequence import configuration
from test_coupled_optimizer import solver

pytestmark = pytest.mark.cupy


@pytest.fixture
def saved(tmp_path):
    old = small_coupled_problem(
        [0.2, 0.2], uniform_capacity=True, consistent=True, streamline_rule="smooth_p8"
    )
    new = small_coupled_problem(
        [0.1] * 4, uniform_capacity=True, consistent=True, streamline_rule="smooth_p8"
    )
    state = np.linspace(0.04, 0.08, old.size)
    ev = old.evaluate(state)
    cfg = configuration(old)
    cfg.update(
        transient=True,
        slabs=2,
        horizon_s=0.4,
        alpha=old.alpha,
        consistent_stabilization=True,
        streamline_rule="smooth_p8",
        method="reference",
    )
    path = tmp_path / "original"
    path.mkdir()
    write_fields(
        path / "target-00.npz",
        state=state,
        control=ev.control,
        desired=state + 0.01,
        velocity=np.stack([f.velocity for f in ev.flows]),
        pressure=np.stack([f.pressure for f in ev.flows]),
    )
    write_report(
        path / "record.json",
        {
            "configuration": cfg,
            "baseline_sha256": "test",
            "cases": [
                {
                    "position": 0,
                    "target": 0,
                    "upper_K": cfg["queries"][0]["upper_K"],
                    "verified": True,
                }
            ],
        },
    )
    cfg = {**cfg, "slabs": 4, "initial_trajectory_directory": str(path)}
    return cfg, new, {"baseline_sha256": "test"}, state


def test_interpolation_includes_initial_condition_and_all_endpoints():
    values = np.array([[2.0, 4.0], [4.0, 8.0]])
    result = interpolate_trajectory(values, np.zeros(2), [0.5, 1.0, 1.5, 2.0], 2.0)
    np.testing.assert_array_equal(result, [[1, 2], [2, 4], [3, 6], [4, 8]])
    np.testing.assert_array_equal(values, [[2, 4], [4, 8]])


def test_refinement_preset_preserves_physical_model_and_final_accuracy():
    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="deflation_example.conf", version_base=None):
        cfg = compose(config_name="coupled_refinement")
    assert cfg.slabs == 256 and cfg.horizon_s == 600 and cfg.target_startup_s == 60
    assert cfg.alpha == 1e-14 and cfg.lower_K == 337.3 and cfg.queries[0].upper_K == 357.3
    assert cfg.temperature_margin_K == 0 and cfg.queries[0].target == 7
    assert cfg.flow_tolerance == cfg.equation_acceptance_tolerance == 1e-12
    assert cfg.conservation_tolerance == 1e-6 and cfg.nonlinear_tolerance == 1e-8
    assert cfg.consistent_stabilization and cfg.streamline_rule == "smooth_p8"
    assert cfg.initial_state_snapshot is None and cfg.initial_trajectory_directory is None
    assert cfg.stage.positions == [0] and cfg.stage.restore is None and cfg.stage.resume is None
    assert cfg.hybrid_block_max_columns == 5 and cfg.hybrid_coarse_device == "cuda"


@pytest.mark.parametrize("times", [[0, 1], [1, 0.5], [1, 3], [np.nan], []])
def test_invalid_times_are_rejected(times):
    with pytest.raises(ValueError):
        interpolate_trajectory(np.ones((2, 3)), np.zeros(3), times, 2.0)


def test_saved_endpoints_retained_and_no_control_or_history_imported(saved):
    cfg, problem, baseline, state = saved
    guess, meta = trajectory_initial_guess(cfg, problem, baseline, 0)
    np.testing.assert_allclose(guess.state.reshape(4, -1)[1::2].ravel(), state, atol=1e-16)
    assert set(vars(guess)) == {"state", "flows"}
    assert meta["source_slabs"] == 2 and meta["new_slabs"] == 4
    assert meta["retained_secant_pairs"] == meta["retained_recycling_directions"] == 0
    assert meta["initial_condition_included"]
    ev = problem.evaluate(guess.state, initial=guess)
    assert ev is not guess and ev.control.shape == (problem.size,)
    assert np.isfinite(ev.control).all()


@pytest.mark.parametrize(
    "key,value",
    [
        ("alpha", 0.3),
        ("horizon_s", 0.8),
        ("slabs", 3),
        ("target_count", 3),
        ("target_startup_s", 0.1),
        ("temperature_margin_K", 0.01),
        ("lower_K", -5),
        ("consistent_stabilization", False),
        ("streamline_rule", "hard_min"),
        ("transport_form", "conservative"),
        ("transient", False),
    ],
)
def test_changed_model_is_not_silently_imported(saved, key, value):
    cfg, problem, baseline, _ = saved
    with pytest.raises(ValueError):
        trajectory_initial_guess({**cfg, key: value}, problem, baseline, 0)


@pytest.mark.parametrize("defect", ["baseline", "target", "bound"])
def test_target_bound_and_baseline_must_match(saved, defect):
    cfg, problem, baseline, _ = saved
    if defect == "baseline":
        baseline = {"baseline_sha256": "other"}
    else:
        query = dict(cfg["queries"][0])
        query["target" if defect == "target" else "upper_K"] += 1
        cfg = {**cfg, "queries": [query]}
    with pytest.raises(ValueError):
        trajectory_initial_guess(cfg, problem, baseline, 0)


@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_finer_optimizer_evaluates_all_slabs_without_history(saved, monkeypatch, method):
    cfg, problem, baseline, _ = saved
    guess, _ = trajectory_initial_guess(cfg, problem, baseline, 0)
    ref = configured_reference(problem, {"rank": 2}, baseline) if method == "reference" else None
    policy = solver(method, ref, rank=2)
    policy.previous = np.ones(problem.size)
    original = sequence.minimize_coupled

    def check(*args, **kwargs):
        assert policy.previous is None
        assert kwargs["resume"] is None
        assert not hasattr(kwargs["initial_evaluation"], "control")
        return original(*args, **kwargs)

    monkeypatch.setattr(sequence, "minimize_coupled", check)
    rows, fields = sequence.optimize_targets(
        problem, policy, cfg, positions=[0], initial_guess=guess, initial_guess_kind="trajectory"
    )
    assert rows[0]["verified"], rows[0]
    assert rows[0]["initial_trajectory_used"] and not rows[0]["initial_state_snapshot_used"]
    assert fields[0]["state"].shape == (problem.size,)
    policy.close()


@pytest.mark.parametrize("other", ["initial_control_directory", "initial_state_snapshot"])
def test_runner_rejects_ambiguous_initialization(saved, tmp_path, other):
    cfg, _, _, _ = saved
    cfg.update(repetition=0, stage={"positions": [0]}, output=str(tmp_path / "rejected"))
    cfg[other] = "another"
    with pytest.raises(ValueError):
        sequence.run(OmegaConf.create(cfg))
    assert not (tmp_path / "rejected").exists()


def test_complete_runner_records_temporal_initialization(saved, monkeypatch, tmp_path):
    cfg, problem, baseline, _ = saved
    cfg.update(
        method="jacobi",
        device="cpu",
        memory_interval=0.01,
        repetition=0,
        rank=0,
        recycle_window=4,
        inner_tolerance=1e-11,
        inner_cap=1000,
        threads=1,
        output=str(tmp_path / "run"),
        stage={"positions": [0]},
    )
    baseline.update(configuration={}, input_sha256={}, seconds=0.25)
    monkeypatch.setattr(sequence, "load_problem", lambda cfg: (problem, baseline))
    report = sequence.run(OmegaConf.create(cfg))
    assert report["all_problems_verified"], report
    assert report["initial_trajectory"]["new_slabs"] == 4
    assert report["cases"][0]["initial_trajectory_used"]
    assert sum(report["components_seconds"].values()) == pytest.approx(report["sequence_seconds"])


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_interpolated_trajectory_with_hybrid_solver(saved, method):
    pytest.importorskip("cupy")
    from deflation_example.coupled_hybrid_solver import HybridCoupledSolver

    cfg, problem, baseline, _ = saved
    guess, _ = trajectory_initial_guess(cfg, problem, baseline, 0)
    ref = configured_reference(problem, {"rank": 2}, baseline) if method == "reference" else None
    policy = HybridCoupledSolver(
        method,
        reference=ref,
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
            problem,
            policy,
            cfg,
            positions=[0],
            initial_guess=guess,
            initial_guess_kind="trajectory",
        )
        assert rows[0]["verified"], rows[0]
        assert max(rows[0]["kkt"].values()) <= cfg["nonlinear_tolerance"]
    finally:
        policy.close()
