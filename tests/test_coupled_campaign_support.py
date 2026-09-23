"""Stage overrides, cross-scheme summaries, prolonged initial controls and v2 statistics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from deflation_example import coupled_campaign as campaign
from deflation_example import coupled_resolution_v2 as v2
from deflation_example.coupled_optimize import prolonged_initial_state
from deflation_example.coupled_pilot import transformer_inputs
from deflation_example.coupled_time_resolution_report import summarize
from deflation_example.reporting import write_fields
from test_coupled_derivatives import small_coupled_problem

ROOT = Path(__file__).parents[1]


def protocol():
    return json.loads(
        (ROOT / "examples/coupled_optimization/final_study/protocol.json").read_text()
    )


def plan(frozen):
    return {
        "grid": {"slabs": 64},
        "baseline": {"directory": "/tmp/baseline"},
        "frozen_selection": frozen,
    }


def test_stage_overrides_compose_with_hydra_before_and_after_the_freeze(tmp_path):
    proto = protocol()
    before = campaign.stage_overrides(
        proto,
        plan(None),
        "reference",
        2,
        [3],
        tmp_path / "out",
        restore=tmp_path / "prev",
        rank=300,
        window=300,
        refresh=1000,
    )
    frozen = {
        "ranks": {
            "jacobi": {"rank": 0, "recycle_window": 1},
            "reference": {"rank": 400, "recycle_window": 400},
            "recycling": {"rank": 200, "recycle_window": 200},
        },
        "inner_refresh": 50000,
    }
    after = campaign.stage_overrides(proto, plan(frozen), "jacobi", 0, [0], tmp_path / "out")
    with initialize_config_dir(
        version_base="1.3", config_dir=str(ROOT / "src/deflation_example/conf")
    ):
        composed_before = compose(config_name="coupled_sequence", overrides=before)
        composed_after = compose(config_name="coupled_sequence", overrides=after)
    assert composed_before.method == "reference" and composed_before.rank == 300
    assert composed_before.stage.positions == [3]
    assert composed_before.stage.restore == str(tmp_path / "prev")
    assert composed_before.stage.resume is None
    assert composed_before.device == "hybrid" and composed_before.hybrid_coarse_device == "cuda"
    assert composed_before.flow_tolerance == 1e-12 and composed_before.flow_cap == 40
    assert len(composed_before.queries) == 5 and composed_before.queries[3].upper_K == 353.3
    assert composed_before.slabs == 64 and composed_before.repetition == 2
    assert composed_after.rank == 0 and composed_after.inner_refresh == 50000
    assert composed_after.stage.restore is None
    with pytest.raises(ValueError):
        campaign.stage_overrides(proto, plan(None), "reference", 0, [0], tmp_path / "x")
    with pytest.raises(ValueError):
        campaign.stage_overrides(proto, plan(frozen), "reference", 0, [0], tmp_path / "x", rank=7)
    with pytest.raises(ValueError):
        campaign.hydra_value("has space")
    assert campaign.hydra_value({"target": 7, "upper_K": 357.3}) == "{target:7,upper_K:357.3}"


def replay_directory(root, name, subdivision, state, times, forward_solver, tracking, excess):
    directory = root / name
    directory.mkdir(parents=True)
    write_fields(directory / "states.npz", state=state, times_s=times)
    record = {
        "status": "converged",
        "subdivision": subdivision,
        "optimization_field_sha256": "source",
        "baseline_sha256": "baseline",
        "configuration": {"slabs": 4, "horizon_s": 600.0},
        "forward_solver": forward_solver,
        "environment": {"git_head": "test"},
        "steps": [{"time_s": float(t), "status": "converged"} for t in times],
        "tracking_integral_refined_K2_m3_s": tracking,
        "maximum_recorded_upper_violation_K": excess,
        "maximum_recorded_lower_violation_K": 0.0,
        "seconds": 1.0,
    }
    (directory / "record.json").write_text(json.dumps(record))
    return directory


def synthetic_states(free, subdivision, bump_node, bump_scale):
    levels = 4 * subdivision
    times = 600.0 * np.arange(1, levels + 1) / levels
    profile = 0.3 * np.sin(np.pi * times / 600.0)[:, None] * np.ones((1, free))
    profile[:, bump_node] += bump_scale * np.exp(-(((times - 250.0) / 40.0) ** 2))
    return profile, times


def test_cross_scheme_summary_and_v2_statistics(tmp_path):
    mesh, parameters, _, _, _, hashes = transformer_inputs(0)
    free = len(mesh.free)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    write_fields(baseline / "baseline.npz", velocity=np.zeros((3, 2)), pressure=np.zeros(2))
    digest = hashlib.sha256((baseline / "baseline.npz").read_bytes()).hexdigest()
    (baseline / "record.json").write_text(
        json.dumps(
            {
                "status": "converged",
                "configuration": {"level": 0},
                "baseline_sha256": digest,
                "input_sha256": hashes,
            }
        )
    )
    euler = {"procedure": "monolithic_newton", "time_scheme": "backward_euler"}
    bdf2 = {"procedure": "monolithic_newton", "time_scheme": "bdf2"}
    bump = 17
    directories = []
    for name, subdivision, scheme, bump_scale, excess in (
        ("s1", 1, euler, 0.00, 0.10),
        ("s2", 2, euler, 0.05, 0.15),
        ("s4", 4, euler, 0.06, 0.175),
    ):
        state, times = synthetic_states(free, subdivision, bump, bump_scale)
        directories.append(
            replay_directory(
                tmp_path, name, subdivision, state, times, scheme, 340.0 + 0.1 * subdivision, excess
            )
        )
    state, times = synthetic_states(free, 8, bump, 0.06)
    other = replay_directory(tmp_path, "b8", 8, state, times, bdf2, 340.9, 0.18)
    with pytest.raises(ValueError):
        summarize([*directories, other], 20.0, initial_value=0.0)
    mixed = summarize([*directories, other], 20.0, initial_value=0.0, cross_scheme=True)
    assert len(mixed["pairs"]) == 3
    report = v2.assess(directories, baseline, 357.3)
    assert report["schema"] == v2.SCHEMA and len(report["pairs"]) == 2
    pair = report["pairs"][-1]
    assert pair["pointwise_peak"]["free_node_index"] == bump
    assert pair["pointwise_maximum_K"] > pair["pointwise_percentiles_K"]["99.9"]
    assert pair["mass_weighted_space_time_rms_K"] < pair["pointwise_maximum_K"]
    assert pair["nodes_exceeding"]["0.05_K"]["count"] >= 1
    assert sum(w["levels"] for w in pair["window_maxima"]) == pair["fine_levels"]
    assessment = report["assessment_v2"]
    assert assessment["pointwise_maximum_K"]["statement"] == "not met"
    assert assessment["recorded_bound_excess_K"] == [0.10, 0.15, 0.175]
    assert assessment["recomputed_bound_excess_K"] == [0.0, 0.0, 0.0]
    assert assessment["bound_excess_richardson"]["limit_K"] == pytest.approx(0.20)
    assert assessment["v1_status"] in {
        "resolution_thresholds_exceeded",
        "refinement_trend_not_decreasing",
    }
    for row in report["rows"]:
        assert row["bound_excess"]["maximum_K"] == 0.0
        assert row["time_scheme"] == "backward_euler"
    cross = v2.assess([*directories, other], baseline, 357.3, cross_scheme=True)
    assert cross["cross_scheme"] and len(cross["pairs"]) == 3
    assert cross["rows"][-1]["time_scheme"] == "bdf2"


def test_prolonged_initial_state_repeats_a_coarser_saved_optimum(tmp_path):
    problem = small_coupled_problem([0.2, 0.35, 0.2, 0.35], uniform_capacity=True)
    spatial = problem.size // 4
    saved = tmp_path / "saved"
    saved.mkdir()
    rng = np.random.default_rng(3)
    state = rng.uniform(0.01, 0.05, size=2 * spatial)
    write_fields(saved / "reference-fields.npz", state=state, control=state * 2, desired=state * 3)
    saved_cfg = {
        "slabs": 2,
        "transient": True,
        "query": 7,
        "upper_K": 357.3,
        "lower_K": 337.3,
        "horizon_s": 600.0,
        "target_count": 16,
        "target_startup_s": 60.0,
    }
    (saved / "record.json").write_text(
        json.dumps(
            {
                "configuration": saved_cfg,
                "baseline_sha256": "base",
                "results": [{"method": "reference", "status": "converged"}],
            }
        )
    )
    cfg = {
        **saved_cfg,
        "slabs": 4,
        "initial_control_directory": str(saved),
        "initial_control_method": "reference",
        "initial_control_position": None,
        "initial_control_repeat": 2,
    }
    prolonged, note = prolonged_initial_state(cfg, problem, {"baseline_sha256": "base"})
    assert prolonged.shape == (problem.size,)
    np.testing.assert_array_equal(prolonged.reshape(4, spatial)[0], state.reshape(2, spatial)[0])
    np.testing.assert_array_equal(prolonged.reshape(4, spatial)[1], state.reshape(2, spatial)[0])
    np.testing.assert_array_equal(prolonged.reshape(4, spatial)[3], state.reshape(2, spatial)[1])
    assert note["temporal_repetition"] == 2 and note["saved_slabs"] == 2
    with pytest.raises(ValueError):
        prolonged_initial_state(
            {**cfg, "initial_control_repeat": 3}, problem, {"baseline_sha256": "base"}
        )
    with pytest.raises(ValueError):
        prolonged_initial_state({**cfg, "upper_K": 353.3}, problem, {"baseline_sha256": "base"})
    with pytest.raises(ValueError):
        prolonged_initial_state(cfg, problem, {"baseline_sha256": "other"})
