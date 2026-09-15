"""Replay commands preserve source fields and expose their forward protocol."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from deflation_example import coupled_resolution as temporal
from deflation_example import coupled_spatial_resolution as spatial
from test_coupled_derivatives import small_coupled_problem


@pytest.mark.parametrize("module", [temporal, spatial])
@pytest.mark.parametrize("relaxation,cap", [(0.5, 100), (0.25, 160)])
@pytest.mark.parametrize("tolerance", [None, 1e-10])
def test_replay_cli_records_and_applies_iteration_options(
    tmp_path, monkeypatch, module, relaxation, cap, tolerance
):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    config = {
        "query": 0,
        "target_count": 16,
        "lower_K": 0,
        "upper_K": 1,
        "transient": True,
        "slabs": problem.slabs,
    }
    fields = {"control": np.ones(problem.size), "state": np.zeros(problem.size)}
    baseline = {
        "baseline_sha256": "same-baseline",
        "configuration": {"convection_form": "advective"},
    }
    monkeypatch.setattr(
        module, "load_saved_solution", lambda *args: ({}, config, fields, "field-hash")
    )

    def load(cfg):
        assert cfg["transient"] is True
        assert cfg["slabs"] == problem.slabs
        return problem, baseline

    monkeypatch.setattr(module, "load_problem", load)
    monkeypatch.setattr(module, "require_matching_baseline", lambda *args: None)
    monkeypatch.setattr(module, "environment", lambda: {"test": True})
    if module is spatial:
        monkeypatch.setattr(
            module,
            "transfer_source",
            lambda coarse, fine, source: (
                source.reshape(problem.slabs, -1),
                sparse.eye(problem.spatial_size),
            ),
        )
    calls = []

    def replay(problem_arg, source, *args, **kwargs):
        calls.append((np.asarray(source).copy(), kwargs))
        return {
            "status": "coupling_iteration_cap",
            "states": np.zeros((1, problem.spatial_size)),
            "steps": [{"time_s": 0.2, "status": "coupling_iteration_cap"}],
            "seconds": 1.0,
        }

    monkeypatch.setattr(module, "replay_controls", replay)
    output = tmp_path / "replay"
    arguments = [
        "replay",
        "--baseline",
        "baseline",
        "--optimization",
        "optimization",
        "--output",
        str(output),
        "--relaxation",
        str(relaxation),
        "--coupling-cap",
        str(cap),
    ]
    if module is spatial:
        arguments.extend(["--fine-baseline", "fine-baseline"])
    if tolerance is not None:
        arguments.extend(["--tolerance", str(tolerance)])
    target = 1e-8 if tolerance is None else tolerance
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    report = json.loads((output / "record.json").read_text())
    assert report["status"] == "coupling_iteration_cap"
    assert "resolution_thresholds_met" not in report
    assert report["forward_solver"] == {
        "tolerance": target,
        "coupling_cap": cap,
        "relaxation": relaxation,
        "flow_method": "newton",
        "flow_cap": problem.flow_cap,
        "momentum_internal_tolerance": target * 0.1,
        "mass_tolerance": 1e-6,
        "energy_tolerance": 1e-6,
    }
    np.testing.assert_array_equal(calls[0][0].ravel(), fields["control"])
    assert calls[0][1]["tolerance"] == target
    assert calls[0][1]["relaxation"] == relaxation
    assert calls[0][1]["coupling_cap"] == cap


@pytest.mark.parametrize("relaxation", [0, -0.1, 1.01, float("nan"), float("inf")])
def test_replay_options_reject_invalid_relaxation(relaxation):
    with pytest.raises(ValueError, match="relaxation"):
        temporal.forward_options(SimpleNamespace(relaxation=relaxation, coupling_cap=100))


def test_replay_options_reject_invalid_cap():
    with pytest.raises(ValueError, match="cap"):
        temporal.forward_options(SimpleNamespace(relaxation=0.5, coupling_cap=0))


@pytest.mark.parametrize("tolerance", [0, -1e-10, 1e-7, float("nan"), float("inf")])
def test_replay_options_only_allow_finite_positive_tighter_targets(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        temporal.forward_options(
            SimpleNamespace(relaxation=0.5, coupling_cap=100, tolerance=tolerance)
        )


def test_spatial_newton_cli_matches_bdf2_and_restart_policy(tmp_path, monkeypatch):
    problem = small_coupled_problem([0.1] * 4, uniform_capacity=True)
    config = {"query": 0, "target_count": 16, "transient": True, "slabs": 2}
    controls = np.linspace(-1, 2, 2 * problem.spatial_size)
    fields = {"control": controls, "state": np.zeros_like(controls)}
    baseline = {
        "baseline_sha256": "baseline",
        "configuration": {"convection_form": "advective"},
    }
    monkeypatch.setattr(spatial, "load_saved_solution", lambda *args: ({}, config, fields, "hash"))
    monkeypatch.setattr(spatial, "load_problem", lambda cfg: (problem, baseline))
    monkeypatch.setattr(spatial, "require_matching_baseline", lambda *args: None)
    monkeypatch.setattr(spatial, "environment", lambda: {"test": True})
    monkeypatch.setattr(
        spatial, "transfer_source", lambda c, f, u: (u, sparse.eye(problem.spatial_size))
    )

    def comparator(directory, p, baseline_digest, source_digest, cfg, **kwargs):
        assert kwargs == {"time_scheme": "bdf2"}
        assert baseline_digest == "baseline" and source_digest == "hash"
        return np.zeros((4, problem.spatial_size)), {"test": True}

    monkeypatch.setattr(spatial, "coarse_replay_states", comparator)

    def replay(p, source, **kwargs):
        np.testing.assert_array_equal(source, np.repeat(controls.reshape(2, -1), 2, axis=0))
        assert kwargs["time_scheme"] == "bdf2"
        assert kwargs["restart_interval"] == 2
        assert kwargs["line_search"] == "fixed_scaled"
        assert kwargs["backtrack_cap"] == 40
        assert kwargs["tolerance"] == 1e-12
        return {
            "status": "newton_iteration_cap",
            "states": np.zeros((1, p.spatial_size)),
            "steps": [{"time_s": 0.1, "status": "newton_iteration_cap"}],
            "seconds": 1.0,
        }

    monkeypatch.setattr(spatial, "newton_trajectory", replay)
    output = tmp_path / "replay"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spatial",
            "--baseline",
            "baseline",
            "--fine-baseline",
            "fine",
            "--optimization",
            "optimization",
            "--coarse-replay",
            "coarse",
            "--subdivision",
            "2",
            "--procedure",
            "monolithic_newton",
            "--time-scheme",
            "bdf2",
            "--line-search",
            "fixed_scaled",
            "--backtrack-cap",
            "40",
            "--tolerance",
            "1e-12",
            "--output",
            str(output),
        ],
    )
    spatial.main()
    record = json.loads((output / "record.json").read_text())
    assert record["status"] == "newton_iteration_cap"
    assert record["forward_solver"]["time_scheme"] == "bdf2"
    assert "resolution_thresholds_met" not in record
