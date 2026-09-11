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
def test_replay_cli_records_and_applies_iteration_options(
    tmp_path, monkeypatch, module, relaxation, cap
):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    config = {"query": 0, "target_count": 16, "lower_K": 0, "upper_K": 1}
    fields = {"control": np.ones(problem.size), "state": np.zeros(problem.size)}
    baseline = {
        "baseline_sha256": "same-baseline",
        "configuration": {"convection_form": "advective"},
    }
    monkeypatch.setattr(
        module, "load_saved_solution", lambda *args: ({}, config, fields, "field-hash")
    )
    monkeypatch.setattr(module, "load_problem", lambda cfg: (problem, baseline))
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
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    report = json.loads((output / "record.json").read_text())
    assert report["status"] == "coupling_iteration_cap"
    assert "resolution_thresholds_met" not in report
    assert report["forward_solver"] == {
        "tolerance": 1e-8,
        "coupling_cap": cap,
        "relaxation": relaxation,
        "flow_method": "newton",
        "flow_cap": problem.flow_cap,
        "momentum_internal_tolerance": 1e-9,
        "mass_tolerance": 1e-6,
        "energy_tolerance": 1e-6,
    }
    np.testing.assert_array_equal(calls[0][0].ravel(), fields["control"])
    assert calls[0][1]["tolerance"] == 1e-8
    assert calls[0][1]["relaxation"] == relaxation
    assert calls[0][1]["coupling_cap"] == cap


@pytest.mark.parametrize("relaxation", [0, -0.1, 1.01, float("nan"), float("inf")])
def test_replay_options_reject_invalid_relaxation(relaxation):
    with pytest.raises(ValueError, match="relaxation"):
        temporal.forward_options(SimpleNamespace(relaxation=relaxation, coupling_cap=100))


def test_replay_options_reject_invalid_cap():
    with pytest.raises(ValueError, match="cap"):
        temporal.forward_options(SimpleNamespace(relaxation=0.5, coupling_cap=0))
