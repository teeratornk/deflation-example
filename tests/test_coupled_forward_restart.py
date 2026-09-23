import json
import sys

import numpy as np
import pytest

import deflation_example.coupled_newton_replay as replay
from test_coupled_derivatives import small_coupled_problem


def test_interrupted_forward_run_restarts_only_from_verified_fields(tmp_path, monkeypatch):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True, consistent=True)
    state = np.linspace(0.04, 0.05, problem.size)
    evaluation = problem.evaluate(state)
    cfg = {
        "transient": True,
        "slabs": 2,
        "horizon_s": 0.55,
        "query": 0,
        "target_count": 16,
        "lower_K": -10.0,
        "upper_K": 10.0,
    }
    fields = {"state": state, "control": evaluation.control, "desired": state}
    monkeypatch.setattr(
        replay, "load_saved_solution", lambda *args: ({}, cfg, fields, "control-hash")
    )
    monkeypatch.setattr(
        replay, "load_problem", lambda *args: (problem, {"baseline_sha256": "baseline-hash"})
    )
    monkeypatch.setattr(replay, "require_matching_baseline", lambda *args: None)
    monkeypatch.setattr(replay, "desired_temperature", lambda *args: state)

    def run(directory, resume=None):
        args = [
            "replay",
            "--baseline",
            str(tmp_path),
            "--optimization",
            str(tmp_path),
            "--output",
            str(directory),
            "--consistent-stabilization",
            "--cap",
            "40",
            "--line-search",
            "fixed_scaled",
            "--threads",
            "1",
        ]
        if resume is not None:
            args += ["--resume-from", str(resume)]
        monkeypatch.setattr(sys, "argv", args)
        replay.main()

    original = replay.newton_step

    def interrupted(problem, source, previous, flow, slab, **kwargs):
        if slab == 1:
            raise KeyboardInterrupt("simulated scheduler interruption")
        return original(problem, source, previous, flow, slab, **kwargs)

    interrupted_path = tmp_path / "interrupted"
    monkeypatch.setattr(replay, "newton_step", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(interrupted_path)
    assert not (interrupted_path / "states.npz").exists()
    assert (interrupted_path / "checkpoints" / "step-00000.npz").exists()
    monkeypatch.setattr(replay, "newton_step", original)
    run(tmp_path / "resumed", interrupted_path)
    run(tmp_path / "uninterrupted")
    with (
        np.load(tmp_path / "resumed" / "states.npz") as resumed,
        np.load(tmp_path / "uninterrupted" / "states.npz") as fresh,
    ):
        for field in resumed.files:
            np.testing.assert_allclose(resumed[field], fresh[field], rtol=0, atol=1e-12)
    report = json.loads((tmp_path / "resumed" / "record.json").read_text())
    assert report["resumed"] and report["restored_steps"] == 1
    assert report["status"] == "converged"
    assert report["restart_verification"]["thermal_relative_residual"] <= 1e-12
