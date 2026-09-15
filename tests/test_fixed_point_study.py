"""Study serialization, input identity and complete-trajectory checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example import fixed_point_study as study
from deflation_example.reporting import write_report
from test_coupled_derivatives import small_coupled_problem
from test_coupled_campaign import Sampler


def identity():
    return {
        "environment": {"source_sha256": {"solver": "source-a"}},
        "optimization_field_sha256": "source-control",
        "baseline_sha256": "baseline",
        "fine_baseline_sha256": "baseline",
        "policy": "anderson3",
        "subdivision": 1,
        "family": "forward",
        "threads": 1,
        "repetition": 0,
        "time_scheme": "backward_euler",
        "steps": [],
        "configuration": {
            "query": 7,
            "target_count": 16,
            "target_startup_s": 0,
            "lower_K": 290,
            "upper_K": 350,
        },
    }


def test_step_commit_survives_interruption(tmp_path, monkeypatch):
    meta = identity()
    write_report(tmp_path / "record.json", meta)
    meta = study.commit_step(
        tmp_path, meta, {"slab_zero_based": 0, "verified": True}, {"state": np.ones(4)}
    )
    original = study.write_report

    def fail(path, value):
        if path.name == "record.json":
            raise RuntimeError("manifest interrupted")
        original(path, value)

    monkeypatch.setattr(study, "write_report", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        study.commit_step(
            tmp_path, meta, {"slab_zero_based": 1, "verified": True}, {"state": np.ones(4) * 2}
        )
    retained = json.loads((tmp_path / "record.json").read_text())
    assert len(retained["steps"]) == 1
    study.verify_restart(retained, identity(), tmp_path)
    altered = identity()
    altered["environment"]["source_sha256"] = {"solver": "source-b"}
    with pytest.raises(ValueError, match="source"):
        study.verify_restart(retained, altered, tmp_path)


@pytest.mark.parametrize("policy", ["anderson3", "newton"])
def test_complete_small_trajectory_matches_recovered_control(tmp_path, monkeypatch, policy):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    state = np.full(problem.size, 0.01)
    controls = problem.evaluate(state).control.reshape(problem.slabs, -1)
    meta = identity()
    meta["policy"] = policy
    monkeypatch.setattr(study, "prepare", lambda args, subdivision: (problem, controls, meta))
    monkeypatch.setattr(study, "desired_temperature", lambda *a: state)
    monkeypatch.setattr(study, "ProcessMemory", lambda **kw: Sampler())
    args = SimpleNamespace(
        family="forward", subdivision=1, output=tmp_path / "run", resume=False, policy=policy
    )
    study.trajectory(args)
    record = json.loads((args.output / "record.json").read_text())
    assert record["status"] == "converged", record["steps"][-1]
    assert len(record["steps"]) == 2
    with np.load(args.output / "states.npz") as data:
        np.testing.assert_allclose(data["state"].ravel(), state, atol=1e-9)
    assert record["all_attempt_process_seconds"] >= record["sum_step_seconds"]
