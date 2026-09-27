"""Capture actual trial inputs atomically, including unfinished flow attempts."""

import json

import numpy as np
import pytest

from deflation_example.coupled_trial_capture import TrialCapture, read_trial
from test_coupled_backtrack import QuadraticProblem


def event(phase="started", **kw):
    return {"phase": phase, "iteration": 0, "attempt": 0, "trial": 0, "step": 1.0, **kw}


def test_started_trial_is_reproducible_before_flow_finishes(tmp_path):
    problem = QuadraticProblem()
    base = problem.evaluate(np.zeros(4))
    writer = TrialCapture(tmp_path / "trials", "frozen-source-config")
    writer(event(), base, np.ones(4), None)
    row, arrays = read_trial(writer.directory, 0)
    assert row["outcome"] is None
    np.testing.assert_array_equal(arrays["candidate"]["state"], np.ones(4))
    np.testing.assert_array_equal(arrays["base"]["state"], np.zeros(4))
    writer(event("finished", status="flow_stagnation"), base, np.ones(4), problem.failed_flow)
    row, arrays = read_trial(writer.directory, 0)
    assert row["outcome"]["status"] == "flow_stagnation"
    np.testing.assert_array_equal(arrays["failed_flow"]["velocity"], problem.failed_flow.velocity)
    assert writer.seconds > 0


def test_base_is_shared_but_candidates_remain_distinct(tmp_path):
    base = QuadraticProblem().evaluate(np.zeros(4))
    writer = TrialCapture(tmp_path / "trials", "test")
    writer(event(), base, np.ones(4), None)
    writer(event("finished"), base, np.ones(4), None)
    writer(event(trial=1, step=0.5), base, np.full(4, 0.5), None)
    a, x = read_trial(writer.directory, 0)
    b, y = read_trial(writer.directory, 1)
    assert a["base"] == b["base"]
    assert a["candidate"] != b["candidate"]
    assert not np.array_equal(x["candidate"]["state"], y["candidate"]["state"])


def test_corrupt_archive_fails_checksum(tmp_path):
    base = QuadraticProblem().evaluate(np.zeros(4))
    writer = TrialCapture(tmp_path / "trials", "test")
    writer(event(), base, np.ones(4), None)
    entry = writer.manifest["trials"][0]["candidate"]
    (writer.directory / entry["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_trial(writer.directory, 0)


def test_repeated_start_completion_and_existing_directory_are_refused(tmp_path):
    base = QuadraticProblem().evaluate(np.zeros(4))
    writer = TrialCapture(tmp_path / "trials", "test")
    with pytest.raises(FileExistsError):
        TrialCapture(writer.directory, "test")
    with pytest.raises(ValueError):
        writer(event("finished"), base, np.ones(4), None)
    writer(event(), base, np.ones(4), None)
    with pytest.raises(ValueError):
        writer(event(), base, np.ones(4), None)
    writer(event("finished"), base, np.ones(4), None)
    with pytest.raises(ValueError):
        writer(event("finished"), base, np.ones(4), None)


def test_manifest_paths_cannot_leave_capture_directory(tmp_path):
    base = QuadraticProblem().evaluate(np.zeros(4))
    writer = TrialCapture(tmp_path / "trials", "test")
    writer(event(), base, np.ones(4), None)
    writer.manifest["trials"][0]["base"]["file"] = "../outside.npz"
    (writer.directory / "manifest.json").write_text(json.dumps(writer.manifest))
    with pytest.raises(ValueError, match="leaves"):
        read_trial(writer.directory, 0)


def test_outcome_cannot_be_attached_to_a_different_candidate(tmp_path):
    base = QuadraticProblem().evaluate(np.zeros(4))
    writer = TrialCapture(tmp_path / "trials", "test")
    writer(event(), base, np.ones(4), None)
    with pytest.raises(ValueError, match="recorded candidate"):
        writer(event("finished"), base, np.zeros(4), None)
    assert read_trial(writer.directory, 0)[0]["outcome"] is None


def test_captured_temperature_replays_on_small_coupled_problem(tmp_path):
    from deflation_example.coupled_trial_replay import replay
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    base = problem.evaluate(np.zeros(problem.size))
    candidate = np.full(problem.size, 0.03)
    writer = TrialCapture(tmp_path / "trials", "test")
    writer(event(), base, candidate, None)
    _, arrays = read_trial(writer.directory, 0)
    result = replay(problem, arrays, np.full(problem.size, 0.1))
    assert result["status"] == "flow_converged"
    np.testing.assert_array_equal(arrays["candidate"]["state"], candidate)


def test_failed_replay_keeps_termination_and_slab(tmp_path):
    from deflation_example.coupled_trial_replay import replay

    problem = QuadraticProblem(0)
    base = problem.evaluate(np.zeros(4))
    writer = TrialCapture(tmp_path / "trials", "test")
    writer(event(), base, np.ones(4), None)
    _, arrays = read_trial(writer.directory, 0)
    result = replay(problem, arrays, np.ones(4))
    assert result["status"] == "flow_stagnation" and result["slab"] == 0
