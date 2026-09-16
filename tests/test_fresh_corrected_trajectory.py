"""Fresh and resumed corrected trajectories preserve inputs and accuracy."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_coupled_newton_replay import data
from deflation_example.reporting import write_arrays, write_report
from deflation_example.coupled_saved import file_digest

directory = Path(__file__).parents[1] / "examples/coupled_optimization/fixed_point"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("fresh_trajectory", directory / "fresh_trajectory.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.pop(0)


class Sampler:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def finish(self):
        return {"test_sampler": True}


@pytest.fixture
def case(tmp_path, monkeypatch):
    problem, state, evaluation, _ = data()
    controls = evaluation.control.reshape(problem.slabs, -1).copy()
    metadata = {
        "schema": "input-schema",
        "environment": {"source_sha256": "source-a"},
        "configuration": {"query": 0, "target_count": 1, "lower_K": 0, "upper_K": 400},
        "policy": "newton",
        "family": "forward",
        "threads": 1,
        "repetition": 0,
        "subdivision": 1,
        "optimization_field_sha256": "control-a",
        "baseline_sha256": "mesh-a",
        "fine_baseline_sha256": "mesh-a",
        "slabs": problem.slabs,
        "spatial_state_dofs": problem.spatial_size,
    }
    monkeypatch.setattr(module, "prepare", lambda *args: (problem, controls, metadata))
    monkeypatch.setattr(module, "ProcessMemory", Sampler)
    monkeypatch.setattr(module, "desired_temperature", lambda *args: state.copy())
    args = SimpleNamespace(output=tmp_path / "trajectory", resume=False, subdivision=1)
    return args, problem, controls, state, metadata


def test_fresh_trajectory_checks_all_steps_and_preserves_source(case):
    args, problem, controls, expected, _ = case
    original = controls.copy()
    record = module.run(args)
    assert record["status"] == "converged"
    assert len(record["steps"]) == problem.slabs
    assert all(row["verified"] for row in record["steps"])
    np.testing.assert_array_equal(original, controls)
    for n, row in enumerate(record["steps"]):
        full, _ = module.checked_field(args.output, row, problem)
        np.testing.assert_allclose(full[problem.free], expected.reshape(2, -1)[n], atol=2e-10)
        assert row["seconds"] >= row["nonlinear_solve_seconds"] >= 0
    assert record["quantities"]["weighted_tracking_squared_K2"] < 1e-14
    assert record["all_attempt_process_seconds"] >= record["sum_step_seconds"]
    assert json.loads((args.output / "record.json").read_text())["status"] == "converged"


def interrupted(case):
    args = case[0]
    record = module.run(args)
    record["steps"] = record["steps"][:1]
    record["status"] = "running"
    record.pop("quantities")
    # A killed process has no exact elapsed time, which must remain unknown.
    record["attempts"][-1].pop("process_seconds")
    write_report(args.output / "record.json", record)
    args.resume = True
    return record


def test_restart_rechecks_prefix_and_keeps_unknown_interrupted_cost(case):
    old = interrupted(case)
    result = module.run(case[0])
    assert result["status"] == "converged"
    assert len(result["steps"]) == 2
    assert result["steps"][0] == old["steps"][0]
    assert len(result["attempts"]) == 2
    assert result["all_attempt_process_seconds"] is None


@pytest.mark.parametrize("mutation", ["driver", "source", "time", "field", "equations", "settings"])
def test_bad_restart_cannot_replace_manifest(case, mutation):
    record = interrupted(case)
    args = case[0]
    if mutation == "driver":
        record["driver_sha256"] = "different"
    elif mutation == "source":
        record["environment"] = {"source_sha256": "different"}
    elif mutation == "time":
        record["steps"][0]["time_s"] += 1
    elif mutation == "settings":
        record["settings"] = {**record["settings"], "equation_tolerance": 1e-8}
    else:
        path = args.output / record["steps"][0]["fields"]
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        arrays["state"] += 0.01
        write_arrays(path, **arrays)
        if mutation == "equations":
            record["steps"][0]["field_sha256"] = file_digest(path)
    write_report(args.output / "record.json", record)
    before = (args.output / "record.json").read_bytes()
    with pytest.raises(ValueError):
        module.run(args)
    assert (args.output / "record.json").read_bytes() == before


def test_terminal_failure_cannot_resume(case):
    record = interrupted(case)
    record["status"] = "numerical_failure"
    write_report(case[0].output / "record.json", record)
    with pytest.raises(ValueError, match="interrupted"):
        module.run(case[0])


def test_new_run_cannot_overwrite_existing_directory(case):
    module.run(case[0])
    with pytest.raises(FileExistsError):
        module.run(case[0])


def test_failed_step_stays_visible_and_has_no_trajectory_summary(case, monkeypatch):
    advance = module.advance

    def failed(*args):
        row, arrays = advance(*args)
        row.update(verified=False, status="test_iteration_cap")
        return row, arrays

    monkeypatch.setattr(module, "advance", failed)
    record = module.run(case[0])
    assert record["status"] == "numerical_failure"
    assert len(record["steps"]) == 1
    assert "quantities" not in record
    assert (case[0].output / record["steps"][0]["fields"]).is_file()


def test_input_failure_has_explicit_record(case, monkeypatch):
    def fail(*args):
        raise ValueError("invalid inputs")

    monkeypatch.setattr(module, "prepare", fail)
    with pytest.raises(ValueError, match="invalid inputs"):
        module.run(case[0])
    record = json.loads((case[0].output / "record.json").read_text())
    assert record["status"] == "execution_failure"
    assert not record["steps"]
