"""Do not pool unmatched local systems or suppress declared failed methods."""

import importlib.util
from pathlib import Path

import pytest

from deflation_example.reporting import write_report

path = Path(__file__).parents[1] / "examples/coupled_optimization/fixed_point/report_step_checks.py"
spec = importlib.util.spec_from_file_location("report_step_checks", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def records(tmp_path):
    directories = []
    for policy in module.LABELS:
        directory = tmp_path / policy
        directory.mkdir()
        write_report(
            directory / "record.json",
            {
                "policy": policy,
                "status": "complete",
                "termination": "coupling_iteration_cap",
                "verified": False,
                "trajectory_record_sha256": "trajectory",
                "initial_field_sha256": "field",
                "slab_zero_based": 58,
                "time_s": 276.5625,
                "accuracy": {"equation": 1e-12, "mass": 1e-6, "energy": 1e-6},
                "environment": {"git_head": "source"},
                "history": [],
            },
        )
        directories.append(directory)
    return directories


def test_failed_outcomes_are_retained(tmp_path):
    report = module.load_comparison(records(tmp_path))
    assert len(report["rows"]) == 6
    assert all(not row["verified"] for row in report["rows"])


def test_missing_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="six"):
        module.load_comparison(records(tmp_path)[:-1])


def test_mismatched_initial_field_is_rejected(tmp_path):
    import json

    directories = records(tmp_path)
    path = directories[-1] / "record.json"
    record = json.loads(path.read_text())
    record["initial_field_sha256"] = "different"
    write_report(path, record)
    with pytest.raises(ValueError, match="different inputs"):
        module.load_comparison(directories)


def test_false_success_is_rejected(tmp_path):
    import json

    directories = records(tmp_path)
    path = directories[0] / "record.json"
    record = json.loads(path.read_text())
    record.update(verified=True, termination="converged")
    write_report(path, record)
    with pytest.raises(ValueError, match="Successful label"):
        module.load_comparison(directories)


def test_failed_methods_still_appear_in_figure(tmp_path):
    report = module.load_comparison(records(tmp_path))
    module.plot(report, tmp_path)
    assert (tmp_path / "local-correction.png").stat().st_size > 1000
