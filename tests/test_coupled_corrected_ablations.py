"""The extra comparisons preserve the frozen problem, source and all outcomes."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from deflation_example import coupled_corrected_ablations as ablations
from deflation_example.coupled_corrected_report import pilot_configuration
from test_coupled_corrected_report import pilot

DESIGN_PATH = (
    Path(__file__).parents[1] / "examples/coupled_optimization/corrected_study/ablations.json"
)


def test_each_case_changes_only_the_declared_factor():
    design, base = ablations.read_design(DESIGN_PATH)
    for case in design["cases"]:
        reference = base["pilots"][1 if case["alpha"] == 1e-14 else 3]
        first, second = pilot_configuration(reference), pilot_configuration(case)
        changes = {key for key in first if first[key] != second[key]}
        expected = {"method"} if case["variant"] == "recycling" else set(case["overrides"])
        assert changes == expected
        assert second["flow_tolerance"] == second["equation_acceptance_tolerance"] == 1e-12
        assert second["rank"] == second["recycle_window"] == 200


def write_population(tmp_path):
    design, base = ablations.read_design(DESIGN_PATH)
    paths = []
    for index, case in enumerate([*base["pilots"], *design["cases"]]):
        path = tmp_path / str(index)
        path.mkdir()
        record = pilot(case)
        record["environment"]["git_head"] = design["numerical_source"]
        (path / "record.json").write_text(json.dumps(record))
        np.savez(path / "target-00.npz", state=np.ones(3), control=np.ones(3), desired=np.ones(3))
        paths.append(path)
    return paths, design, base


def test_summary_keeps_all_ten_outcomes_and_both_ratio_directions(tmp_path):
    paths, design, base = write_population(tmp_path)
    record_path = paths[4] / "record.json"
    record = json.loads(record_path.read_text())
    record.update(sequence_seconds=2.0, preparation_inclusive_seconds=3.5)
    record["components_seconds"]["queries"] = 1.7
    record_path.write_text(json.dumps(record))
    report = ablations.summarize(paths[:4], paths[4:], design, base)
    assert len(report["rows"]) == 10
    assert len(report["pairs"]) == 8
    assert all(row["optimization_verified"] for row in report["rows"])
    pair = next(
        row for row in report["pairs"] if row["variant"] == "recycling" and row["alpha"] == 1e-14
    )
    assert pair["variant_seconds_over_full_reference_seconds"] == 0.5
    assert not report["submission_ready"]


@pytest.mark.parametrize(
    "failure", ["failed", "different_state", "wrong_configuration", "missing", "different_device"]
)
def test_unqualified_pair_has_no_timing_ratio(tmp_path, failure):
    paths, design, base = write_population(tmp_path)
    path = paths[5] / "record.json"
    record = json.loads(path.read_text())
    if failure == "failed":
        record["cases"][0].update(status="nonlinear_iteration_cap", verified=False)
        record.update(verified_problems=0, all_problems_verified=False)
    elif failure == "different_state":
        np.savez(
            paths[5] / "target-00.npz",
            state=np.ones(3) + 0.01,
            control=np.ones(3),
            desired=np.ones(3),
        )
    elif failure == "wrong_configuration":
        record["configuration"]["reference_transfer"] = "full"
    elif failure == "different_device":
        record["device"] = {"name": "other"}
    else:
        path.unlink()
    if failure != "missing":
        path.write_text(json.dumps(record))
    report = ablations.summarize(paths[:4], paths[4:], design, base)
    assert len(report["rows"]) == 10
    pair = next(
        row for row in report["pairs"] if row["variant"] == "sequential" and row["alpha"] == 1e-14
    )
    assert pair["variant_seconds_over_full_reference_seconds"] is None


def test_duplicate_and_missing_runs_are_not_silently_replaced(tmp_path):
    design, base = ablations.read_design(DESIGN_PATH)
    paths = [tmp_path / str(i) for i in range(10)]
    report = ablations.summarize(paths[:4], paths[4:], design, base)
    assert all(row["status"] == "missing" for row in report["rows"])
    with pytest.raises(ValueError, match="distinct"):
        ablations.summarize(paths[:4], [paths[0]] * 6, design, base)


@pytest.mark.parametrize("dirty", [False, True])
def test_launch_uses_measured_source_and_refuses_modified_tree(tmp_path, monkeypatch, dirty):
    design, base = ablations.read_design(DESIGN_PATH)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "record.json").write_text(json.dumps(base))
    monkeypatch.setattr(
        ablations.subprocess,
        "check_output",
        lambda cmd, **kwargs: (
            design["numerical_source"]
            if cmd[1] == "rev-parse"
            else (" M src/solver.py" if dirty else "")
        ),
    )
    args = (design, base, 1, tmp_path / "numerical", baseline, tmp_path / "run")
    if dirty:
        with pytest.raises(ValueError, match="frozen clean"):
            ablations.launch_command(*args)
    else:
        command = ablations.launch_command(*args)
        assert "reference_transfer=sequential" in command
        assert "rank=200" in command
        assert "device=hybrid" in command
        (tmp_path / "run").mkdir()
        with pytest.raises(FileExistsError):
            ablations.launch_command(*args)


def test_arbitrary_configuration_changes_are_rejected():
    design, _ = ablations.read_design(DESIGN_PATH)
    case = deepcopy(design["cases"][0])
    case["overrides"] = {"inner_tolerance": 0.1}
    with pytest.raises(ValueError, match="Only the declared"):
        pilot_configuration(case)
