"""Corrected pilots retain failures and cannot stand in for final comparisons."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from deflation_example.coupled_ablations import derivative_identity
from deflation_example.coupled_corrected_report import (
    audit_record,
    field_agreement,
    pilot_configuration,
    summarize,
    physical_temperature_scale,
)
from deflation_example.reporting import file_sha256
import deflation_example.coupled_corrected_report as module
from test_coupled_report import record

SOURCE = "a" * 40
DESIGN = json.loads(
    (
        Path(__file__).parents[1] / "examples/coupled_optimization/corrected_study/protocol.json"
    ).read_text()
)


def pilot(case):
    saved = record(case["method"], 0, 3 if case["method"] == "jacobi" else 4)
    saved["schema"] = "coupled-sequence-stage-v1"
    saved["configuration"] = pilot_configuration(case)
    saved["environment"].update(git_head=SOURCE, source_tree_clean=True)
    saved["baseline_sha256"] = DESIGN["baseline_sha256"]
    parameter_file = Path(module.__file__).parent / "data/transformer_2d/parameters.json"
    saved["input_sha256"] = {"parameters.json": file_sha256(parameter_file)}
    saved["state_dofs_per_problem"] = 3
    saved["stage"] = {"positions": [0], "restore": None, "resume": None}
    saved["cases"][0]["equations"][0]["momentum_relative_residual"] = 1e-12
    saved["cases"][0].update(objective=1.0, inner_iterations=0, history=[])
    return saved


def test_running_record_can_precede_baseline_metadata_but_cannot_verify():
    case = DESIGN["pilots"][0]
    saved = pilot(case)
    saved["status"] = "running"
    saved.pop("baseline_sha256")
    row = audit_record(saved, case, DESIGN, SOURCE)
    assert row == {"status": "running", "optimization_verified": False}
    saved["status"] = "complete"
    with pytest.raises(ValueError, match="baseline"):
        audit_record(saved, case, DESIGN, SOURCE)


def test_temperature_units_require_the_actual_model_checksum():
    saved = pilot(DESIGN["pilots"][0])
    assert physical_temperature_scale(saved) == 20.0
    saved["input_sha256"]["parameters.json"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        physical_temperature_scale(saved)


def test_strict_criteria_and_formulation_are_checked_independently_of_labels():
    case = DESIGN["pilots"][0]
    saved = pilot(case)
    assert audit_record(saved, case, DESIGN, SOURCE)["optimization_verified"]
    bad = deepcopy(saved)
    bad["cases"][0]["equations"][0]["momentum_relative_residual"] = 2e-12
    with pytest.raises(ValueError, match="accuracy"):
        audit_record(bad, case, DESIGN, SOURCE)
    for key, value in (
        ("consistent_stabilization", False),
        ("temperature_margin_K", 0.01),
        ("alpha", 1e-12),
        ("rank", 100),
    ):
        bad = deepcopy(saved)
        bad["configuration"][key] = value
        with pytest.raises(ValueError, match="Configuration"):
            audit_record(bad, case, DESIGN, SOURCE)


def test_derivative_gate_cannot_cross_thermal_formulations():
    cfg = pilot_configuration(DESIGN["pilots"][0])
    other = {**cfg, "consistent_stabilization": False}
    assert derivative_identity(cfg) != derivative_identity(other)
    assert derivative_identity(cfg) != derivative_identity({**cfg, "transport_form": "skew"})


def test_failed_pilot_is_retained_without_a_verified_timing():
    case = DESIGN["pilots"][0]
    saved = pilot(case)
    saved["cases"][0].update(status="nonlinear_iteration_cap", verified=False)
    saved.update(verified_problems=0, all_problems_verified=False)
    result = audit_record(saved, case, DESIGN, SOURCE)
    assert not result["optimization_verified"]
    assert result["status"] == "nonlinear_iteration_cap"
    assert result["sequence_seconds"] == 3


def test_resumed_stage_requires_attempt_accounting():
    case = DESIGN["pilots"][0]
    saved = pilot(case)
    saved["stage"]["resume"] = {"prior_seconds": 1}
    with pytest.raises(ValueError, match="attempt-cost"):
        audit_record(saved, case, DESIGN, SOURCE)


def test_field_comparison_checks_targets_finiteness_shapes_and_physical_units():
    first = {key: np.ones(3) for key in ("state", "control", "desired")}
    second = deepcopy(first)
    second["state"] += 0.01
    assert field_agreement(first, second, temperature_scale=20) == pytest.approx(0.2)
    second["desired"][0] += 1
    with pytest.raises(ValueError, match="target"):
        field_agreement(first, second, temperature_scale=20)


def test_all_cases_and_a_ratio_below_one_remain_in_the_summary(tmp_path):
    paths = []
    for case in DESIGN["pilots"]:
        path = tmp_path / str(case["case"])
        path.mkdir()
        (path / "record.json").write_text(json.dumps(pilot(case)))
        np.savez(path / "target-00.npz", state=np.ones(3), control=np.ones(3), desired=np.ones(3))
        paths.append(path)
    report = summarize(paths, DESIGN, SOURCE)
    assert len(report["rows"]) == 4
    assert all(row["optimization_verified"] for row in report["rows"])
    assert all(row["paired_complete_cost_ratio"] == 0.75 for row in report["pairs"])
    assert not report["submission_ready"]
    assert not any(row["resolution_and_feasibility_qualified"] for row in report["pairs"])
    (paths[0] / "target-00.npz").unlink()
    result = summarize(paths, DESIGN, SOURCE)
    assert result["pairs"][0]["paired_complete_cost_ratio"] is None
    assert str(tmp_path) not in result["pairs"][0]["comparison_error"]


def test_missing_cases_are_explicit_and_duplicate_inputs_rejected(tmp_path):
    paths = [tmp_path / str(k) for k in range(4)]
    result = summarize(paths, DESIGN, SOURCE)
    assert [row["status"] for row in result["rows"]] == ["missing"] * 4
    assert all(row["paired_complete_cost_ratio"] is None for row in result["pairs"])
    with pytest.raises(ValueError, match="distinct"):
        summarize([paths[0]] * 4, DESIGN, SOURCE)
