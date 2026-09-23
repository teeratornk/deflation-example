"""A resolution label requires recorded independent checks on the declared grid."""

from copy import deepcopy
import json

import numpy as np
import pytest

from deflation_example.coupled_time_resolution_report import audit_forward_equations, summarize
from deflation_example.reporting import write_report
from test_coupled_time_resolution_report import case


def strict_case(tmp_path, subdivision, status="converged"):
    directory = case(tmp_path, subdivision, status)
    filename = directory / "record.json"
    record = json.loads(filename.read_text())
    record["forward_slabs"] = 2 * subdivision
    record["configuration"]["horizon_s"] = 2.0
    checks = {
        "procedure": "returned_state_verification",
        "momentum_relative_residual": 2e-13,
        "continuity_relative_residual": 1e-15,
        "thermal_relative_residual": 3e-13,
        "mass_relative_imbalance": 1e-14,
        "energy_relative_defect": 1e-10,
    }
    for row in record["steps"]:
        row["history"] = [deepcopy(checks)]
    write_report(filename, record)
    return directory, record


def test_strict_resolution_reports_verified_counts_and_equation_maxima(tmp_path):
    paths = [strict_case(tmp_path, n)[0] for n in (4, 8, 16)]
    result = summarize(paths, 1, initial_value=0, verify_equations=True)
    assert result["independent_equation_audit"]
    assert result["resolution_assessment"]["discrete_time_resolution_met"]
    for row, n in zip(result["rows"], (8, 16, 32), strict=True):
        checks = row["independent_equations"]
        assert checks["verified_steps"] == n
        assert checks["all_steps_verified"]
        assert checks["verified_prefix_maxima"]["momentum_relative_residual"] == 2e-13


@pytest.mark.parametrize("value", [2e-12, -1e-15, np.nan])
def test_converged_label_cannot_override_invalid_equations(tmp_path, value):
    path, record = strict_case(tmp_path, 1)
    record["steps"][0]["history"][-1]["thermal_relative_residual"] = value
    (path / "record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="original equations"):
        summarize([path], 1, initial_value=0, verify_equations=True)


@pytest.mark.parametrize("key", ["mass_relative_imbalance", "energy_relative_defect"])
def test_conservation_failures_block_strict_resolution(tmp_path, key):
    _, record = strict_case(tmp_path, 1)
    record["steps"][0]["history"][-1][key] = 2e-6
    with pytest.raises(ValueError, match="original equations"):
        audit_forward_equations(record)


def test_missing_independent_checks_cannot_be_inferred_from_status(tmp_path):
    path = case(tmp_path, 1)
    with pytest.raises(ValueError, match="equation checks"):
        summarize([path], 1, initial_value=0, verify_equations=True)


@pytest.mark.parametrize("mutation", ["horizon", "count", "weak_tolerance", "procedure", "gap"])
def test_strict_audit_rejects_inconsistent_time_or_verification_protocol(tmp_path, mutation):
    _, record = strict_case(tmp_path, 2)
    if mutation == "horizon":
        record["configuration"]["horizon_s"] = 3.0
    elif mutation == "count":
        record["forward_slabs"] = 8
    elif mutation == "weak_tolerance":
        record["forward_solver"]["tolerance"] = 1e-8
    elif mutation == "procedure":
        record["steps"][0]["history"][-1]["procedure"] = "iteration_recurrence"
    else:
        record["steps"][1]["status"] = "failed"
    with pytest.raises(ValueError):
        audit_forward_equations(record)


def test_failed_and_missing_trajectories_remain_in_strict_population(tmp_path):
    paths = [strict_case(tmp_path, n)[0] for n in (4, 8, 16)]
    paths.extend((strict_case(tmp_path, 32, "hybrid_no_convergence")[0], tmp_path / "missing"))
    report = summarize(paths, 1, initial_value=0, verify_equations=True)
    assert len(report["rows"]) == 5
    assert not report["rows"][-2]["independent_equations"]["all_steps_verified"]
    assert report["rows"][-1]["status"] == "missing"
    assert not report["resolution_assessment"]["discrete_time_resolution_met"]


def test_strict_resolution_requires_initial_temperature(tmp_path):
    path, _ = strict_case(tmp_path, 1)
    with pytest.raises(ValueError, match="initial temperature"):
        summarize([path], 1, verify_equations=True)


def test_strict_cli_writes_audited_summary_and_figure(tmp_path, monkeypatch):
    from deflation_example.coupled_time_resolution_report import main

    paths = [strict_case(tmp_path, n)[0] for n in (4, 8, 16)]
    output = tmp_path / "audited"
    monkeypatch.setattr(
        "sys.argv",
        [
            "resolution",
            "--replays",
            *map(str, paths),
            "--temperature-scale",
            "1",
            "--initial-value",
            "0",
            "--verify-equations",
            "--plot",
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads((output / "summary.json").read_text())
    assert report["independent_equation_audit"]
    assert report["resolution_assessment"]["discrete_time_resolution_met"]
    assert (output / "time_resolution.pdf").stat().st_size > 0
