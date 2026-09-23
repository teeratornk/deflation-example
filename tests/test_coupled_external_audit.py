"""Separate field verification cannot silently replace inputs or failed solves."""

import json

import pytest

from deflation_example.coupled_external_audit import read_equation_audit
from deflation_example.coupled_saved import file_digest
from deflation_example.coupled_time_resolution_report import summarize
from deflation_example.reporting import write_report
from test_coupled_resolution_equation_audit import strict_case


def fixture(tmp_path, subdivision=1, status="converged"):
    directory, record = strict_case(tmp_path, subdivision, status)
    rows = []
    for n, row in enumerate(record["steps"]):
        checks = row["history"][-1].copy()
        checks.pop("procedure")
        row.pop("history")  # Pure Newton lacks the old hybrid-only history marker.
        rows.append(
            {
                "slab_zero_based": n,
                "time_s": row["time_s"],
                "recorded_status": row["status"],
                "equations": checks,
                "velocity_boundary_maximum_absolute_error": 0.0,
                "pressure_gauge_absolute_error": 0.0,
                "verified": status == "converged",
            }
        )
    write_report(directory / "record.json", record)
    path = directory / "audit.json"
    audit = {
        "schema": "coupled-saved-forward-equation-check-v1",
        "forward_formulation": None,
        "forward_source": "source",
        "baseline_sha256": "same-baseline",
        "optimization_fields_sha256": "same-source",
        "forward_record_sha256": file_digest(directory / "record.json"),
        "forward_fields_sha256": file_digest(directory / "states.npz"),
        "equation_tolerance": 1e-12,
        "conservation_tolerance": 1e-6,
        "saved_steps": len(rows),
        "declared_steps": 2 * subdivision,
        "steps": rows,
        "all_saved_steps_verified": status == "converged",
        "complete_trajectory_verified": status == "converged",
    }
    write_report(path, audit)
    return directory, record, path, audit


def test_separate_verified_fields_support_complete_report_without_rewriting_records(tmp_path):
    data = [fixture(tmp_path, n) for n in (4, 8, 16)]
    before = [file_digest(row[0] / "record.json") for row in data]
    result = summarize(
        [r[0] for r in data], 1, 0, verify_equations=True, equation_audits=[r[2] for r in data]
    )
    assert result["resolution_assessment"]["discrete_time_resolution_met"]
    assert all(r["independent_equations"]["all_steps_verified"] for r in result["rows"])
    assert before == [file_digest(row[0] / "record.json") for row in data]


@pytest.mark.parametrize(
    "key",
    [
        "forward_record_sha256",
        "forward_fields_sha256",
        "optimization_fields_sha256",
        "baseline_sha256",
        "forward_source",
        "forward_formulation",
    ],
)
def test_wrong_inputs_or_stale_independent_verification_rejected(tmp_path, key):
    directory, record, path, audit = fixture(tmp_path)
    audit[key] = "wrong"
    write_report(path, audit)
    with pytest.raises(ValueError, match="exact fields"):
        read_equation_audit(directory, record, path)


@pytest.mark.parametrize(
    "mutation", ["norm", "boundary", "gauge", "time", "count", "flag", "summary", "negative", "nan"]
)
def test_flags_do_not_override_numerical_checks(tmp_path, mutation):
    directory, record, path, audit = fixture(tmp_path)
    row = audit["steps"][0]
    if mutation in {"norm", "negative", "nan"}:
        row["equations"]["thermal_relative_residual"] = {
            "norm": 2e-12,
            "negative": -1e-15,
            "nan": float("nan"),
        }[mutation]
    elif mutation == "boundary":
        row["velocity_boundary_maximum_absolute_error"] = 2e-12
    elif mutation == "gauge":
        row["pressure_gauge_absolute_error"] = 2e-12
    elif mutation == "time":
        row["time_s"] += 0.1
    elif mutation == "count":
        audit["declared_steps"] += 1
    elif mutation == "flag":
        row["verified"] = False
    else:
        audit["complete_trajectory_verified"] = False
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        read_equation_audit(directory, record, path)


def test_failed_trajectory_stays_in_declared_population(tmp_path):
    data = [fixture(tmp_path, n) for n in (4, 8, 16)]
    data.append(fixture(tmp_path, 32, "newton_line_search_stagnation"))
    result = summarize(
        [r[0] for r in data], 1, 0, verify_equations=True, equation_audits=[r[2] for r in data]
    )
    assert len(result["rows"]) == 4
    assert result["rows"][-1]["independent_equations"]["verified_steps"] == 0
    assert result["rows"][-1]["declared_slabs"] == 64
    assert result["rows"][-1]["last_verified_time_s"] == 0
    assert result["rows"][-1]["last_recorded_time_s"] > 0
    assert not result["resolution_assessment"]["discrete_time_resolution_met"]


def test_audit_count_and_verification_mode_must_be_explicit(tmp_path):
    directory, _, path, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        summarize([directory], 1, 0, equation_audits=[path])
    with pytest.raises(ValueError):
        summarize([directory], 1, 0, verify_equations=True, equation_audits=[])


def test_resolution_figure_identifies_the_incomplete_replay(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from deflation_example.coupled_time_resolution_report import main

    data = [fixture(tmp_path, 4), fixture(tmp_path, 8, "newton_line_search_stagnation")]
    output = tmp_path / "figure"
    labels = []
    original = Axes.axvline

    def line(self, x, **kwargs):
        labels.append((x, kwargs.get("label")))
        return original(self, x, **kwargs)

    monkeypatch.setattr(Axes, "axvline", line)
    monkeypatch.setattr(
        "sys.argv",
        [
            "report",
            "--replays",
            *[str(r[0]) for r in data],
            "--equation-audits",
            *[str(r[2]) for r in data],
            "--verify-equations",
            "--temperature-scale",
            "1",
            "--initial-value",
            "0",
            "--plot",
            "--output",
            str(output),
        ],
    )
    main()
    assert labels == [(0.125, "16-step replay stopped (incomplete)")]
    assert (output / "time_resolution.pdf").stat().st_size > 0
