"""Presentation tables come only from assembled, verified records."""

from copy import deepcopy
import json

from omegaconf import OmegaConf
import pytest

from deflation_example import coupled_campaign as campaign
from deflation_example import coupled_presentation as presentation
from deflation_example.coupled_report import summarize
from deflation_example.reporting import write_report
from test_coupled_campaign import base_config, prepared, run_stage
from test_coupled_derivatives import small_coupled_problem


def chain(problem, method, root):
    first = run_stage(problem, method, root / "stage-0", positions=[0])
    assert first["all_problems_verified"]
    second = run_stage(problem, method, root / "stage-1", positions=[1], restore=root / "stage-0")
    assert second["all_problems_verified"]
    return campaign.assemble([root / "stage-0", root / "stage-1"])


def test_presentation_reports_split_ratios_and_screen(monkeypatch, tmp_path):
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    prepared(monkeypatch, problem)
    records = [chain(problem, method, tmp_path / method) for method in ("jacobi", "reference")]
    summary_dir = tmp_path / "summary"
    (summary_dir / "assembled").mkdir(parents=True)
    for record in records:
        method = record["configuration"]["method"]
        target = summary_dir / "assembled" / f"{method}-rep-0"
        target.mkdir()
        write_report(target / "record.json", record)
    report = summarize(records, repetitions=1, population=["jacobi", "reference"])
    assert report["all_declared_sequences_verified"]
    write_report(
        summary_dir / "summary.json",
        {"schema": "coupled-campaign-report-v1", "chains": [], "summary": report},
    )
    screen = sequence_screen(monkeypatch, problem, tmp_path)
    built = presentation.build(summary_dir, tmp_path / "tables", [screen], plots=True)
    assert built["schema"] == presentation.SCHEMA
    ratios = built["ratios"]
    assert ratios["fastest_tested_alternative"] == "jacobi"
    assert ratios["wall_ratio"] == pytest.approx(
        report["fastest_tested_alternative_over_reference"]
    )
    assert ratios["cg_iteration_ratio"] > 0 and ratios["inner_solve_ratio"] > 0
    for method in ("jacobi", "reference"):
        row = built["methods"][method]
        assert row["verified"] == 1 and 0 < row["median_linear_fraction"] < 1
        assert row["median_cg_iterations"] > 0 and row["median_outer_iterations"] > 0
        assert len(row["per_position_median_seconds"]) == 2
        assert row["restart_counts"] == [1]
    assert built["methods"]["recycling"]["verified"] == 0
    assert built["screen"][0]["method"] == "jacobi" and built["screen"][0]["verified"]
    for name in (
        "complete_rows.tex",
        "split_rows.tex",
        "ratio_rows.tex",
        "accuracy_rows.tex",
        "memory_rows.tex",
        "execution_rows.tex",
        "screen_rows.tex",
        "target_rows.tex",
    ):
        text = (tmp_path / "tables" / name).read_text()
        assert text.strip() and text.count(r"\\") >= 1
    macros = (tmp_path / "tables" / "macros.tex").read_text()
    assert f"\\newcommand{{\\coupledWallRatio}}{{{ratios['wall_ratio']:.2f}}}" in macros
    assert "\\newcommand{\\coupledReferenceVerified}{1}" in macros
    assert "\\newcommand{\\coupledRecyclingMedianHours}{---}" in macros
    assert (tmp_path / "tables" / "cumulative.pdf").is_file()
    # The mechanism figure: one point per outer iterate, both methods on one axis.
    assert (tmp_path / "tables" / "iterations.pdf").is_file()
    series = presentation.iterate_series(
        [r for r in records if r["configuration"]["method"] == "reference"]
    )
    zero = presentation.iterate_series(
        [r for r in records if r["configuration"]["method"] == "jacobi"]
    )
    assert series and zero and len(series) == len(zero)
    assert all(count > 0 for count in series + zero)
    # An unverified sequence never enters the median.
    unverified = deepcopy(records[0])
    unverified["all_problems_verified"] = False
    assert presentation.iterate_series([unverified]) == []
    stored = json.loads((tmp_path / "tables" / "summary.json").read_text())
    assert "generator_environment" in stored
    with pytest.raises(FileExistsError):
        presentation.build(summary_dir, tmp_path / "tables")


def sequence_screen(monkeypatch, problem, tmp_path):
    from deflation_example import coupled_sequence as sequence

    config = base_config(problem, "jacobi", tmp_path / "screen-jacobi")
    config["stage"] = {"positions": [1], "restore": None, "resume": None}
    record = sequence.run(OmegaConf.create(config))
    assert record["all_problems_verified"]
    return tmp_path / "screen-jacobi"


def test_regime_artifacts_ride_along_without_entering_the_headline(monkeypatch, tmp_path):
    """The regime ablations share one manifest with the headline but never pool into it."""
    from deflation_example import coupled_regime as regime
    from test_coupled_regime import populate, tangent_design

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    prepared(monkeypatch, problem)
    records = [chain(problem, method, tmp_path / method) for method in ("jacobi", "reference")]
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    report = summarize(records, repetitions=1, population=["jacobi", "reference"])
    write_report(
        summary_dir / "summary.json",
        {"schema": "coupled-campaign-report-v1", "chains": [], "summary": report},
    )
    (summary_dir / "assembled").mkdir()
    for record in records:
        target = summary_dir / "assembled" / f"{record['configuration']['method']}-rep-0"
        target.mkdir()
        write_report(target / "record.json", record)
    runs = tmp_path / "regime-runs"
    runs.mkdir()
    populate(runs)
    design_path = tmp_path / "tangent.json"
    design_path.write_text(json.dumps(tangent_design()))
    regime_output = tmp_path / "regime"
    regime.build(design_path, runs, None, None, regime_output)

    built = presentation.build(
        summary_dir, tmp_path / "tables", regime=regime_output / "regime.json"
    )
    assert built["regime"]["tangent"]["verdict"]["status"] == "supported"
    # The headline population is untouched by the ablation records.
    assert built["methods"]["jacobi"]["verified"] == 1
    assert built["ratios"]["fastest_tested_alternative"] == "jacobi"
    macros = (tmp_path / "tables" / "macros.tex").read_text()
    assert "\\newcommand{\\coupledTangentVerdict}{supported}" in macros
    assert "\\newcommand{\\coupledFrozenSpeedup}" in macros
    rows = (tmp_path / "tables" / "regime_rows.tex").read_text()
    assert rows.count(r"\\") == 2 and "Coupled" in rows and "Frozen" in rows


def test_an_interrupted_screen_attempt_is_retained_with_null_timings(tmp_path):
    """A scheduler termination leaves a record with no timers; it is an outcome, not a gap."""
    directory = tmp_path / "jacobi-r0" / "stage-1"
    directory.mkdir(parents=True)
    (directory / "record.json").write_text(
        json.dumps(
            {
                "schema": "coupled-sequence-stage-v1",
                "status": "running",
                "configuration": {
                    "method": "jacobi",
                    "rank": 0,
                    "recycle_window": 1,
                    "device": "hybrid",
                    "hybrid_coarse_device": "cuda",
                },
                "cases": [],
            }
        )
    )
    rows = presentation.load_screen([directory])
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "running" and row["verified"] is False
    assert row["sequence_seconds"] is None
    assert row["preparation_inclusive_seconds"] is None
    assert row["cg_iterations"] is None and row["outer_iterations"] is None
    # The table renders the missing numbers rather than failing on them.
    rendered = " & ".join(
        [
            presentation.LABELS[row["method"]],
            str(row["rank"]),
            f"{row['device']} ({row['coarse_device']} coarse)",
            row["status"],
            presentation._fmt(row["preparation_inclusive_seconds"], ".0f"),
            presentation._fmt(row["cg_iterations"], ".0f"),
        ]
    )
    assert rendered.count("---") == 2


def spatial_record(directory, slabs, procedure, to_time, maximum):
    directory.mkdir()
    (directory / "record.json").write_text(
        json.dumps(
            {
                "schema": "coupled-spatial-pair-assessment-v1",
                "slabs": slabs,
                "forward_procedures_match": True,
                "coarse": {
                    "procedure": {"procedure": procedure},
                    "spatial_state_dofs": 10830,
                    "terminating_status": "converged",
                },
                "fine": {
                    "procedure": {"procedure": procedure},
                    "spatial_state_dofs": 42180,
                    "terminating_status": "newton_line_search_stagnation",
                },
                "assessed_interval": {
                    "levels": 7,
                    "declared_levels": slabs,
                    "to_time_s": to_time,
                    "horizon_s": 600.0,
                    "fraction_of_horizon": to_time / 600.0,
                    "covers_the_horizon": False,
                },
                "statistics": {
                    "pointwise_maximum_K": maximum,
                    "mass_weighted_space_time_rms_K": 0.34,
                    "pointwise_peak": {"time_s": to_time},
                },
                "criteria": {"pointwise_met": None, "rms_met": None},
            }
        )
    )
    return directory


def test_spatial_assessments_reach_the_tables_with_their_coverage(tmp_path):
    """The widest assessed window carries the prose number, and its coverage goes with it."""
    directories = [
        spatial_record(tmp_path / "64-newton", 64, "newton", 65.625, 6.32),
        spatial_record(tmp_path / "64-anderson5", 64, "anderson5", 46.875, 3.69),
    ]
    rows = presentation.load_spatial(directories)
    assert [row["procedure"] for row in rows] == ["anderson5", "newton"]
    empty = dict.fromkeys(
        (
            "median_sequence_seconds",
            "median_linear_fraction",
            "median_cg_iterations",
            "deployed_rank_range",
        )
    )
    methods = {
        name: {**empty, "verified": 0, "recorded": 0, "restart_counts": []}
        for name in presentation.METHODS
    }
    values = presentation.macros(
        {"methods": methods, "ratios": None, "regime": None, "spatial": rows}
    )
    # The widest window is the Newton one, so it supplies the prose numbers.
    assert values["coupledSpatialMaximumK"] == "6.32"
    assert values["coupledSpatialAssessedSeconds"] == "65.6"
    assert values["coupledSpatialAssessedPercent"] == "11"
    assert values["coupledSpatialProcedure"] == "newton"
    assert values["coupledSpatialAssessments"] == "2"
    # A directory that is not a spatial assessment is refused rather than guessed at.
    other = tmp_path / "other"
    other.mkdir()
    (other / "record.json").write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(ValueError, match="not a spatial pair assessment"):
        presentation.load_spatial([other])
