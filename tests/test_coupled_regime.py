"""The regime analysis applies the declared rules and never invents a verdict."""

import json

import pytest

from deflation_example.coupled_regime import (
    build,
    macros,
    pair,
    scaling_verdict,
    tangent_analysis,
    tangent_verdict,
)


def quadratic(cg, seconds):
    return {
        "linear_iterations": cg,
        "timing": {"total_seconds": seconds, "components_seconds": {"iteration": seconds}},
    }


def stage_record(
    tangent, method, rank, outer, cg_each, seconds_each, total, objective=1.0, slabs=64
):
    history = [
        {
            "iteration": index,
            "objective": objective,
            "kkt": {"stationarity": 1e-9},
            "evaluation_seconds": 1.0,
            "attempts": [{"qp_history": [quadratic(cg_each, seconds_each)], "trials": []}],
        }
        for index in range(outer)
    ]
    return {
        "schema": "coupled-sequence-stage-v1",
        "status": "complete",
        "sequence_seconds": total,
        "configuration": {"method": method, "rank": rank, "tangent": tangent, "slabs": slabs},
        "cases": [
            {
                "position": 0,
                "status": "converged",
                "verified": True,
                "objective": objective,
                "kkt": {
                    "primal_absolute": 0.0,
                    "stationarity": 1e-9,
                    "dual_feasibility": 0.0,
                    "lower_complementarity": 0.0,
                    "upper_complementarity": 0.0,
                },
                "history": history,
            }
        ],
    }


def write(root, name, record):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "record.json").write_text(json.dumps(record))
    return directory


def test_pair_computes_the_declared_quantities():
    zero = stage_record(
        "coupled", "jacobi", 0, outer=10, cg_each=100, seconds_each=100.0, total=1200.0
    )
    deflated = stage_record(
        "coupled", "reference", 100, outer=10, cg_each=50, seconds_each=55.0, total=700.0
    )
    from deflation_example.coupled_presentation import linear_split

    first = {**linear_split(zero), "outer_iterations": 10, "objective": 1.0}
    second = {**linear_split(deflated), "outer_iterations": 10, "objective": 1.0}
    measured = pair(first, second)
    assert measured["S"] == pytest.approx(1200 / 700)
    assert measured["C"] == pytest.approx(2.0)
    # Per conjugate gradient iteration the deflated arm costs 1.1 times as much.
    assert measured["B"] == pytest.approx(1.1)
    assert measured["equal_outer_iterations"]


def tangent_design():
    return {
        "schema": "coupled-tangent-regime-design-v1",
        "tangents": ["coupled", "frozen"],
        "rows": [
            {"tangent": "coupled", "method": "jacobi", "rank": 0, "role": "control"},
            {"tangent": "coupled", "method": "reference", "rank": 100, "role": "control"},
            {"tangent": "frozen", "method": "jacobi", "rank": 0, "role": "arm"},
            {"tangent": "frozen", "method": "reference", "rank": 100, "role": "arm"},
        ],
    }


def populate(root, frozen_reference_seconds=1100.0, frozen_objective=1.0):
    """The regime the designs predict: deflation pays on the coupled operator, not the cheap one."""
    write(
        root,
        "coupled-jacobi-r0/rep-0",
        stage_record("coupled", "jacobi", 0, 10, 100, 200.0, 2200.0),
    )
    write(
        root,
        "coupled-reference-r100/rep-0",
        stage_record("coupled", "reference", 100, 10, 50, 110.0, 1300.0),
    )
    write(root, "frozen-jacobi-r0/rep-0", stage_record("frozen", "jacobi", 0, 12, 100, 3.4, 300.0))
    write(
        root,
        "frozen-reference-r100/rep-0",
        stage_record(
            "frozen",
            "reference",
            100,
            12,
            70,
            6.7,
            frozen_reference_seconds,
            objective=frozen_objective,
        ),
    )


def test_the_predicted_regime_is_supported_and_names_its_clauses(tmp_path):
    populate(tmp_path)
    analysis = tangent_analysis(tangent_design(), tmp_path)
    assert analysis["verdict"]["status"] == "supported"
    assert all(analysis["verdict"]["clauses"].values())
    assert analysis["pairs"]["coupled"]["S"] > 1.2
    assert analysis["pairs"]["frozen"]["S"] < 1
    # The outer-iteration inflation is what the pre-check exists to measure.
    assert analysis["outer_iteration_inflation"]["jacobi"] == pytest.approx(1.2)


def test_a_disagreeing_optimum_is_inconclusive_not_supported(tmp_path):
    populate(tmp_path, frozen_objective=1.5)
    analysis = tangent_analysis(tangent_design(), tmp_path)
    assert analysis["verdict"]["status"] == "inconclusive"
    assert "different optima" in analysis["verdict"]["reason"]


def test_deflation_still_paying_on_the_cheap_operator_is_refuted(tmp_path):
    populate(tmp_path, frozen_reference_seconds=200.0)
    analysis = tangent_analysis(tangent_design(), tmp_path)
    assert analysis["verdict"]["status"] == "refuted"


def test_an_unmeasured_arm_leaves_the_verdict_pending(tmp_path):
    write(
        tmp_path,
        "coupled-jacobi-r0/rep-0",
        stage_record("coupled", "jacobi", 0, 10, 100, 200.0, 2200.0),
    )
    analysis = tangent_analysis(tangent_design(), tmp_path)
    assert analysis["verdict"]["status"] == "pending"


def test_an_unverified_arm_is_inconclusive(tmp_path):
    populate(tmp_path)
    path = tmp_path / "frozen-reference-r100/rep-0/record.json"
    record = json.loads(path.read_text())
    record["cases"][0]["verified"] = False
    path.write_text(json.dumps(record))
    analysis = tangent_analysis(tangent_design(), tmp_path)
    assert analysis["verdict"]["status"] == "inconclusive"
    assert "did not verify" in analysis["verdict"]["reason"]


def measured(R, B, share=0.8):
    return {"R": R, "B": B, "C": 2.0, "share": {"zero": share, "deflated": share}}


def test_scaling_rules_cover_growth_stagnation_and_reversal():
    arms = {
        (64, "jacobi"): {"measured": True, "outer_iterations": 10, "objective": 1.0},
        (64, "reference"): {"measured": True, "outer_iterations": 10, "objective": 1.0},
        (128, "jacobi"): {"measured": True, "outer_iterations": 10, "objective": 2.0},
        (128, "reference"): {"measured": True, "outer_iterations": 10, "objective": 2.0},
    }
    grew = scaling_verdict(arms, {64: measured(1.7, 1.15), 128: measured(2.1, 1.08)}, 10)
    assert grew["status"] == "supported" and grew["relative_change"] > 0
    flat = scaling_verdict(arms, {64: measured(1.70, 1.15), 128: measured(1.73, 1.08)}, 10)
    assert flat["status"] == "inconclusive" and "twentieth" in flat["reason"]
    fell = scaling_verdict(arms, {64: measured(1.7, 1.15), 128: measured(1.4, 1.20)}, 10)
    assert fell["status"] == "refuted"
    # Growth without a falling per-iteration coarse cost is not the claimed mechanism.
    other = scaling_verdict(arms, {64: measured(1.7, 1.10), 128: measured(2.1, 1.20)}, 10)
    assert other["status"] == "inconclusive" and "did not fall" in other["reason"]


def test_an_arm_short_of_the_budget_is_inconclusive():
    arms = {
        (64, "jacobi"): {"measured": True, "outer_iterations": 10, "objective": 1.0},
        (64, "reference"): {"measured": True, "outer_iterations": 4, "objective": 1.0},
        (128, "jacobi"): {"measured": True, "outer_iterations": 10, "objective": 2.0},
        (128, "reference"): {"measured": True, "outer_iterations": 10, "objective": 2.0},
    }
    short = scaling_verdict(arms, {64: measured(1.7, 1.15), 128: measured(2.1, 1.08)}, 10)
    assert short["status"] == "inconclusive" and "before the matched budget" in short["reason"]


def test_build_writes_rows_and_macros_without_a_scaling_design(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    populate(runs)
    design_path = tmp_path / "tangent.json"
    design_path.write_text(json.dumps(tangent_design()))
    output = tmp_path / "out"
    report = build(design_path, runs, None, None, output)
    assert (output / "regime_rows.tex").read_text().count("\\\\") == 2
    assert (output / "scaling_rows.tex").read_text() == ""
    values = macros(report)
    assert values["coupledTangentVerdict"] == "supported"
    assert float(values["coupledCoupledSpeedup"]) > 1.2
    assert float(values["coupledFrozenSpeedup"]) < 1
    assert "coupledScalingVerdict" not in values


def test_an_unknown_design_schema_is_rejected(tmp_path):
    design_path = tmp_path / "tangent.json"
    design_path.write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(ValueError, match="unexpected schema"):
        build(design_path, tmp_path, None, None, tmp_path / "out")


def test_tangent_verdict_refuses_to_guess_without_both_models():
    assert (
        tangent_verdict(tangent_design(), {}, {"frozen": None, "coupled": None}, {}, {})["status"]
        == "pending"
    )


def test_the_matched_cap_fallback_is_found_and_read_as_the_declared_comparison(tmp_path):
    """A fired kill switch renames the rows; the analysis must still find and read them."""
    populate(tmp_path)
    for name in ("frozen-jacobi-r0", "frozen-reference-r100"):
        plain = tmp_path / name / "rep-0"
        plain.rename(tmp_path / name / "rep-0-cap44")
    analysis = tangent_analysis(tangent_design(), tmp_path)
    assert analysis["matched_outer_iteration_caps"] == [44]
    verdict = analysis["verdict"]
    assert verdict["status"] == "inconclusive"
    assert "matched cap of 44" in verdict["reason"]
    # The numbers are still reported, because the comparison is the declared fallback.
    frozen = verdict["equal_outer_iteration_comparison"]["frozen"]
    assert frozen["equal_outer_iterations"] and frozen["C"] > 1
    assert analysis["pairs"]["frozen"]["S"] < 1


def test_an_uncapped_record_wins_over_a_capped_one(tmp_path):
    populate(tmp_path)
    source = tmp_path / "frozen-jacobi-r0" / "rep-0"
    capped = tmp_path / "frozen-jacobi-r0" / "rep-0-cap44"
    capped.mkdir()
    (capped / "record.json").write_text((source / "record.json").read_text())
    analysis = tangent_analysis(tangent_design(), tmp_path)
    assert analysis["matched_outer_iteration_caps"] == []
    assert analysis["verdict"]["status"] == "supported"
