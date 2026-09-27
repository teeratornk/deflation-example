"""Ablations preserve physical inputs, final accuracy and unfavorable outcomes."""

from copy import deepcopy
import json

import numpy as np
from omegaconf import OmegaConf
import pytest

from deflation_example.linear_spacetime_ablation import (
    ARM_KEYS,
    ablation_configuration,
    ablation_settings,
    checked_ablation_settings,
)
from deflation_example.linear_spacetime_ablation_report import summarize, plot
from test_coupled_regularization_complete import screen
from test_coupled_regularization import protocol


def settings(tmp_path):
    path = tmp_path / "screen.json"
    path.write_text(json.dumps(screen()))
    return ablation_settings(path)


def test_complete_predeclared_population_and_rotation(tmp_path):
    frozen = settings(tmp_path)
    assert len(frozen["arms"]) == 22
    assert len(frozen["schedule"]) == 66
    assert len({(r["arm"], r["repetition"]) for r in frozen["schedule"]}) == 66
    assert frozen["schedule"][0]["arm"] != frozen["schedule"][22]["arm"]
    nominal = {
        "configuration": {**protocol(), "queries": [{"target": 7, "upper_K": 357.3}]},
        "all_problems_verified": True,
        "cases": [{}],
    }
    configurations = [ablation_configuration(nominal, frozen, i) for i in range(66)]
    common = [
        {k: v for k, v in c.items() if k not in ARM_KEYS | {"speedup_ablation_arm", "repetition"}}
        for c in configurations
    ]
    assert all(c == common[0] for c in common)
    assert common[0]["alpha"] == 1e-11
    assert common[0]["reference_krylov_steps"] == 96
    assert common[0]["physics"] == "prescribed_flow"
    assert common[0]["inner_tolerance"] == 1e-10
    with pytest.raises(ValueError, match="Task"):
        ablation_configuration(nominal, frozen, 66)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(frozen))
    assert checked_ablation_settings(tmp_path / "screen.json", path) == frozen
    frozen["alpha"] = 1e-5
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="predeclared"):
        checked_ablation_settings(tmp_path / "screen.json", path)


@pytest.mark.parametrize("selection", ["lowest", "alternating_low_high"])
@pytest.mark.parametrize("transfer", ["full", "sequential"])
def test_complete_small_trajectory_and_independent_agreement(
    monkeypatch, tmp_path, selection, transfer
):
    from deflation_example import linear_spacetime_sequence as sequence
    from deflation_example.coupled_report import validate_record
    from deflation_example.coupled_confirmation_report import inner_evidence
    from test_linear_spacetime import setup

    _, problem, cfg = setup()
    baseline = {"baseline_sha256": "test", "configuration": {}, "input_sha256": {}, "seconds": 0.0}
    monkeypatch.setattr(sequence, "load_fixed_flow", lambda cfg: (problem, baseline))
    cfg.update(
        method="reference",
        rank=2,
        recycle_window=2,
        inner_preconditioner="frozen",
        reference_transfer=transfer,
        reference_krylov_selection=selection,
        output=str(tmp_path / "reference"),
    )
    actual = sequence.run(OmegaConf.create(cfg))
    assert actual["all_problems_verified"], actual
    assert validate_record(actual), actual
    assert inner_evidence(actual)["complete_histories"]
    plain = sequence.run(
        OmegaConf.create(
            {
                **cfg,
                "method": "jacobi",
                "rank": 0,
                "reference_transfer": "full",
                "output": str(tmp_path / "plain"),
            }
        )
    )
    assert validate_record(plain), plain
    for position in range(len(cfg["queries"])):
        with np.load(tmp_path / "reference" / f"target-{position:02d}.npz") as a:
            with np.load(tmp_path / "plain" / f"target-{position:02d}.npz") as b:
                np.testing.assert_allclose(a["state"], b["state"], atol=1e-6, rtol=0)
    assert actual["components_seconds"]["reference_construction"] > 0
    assert actual["sequence_seconds"] == pytest.approx(sum(actual["components_seconds"].values()))


def population(tmp_path):
    from test_coupled_confirmation_report import population as existing_population

    frozen = settings(tmp_path)
    _, templates, _ = existing_population()
    template = templates[0]
    records, agreements = [], {}
    for name, arm in frozen["arms"].items():
        for rep in range(3):
            row = deepcopy(template)
            cfg = row["configuration"]
            cfg.update(
                arm,
                speedup_ablation_arm=name,
                repetition=rep,
                physics="prescribed_flow",
                feedback_multiplier=0.0,
                alpha=1e-11,
                slabs=64,
                horizon_s=600.0,
                target_startup_s=60.0,
                lower_K=337.3,
                reference_krylov_steps=96,
                reference_krylov_seed=20260923,
                warm_start=True,
                stage=None,
                queries=[{"target": t, "upper_K": 357.3} for t in frozen["targets"]],
            )
            row["fixed_flow_preparation"] = {"velocity_sha256": "matched"}
            for case, target in zip(row["cases"], frozen["targets"], strict=True):
                case.update(target=target, upper_K=357.3)
                case["adjoint"].update(
                    maximum_source_adjoint_relative_residual=1e-13,
                    gradient_weight_normalized_difference=1e-13,
                )
                step = case.pop("history")[0]["attempts"][0]["qp_history"][0]
                step["deployed_rank"] = arm["rank"]
                case["pdas_history"] = [step]
            records.append(row)
            agreements[name, rep] = {"state_absolute": 1e-9, "objective_relative": 1e-9}
    return frozen, records, agreements


def test_summary_preserves_all_configurations_and_requires_agreement(tmp_path):
    frozen, records, agreements = population(tmp_path)
    result = summarize(records, frozen, agreements)
    assert result["all_declared_sequences_verified"]
    assert len(result["rows"]) == 22
    assert all(c["fastest_alternative_over_reference"] == 1 for c in result["comparisons"])
    no_fields = summarize(records, frozen)
    assert all(c["fastest_alternative_over_reference"] is None for c in no_fields["comparisons"])
    agreements["frozen", 0]["state_absolute"] = 1e-3
    bad = summarize(records, frozen, agreements)
    assert all(c["fastest_alternative_over_reference"] is None for c in bad["comparisons"])


def test_missing_baseline_has_no_completed_solve_speedup(tmp_path):
    frozen, records, agreements = population(tmp_path)
    records = [r for r in records if r["configuration"]["speedup_ablation_arm"] != "frozen"]
    result = summarize(records, frozen, agreements)
    assert not result["all_declared_sequences_verified"]
    assert all(c["fastest_alternative_over_reference"] is None for c in result["comparisons"])
    row = next(r for r in result["rows"] if r["arm"] == "frozen")
    assert all(r["status"] == "missing_record" for r in row["outcomes"])
    plot(result, tmp_path)
    assert (tmp_path / "rank-time-memory.pdf").stat().st_size > 1000


@pytest.mark.parametrize("status", ["running", "failed", "construction_failed"])
def test_unfinished_or_failed_comparator_is_retained_without_a_ratio(tmp_path, status):
    frozen, records, agreements = population(tmp_path)
    row = next(r for r in records if r["configuration"]["speedup_ablation_arm"] == "frozen")
    row["status"] = status
    if status == "running":
        for key in ("sequence_seconds", "cases", "memory", "all_problems_verified"):
            del row[key]
    else:
        row["all_problems_verified"] = False
        row["verified_problems"] -= 1
        row["cases"][0].update(status="iteration_cap", verified=False)
        if status == "construction_failed":
            row.update(cases=[], verified_problems=0, error_type="MemoryError")
            del row["fixed_flow_preparation"]
    result = summarize(records, frozen, agreements)
    assert not result["all_declared_sequences_verified"]
    assert all(c["fastest_alternative_over_reference"] is None for c in result["comparisons"])
    outcome = next(r for r in result["rows"] if r["arm"] == "frozen")["outcomes"][0]
    assert not outcome["verified"]
    assert outcome["status"] == ("unfinished_record" if status == "running" else status)


@pytest.mark.parametrize(
    "defect", ["source", "velocity", "accuracy", "time_sum", "duplicate", "selection"]
)
def test_mismatched_records_are_rejected(tmp_path, defect):
    frozen, records, agreements = population(tmp_path)
    last = records[-1]
    if defect == "source":
        last["environment"]["source_sha256"] = {"changed": "source"}
    elif defect == "velocity":
        last["fixed_flow_preparation"]["velocity_sha256"] = "different"
    elif defect == "accuracy":
        last["configuration"]["inner_tolerance"] = 1e-8
    elif defect == "time_sum":
        last["sequence_seconds"] += 1
    elif defect == "duplicate":
        records.append(deepcopy(last))
    else:
        last["configuration"]["reference_krylov_selection"] = "alternating_low_high"
    with pytest.raises(ValueError):
        summarize(records, frozen, agreements)
