"""Only the predeclared, verified screening choice starts complete optimization."""

import copy
import json

import pytest

from deflation_example.coupled_confirmation_report import ARM_KEYS
from deflation_example.coupled_regularization import ALPHAS, RANKS
from deflation_example.coupled_regularization_complete import (
    checked_settings,
    complete_configuration,
    selected_settings,
)
from test_coupled_regularization import protocol


def screen():
    groups = []
    for alpha in ALPHAS:
        rows = []
        for rank in RANKS:
            passes = alpha == ALPHAS[-1] and rank in (8, 16)
            rows.append(
                {
                    "rank": rank,
                    "repetitions": 3,
                    "eligible_accuracy": True,
                    "reasons": [],
                    "failed_cases": [],
                    "gate_passed": passes,
                    "online_saving_fraction": 0.1 if passes else -0.1,
                    "constant_cost_break_even_three_quadratic_sequences": 1 if passes else None,
                    "predicted_saving_over_nine_sequences_seconds": rank * 100,
                }
            )
        groups.append({"alpha": alpha, "status": "complete", "rows": rows})
    return {
        "schema": "coupled-regularization-summary-v1",
        "all_alpha_attempts_finished": True,
        "decision": "evaluate_one_complete_optimization_comparison",
        "groups": groups,
        "selected": {"alpha": ALPHAS[-1], **groups[-1]["rows"][-1]},
    }


def test_single_selected_setting_freezes_four_matched_arms():
    settings = selected_settings(screen(), "screen-hash")
    assert settings["alpha"] == 1e-11
    assert settings["targets"] == [7, 15, 14]
    assert settings["phase"] == "development" and settings["repetitions"] == 3
    assert set(settings["arms"]) == {"jacobi", "frozen", "reference", "recycling"}
    for arm in settings["arms"].values():
        assert set(arm) == ARM_KEYS
    assert settings["arms"]["reference"]["rank"] == settings["arms"]["recycling"]["rank"] == 16
    assert settings["screen_sha256"] == "screen-hash"


@pytest.mark.parametrize(
    "defect",
    [
        "unfinished",
        "missing_alpha",
        "missing_rank",
        "missing_repeat",
        "accuracy",
        "saving",
        "amortization",
        "different_selection",
    ],
)
def test_ineligible_or_changed_screen_cannot_launch(defect):
    summary = screen()
    last = summary["groups"][-1]
    if defect == "unfinished":
        summary["all_alpha_attempts_finished"] = False
    elif defect == "missing_alpha":
        summary["groups"].pop(0)
    elif defect == "missing_rank":
        last["rows"].pop(0)
    elif defect == "missing_repeat":
        last["rows"][0]["repetitions"] = 2
    elif defect == "accuracy":
        last["rows"][-1]["eligible_accuracy"] = False
    elif defect == "saving":
        last["rows"][-1]["online_saving_fraction"] = 0.04
    elif defect == "amortization":
        last["rows"][-1]["constant_cost_break_even_three_quadratic_sequences"] = 10
    else:
        summary["selected"] = {"alpha": ALPHAS[-1], **last["rows"][-2]}
    with pytest.raises(ValueError):
        selected_settings(summary, "screen-hash")


def test_complete_configuration_varies_only_solver_policy_and_regularization():
    settings = selected_settings(screen(), "screen-hash")
    original = {
        **protocol(),
        "alpha": 1e-14,
        "queries": [{"target": 7, "upper_K": 357.3}],
        "method": "jacobi",
        "rank": 0,
        "recycle_window": 1,
        "reference_transfer": "full",
    }
    record = {"configuration": original, "all_problems_verified": True, "cases": [{}]}
    before = copy.deepcopy(record)
    configs = [complete_configuration(record, settings, arm, 0) for arm in settings["arms"]]
    identities = [{k: v for k, v in c.items() if k not in ARM_KEYS} for c in configs]
    assert all(c == identities[0] for c in identities)
    for cfg in configs:
        assert cfg["stage"] is None
        assert cfg["alpha"] == 1e-11
        assert cfg["inner_tolerance"] == 1e-10 and cfg["nonlinear_tolerance"] == 1e-8
        assert cfg["initial_state_alpha_policy"] == "shared_temperature"
        assert cfg["reference_krylov_steps"] == 48
        assert cfg["warm_start"] is True
        assert [q["target"] for q in cfg["queries"]] == settings["targets"]
    assert record == before
    with pytest.raises(ValueError, match="three repetitions"):
        complete_configuration(record, settings, "reference", 3)
    with pytest.raises(ValueError, match="four declared"):
        complete_configuration(record, settings, "another", 0)
    with pytest.raises(ValueError, match="verified nominal"):
        complete_configuration({**record, "all_problems_verified": False}, settings, "reference", 0)


def test_frozen_settings_reject_manual_edits(tmp_path):
    path, frozen = tmp_path / "screen.json", tmp_path / "settings.json"
    path.write_text(json.dumps(screen()))
    settings = checked_settings(path)
    frozen.write_text(json.dumps(settings))
    assert checked_settings(path, frozen) == settings
    settings["alpha"] = 1e-10
    frozen.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="frozen screening"):
        checked_settings(path, frozen)
