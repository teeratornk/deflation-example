"""Smooth-study summaries retain costs and verify the measured implementation."""

from copy import deepcopy
import json

import numpy as np
import pytest

from deflation_example.coupled_smooth_report import audit, summarize
from test_coupled_corrected_report import pilot, DESIGN, SOURCE


def smooth_record(method):
    case = next(c for c in DESIGN["pilots"] if c["method"] == method)
    saved = pilot(case)
    saved["configuration"].update(
        streamline_rule="smooth_p8",
        rank=0 if method == "jacobi" else 200,
        recycle_window=1 if method == "jacobi" else 200,
    )
    saved["initial_state"] = {
        "policy": "fresh_optimization_from_assessed_checkpoint_temperature",
        "evaluated_streamline_rule": "smooth_p8",
        "retained_secant_pairs": 0,
        "retained_recycling_directions": 0,
        "snapshot": "same",
    }
    saved["initialization_scope"] = "initial field; no old history or cost"
    saved["components_seconds"]["reference_construction"] = 0
    case = saved["cases"][0]
    case["adjoint"].update(
        maximum_source_adjoint_relative_residual=1e-12,
        gradient_weight_normalized_difference=1e-12,
    )
    case["nonlinear_iterations"] = 1
    case["inner_iterations"] = 2
    case["history"] = [
        {
            "iteration": 0,
            "kkt": case["kkt"],
            "attempts": [
                {
                    "qp_history": [
                        {
                            "linear_status": "converged",
                            "linear_iterations": 2,
                            "linear_residual": 1e-12,
                            "deployed_rank": saved["configuration"]["rank"],
                            "fallback": None,
                            "timing": {
                                "total_seconds": 1.0,
                                "components_seconds": {"iteration": 1.0},
                                "hybrid_coarse_correction": {"seconds": 0.4},
                            },
                        }
                    ]
                }
            ],
        }
    ]
    return saved


def test_nested_costs_are_counted_once():
    row = audit(smooth_record("reference"), SOURCE)
    assert row["verified"]
    costs = row["nonoverlapping_cost_seconds"]
    assert costs["coarse_processing_and_application"] == 0.4
    assert costs["remaining_inner_solve"] == 0.6
    assert sum(costs.values()) == pytest.approx(row["sequence_seconds"])


def test_report_preserves_unfavorable_comparison_and_declared_source(tmp_path):
    paths = {}
    for method in ("jacobi", "reference"):
        paths[method] = tmp_path / method
        paths[method].mkdir()
        (paths[method] / "record.json").write_text(json.dumps(smooth_record(method)))
        np.savez(
            paths[method] / "target-00.npz",
            **{k: np.ones(3) for k in ("state", "control", "desired")},
        )
    result = summarize(paths, SOURCE)
    assert result["both_verified_and_field_agreement_met"]
    assert result["jacobi_over_reference_time"] == 0.75
    assert result["maximum_temperature_difference_K"] == 0
    assert str(tmp_path) not in json.dumps(result)
    bad = smooth_record("reference")
    bad["initial_state"]["snapshot"] = "other"
    (paths["reference"] / "record.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="same assessed"):
        summarize(paths, SOURCE)


def test_report_rechecks_source_adjoint_and_residuals():
    original = smooth_record("reference")
    with pytest.raises(ValueError, match="clean source"):
        audit(original, "b" * 40)
    bad = deepcopy(original)
    bad["cases"][0]["adjoint"]["maximum_source_adjoint_relative_residual"] = 1e-5
    with pytest.raises(ValueError, match="Independent source"):
        audit(bad, SOURCE)
    bad = deepcopy(original)
    bad["cases"][0]["history"][0]["attempts"][0]["qp_history"][0]["linear_residual"] = 1e-4
    with pytest.raises(ValueError, match="original-system"):
        audit(bad, SOURCE)


@pytest.mark.parametrize("nested", [-1.0, 2.0, np.nan])
def test_invalid_nested_times_are_rejected(nested):
    saved = smooth_record("reference")
    timing = saved["cases"][0]["history"][0]["attempts"][0]["qp_history"][0]["timing"]
    timing["hybrid_coarse_correction"]["seconds"] = nested
    with pytest.raises(ValueError, match="timers"):
        audit(saved, SOURCE)


def test_unsuccessful_outcome_is_retained():
    saved = smooth_record("reference")
    saved["cases"][0].update(status="nonlinear_iteration_cap", verified=False)
    saved.update(verified_problems=0, all_problems_verified=False)
    result = audit(saved, SOURCE)
    assert not result["verified"]
    assert result["status"] == "nonlinear_iteration_cap"
    assert result["sequence_seconds"] == 4
