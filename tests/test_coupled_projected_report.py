"""Intermediate accuracy cannot be substituted for complete nonlinear verification."""

import copy

import pytest

from deflation_example.coupled_projected_report import KKT_NAMES, audit


def record():
    zero = dict.fromkeys(KKT_NAMES, 0.0)
    attempt = dict(
        linear_tolerance=0.01,
        qp_tolerance=0.01,
        strict_accuracy=False,
        qp_status="converged",
        qp_kkt=zero,
        qp_history=[{"linear_status": "converged", "linear_residual": 0.009}],
    )
    strict = {
        **attempt,
        "linear_tolerance": 1e-10,
        "qp_tolerance": 1e-10,
        "strict_accuracy": True,
        "qp_history": [{"linear_status": "converged", "linear_residual": 9e-11}],
    }
    case = dict(
        position=0,
        target=7,
        status="converged",
        verified=True,
        kkt=zero,
        equations=[
            dict(
                slab=0,
                momentum_relative_residual=1e-13,
                continuity_relative_residual=1e-13,
                thermal_relative_residual=1e-13,
                mass_relative_imbalance=1e-9,
                energy={"relative_defect": 1e-9},
            )
        ],
        adjoint=dict(
            maximum_momentum_adjoint_relative_residual=1e-13,
            maximum_source_adjoint_relative_residual=1e-13,
            gradient_weight_normalized_difference=1e-13,
        ),
        history=[{"attempts": [attempt, strict]}],
    )
    return dict(
        schema="coupled-trust-development-v1",
        configuration=dict(
            qp_solver="projected",
            trust_accuracy="adaptive_projected",
            inner_tolerance=1e-10,
            qp_tolerance=1e-10,
            nonlinear_tolerance=1e-8,
            equation_acceptance_tolerance=1e-12,
            conservation_tolerance=1e-6,
            slabs=1,
            queries=[{"target": 7}],
        ),
        status="complete",
        all_problems_verified=True,
        cases=[case],
        attempt_seconds=12.0,
        prior_attempt_seconds=5.0,
        cumulative_attempt_seconds=17.0,
        components_seconds={"construction": 2.0, "optimization": 10.0},
    )


def test_updated_policy_checks_intermediate_and_final_targets():
    source = record()
    before = copy.deepcopy(source)
    result = audit(source)
    assert result["verified"], result
    assert result["cases"][0]["maximum_intermediate_original_residual"] == 0.009
    assert source == before
    source["cases"][0]["history"][0]["attempts"].pop()
    assert "final_strict_phase" in audit(source)["cases"][0]["failures"]


@pytest.mark.parametrize("rho", [0.02, -1, float("nan"), None])
def test_every_projected_solve_requires_original_residual(rho):
    source = record()
    source["cases"][0]["history"][0]["attempts"][0]["qp_history"][0]["linear_residual"] = rho
    assert "original_residual" in audit(source)["cases"][0]["failures"]


@pytest.mark.parametrize("missing", ["kkt", "equations", "adjoint", "history"])
def test_missing_final_evidence_fails(missing):
    source = record()
    del source["cases"][0][missing]
    assert not audit(source)["verified"]


def test_already_converged_initial_guess_needs_no_inner_work():
    source = record()
    source["cases"][0]["history"] = []
    assert audit(source)["verified"]


def test_truncated_target_population_and_duplicate_slabs_fail():
    source = record()
    source["configuration"]["queries"].append({"target": 15})
    assert "target_coverage" in audit(source)["failures"]
    source = record()
    source["configuration"]["slabs"] = 2
    source["cases"][0]["equations"] *= 2
    assert "equations_or_conservation" in audit(source)["cases"][0]["failures"]


def test_failed_and_missing_attempts_remain_visible():
    assert audit(None)["status"] == "missing"
    source = record()
    source.update(status="budget_exhausted", all_problems_verified=False)
    assert not audit(source)["verified"]
    assert audit(source)["status"] == "budget_exhausted"


def test_timing_requires_complete_construction_and_prior_intervals():
    source = record()
    source["components_seconds"]["construction"] = 0.0
    assert "elapsed_accounting" in audit(source)["failures"]
    source = record()
    source["cumulative_attempt_seconds"] = 12.0
    assert "elapsed_accounting" in audit(source)["failures"]


def test_other_protocols_and_looser_final_targets_cannot_be_relabelled():
    source = record()
    source["configuration"]["trust_accuracy"] = "adaptive"
    with pytest.raises(ValueError, match="adaptive_projected"):
        audit(source)
    source = record()
    source["configuration"]["nonlinear_tolerance"] = 0.01
    assert "final_tolerances" in audit(source)["failures"]
