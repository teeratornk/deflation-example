"""Exercise artifact generation with synthetic records and no external data."""

import json

import pytest

from deflation_example.coupled_corrected_evidence import build


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """Small artificial records test formatting, coverage and derived quantities.

    These values are test inputs, not measurements or manuscript evidence.
    Real-data reproduction uses the public builder's explicit --evidence input.
    """
    root = tmp_path_factory.mktemp("synthetic-evidence")

    def write(name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    arm = {
        "status": "converged",
        "verified": True,
        "iterations": 3,
        "expensive_applications": 20,
        "hours": 1.0,
        "optimality": 1e-9,
    }
    write(
        "coupled-smooth-preconditioner-v1-outcome.json",
        {
            "arms": {
                "diagonal": {**arm, "expensive_applications": 100, "hours": 2.0},
                "frozen": arm,
            },
            "E": 5.0,
            "R": 2.0,
            "reading": "synthetic test",
            "checks": {"the_answer_did_not_move": {"relative_objective_difference": 1e-12}},
        },
    )
    write(
        "coupled-preconditioner-ablation-v1-outcome.json",
        {
            "readings": {
                "coarse_space": {"measured": {"applications": 20, "hours": 1.2}},
                "sweeps": {"measured_spread": 0.01},
            }
        },
    )
    write(
        "coupled-hard-regime-pilot-v1-outcome.json",
        {
            "regimes": {
                name: {"ratios": {"C2": 1.2, "R": 1.1}, "reading": "qualifies"}
                for name in ("alpha-1e-13", "alpha-1e-12", "alpha-1e-11")
            },
            "refined_v3": {"ratios": {"C2": 0.9, "R": 0.8}, "reading": "not harder"},
        },
    )
    write(
        "coupled-matched-reference-v1-outcome.json",
        {
            "ratios": {"C": 1.02, "R": 0.9},
            "reading": "synthetic test",
            "reference_numbers": {
                "shipped_operator": {"C": 2.0, "R": 1.3},
                "corrected_operator_shipped_reference": {"C": 1.01},
            },
        },
    )
    write(
        "coupled-hard-regime-confirmation-v1-outcome.json",
        {"reading": "synthetic test", "why": "unit fixture"},
    )
    write(
        "coupled-smooth-optimization-v6-195465-1/record.json",
        {
            "sequence_seconds": 4000.0,
            "cases": [
                {
                    "status": "converged",
                    "verified": True,
                    "nonlinear_iterations": 3,
                    "history": [{"attempts": [{"qp_history": [{"linear_iterations": 80}]}]}],
                    "kkt": {"stationarity": 1e-9},
                    "equations": [
                        {"thermal_relative_residual": 1e-13, "mass_relative_imbalance": 1e-14}
                    ],
                    "adjoint": {"steps": [{"momentum_adjoint_relative_residual": 1e-13}]},
                }
            ],
        },
    )
    for name, condition in (("shipped-reference-r200", 200.0), ("consistent-reference-r200", 20.0)):
        write(
            f"coupled-consistent-scaling-v1-runs/{name}/stage-0/record.json",
            {"coarse_condition": condition},
        )
    for method, count in (("reference", 20), ("jacobi", 100)):
        write(
            f"mesh-public-data-v1/final/transformer-transient8/{method}-0/record.json",
            {"coarse_condition": 10.0, "iterations": count},
        )
    write(
        "coupled-coarse-placement-v2.json",
        {
            "rows": [
                {
                    "model": "uncorrected",
                    "scaled_rayleigh_quotients": {"0.0": 0.001, "1.0": 0.1},
                    "largest_scaled_eigenvalue_power_estimate": 1.0,
                },
                {
                    "model": "corrected",
                    "scaled_rayleigh_quotients": {"0.0": 0.01, "1.0": 0.1},
                    "largest_scaled_eigenvalue_power_estimate": 1.0,
                },
            ]
        },
    )
    return root


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, study):
    return build(study, tmp_path_factory.mktemp("coupled"))


def test_the_headline_macros_carry_the_measured_ratios(artifacts):
    values = artifacts["macros"]
    head = artifacts["headline"]
    assert values["coupledFrozenApplicationRatio"] == f"{head['application_ratio']:.2f}"
    assert values["coupledFrozenWallRatio"] == f"{head['wall_ratio']:.2f}"
    assert values["coupledFrozenApplications"] == str(
        head["arms"]["frozen"]["expensive_applications"]
    )
    assert values["coupledDiagonalApplications"] == str(
        head["arms"]["diagonal"]["expensive_applications"]
    )


def test_both_headline_arms_converged_and_verified(artifacts):
    for arm in artifacts["headline"]["arms"].values():
        assert arm["status"] == "converged"
        assert arm["verified"] is True
        assert arm["optimality"] <= 1e-8


def test_the_regime_map_runs_from_the_uncorrected_operator_to_the_refined_mesh(artifacts):
    wheres = [row["where"] for row in artifacts["regime"]]
    assert wheres[0] == "uncorrected operator"
    assert any("1e-11" in w for w in wheres)
    assert any("refined" in w for w in wheres)
    # The corrected operator's nominal ratio is the one that refutes the old headline.
    nominal = artifacts["regime"][1]
    assert nominal["cg_ratio"] < 1.15


def test_the_generated_files_exist_and_parse(tmp_path, study):
    summary = build(study, tmp_path)
    macro_text = (tmp_path / "macros.tex").read_text()
    assert macro_text.count("\\newcommand") == len(summary["macros"])
    for name in ("regime_rows.tex", "preconditioner_rows.tex", "summary.json"):
        assert (tmp_path / name).is_file()
    json.loads((tmp_path / "summary.json").read_text())


def test_a_missing_record_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(tmp_path, tmp_path / "out")


def test_the_comparison_includes_the_method_the_paper_advocates(artifacts):
    """Against plain Jacobi alone a reader cannot see how deflation fares."""
    deflation = artifacts["deflation_reference"]
    assert deflation is not None
    assert deflation["status"] == "converged" and deflation["verified"] is True
    values = artifacts["macros"]
    frozen = artifacts["headline"]["arms"]["frozen"]
    assert values["coupledDeflationApplications"] == str(deflation["expensive_applications"])
    expected = deflation["expensive_applications"] / frozen["expensive_applications"]
    assert values["coupledFrozenAgainstDeflationApplications"] == f"{expected:.2f}"


def test_every_regime_row_states_the_budget_it_was_measured_on(artifacts):
    """Two-quadratic and ten-quadratic ratios must not read as one series."""
    budgets = {row["budget"] for row in artifacts["regime"]}
    assert budgets == {"ten quadratics", "two quadratics"}
    for row in artifacts["regime"]:
        assert row["conditions"]
        assert row["source"]


def test_the_refined_row_declares_the_conditions_that_differ(artifacts):
    refined = next(row for row in artifacts["regime"] if "refined" in row["where"])
    assert "flow tolerance 1e-11" in refined["conditions"]
    assert "host device" in refined["conditions"]


def test_coarse_spectrum_reports_the_controlled_pair_and_the_refuted_placement(study):
    """The mechanism evidence is measured, and the placement probe's refutation is kept."""
    from deflation_example.coupled_corrected_evidence import coarse_spectrum

    screen, values, _ = coarse_spectrum(study)
    assert screen and screen["uncorrected"] and screen["corrected"]
    # The controlled pair: the uncorrected space resolves more of the spectrum.
    assert float(values["coupledUncorrectedCoarseCondition"]) > float(
        values["coupledCorrectedCoarseCondition"]
    )
    # Placement was measured and refuted the bulk reading: both spaces stay far below
    # the scaled bulk of one, so no macro may imply the corrected space reached it.
    for name in ("coupledUncorrectedPlacementRange", "coupledCorrectedPlacementRange"):
        high = float(values[name].split(" to ")[1])
        assert high < 0.5, f"{name} must stay below the spectral bulk"
    assert float(values["coupledUncorrectedPlacementSpread"]) > float(
        values["coupledCorrectedPlacementSpread"]
    )


def test_mechanism_rows_quote_no_hand_typed_iteration_counts(study):
    """Every number in the mechanism table is derived from a record."""
    from deflation_example.coupled_corrected_evidence import coarse_spectrum

    _, values, _ = coarse_spectrum(study)
    counts = (
        values["coupledTransformerReferenceIterations"],
        values["coupledTransformerJacobiIterations"],
    )
    assert all(count.isdigit() for count in counts)
    assert int(counts[0]) < int(counts[1])
