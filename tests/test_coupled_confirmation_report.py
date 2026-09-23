"""Confirmation requires matched strong controls, complete timings and solutions."""

from copy import deepcopy

import numpy as np
import pytest

from deflation_example.coupled_confirmation_report import summarize
from test_coupled_report import record


def population():
    settings = {"targets": [7, 1, 2], "repetitions": 5, "phase": "confirmation", "arms": {}}
    records, fields = [], []
    for name, method, preconditioner, seconds in (
        ("jacobi", "jacobi", "jacobi", 20.0),
        ("frozen", "jacobi", "frozen", 10.0),
        ("recycling", "recycling", "frozen", 12.0),
        ("reference", "reference", "frozen", 8.0),
    ):
        arm = {
            "method": method,
            "inner_preconditioner": preconditioner,
            "rank": 20 if method != "jacobi" else 0,
            "recycle_window": 20 if method != "jacobi" else 1,
            "reference_selection": "preconditioned_coupled" if method == "reference" else "thermal",
        }
        settings["arms"][name] = arm
        for repeat in range(5):
            row = record(method, repeat, seconds + repeat * 0.01)
            row["configuration"].update(
                arm,
                inner_tolerance=1e-10,
                equation_acceptance_tolerance=1e-12,
                conservation_tolerance=1e-6,
            )
            row["cases"] = [deepcopy(row["cases"][0]) for _ in range(3)]
            for i, case in enumerate(row["cases"]):
                case.update(position=i, target=settings["targets"][i], objective=1e-5)
                case["equations"][0]["momentum_relative_residual"] = 1e-13
            row["configuration"]["queries"] = [
                {"target": t, "upper_K": 357.3} for t in settings["targets"]
            ]
            row["verified_problems"] = 3
            records.append(row)
            fields.append([np.ones(4) for _ in range(3)])
    return settings, records, fields


def test_all_four_arms_and_verified_fields_are_required():
    settings, records, fields = population()
    result = summarize(records, settings, fields)
    assert result["publication_gate_passed"]
    assert result["fastest_tested_alternative_over_reference"] == pytest.approx(10.02 / 8.02)
    assert not summarize(records[:-1], settings, fields[:-1])["publication_gate_passed"]
    assert not summarize(records, settings)["publication_gate_passed"]


@pytest.mark.parametrize("defect", ["state", "objective", "failure", "overlapping_times"])
def test_bad_or_unclear_outcomes_do_not_pass_the_publication_gate(defect):
    settings, records, fields = population()
    if defect == "state":
        fields[-1][0][0] += 0.1
    elif defect == "objective":
        records[-1]["cases"][0]["objective"] *= 2
    elif defect == "failure":
        records[-1]["cases"][0].update(status="nonlinear_iteration_cap", verified=False)
        records[-1].update(all_problems_verified=False, verified_problems=2)
    else:
        records[-1]["sequence_seconds"] += 10
        records[-1]["components_seconds"]["queries"] += 10
        records[-1]["preparation_inclusive_seconds"] += 10
    assert not summarize(records, settings, fields)["publication_gate_passed"]


@pytest.mark.parametrize("defect", ["duplicate", "tolerance", "source", "targets"])
def test_protocol_changes_are_refused(defect):
    settings, records, fields = population()
    if defect == "duplicate":
        records.append(records[-1])
    elif defect == "tolerance":
        records[-1]["configuration"]["inner_tolerance"] = 1e-8
    elif defect == "source":
        records[-1]["environment"]["source_sha256"] = {"solver.py": "changed"}
    else:
        records[-1]["configuration"]["queries"][0]["target"] = 0
    with pytest.raises(ValueError):
        summarize(records, settings, fields)


def test_development_outcomes_are_not_confirmation():
    settings, records, fields = population()
    settings["phase"] = "development"
    assert not summarize(records, settings, fields)["publication_gate_passed"]
