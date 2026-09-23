"""Matched starts, unchanged equations, explicit root agreement and all outcomes."""

import numpy as np
import pytest

from deflation_example.coupled_repair_protocol import (
    POLICIES,
    compare_repairs,
    derivative_check,
    paired_selection,
)
from test_coupled_newton_replay import data


def test_complete_derivative_and_four_policy_agreement():
    problem, _, _, source = data(consistent=True, streamline_rule="smooth_p8", inlet=0.2)
    previous = problem.full_temperature(problem.initial)
    original = source.copy()
    assert derivative_check(
        problem, source, previous, problem.initial_flow, previous, problem.initial_flow, 0
    )["passed"]
    rows, fields, lengths = [], {}, []
    for initialization in ("saved", "previous"):
        current, arrays = compare_repairs(
            problem,
            source,
            previous,
            problem.initial_flow,
            previous,
            problem.initial_flow,
            0,
            initialization=initialization,
            callback=lambda r, f: lengths.append((len(r), len(f))),
        )
        rows.extend(current)
        fields.update(arrays)
    np.testing.assert_array_equal(source, original)
    assert lengths == [(i, 3 * i) for i in range(1, 5)] * 2
    assert [r["policy"] for r in rows[:4]] == [p[0] for p in POLICIES]
    assert all(r["independent_criteria_met"] for r in rows)
    assert (
        paired_selection(rows, fields, problem.temperature_scale)["selected_policy"]
        == "equation_max"
    )


@pytest.mark.parametrize("field", ["state", "velocity", "pressure"])
def test_matching_temperature_alone_does_not_establish_same_solution(field):
    rows = [
        {"policy": "equation_max", "initialization": start, "independent_criteria_met": True}
        for start in ("saved", "previous")
    ]
    fields = {
        f"{start}_equation_max_{name}": np.ones(2)
        for start in ("saved", "previous")
        for name in ("state", "velocity", "pressure")
    }
    fields[f"previous_equation_max_{field}"][0] += 0.1
    assert paired_selection(rows, fields, 20)["selected_policy"] is None


def test_duplicate_start_or_unsuccessful_step_cannot_select_policy():
    row = {"policy": "equation_max", "initialization": "saved", "independent_criteria_met": True}
    assert paired_selection([row, row], {}, 20)["selected_policy"] is None
    failed = {**row, "initialization": "previous", "independent_criteria_met": False}
    assert paired_selection([row, failed], {}, 20)["selected_policy"] is None


def test_wrong_complete_jacobian_fails_derivative_gate(monkeypatch):
    import deflation_example.coupled_repair_protocol as module

    problem, _, _, source = data(consistent=True, streamline_rule="smooth_p8", inlet=0.2)
    previous = problem.full_temperature(problem.initial)
    original = module.step_linearization

    def corrupt(*args, **kwargs):
        matrices = list(original(*args, **kwargs))
        matrices[0] = 2 * matrices[0]
        return tuple(matrices)

    monkeypatch.setattr(module, "step_linearization", corrupt)
    assert not derivative_check(
        problem, source, previous, problem.initial_flow, previous, problem.initial_flow, 0
    )["passed"]
