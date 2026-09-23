"""A post-run equation check uses saved fields and their own temporal history."""

from copy import deepcopy
import numpy as np
import pytest

from deflation_example.coupled_forward_verify import verify_fields
from test_coupled_derivatives import small_coupled_problem


def data():
    problem = small_coupled_problem(
        [0.2, 0.35], uniform_capacity=True, consistent=True, streamline_rule="smooth_p8"
    )
    problem.flow_tolerance = 1e-13
    state = np.linspace(0.04, 0.1, problem.size)
    result = problem.evaluate(state)
    fields = {
        "state": state.reshape(2, -1),
        "velocity": np.stack([f.velocity for f in result.flows]),
        "pressure": np.stack([f.pressure for f in result.flows]),
        "times_s": np.cumsum(problem.physical_steps),
    }
    rows = [{"status": "converged", "time_s": t} for t in fields["times_s"]]
    return problem, result.control.reshape(2, -1), fields, rows


def test_saved_field_equations_are_recomputed_without_modifying_inputs():
    problem, source, fields, rows = data()
    original = deepcopy(fields)
    report = verify_fields(
        problem, source, fields, rows, time_scheme="backward_euler", subdivision=1
    )
    assert report["complete_trajectory_verified"]
    for key in fields:
        np.testing.assert_array_equal(fields[key], original[key])
    fields["state"][1, 0] += 1e-3
    changed = verify_fields(
        problem, source, fields, rows, time_scheme="backward_euler", subdivision=1
    )
    assert not changed["complete_trajectory_verified"]
    assert changed["steps"][0]["verified"]
    assert not changed["steps"][1]["verified"]


def test_verified_prefix_does_not_replace_a_complete_trajectory():
    problem, source, fields, rows = data()
    prefix = {k: v[:1] for k, v in fields.items()}
    result = verify_fields(
        problem, source, prefix, rows[:1], time_scheme="backward_euler", subdivision=1
    )
    assert result["all_saved_steps_verified"]
    assert not result["complete_trajectory_verified"]
    rows[-1]["status"] = "newton_iteration_cap"
    result = verify_fields(
        problem, source, fields, rows, time_scheme="backward_euler", subdivision=1
    )
    assert not result["complete_trajectory_verified"]


@pytest.mark.parametrize("mutation", ["time", "nonfinite", "pressure_gauge", "source"])
def test_saved_labels_cannot_override_wrong_fields(mutation):
    problem, source, fields, rows = data()
    if mutation == "time":
        fields["times_s"][0] += 0.01
    elif mutation == "nonfinite":
        fields["velocity"][0, 0, 0] = np.nan
    elif mutation == "pressure_gauge":
        fields["pressure"][:, problem.pressure_gauge[0]] += 1e-3
    else:
        source = source + 1e-3
    if mutation in {"time", "nonfinite", "pressure_gauge"}:
        with pytest.raises(ValueError):
            verify_fields(
                problem, source, fields, rows, time_scheme="backward_euler", subdivision=1
            )
    else:
        result = verify_fields(
            problem, source, fields, rows, time_scheme="backward_euler", subdivision=1
        )
        assert not result["complete_trajectory_verified"]
