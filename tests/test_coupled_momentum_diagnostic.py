"""Full-horizon momentum checks preserve temperature, time coupling and failures."""

import copy
import json

import numpy as np
import pytest

from deflation_example import coupled_momentum_diagnostic as diagnostic
from deflation_example import coupled_momentum_example as example
from test_coupled_derivatives import small_coupled_problem


def inputs():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    old = np.linspace(0.04, 0.12, problem.size)
    new = old + np.linspace(0.001, 0.003, problem.size)
    expected = problem.evaluate(new)
    retained = problem.evaluate(old)
    base = {
        "state": old,
        "velocity": np.stack([r.velocity for r in retained.flows]),
        "pressure": np.stack([r.pressure for r in retained.flows]),
    }
    return problem, base, new, expected


@pytest.mark.parametrize("subdivision", [1, 2, 4, 8])
def test_complete_interpolation_and_original_equations(subdivision):
    problem, base, new, expected = inputs()
    before = copy.deepcopy(base)
    report, fields = diagnostic.integrate(problem, new, base, subdivision)
    assert report["verified"] and report["completed_steps"] == 2 * subdivision
    np.testing.assert_array_equal(fields["previous_velocity"][0], problem.initial_flow.velocity)
    np.testing.assert_array_equal(fields["previous_velocity"][1:], fields["velocity"][:-1])
    assert fields["time_s"][-1] == pytest.approx(0.55)
    if subdivision == 1:
        np.testing.assert_allclose(
            fields["velocity"], np.stack([r.velocity for r in expected.flows]), atol=1e-10
        )
    for index, row in enumerate(report["steps"]):
        slab, j = row["slab_zero_based"], row["substep"]
        Y = new.reshape(problem.slabs, -1)
        first = problem.initial if slab == 0 else Y[slab - 1]
        state = Y[slab] if j == subdivision else first + (j / subdivision) * (Y[slab] - first)
        flow = diagnostic.FlowResult(
            fields["velocity"][index], fields["pressure"][index], row["status"], []
        )
        checks = problem.flow.verify(
            flow,
            diagnostic.acceleration(problem, state),
            problem.boundary_indices,
            problem.boundary_values,
            previous=fields["previous_velocity"][index],
            time_step=problem.physical_steps[slab] / subdivision,
            pressure_gauge=problem.pressure_gauge,
        )
        assert max(checks.values()) <= problem.flow_tolerance
    for key in base:
        np.testing.assert_array_equal(base[key], before[key])


def test_separate_prefix_and_local_reproduce_complete_integration():
    problem, base, new, _ = inputs()
    prefix, a = diagnostic.integrate(problem, new, base, stop_slab=1)
    previous = diagnostic.FlowResult(a["velocity"][-1], a["pressure"][-1], "converged", [])
    local, b = diagnostic.integrate(problem, new, base, start_slab=1, previous=previous)
    full, c = diagnostic.integrate(problem, new, base)
    assert prefix["verified"] and local["verified"] and full["verified"]
    np.testing.assert_array_equal(b["velocity"][-1], c["velocity"][-1])


def test_failed_step_remains_in_output_and_stops_propagation(monkeypatch):
    problem, base, new, _ = inputs()
    solve = diagnostic.solve_momentum
    calls = []

    def failed(*args, **kwargs):
        result = solve(*args, **kwargs)
        calls.append(1)
        result.status = "iteration_cap"
        return result

    monkeypatch.setattr(diagnostic, "solve_momentum", failed)
    report, fields = diagnostic.integrate(problem, new, base, 4)
    assert not report["verified"] and report["status"] == "iteration_cap"
    assert report["completed_steps"] == len(fields["velocity"]) == len(calls) == 1


def test_history_path_keeps_failed_results_out_of_seed(monkeypatch):
    problem, base, new, expected = inputs()
    solve = diagnostic.solve_momentum
    seeds = []

    def failed(*args, **kwargs):
        seeds.append(kwargs["initial"].velocity.copy())
        result = solve(*args, **kwargs)
        if len(seeds) == 2:
            result.status = "iteration_cap"
        return result

    monkeypatch.setattr(diagnostic, "solve_momentum", failed)
    report, fields = diagnostic.history_path(
        problem, base, new, 1, expected.flows[0].velocity, [0, 0.5, 1]
    )
    assert report["verified_count"] == 2
    assert report["cases"][2]["seed_fraction"] == 0
    np.testing.assert_array_equal(seeds[1], seeds[2])
    np.testing.assert_allclose(fields["velocity"][-1], expected.flows[1].velocity, atol=1e-10)


@pytest.mark.parametrize(
    "options",
    [{"subdivision": 0}, {"budget_seconds": float("nan")}, {"start_slab": 1}, {"stop_slab": 3}],
)
def test_invalid_controls_fail_before_integration(options):
    problem, base, new, _ = inputs()
    with pytest.raises(ValueError):
        diagnostic.integrate(problem, new, base, **options)


@pytest.mark.parametrize("fractions", [[0.1, 0.5], [0, 0.5, 0.5], [0, float("nan")], [0, 1.1]])
def test_invalid_path_rejected(fractions):
    problem, base, new, expected = inputs()
    with pytest.raises(ValueError):
        diagnostic.history_path(problem, base, new, 1, expected.flows[0].velocity, fractions)


@pytest.mark.parametrize("mode", ["trajectory", "local", "path"])
def test_self_contained_commands(tmp_path, monkeypatch, mode):
    output = tmp_path / mode
    monkeypatch.setattr(
        "sys.argv",
        ["momentum", "--example", "--mode", mode, "--subdivision", "2", "--output", str(output)],
    )
    example.main()
    report = json.loads((output / "record.json").read_text())
    assert report["source"]["kind"] == "self_contained_verification"
    assert report["status"] == ("complete" if mode == "path" else "converged")
    assert (output / "fields.npz").is_file()
