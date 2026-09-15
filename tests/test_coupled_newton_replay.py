"""Check unchanged controls, coupled equations and retained Newton candidates."""

import numpy as np
import pytest

from deflation_example.coupled_newton_replay import newton_step, step_equations, criteria_met
from deflation_example.coupled_resolution import forward_model
from test_coupled_derivatives import small_coupled_problem


def data():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    state = np.linspace(0.04, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    source = np.zeros(len(problem.mesh.nodes))
    source[problem.free] = evaluation.control.reshape(problem.slabs, -1)[0]
    return problem, state, evaluation, source


@pytest.mark.parametrize("line_search", ["equation_max", "fixed_scaled"])
def test_forward_newton_recovers_complete_known_trajectory_from_fixed_sources(line_search):
    problem, expected, evaluation, _ = data()
    state = problem.full_temperature(problem.initial)
    flow = problem.initial_flow
    for n in range(problem.slabs):
        source = np.zeros(len(problem.mesh.nodes))
        source[problem.free] = evaluation.control.reshape(problem.slabs, -1)[n]
        original = source.copy()
        result = newton_step(
            problem, source, state, flow, n, tolerance=1e-11, line_search=line_search
        )
        assert result.status == "converged", result.history
        _, metrics = step_equations(
            problem, forward_model(problem), result.state, result.flow, source, state, flow, n
        )
        assert criteria_met(metrics, 1e-11)
        for name, value in metrics.items():
            assert result.history[-1][name] == value
        np.testing.assert_allclose(
            result.state[problem.free], expected.reshape(2, -1)[n], atol=2e-10
        )
        np.testing.assert_array_equal(source, original)
        assert len(result.history) <= 10
        state, flow = result.state, result.flow


def test_fixed_scaled_merit_uses_the_original_equations_for_final_acceptance():
    from deflation_example.coupled_newton_repair import compare_step

    problem, _, _, source = data()
    previous = problem.full_temperature(problem.initial)
    snapshots = []
    rows, fields = compare_step(
        problem,
        source,
        previous,
        problem.initial_flow,
        previous,
        problem.initial_flow,
        0,
        callback=lambda rows, fields: snapshots.append((rows, fields)),
    )
    assert all(row["status"] == "converged" and row["independent_criteria_met"] for row in rows)
    np.testing.assert_allclose(
        fields["equation_max_state"], fields["fixed_scaled_state"], atol=1e-11
    )
    assert rows[1]["backtrack_cap"] == 40
    assert [len(rows) for rows, _ in snapshots] == [1, 2]
    assert set(snapshots[0][1]) == {
        "equation_max_state",
        "equation_max_velocity",
        "equation_max_pressure",
    }


@pytest.mark.parametrize("options", [{"line_search": "unknown"}, {"backtrack_cap": 0}])
def test_invalid_globalization_policy_rejected(options):
    problem, _, _, source = data()
    with pytest.raises(ValueError):
        newton_step(
            problem,
            source,
            problem.full_temperature(problem.initial),
            problem.initial_flow,
            0,
            **options,
        )


def test_converged_initial_guard_and_zero_budget_keep_matching_fields():
    problem, expected, evaluation, source = data()
    previous = problem.full_temperature(problem.initial)
    initial = problem.full_temperature(expected.reshape(2, -1)[0])
    saved = newton_step(
        problem,
        source,
        previous,
        problem.initial_flow,
        0,
        initial_state=initial,
        initial_flow=evaluation.flows[0],
        tolerance=1e-10,
        max_iterations=0,
    )
    assert saved.status == "converged"
    assert len(saved.history) == 1
    np.testing.assert_array_equal(saved.state, initial)
    capped = newton_step(problem, source, previous, problem.initial_flow, 0, max_iterations=0)
    assert capped.status == "newton_iteration_cap"
    np.testing.assert_array_equal(capped.state, previous)


def test_linear_failure_retains_previous_temperature_velocity_and_residual(monkeypatch):
    import deflation_example.coupled_newton_replay as module

    problem, _, _, source = data()
    previous = problem.full_temperature(problem.initial)

    def fail(*args, **kwargs):
        raise RuntimeError("test factorization failure")

    monkeypatch.setattr(module, "splu", fail)
    result = newton_step(problem, source, previous, problem.initial_flow, 0)
    assert result.status == "newton_linear_failure"
    np.testing.assert_array_equal(result.state, previous)
    np.testing.assert_array_equal(result.flow.velocity, problem.initial_flow.velocity)
    _, metrics = step_equations(
        problem,
        forward_model(problem),
        result.state,
        result.flow,
        source,
        previous,
        problem.initial_flow,
        0,
    )
    for name, value in metrics.items():
        assert result.history[-1][name] == value


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, 1e-7])
def test_invalid_forward_targets_are_rejected(tolerance):
    problem, _, _, source = data()
    with pytest.raises(ValueError):
        newton_step(
            problem,
            source,
            problem.full_temperature(problem.initial),
            problem.initial_flow,
            0,
            tolerance=tolerance,
        )


def test_zero_temperature_and_source_need_no_newton_factorization(monkeypatch):
    import deflation_example.coupled_newton_replay as module

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    problem.initial[:] = 0
    problem.thermal_boundary[:] = 0
    problem.source[:] = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("The verified zero state must return before factorization")

    monkeypatch.setattr(module, "splu", forbidden)
    result = newton_step(
        problem,
        np.zeros(len(problem.mesh.nodes)),
        problem.full_temperature(problem.initial),
        problem.initial_flow,
        0,
        tolerance=1e-10,
    )
    assert result.status == "converged"
    assert len(result.history) == 1
    assert not np.any(result.state)


def test_rejected_newton_trials_keep_the_initial_fields_and_metrics(monkeypatch):
    import deflation_example.coupled_newton_replay as module

    problem, _, _, source = data()
    previous = problem.full_temperature(problem.initial)
    evaluate = module.step_equations
    first = None

    def reject_trials(*args, **kwargs):
        nonlocal first
        residual, metrics = evaluate(*args, **kwargs)
        if first is None:
            first = dict(metrics)
        else:
            metrics = {name: 2 * value for name, value in first.items()}
        return residual, metrics

    monkeypatch.setattr(module, "step_equations", reject_trials)
    result = newton_step(problem, source, previous, problem.initial_flow, 0)
    assert result.status == "newton_line_search_stagnation"
    assert len(result.history[-1]["trials"]) == 21
    assert not result.history[-1]["candidate_retained"]
    np.testing.assert_array_equal(result.state, previous)
    np.testing.assert_array_equal(result.flow.velocity, problem.initial_flow.velocity)
    for name, value in first.items():
        assert result.history[-1][name] == value


@pytest.mark.parametrize("subdivision", [1, 2])
@pytest.mark.parametrize("time_scheme", ["backward_euler", "bdf2"])
def test_resolution_cli_preserves_source_history_and_recomputes_comparisons(
    monkeypatch, tmp_path, subdivision, time_scheme
):
    import json
    import sys
    import deflation_example.coupled_newton_replay as module
    from deflation_example.coupled_resolution import replay_controls
    from deflation_example.coupled_targets import desired_temperature

    original_problem, original_state, evaluation, _ = data()
    desired = desired_temperature(original_problem, 0, 2)
    config = {
        "slabs": 2,
        "transient": True,
        "query": 0,
        "target_count": 2,
        "lower_K": 299.0,
        "upper_K": 301.0,
    }
    fields = {"state": original_state, "control": evaluation.control.copy(), "desired": desired}
    source_before = fields["control"].copy()
    monkeypatch.setattr(
        module,
        "load_saved_solution",
        lambda *args: ({"baseline_sha256": "test"}, config, fields, "saved-field-digest"),
    )

    def load(cfg):
        assert cfg["slabs"] == 2 * subdivision
        problem = small_coupled_problem(
            np.repeat(np.array([0.2, 0.35]) / subdivision, subdivision), uniform_capacity=True
        )
        return problem, {"baseline_sha256": "test"}

    monkeypatch.setattr(module, "load_problem", load)
    output = tmp_path / "newton"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay",
            "--baseline",
            "baseline",
            "--optimization",
            "saved",
            "--subdivision",
            str(subdivision),
            "--time-scheme",
            time_scheme,
            "--tolerance",
            "1e-11",
            "--output",
            str(output),
        ],
    )
    module.main()
    report = json.loads((output / "record.json").read_text())
    assert report["status"] == "converged"
    assert report["subdivision"] == subdivision
    assert report["forward_slabs"] == 2 * subdivision
    assert report["forward_solver"]["time_scheme"] == time_scheme
    np.testing.assert_array_equal(fields["control"], source_before)
    if time_scheme == "backward_euler":
        comparison = replay_controls(
            original_problem, source_before, subdivision=subdivision, tolerance=1e-10
        )
    else:
        problem, _ = load({"slabs": 2 * subdivision})
        comparison = module.newton_trajectory(
            problem,
            np.repeat(source_before.reshape(2, -1), subdivision, axis=0),
            tolerance=1e-11,
            time_scheme="bdf2",
            restart_interval=subdivision,
        )
    assert comparison["status"] == "converged"
    with np.load(output / "states.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["state"], comparison["states"], atol=1e-8)
        endpoint_error = (
            np.max(
                np.abs(
                    arrays["state"][subdivision - 1 :: subdivision] - original_state.reshape(2, -1)
                )
            )
            * original_problem.temperature_scale
        )
    assert report["maximum_endpoint_difference_K"] == pytest.approx(endpoint_error)
    assert (report["resolution_thresholds_met"] is None) == (subdivision == 1)


def test_newton_trajectory_replays_saved_controls_and_retains_partial_failure():
    from deflation_example.coupled_newton_replay import newton_trajectory

    problem, state, evaluation, _ = data()
    source = evaluation.control.copy()
    progress = []
    result = newton_trajectory(problem, source, callback=progress.append)
    assert result["status"] == "converged"
    assert len(progress) == problem.slabs
    np.testing.assert_allclose(result["states"].ravel(), state, atol=1e-8)
    np.testing.assert_array_equal(source, evaluation.control)
    failed = newton_trajectory(problem, source, cap=0)
    assert failed["status"] == "newton_iteration_cap"
    assert len(failed["states"]) == 1
    assert len(failed["steps"]) == 1


@pytest.mark.parametrize("subdivision", [1, 2])
@pytest.mark.parametrize("line_search", ["equation_max", "fixed_scaled"])
def test_newton_spatial_cli_preserves_forward_protocol_and_fields(
    tmp_path, monkeypatch, subdivision, line_search
):
    import json
    import sys
    from scipy import sparse
    import deflation_example.coupled_spatial_resolution as module
    from deflation_example.coupled_targets import desired_temperature

    problem, state, evaluation, _ = data()
    config = {
        "slabs": 2,
        "transient": True,
        "query": 0,
        "target_count": 2,
        "lower_K": 299.0,
        "upper_K": 301.0,
    }
    fields = {
        "state": state,
        "control": evaluation.control,
        "desired": desired_temperature(problem, 0, 2),
    }
    baseline = {"baseline_sha256": "test", "configuration": {"convection_form": "advective"}}
    monkeypatch.setattr(
        module, "load_saved_solution", lambda *a: (baseline, config, fields, "fields")
    )
    comparison = small_coupled_problem(
        np.repeat(np.array([0.2, 0.35]) / subdivision, subdivision), uniform_capacity=True
    )

    def load(cfg):
        assert cfg["slabs"] == 2 * subdivision
        return comparison, baseline

    monkeypatch.setattr(module, "load_problem", load)
    monkeypatch.setattr(
        module,
        "transfer_source",
        lambda c, f, u: (
            u.reshape(comparison.slabs, -1).copy(),
            sparse.eye(problem.spatial_size, format="csr"),
        ),
    )
    output = tmp_path / "spatial"
    extra = []
    expected = state.reshape(2, -1)
    if subdivision > 1:
        from deflation_example.coupled_newton_replay import newton_trajectory

        repeated = np.repeat(evaluation.control.reshape(2, -1), subdivision, axis=0)
        replay = newton_trajectory(comparison, repeated, tolerance=1e-12)
        assert replay["status"] == "converged"
        expected = replay["states"]
        directory = tmp_path / "coarse-replay"
        directory.mkdir()
        (directory / "record.json").write_text(
            json.dumps(
                {
                    "configuration": config,
                    "optimization_field_sha256": "fields",
                    "baseline_sha256": "test",
                    "status": "converged",
                    "steps": replay["steps"],
                    "forward_solver": {"tolerance": 1e-12},
                }
            )
        )
        np.savez(
            directory / "states.npz", state=expected, times_s=np.cumsum(comparison.physical_steps)
        )
        extra = ["--subdivision", str(subdivision), "--coarse-replay", str(directory)]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spatial",
            "--baseline",
            "coarse",
            "--fine-baseline",
            "fine",
            "--optimization",
            "saved",
            "--procedure",
            "monolithic_newton",
            "--line-search",
            line_search,
            "--backtrack-cap",
            "40",
            "--tolerance",
            "1e-12",
            "--output",
            str(output),
            *extra,
        ],
    )
    module.main()
    report = json.loads((output / "record.json").read_text())
    assert report["status"] == "converged"
    assert report["forward_solver"]["procedure"] == "monolithic_newton"
    assert report["forward_solver"]["tolerance"] == 1e-12
    assert report["forward_solver"]["line_search"] == line_search
    assert report["forward_solver"]["backtrack_cap"] == 40
    assert report["maximum_temperature_difference_K"] < 1e-8
    assert report["subdivision"] == subdivision
    assert (report["coarse_replay"] is not None) == (subdivision > 1)
    with np.load(output / "states.npz") as arrays:
        assert {"velocity", "pressure", "state", "times_s"} <= set(arrays.files)
        np.testing.assert_allclose(arrays["state"], expected, atol=1e-8)
