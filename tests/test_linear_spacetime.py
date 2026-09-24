"""Exact fixed-flow trajectories, independent optimality and retained outcomes."""

import copy
import json
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
from scipy import sparse
from scipy.optimize import minimize

from deflation_example import linear_spacetime_sequence as sequence
from deflation_example.coupled_confirmation_report import inner_evidence
from deflation_example.coupled_nominal_krylov import configured_krylov_reference
from deflation_example.coupled_optimizer import box_quadratic
from deflation_example.coupled_report import validate_record
from deflation_example.linear_spacetime import FixedFlowProblem, solve_trajectory
from deflation_example.linear_spacetime_complete import linear_configuration, linear_settings
from test_coupled_derivatives import small_coupled_problem
from test_coupled_optimizer import solver
from test_coupled_regularization import protocol
from test_coupled_regularization_complete import screen
from test_coupled_sequence import configuration


def setup(steps=(0.2, 0.35, 0.15), consistent=True):
    physical = small_coupled_problem(
        steps,
        feedback=0,
        uniform_capacity=True,
        consistent=consistent,
        streamline_rule="smooth_p8",
    )
    problem = FixedFlowProblem(physical)
    cfg = configuration(problem)
    cfg.update(
        physics="prescribed_flow",
        feedback_multiplier=0.0,
        device="cpu",
        method="jacobi",
        rank=0,
        recycle_window=2,
        inner_preconditioner="jacobi",
        frozen_sweeps=3,
        inner_tolerance=1e-10,
        inner_cap=1000,
        threads=1,
        reference_krylov_steps=6,
        reference_krylov_seed=20260923,
        reference_selection="krylov_coupled",
        memory_interval=0.01,
        repetition=0,
    )
    cfg["queries"][1]["upper_K"] = cfg["queries"][0]["upper_K"]
    return physical, problem, cfg


@pytest.mark.parametrize("steps", [None, [0.2, 0.35, 0.15]])
@pytest.mark.parametrize("consistent", [False, True])
def test_exact_control_and_transpose_match_feedback_disabled_coupled_model(steps, consistent):
    physical, problem, cfg = setup(steps, consistent)
    rng = np.random.default_rng(217)
    J = problem.jacobian @ np.eye(problem.size)
    np.testing.assert_allclose(problem.jacobian.T @ np.eye(problem.size), J.T, atol=2e-14)
    for state in [np.zeros(problem.size), rng.uniform(-0.05, 0.15, problem.size)]:
        actual = problem.evaluate(state)
        expected = physical.evaluate(state)
        np.testing.assert_allclose(actual.control, expected.control, atol=2e-12, rtol=2e-11)
        np.testing.assert_allclose(J, expected.jacobian @ np.eye(problem.size), atol=2e-11)
        desired = np.full(problem.size, 0.12)
        report = problem.verify_adjoint(actual, desired)
        assert report["gradient_weight_normalized_difference"] < 1e-12
        assert report["maximum_source_adjoint_relative_residual"] < 1e-12
        from deflation_example.coupled_optimize import equation_acceptance

        assert equation_acceptance(problem.verify(actual, local_mass=True), cfg)
    H = problem.H @ np.eye(problem.size)
    np.testing.assert_allclose(
        H, np.diag(problem.weights) + problem.alpha * J.T @ np.diag(problem.weights) @ J
    )
    np.testing.assert_allclose(H, H.T, atol=2e-13)
    assert np.linalg.eigvalsh(H).min() > 0
    if consistent:
        assert np.linalg.norm(J - problem.frozen_operator.toarray()) > 0.01
    if steps:
        n = problem.spatial_size
        S = problem.assembly.source_action[problem.free][:, problem.free].toarray()
        C = problem.assembly.storage[problem.free][:, problem.free].toarray()
        for level in range(1, len(steps)):
            np.testing.assert_allclose(
                J[level * n : (level + 1) * n, (level - 1) * n : level * n],
                -np.linalg.solve(S, C) / problem.steps[level],
                atol=2e-14,
            )
        assert np.count_nonzero(J[:n, n:]) == 0
        vector = np.zeros(problem.size)
        vector[-n:] = 1
        assert np.linalg.norm((problem.jacobian.T @ vector)[-2 * n : -n]) > 0


def test_fixed_flow_evaluation_never_runs_momentum_or_uses_saved_flow(monkeypatch):
    physical, problem, _ = setup()

    def forbidden(*args, **kwargs):
        raise AssertionError("Prescribed-flow application must not solve momentum")

    monkeypatch.setattr(physical, "evaluate", forbidden)
    monkeypatch.setattr(physical.flow, "solve", forbidden)
    actual = problem.evaluate(np.zeros(problem.size), initial=object())
    assert all(flow is physical.initial_flow for flow in actual.flows)
    assert problem.jacobian.factors == (None,) * problem.slabs
    assert problem.jacobian.thermal_only
    with pytest.raises(ValueError, match="zero|feedback"):
        FixedFlowProblem(small_coupled_problem())
    for state in [np.zeros(problem.size - 1), np.full(problem.size, np.nan)]:
        with pytest.raises(ValueError):
            problem.evaluate(state)


def test_independent_checks_detect_corrupt_control_and_temporal_transpose():
    _, problem, cfg = setup()
    from deflation_example.coupled_optimize import equation_acceptance

    state = np.linspace(0.03, 0.1, problem.size)
    final = problem.evaluate(state)
    final.control[0] += 0.01
    assert not equation_acceptance(problem.verify(final), cfg)
    final = problem.evaluate(state)
    thermal = problem.jacobian.thermal.copy().tolil()
    n = problem.spatial_size
    thermal[-n:, -2 * n : -n] = 0
    problem.jacobian.thermal = thermal.tocsr()
    report = problem.verify_adjoint(final, np.full(problem.size, 0.2))
    assert report["gradient_weight_normalized_difference"] > 1e-6


@pytest.mark.parametrize("consistent", [False, True])
def test_nonuniform_capacity_and_physical_time_weights(consistent):
    physical = small_coupled_problem([0.2, 0.4], feedback=0, consistent=consistent)
    problem = FixedFlowProblem(physical)
    state = np.linspace(0.03, 0.11, problem.size)
    np.testing.assert_allclose(
        problem.evaluate(state).control, physical.evaluate(state).control, atol=1e-11
    )
    mass = physical.assembly.mass[physical.free]
    np.testing.assert_allclose(
        problem.weights, np.kron(np.array([0.2, 0.4]) / 0.3, mass / mass.mean())
    )
    assert problem.objective_scale == pytest.approx(mass.mean() * physical.steps.mean())


def test_isothermal_refinement_is_verified_and_ignores_targets(monkeypatch):
    physical, _, cfg = setup()
    cfg.update(equation_acceptance_tolerance=1e-12, flow_cap=20)
    report = sequence.refine_isothermal(physical, cfg)
    assert max(report["after"].values()) <= 1e-13
    assert report["velocity_relative_change"] <= 1e-6
    assert report["velocity_sha256"]
    assert report["seconds"] >= 0
    calls = []
    verify = physical.flow.verify

    def residual(*args, **kwargs):
        actual = verify(*args, **kwargs)
        calls.append(actual)
        if 2 <= len(calls) <= 4:
            actual = {**actual, "momentum_relative_residual": 1e-10}
        return actual

    monkeypatch.setattr(physical.flow, "verify", residual)
    forced = sequence.refine_isothermal(physical, cfg)
    assert forced["history"]
    assert max(forced["after"].values()) <= 1e-13


def test_failed_isothermal_candidate_keeps_initial_state_and_its_residual(monkeypatch):
    from dataclasses import replace

    physical, _, cfg = setup()
    cfg.update(equation_acceptance_tolerance=1e-12, flow_cap=4)
    original = physical.initial_flow
    candidate = replace(original, velocity=original.velocity + 1e-14, status="iteration_cap")

    def verify(result, *args, **kwargs):
        return {
            "momentum_relative_residual": 2e-12 if result is original else 3e-12,
            "continuity_relative_residual": 1e-16,
        }

    monkeypatch.setattr(physical.flow, "verify", verify)
    monkeypatch.setattr(physical.flow, "solve", lambda *a, **k: candidate)
    with pytest.raises(sequence.IsothermalRefinementError) as failure:
        sequence.refine_isothermal(physical, cfg)
    report = failure.value.diagnostics
    assert report["after"]["momentum_relative_residual"] == 2e-12
    assert report["history"][0]["trajectory_checks"]["momentum_relative_residual"] == 3e-12
    assert report["history"][0]["candidate_retained"] is False
    assert report["velocity_relative_change"] == 0
    assert physical.initial_flow is original


def test_initial_isothermal_accuracy_bypasses_corrections(monkeypatch):
    physical, _, cfg = setup()
    cfg.update(equation_acceptance_tolerance=1e-12, flow_cap=4)

    def forbidden(*args, **kwargs):
        raise AssertionError("An already verified flow needs no Newton step")

    monkeypatch.setattr(physical.flow, "solve", forbidden)
    report = sequence.refine_isothermal(physical, cfg)
    assert report["status"] == "verified" and report["history"] == []


@pytest.mark.parametrize("arm", ["jacobi", "frozen", "reference", "recycling"])
def test_four_linear_solvers_match_independent_constrained_optimizer(arm):
    _, problem, cfg = setup()
    desired = np.linspace(-0.5, 0.8, problem.size)
    lower, upper = -0.04, 0.15
    initial = np.full(problem.size, upper)
    cfg["inner_preconditioner"] = "jacobi" if arm == "jacobi" else "frozen"
    reference = None
    if arm == "reference":
        cfg.update(method="reference", rank=2)
        reference = configured_krylov_reference(problem, cfg, initial_state=np.zeros(problem.size))
        assert "fixed-flow" in reference.description["nominal_policy"]
        saved = reference.restrict(np.arange(problem.size)).copy()
    method = arm if arm in {"reference", "recycling"} else "jacobi"
    inner = solver(method, reference, rank=2 if reference is not None or arm == "recycling" else 0)
    final, result, objective, kkt = solve_trajectory(
        problem, desired, lower, upper, inner, cfg, initial
    )
    J = problem.jacobian @ np.eye(problem.size)
    weights = problem.weights

    def independent(state):
        control = J @ state + problem.offset
        value = 0.5 * np.sum(weights * ((state - desired) ** 2 + problem.alpha * control**2))
        gradient = weights * (state - desired) + problem.alpha * J.T @ (weights * control)
        return value, gradient

    expected = minimize(
        independent,
        np.zeros(problem.size),
        jac=True,
        method="SLSQP",
        bounds=[(lower, upper)] * problem.size,
        options={"ftol": 1e-13, "maxiter": 500},
    )
    assert expected.success
    assert result.status == "converged", (result.status, kkt)
    assert max(kkt.values()) <= 1e-8
    np.testing.assert_allclose(final.state, expected.x, atol=2e-6)
    assert objective == pytest.approx(expected.fun, abs=1e-11)
    assert any(final.state == lower) and any(final.state == upper)
    assert any((final.state > lower) & (final.state < upper))
    assert all(step.get("linear_residual", 0) <= inner.rtol for step in result.history)
    again = solve_trajectory(problem, desired, lower, upper, inner, cfg, final.state)
    assert again[1].history == []
    if reference is not None:
        np.testing.assert_array_equal(reference.restrict(np.arange(problem.size)), saved)
    inner.close()


def test_warm_initial_quadratic_guard_and_custom_kkt():
    H = sparse.eye(3, format="csr")
    g = -np.array([0.1, -0.2, 0.3])
    initial = -g.copy()
    seen = []

    def check(x, gradient):
        seen.append(x.copy())
        return {"stationarity": float(np.max(np.abs(gradient)))}

    result = box_quadratic(H, g, np.ones(3), -1, 1, solver(), initial=initial, kkt_evaluator=check)
    assert result.status == "converged" and not result.history
    np.testing.assert_array_equal(result.x, initial)
    np.testing.assert_array_equal(seen[-1], result.x)
    zero = box_quadratic(H, np.zeros(3), np.ones(3), -1, 1, solver(), initial=np.zeros(3))
    assert zero.status == "converged" and zero.history == []
    assert max(zero.kkt.values()) == 0
    for bad in [np.zeros(2), np.full(3, np.inf)]:
        with pytest.raises(ValueError, match="Initial quadratic"):
            box_quadratic(H, g, np.ones(3), -1, 1, solver(), initial=bad)


@pytest.mark.parametrize("failure", [None, "cap", "memory"])
def test_linear_sequence_retains_failures_and_uses_only_verified_warm_states(monkeypatch, failure):
    _, problem, cfg = setup()
    original = sequence.solve_trajectory
    starts = []

    def solve(*args):
        starts.append(args[-1].copy())
        if failure == "memory" and len(starts) == 1:
            raise MemoryError("controlled test")
        result = original(*args)
        if failure == "cap" and len(starts) == 1:
            result[1].status = "active_set_cap"
        return result

    monkeypatch.setattr(sequence, "solve_trajectory", solve)
    seed = SimpleNamespace(state=np.full(problem.size, 0.04))
    cases, fields = sequence.optimize_targets(problem, solver(), cfg, initial_guess=seed)
    assert len(cases) == len(fields) == 2
    np.testing.assert_array_equal(starts[0], seed.state)
    if failure == "memory":
        assert [c["status"] for c in cases] == ["memory_limited", "not_run_after_numerical_error"]
    elif failure == "cap":
        assert not cases[0]["verified"] and cases[1]["verified"]
        np.testing.assert_array_equal(starts[1], np.zeros(problem.size))
    else:
        assert all(c["verified"] for c in cases), cases
        np.testing.assert_array_equal(starts[1], fields[0]["state"])
    assert cases[1]["warm_start_used"] == (failure is None)


def test_complete_linear_record_has_direct_histories_and_matched_timing(monkeypatch, tmp_path):
    _, problem, cfg = setup()
    cfg.update(output=str(tmp_path / "complete"), evaluation_progress=True, linear_progress=True)
    monkeypatch.setattr(
        sequence,
        "load_fixed_flow",
        lambda cfg: (
            problem,
            {
                "baseline_sha256": "test",
                "configuration": {},
                "input_sha256": {},
                "seconds": 0.25,
            },
        ),
    )
    report = sequence.run(OmegaConf.create(cfg))
    assert report["all_problems_verified"], report
    assert validate_record(report)
    assert report["numerical_policy"] == sequence.NUMERICAL_POLICY
    assert "linear-quadratic" in report["scope"]
    assert sum(report["components_seconds"].values()) == pytest.approx(report["sequence_seconds"])
    assert report["preparation_inclusive_seconds"] == pytest.approx(
        report["sequence_seconds"] + report["process_preparation_seconds"] + 0.25
    )
    evidence = inner_evidence(report)
    assert evidence["complete_histories"]
    assert evidence["recorded_inner_iterations"] == sum(
        c["inner_iterations"] for c in report["cases"]
    )
    assert evidence["maximum_converged_original_residual"] <= cfg["inner_tolerance"]
    assert all("history" not in c and "nonlinear_iterations" not in c for c in report["cases"])
    with pytest.raises(FileExistsError):
        sequence.run(OmegaConf.create(cfg))
    report["configuration"].pop("physics")
    with pytest.raises(ValueError, match="prescribed-flow"):
        inner_evidence(report)


def test_linear_protocol_changes_only_feedback_and_declared_physics(tmp_path):
    path = tmp_path / "screen.json"
    path.write_text(json.dumps(screen()))
    settings = linear_settings(path)
    original = {**protocol(), "queries": [{"target": 7, "upper_K": 357.3}]}
    record = {"all_problems_verified": True, "cases": [{}], "configuration": original}
    before = copy.deepcopy(record)
    from deflation_example.coupled_regularization_complete import complete_configuration

    for arm in settings["arms"]:
        actual = linear_configuration(record, settings, arm, 0)
        expected = complete_configuration(record, settings, arm, 0)
        assert actual == {**expected, "physics": "prescribed_flow", "feedback_multiplier": 0.0}
        assert actual["slabs"] == 64 and actual["horizon_s"] == 600
        assert actual["target_startup_s"] == 60
        assert actual["alpha"] == 1e-11
    assert record == before
    frozen = tmp_path / "settings.json"
    frozen.write_text(json.dumps(settings))
    assert linear_settings(path, frozen) == settings
    settings["feedback_multiplier"] = 1
    frozen.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="declared matched"):
        linear_settings(path, frozen)


def test_linear_population_gates_and_physics_separation():
    from deflation_example.coupled_confirmation_report import summarize
    from test_coupled_confirmation_report import population

    settings, records, fields = population()
    for record in records:
        record["configuration"]["physics"] = "prescribed_flow"
        record["fixed_flow_preparation"] = {"velocity_sha256": "test-fixed-flow"}
        for case in record["cases"]:
            case["pdas_history"] = case.pop("history")[0]["attempts"][0]["qp_history"]
            case["pdas_steps"] = len(case["pdas_history"])
    assert summarize(records, settings, fields)["publication_gate_passed"]
    records[-1]["fixed_flow_preparation"]["velocity_sha256"] = "changed"
    with pytest.raises(ValueError, match="settings must match"):
        summarize(records, settings, fields)
    records[-1]["fixed_flow_preparation"]["velocity_sha256"] = "test-fixed-flow"
    records[-1]["configuration"]["physics"] = "coupled"
    with pytest.raises(ValueError, match="settings must match"):
        summarize(records, settings, fields)
    records[-1]["configuration"]["physics"] = "prescribed_flow"
    case = records[-1]["cases"][0]
    case.update(status="linear_iteration_cap", verified=False)
    case["pdas_history"][0].update(linear_status="iteration_cap", candidate_retained=False)
    records[-1].update(all_problems_verified=False, verified_problems=2)
    assert not summarize(records, settings, fields)["publication_gate_passed"]
