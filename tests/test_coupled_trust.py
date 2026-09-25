"""Trust-region accuracy, physical bounds and complete recovery checks."""

import copy
import json

import numpy as np
import pytest
from scipy import sparse

from deflation_example.coupled_optimizer import box_quadratic
from deflation_example.coupled_recovery import RecoveryStore
from deflation_example.coupled_trust import (
    POLICY,
    intermediate_targets,
    minimize_trust,
    radius_update,
)
from test_coupled_derivatives import small_coupled_problem
from test_coupled_optimizer import solver


def test_radius_rules():
    assert radius_update(0.25, 0.09, True) == (0.125, False)
    assert radius_update(0.25, 0.1, True) == (0.25, True)
    assert radius_update(0.25, 0.8, True) == (0.5, True)
    assert radius_update(2.0, 0.9, True) == (2.0, True)
    assert radius_update(0.25, float("nan"), False) == (0.125, False)
    assert radius_update(0.25, 1.0, True, flow_failed=True) == (0.125, False)


def test_intermediate_targets_restore_strict_accuracy():
    assert intermediate_targets(1, "strict", 1e-10, 1e-10) == (1e-10, 1e-10, True)
    q, linear, strict = intermediate_targets(1, "adaptive", 1e-10, 1e-10)
    assert q == 1e-2 and linear == 1e-4 and not strict
    assert intermediate_targets(1e-5, "adaptive", 1e-10, 1e-10) == (1e-10, 1e-10, True)
    with pytest.raises(ValueError):
        intermediate_targets(1, "unknown", 1e-10, 1e-10)


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
@pytest.mark.parametrize("accuracy", ["strict", "adaptive"])
def test_trust_solves_small_coupled_problem(steps, accuracy):
    from deflation_example.coupled_optimizer import minimize_coupled

    problem = small_coupled_problem(steps)
    desired = np.linspace(-0.5, 0.8, problem.size)
    result = minimize_trust(
        problem,
        desired,
        -0.04,
        0.15,
        solver(),
        inner_preconditioner="jacobi",
        accuracy=accuracy,
        max_iterations=80,
    )
    independent = minimize_coupled(problem, desired, -0.04, 0.15, solver(), max_iterations=80)
    assert result.status == independent.status == "converged", (result.status, result.kkt)
    assert max(result.kkt.values()) <= 1e-8
    np.testing.assert_allclose(result.evaluation.state, independent.evaluation.state, atol=2e-7)
    assert result.objective == pytest.approx(independent.objective, abs=1e-10)
    for row in result.history:
        for attempt in row["attempts"]:
            assert attempt["temperature_step_K"] <= attempt["radius_K"] * (1 + 1e-10)


def test_temporary_radius_is_not_a_physical_kkt_bound(monkeypatch):
    problem = small_coupled_problem()
    monkeypatch.setitem(POLICY, "initial_radius_K", 0.0025)
    desired = np.full(problem.size, 0.2)
    result = minimize_trust(
        problem, desired, -0.3, 0.3, solver(), inner_preconditioner="jacobi", max_iterations=1
    )
    assert result.status == "nonlinear_iteration_cap"
    assert max(result.kkt.values()) > 1e-8
    assert np.max(np.abs(result.evaluation.state)) <= 0.0025 * (1 + 1e-10)


def test_already_optimal_initial_state_needs_no_inner_solves():
    problem = small_coupled_problem()
    state = np.zeros(problem.size)
    ev = problem.evaluate(state)
    desired = (
        state + problem.alpha * (ev.jacobian.T @ (problem.weights * ev.control)) / problem.weights
    )
    result = minimize_trust(problem, desired, -0.1, 0.3, solver(), initial=state)
    assert result.status == "converged"
    assert result.history == []


def test_qp_resume_preserves_partition_and_solution():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(12, 12))
    H = sparse.csr_matrix(A.T @ A + np.eye(12))
    g = rng.normal(size=12)
    saved = []

    class Interrupted(Exception):
        pass

    def stop(payload):
        saved.append(copy.deepcopy(payload))
        raise Interrupted

    plain = box_quadratic(H, g, H.diagonal(), -0.1, 0.15, solver())
    with pytest.raises(Interrupted):
        box_quadratic(H, g, H.diagonal(), -0.1, 0.15, solver(), checkpoint=stop)
    resumed = box_quadratic(H, g, H.diagonal(), -0.1, 0.15, solver(), resume=saved[0])
    assert plain.status == resumed.status == "converged"
    np.testing.assert_allclose(plain.x, resumed.x, atol=1e-12)
    assert len(plain.history) == len(resumed.history)
    assert max(resumed.kkt.values()) <= 1e-9


@pytest.mark.parametrize("at_qp", [True, False])
def test_trust_resume_matches_uninterrupted(at_qp):
    problem = small_coupled_problem([0.2, 0.35])
    desired = np.linspace(-0.2, 0.4, problem.size)
    saved = []

    class Interrupted(Exception):
        pass

    def stop(payload):
        if (payload["qp"] is not None) if at_qp else payload["iteration"] == 1:
            saved.append(copy.deepcopy(payload))
            raise Interrupted

    arguments = dict(inner_preconditioner="jacobi", max_iterations=80)
    plain = minimize_trust(problem, desired, -0.05, 0.15, solver(), **arguments)
    with pytest.raises(Interrupted):
        minimize_trust(problem, desired, -0.05, 0.15, solver(), checkpoint=stop, **arguments)
    resumed = minimize_trust(problem, desired, -0.05, 0.15, solver(), resume=saved[0], **arguments)
    assert plain.status == resumed.status == "converged"
    np.testing.assert_allclose(plain.evaluation.state, resumed.evaluation.state, atol=1e-8)
    assert max(resumed.kkt.values()) <= 1e-8


def test_atomic_checkpoint_and_mismatch(tmp_path, monkeypatch):
    store = RecoveryStore(tmp_path, "source-and-config")
    store.save({"state": np.arange(3.0), "history": [{"residual": 1e-12}], "none": None})
    np.testing.assert_array_equal(store.load()["state"], np.arange(3.0))
    import deflation_example.coupled_recovery as recovery

    original = recovery.write_report
    monkeypatch.setattr(
        recovery, "write_report", lambda *args: (_ for _ in ()).throw(OSError("test"))
    )
    with pytest.raises(OSError):
        store.save({"state": np.ones(3)})
    np.testing.assert_array_equal(store.load()["state"], np.arange(3.0))
    monkeypatch.setattr(recovery, "write_report", original)
    with pytest.raises(ValueError, match="differs"):
        RecoveryStore(tmp_path, "changed-config").load()
    meta = json.loads(store.manifest.read_text())
    (tmp_path / f"state-{meta['slot']}.npz").write_bytes(b"corrupt test archive")
    with pytest.raises(ValueError, match="checksum"):
        store.load()


def test_nonfinite_checkpoint_arrays_rejected(tmp_path):
    store = RecoveryStore(tmp_path, "test")
    with pytest.raises(ValueError):
        store.save({"x": np.array([float("nan")])})


def test_saved_state_derivative_check():
    from deflation_example.coupled_trust_check import check

    problem = small_coupled_problem(
        [0.2, 0.35], consistent=True, streamline_rule="smooth_p8", uniform_capacity=True
    )
    state = np.linspace(0.02, 0.08, problem.size)
    result = check(problem, state, None, np.full(problem.size, 0.12))
    assert result["derivatives_passed"], result


def test_budget_keeps_state_and_residual_correspondence():
    problem = small_coupled_problem()
    desired = np.linspace(-0.1, 0.2, problem.size)
    result = minimize_trust(problem, desired, -0.2, 0.3, solver(), budget_seconds=1e-12)
    assert result.status == "budget_exhausted"
    objective, gradient = problem.objective_gradient(result.evaluation, desired)
    assert objective == result.objective
    np.testing.assert_array_equal(gradient, result.gradient)
    assert POLICY["minimum_radius_K"] > 0


@pytest.mark.parametrize("permanent", [False, True])
def test_failed_flow_trials_shrink_radius_without_changing_retained_state(monkeypatch, permanent):
    from deflation_example.axisymmetric_flow import FlowResult
    from deflation_example.coupled_control import FlowEvaluationError

    problem = small_coupled_problem()
    evaluate = problem.evaluate
    desired = np.full(problem.size, 0.3)

    def guarded(state, initial=None):
        if initial is not None and np.max(np.abs(state - initial.state)) > (
            0 if permanent else 0.02
        ):
            raise FlowEvaluationError(
                0,
                FlowResult(np.zeros((1, 2)), np.zeros(1), "stagnation", []),
                {"momentum_relative_residual": 1.0},
            )
        return evaluate(state, initial=initial)

    monkeypatch.setattr(problem, "evaluate", guarded)
    result = minimize_trust(
        problem, desired, -0.05, 0.2, solver(), inner_preconditioner="jacobi", max_iterations=80
    )
    failures = [
        a
        for row in result.history
        for a in row["attempts"]
        if a["trials"] and a["trials"][-1]["status"] == "flow_stagnation"
    ]
    assert failures
    assert result.status == ("trust_radius_exhausted" if permanent else "converged")
    if permanent:
        np.testing.assert_array_equal(result.evaluation.state, np.zeros(problem.size))
    objective, gradient = problem.objective_gradient(result.evaluation, desired)
    assert result.objective == objective
    np.testing.assert_array_equal(result.gradient, gradient)


def test_trust_matches_independent_optimizer():
    from scipy.optimize import minimize

    problem = small_coupled_problem([0.2, 0.35])
    desired = np.linspace(-0.2, 0.4, problem.size)
    result = minimize_trust(problem, desired, -0.04, 0.15, solver(), inner_preconditioner="jacobi")
    independent = minimize(
        lambda x: problem.objective_gradient(problem.evaluate(x), desired),
        np.zeros(problem.size),
        jac=True,
        bounds=[(-0.04, 0.15)] * problem.size,
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 200},
    )
    assert independent.success and result.status == "converged"
    np.testing.assert_allclose(result.evaluation.state, independent.x, atol=2e-6)


def test_budget_within_qp_retains_partial_work(monkeypatch):
    import deflation_example.coupled_trust as trust

    problem = small_coupled_problem()
    original = trust.box_quadratic
    clock = [0.0]
    monkeypatch.setattr(trust.time, "perf_counter", lambda: clock[0])

    def slowed(*args, checkpoint, **kwargs):
        def complete(payload):
            clock[0] += 2.0
            checkpoint(payload)

        return original(*args, checkpoint=complete, **kwargs)

    monkeypatch.setattr(trust, "box_quadratic", slowed)
    saved = []
    result = minimize_trust(
        problem,
        np.full(problem.size, 0.3),
        -0.05,
        0.2,
        solver(),
        inner_preconditioner="jacobi",
        budget_seconds=1.0,
        checkpoint=lambda p: saved.append(copy.deepcopy(p)),
    )
    assert result.status == "budget_exhausted"
    assert saved[-1]["qp"] is not None
    assert result.history[-1]["attempts"][-1]["qp_history"]


def test_stop_inside_cg_keeps_retained_nonlinear_state_and_records_work():
    problem = small_coupled_problem([0.2, 0.35])
    desired = np.full(problem.size, 0.3)
    adapter = solver()
    events, saved = [], []
    adapter.progress_callback = events.append
    adapter.stop_requested = lambda: any(e["iteration"] >= 1 for e in events)
    result = minimize_trust(
        problem,
        desired,
        -0.05,
        0.2,
        adapter,
        inner_preconditioner="jacobi",
        checkpoint=lambda p: saved.append(copy.deepcopy(p)),
    )
    assert result.status == "budget_exhausted"
    attempt = result.history[-1]["attempts"][-1]
    assert attempt["qp_status"] == "linear_budget_exhausted"
    assert sum(s["linear_iterations"] for s in attempt["qp_history"]) == 1
    assert attempt["qp_history"][-1]["candidate_retained"] is False
    np.testing.assert_array_equal(result.evaluation.state, np.zeros(problem.size))
    objective, gradient = problem.objective_gradient(result.evaluation, desired)
    assert result.objective == objective
    np.testing.assert_array_equal(result.gradient, gradient)
    np.testing.assert_array_equal(saved[-1]["state"], result.evaluation.state)
    assert saved[-1]["attempts"][-1]["qp_status"] == "linear_budget_exhausted"


def test_progress_reports_retained_objective_and_stationarity_scale():
    from deflation_example.coupled_trust import optimality

    problem = small_coupled_problem([0.2, 0.35])
    desired = np.full(problem.size, 0.3)
    rows = []
    result = minimize_trust(
        problem,
        desired,
        -0.05,
        0.2,
        solver(),
        inner_preconditioner="jacobi",
        max_iterations=1,
        callback=lambda row, ev: rows.append(copy.deepcopy(row)),
    )
    kkt, scale = optimality(problem, result.evaluation, desired, result.gradient, -0.05, 0.2)
    assert len(rows) == 1
    retained = rows[0]["retained"]
    assert retained["objective"] == result.objective
    assert retained["kkt"] == kkt
    assert retained["stationarity_scale"] == scale
    assert retained["stationarity_numerator"] == pytest.approx(kkt["stationarity"] * scale)


@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_recovery_runner_verifies_complete_small_sequence(tmp_path, monkeypatch, method, interrupt):
    import deflation_example.coupled_trust_run as runner
    from deflation_example.study_solvers import ArrayReference

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    baseline = {"baseline_sha256": "test", "seconds": 0.0}
    monkeypatch.setattr(
        runner,
        "snapshot_initial_guess",
        lambda *args: (problem.evaluate(np.zeros(problem.size)), {}),
    )
    constructions = []

    def construct(*args, **kwargs):
        constructions.append(1)
        return ArrayReference(np.eye(problem.size)[:, :2], {"test": True})

    monkeypatch.setattr(runner, "configured_krylov_reference", construct)
    cfg = dict(
        method=method,
        rank=2 if method != "jacobi" else 0,
        recycle_window=3,
        threads=1,
        inner_tolerance=1e-10,
        inner_cap=1000,
        inner_refresh=1000,
        queries=[{"target": 0, "upper_K": 300.2}, {"target": 1, "upper_K": 300.2}],
        target_count=2,
        target_startup_s=0.0,
        lower_K=299.95,
        nonlinear_tolerance=1e-8,
        nonlinear_cap=80,
        qp_tolerance=1e-10,
        qp_cap=100,
        secant_memory=3,
        inner_preconditioner="jacobi",
        frozen_sweeps=3,
        trust_accuracy="strict",
        equation_acceptance_tolerance=1e-9,
        conservation_tolerance=1e-6,
    )
    directory = tmp_path / method
    if interrupt:

        class Interrupted(Exception):
            pass

        save = RecoveryStore.save

        def interrupt_after_qp(self, payload):
            save(self, payload)
            if payload["optimizer"] is not None and payload["optimizer"]["qp"] is not None:
                raise Interrupted

        monkeypatch.setattr(RecoveryStore, "save", interrupt_after_qp)
        with pytest.raises(Interrupted):
            runner.run(cfg, directory, problem_loader=lambda cfg: (problem, baseline))
        monkeypatch.setattr(RecoveryStore, "save", save)
        result = runner.run(
            cfg,
            tmp_path / (method + "-resumed"),
            resume_from=directory,
            problem_loader=lambda cfg: (problem, baseline),
        )
        assert result["resumed"]
        assert result["prior_attempt_seconds"] > 0
        assert result["cumulative_attempt_seconds"] > result["attempt_seconds"]
    else:
        result = runner.run(cfg, directory, problem_loader=lambda cfg: (problem, baseline))
    assert result["all_problems_verified"], [(c["status"], c["kkt"]) for c in result["cases"]]
    assert result["attempt_seconds"] == pytest.approx(sum(result["components_seconds"].values()))
    assert result["verified_problems"] == 2
    assert (tmp_path / method / "recovery/latest.json").is_file()
    assert len(constructions) == (1 if method == "reference" else 0)
    from deflation_example.coupled_trust_report import summarize

    result_directory = tmp_path / (method + "-resumed") if interrupt else directory
    summary = summarize(result_directory / "record.json")
    assert summary["all_problems_verified"]
    assert summary["inner_iterations"] == sum(c["inner_iterations"] for c in result["cases"])
