"""Cooperative stops, residual identities and nonintrusive CPU progress."""

import json

import numpy as np
import pytest
from scipy import sparse

from deflation_example.coupled_optimize import observe_linear_solves
from deflation_example.linear_progress import LinearProgressWriter
from deflation_example.refinement import verified_refinement
from deflation_example.solvers import deflated_cg, independent_residual
from deflation_example.study_solvers import ArrayReference, StudySolver
from test_refinement import controlled_solver


def system():
    rng = np.random.default_rng(19)
    Q, _ = np.linalg.qr(rng.standard_normal((24, 24)))
    A = sparse.csr_matrix(Q @ np.diag(np.linspace(1, 300, 24)) @ Q.T)
    return A, rng.standard_normal(24), Q[:, :3]


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("refresh", [3, 1000])
@pytest.mark.parametrize("flexible", [False, True])
def test_progress_preserves_arithmetic_and_original_acceptance(rank, refresh, flexible):
    A, b, Z = system()
    kwargs = dict(
        basis=Z[:, :rank],
        diagonal=A.diagonal(),
        refresh=refresh,
        preconditioner=(lambda r: r / A.diagonal()) if flexible else None,
    )
    plain = deflated_cg(A, b, **kwargs)
    events = []
    observed = deflated_cg(
        A, b, **kwargs, progress_callback=events.append, stop_requested=lambda: False
    )
    assert plain.status == observed.status == "converged"
    assert plain.iterations == observed.iterations
    np.testing.assert_array_equal(plain.x, observed.x)
    assert plain.residual == observed.residual
    assert events[0]["stage"] == "initial" and events[-1]["stage"] == "final"
    assert events[-1]["residual"] == independent_residual(A, observed.x, b)
    assert events[-1]["residual_kind"] == "independently_recomputed"
    assert all(e["residual_scope"] == "current_kernel_rhs" for e in events)
    for event in events[1:-1]:
        assert event["residual_kind"] == (
            "independently_recomputed" if event["iteration"] % refresh == 0 else "recurrence"
        )


@pytest.mark.parametrize("already_solved", [False, True])
def test_expired_budget_skips_coarse_setup_but_verifies_state(already_solved):
    A, b, Z = system()
    initial = np.linalg.solve(A.toarray(), b) if already_solved else np.zeros_like(b)

    def forbidden(*args):
        raise AssertionError("Expired budget must not build a coarse space")

    events = []
    result = deflated_cg(
        A,
        b,
        Z,
        x0=initial,
        coarse_factory=forbidden,
        stop_requested=lambda: True,
        progress_callback=events.append,
    )
    assert result.status == ("converged" if already_solved else "budget_exhausted")
    assert result.iterations == 0
    np.testing.assert_array_equal(result.x, initial)
    assert result.residual == independent_residual(A, result.x, b)
    assert len(events) == 1 and events[0]["stage"] == "final"


def test_deadline_during_cg_returns_current_verified_candidate():
    A, b, _ = system()
    events = []
    result = deflated_cg(
        A,
        b,
        diagonal=A.diagonal(),
        progress_callback=events.append,
        stop_requested=lambda: any(e["iteration"] >= 2 for e in events),
    )
    assert result.status == "budget_exhausted" and result.iterations == 2
    assert result.residual == independent_residual(A, result.x, b)
    assert events[-1]["status"] == "budget_exhausted"


@pytest.mark.parametrize("quality", [0.5, 0.0, -1.0, 1.0])
def test_refinement_stops_on_deadline_and_retains_best_state(quality):
    solve, calls = controlled_solver(quality=quality, status="budget_exhausted")

    def first_then_stopped(*args):
        result, timing = solve(*args)
        if len(calls) == 1:
            result.status = "residual_failed"
        return result, timing

    A, b = sparse.eye(3), np.ones(3)
    result, metrics = verified_refinement(first_then_stopped, A, b, None, 1e-10, 100)
    assert len(calls) == 2
    assert result.status == ("converged" if quality == 1 else "budget_exhausted")
    assert result.residual == independent_residual(A, result.x, b)
    assert metrics["best_candidate_attempt"] == (1 if quality > 0 else 0)
    assert metrics["refinement_attempts"][-1]["status"] == "budget_exhausted"


@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_adapter_stops_without_launching_error_solves(method, tmp_path):
    A, b, Z = system()
    events = []
    adapter = StudySolver(
        method,
        rank=3,
        reference=ArrayReference(Z, {}),
        residual_policy="refine",
        progress_callback=events.append,
        stop_requested=lambda: any(e["iteration"] >= 2 for e in events),
    )
    previous_callback = adapter.progress_callback
    observe_linear_solves(adapter, tmp_path / "progress.json", heartbeat_seconds=1e-9)
    result, metrics = adapter.solve(A, b, np.arange(len(b)))
    assert result.status == "budget_exhausted"
    assert result.iterations == 2 and len(metrics["refinement_attempts"]) == 1
    assert adapter.progress_callback is previous_callback
    final = json.loads((tmp_path / "progress.json").read_text())
    assert final["status"] == result.status and final["residual"] == result.residual
    assert all(e["kernel_call"] == 1 for e in events)
    assert sum(metrics["components_seconds"].values()) == pytest.approx(metrics["total_seconds"])
    adapter.close()


def test_heartbeat_throttles_and_keeps_kernel_scope(tmp_path):
    clock = [0.0]
    path = tmp_path / "progress.json"
    writer = LinearProgressWriter(path, {"call": 3}, interval=30, clock=lambda: clock[0])
    event = dict(
        stage="initial",
        iteration=0,
        residual=1.0,
        residual_kind="independently_recomputed",
        residual_scope="current_kernel_rhs",
        status="running",
    )
    writer(event)
    first = path.read_bytes()
    clock[0] = 29
    writer({**event, "stage": "iteration", "iteration": 5})
    assert path.read_bytes() == first
    clock[0] = 30
    writer({**event, "stage": "iteration", "iteration": 6, "residual_kind": "recurrence"})
    row = json.loads(path.read_text())
    assert row["status"] == "running" and row["elapsed_seconds"] == 30
    assert row["kernel"]["residual_kind"] == "recurrence"
    clock[0] = 31
    writer({**event, "stage": "final", "iteration": 7, "status": "converged"})
    row = json.loads(path.read_text())
    assert row["status"] == "running"  # the outer refinement still verifies its original RHS
    assert row["kernel"]["status"] == "converged"


def test_callback_exception_propagates_and_restores_observer(tmp_path):
    A, b, _ = system()

    def broken(event):
        raise OSError("diagnostic writer unavailable")

    adapter = StudySolver("jacobi", progress_callback=broken)
    observe_linear_solves(adapter, tmp_path / "progress.json", heartbeat_seconds=1)
    with pytest.raises(OSError, match="diagnostic writer"):
        adapter.solve(A, b, np.arange(len(b)))
    assert adapter.progress_callback is broken


@pytest.mark.parametrize("argument", ["progress_callback", "stop_requested"])
def test_callbacks_require_supported_backend_and_callable(argument):
    with pytest.raises(ValueError, match="callable"):
        StudySolver("jacobi", **{argument: 1})
    with pytest.raises(ValueError, match="CPU"):
        StudySolver("jacobi", device="cuda", **{argument: lambda: False})
    with pytest.raises(ValueError, match="callable"):
        deflated_cg(np.eye(2), np.ones(2), **{argument: 1})


def test_zero_rhs_with_expired_budget_is_verified():
    result = deflated_cg(np.eye(3), np.zeros(3), stop_requested=lambda: True)
    assert result.status == "converged" and result.residual == 0
