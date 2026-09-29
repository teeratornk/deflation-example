"""Same systems, retained transfer history, and independently checked error metrics."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse.linalg import aslinearoperator

from deflation_example import coupled_matched_transfer as replay
from deflation_example.coupled_trace import CoupledTrace
from deflation_example.reporting import file_sha256, write_arrays, write_report
from deflation_example.solvers import LinearResult


def test_transition_counts_do_not_confuse_initial_mask_with_release():
    first = replay.transfer_counts(None, np.array([0, 2]), 4)
    assert first == {"inactive": 2, "active": 2, "newly_active": None, "newly_inactive": None}
    assert replay.transfer_counts(np.array([0, 2]), np.array([1, 2, 3]), 4) == {
        "inactive": 3,
        "active": 1,
        "newly_active": 1,
        "newly_inactive": 2,
    }


def test_energy_fraction_uses_error_for_the_recorded_start():
    A = np.diag([0.1, 1, 5, 20])
    error = np.array([0.1, 1, 2, 3])
    basis = np.eye(4)[:, :1]
    result = replay.coarse_energy(aslinearoperator(A), A @ error, error, basis)
    assert result["energy_fraction_removed"] == pytest.approx(0.001 / (error @ A @ error))
    assert result["deployed_rank"] == 1
    empty = replay.coarse_energy(aslinearoperator(A), A @ error, error, basis[:, :0])
    assert empty["energy_fraction_removed"] == pytest.approx(0)
    dependent = replay.coarse_energy(
        aslinearoperator(A), A @ error, error, np.column_stack((basis, basis))
    )
    assert dependent["deployed_rank"] == 1


def test_independent_error_solve_and_zero_load():
    A = aslinearoperator(np.diag([0.1, 1, 3, 5]))
    A.diagonal = lambda: np.array([0.1, 1, 3, 5])
    for rhs in (np.zeros(4), np.ones(4)):
        initial = np.zeros(4)
        error, report = replay.error_solution(A, rhs, initial, np.arange(4), 100, lambda: False)
        if np.any(rhs):
            np.testing.assert_allclose(A @ error, rhs)
            assert report["status"] == "verified"
        else:
            assert error is None and report["status"] == "initially_converged"


def fixture_capture(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    cfg = {
        "method": "reference",
        "rank": 2,
        "capture_linear_systems": True,
        "reference_transfer": "full",
        "threads": 1,
        "inner_cap": 100,
        "inner_refresh": 50000,
        "frozen_sweeps": 3,
    }
    trace = CoupledTrace(root / "inactive-trace-00", cfg, "baseline")
    basis = np.eye(4)[:, [1, 3]]
    diagonal = np.array([0.1, 1, 5, 20])
    evaluation = SimpleNamespace(
        state=np.zeros(4),
        flows=[SimpleNamespace(velocity=np.zeros(2), pressure=np.zeros(1))],
    )
    masks = [np.arange(4), np.array([0, 2, 3]), np.arange(4)]
    for number, indices in enumerate(masks):
        trace.begin_quadratic(
            number, 0, evaluation, np.ones(4), np.ones(4), diagonal, 0, [], np.zeros(4), np.ones(4)
        )
        trace.record["quadratics"][-1]["linear_tolerance"] = 1e-4
        rhs = np.ones(len(indices))
        trace.before_solve(0, indices, rhs, np.full_like(rhs, 0.001), False)
        trace.after_solve(LinearResult(rhs / diagonal[indices], 1, 0, "converged"), {})
    trace.finish()
    write_arrays(root / "reference.npz", basis=basis)
    record = {
        "status": "complete",
        "all_problems_verified": True,
        "configuration": cfg,
        "environment": trace.record["environment"],
        "baseline_sha256": "baseline",
        "reference_sha256": file_sha256(root / "reference.npz"),
        "reference_description": {},
        "components_seconds": {"reference_construction_or_restore": 1.0},
    }
    write_report(root / "record.json", record)
    return root, record, trace.record, diagonal


@pytest.mark.parametrize("defect", ["incomplete", "source", "config", "order", "sequential"])
def test_capture_gate_rejects_changed_or_incomplete_inputs(tmp_path, defect):
    _, record, manifest, _ = fixture_capture(tmp_path)
    manifest = copy.deepcopy(manifest)
    if defect == "incomplete":
        manifest["status"] = "capturing"
    elif defect == "source":
        manifest["environment"]["source_sha256"] = {"different": "source"}
    elif defect == "config":
        manifest["configuration"]["frozen_sweeps"] = 5
    elif defect == "order":
        manifest["systems"].reverse()
    else:
        record["configuration"]["reference_transfer"] = "sequential"
    with pytest.raises(ValueError):
        replay.require_capture(record, manifest)


def test_complete_replay_retains_zero_extension_history_and_all_systems(tmp_path, monkeypatch):
    root, _, _, diagonal = fixture_capture(tmp_path)
    problem = SimpleNamespace(size=4, verify=lambda e: [])

    class Parent:
        def restrict(self, indices):
            return aslinearoperator(np.diag(diagonal[indices]))

    class Preconditioner:
        def __init__(self, *args):
            pass

        def attach(self, B, indices):
            B.preconditioner = lambda r: r / diagonal[indices]

        def close(self):
            pass

    monkeypatch.setattr(
        replay, "load_problem", lambda cfg: (problem, {"baseline_sha256": "baseline"})
    )
    monkeypatch.setattr(
        replay, "rebuild", lambda *args: (None, Parent(), diagonal.copy(), np.ones(4))
    )
    monkeypatch.setattr(replay, "equation_acceptance", lambda *args: True)
    monkeypatch.setattr(replay, "ReplayPreconditioner", Preconditioner)
    result = replay.replay(root, tmp_path, tmp_path / "replay")
    assert result["all_systems_verified"]
    assert result["expected_systems"] == len(result["rows"]) == 3
    first, _, last = result["rows"]
    assert first["relative_basis_difference"] == 0
    assert last["newly_inactive"] == 1
    assert last["released_entries_difference_norm"] == pytest.approx(1)
    a, b = (last["methods"][p]["coarse_error"] for p in ("full", "sequential"))
    assert a["deployed_rank"] == 2 and b["deployed_rank"] == 1
    assert a["energy_fraction_removed"] > b["energy_fraction_removed"]
    for row in result["rows"]:
        assert row["error_equation"]["status"] == "verified"
        for method in row["methods"].values():
            assert method["total_seconds"] == pytest.approx(
                method["solve_seconds"] + method["independent_verification_seconds"]
            )
    stored = json.loads((tmp_path / "replay" / "record.json").read_text())
    assert stored["all_systems_verified"]
