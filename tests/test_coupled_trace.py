"""Diagnostic capture must preserve numerical results and bind every input."""

import numpy as np
import pytest

from deflation_example.coupled_optimizer import minimize_coupled
from deflation_example.coupled_trace import CoupledTrace, read_arrays, read_manifest
from test_coupled_derivatives import small_coupled_problem
from test_coupled_optimizer import solver


def test_capture_preserves_complete_optimization_and_every_solve(tmp_path):
    problem = small_coupled_problem([0.2, 0.35])
    desired = np.linspace(-0.1, 0.3, problem.size)
    plain = minimize_coupled(problem, desired, -0.05, 0.15, solver(), qp_tolerance=1e-11)
    trace = CoupledTrace(tmp_path / "trace", {"initial_state_snapshot": "/private/input"}, "abc")
    captured = minimize_coupled(
        problem, desired, -0.05, 0.15, solver(), qp_tolerance=1e-11, observer=trace
    )
    trace.finish()
    np.testing.assert_array_equal(captured.evaluation.state, plain.evaluation.state)
    assert captured.status == plain.status == "converged"
    record = read_manifest(tmp_path / "trace")
    assert record["status"] == "complete"
    assert "initial_state_snapshot" not in record["configuration"]
    assert len(record["systems"]) == sum(
        len(a["qp_history"]) for r in captured.history for a in r["attempts"]
    )
    for row in record["systems"]:
        arrays = read_arrays(tmp_path / "trace", row["file"], row["sha256"])
        assert arrays["initial"].shape == arrays["rhs"].shape == arrays["indices"].shape
        assert row["status"] == "converged"
        read_arrays(tmp_path / "trace", row["solution_file"], row["solution_sha256"])
    with pytest.raises(ValueError, match="checksum"):
        read_arrays(tmp_path / "trace", record["systems"][0]["file"], "wrong")


def test_trace_rejects_incomplete_and_out_of_order_calls(tmp_path):
    trace = CoupledTrace(tmp_path / "trace", {}, "abc")
    with pytest.raises(RuntimeError, match="Begin"):
        trace.before_solve(0, np.arange(2), np.ones(2), np.zeros(2), False)
    trace.current = 0
    trace.before_solve(0, np.arange(2), np.ones(2), np.zeros(2), False)
    with pytest.raises(RuntimeError, match="incomplete"):
        trace.finish()
    assert read_manifest(tmp_path / "trace")["systems"][0]["status"] == "pending"
