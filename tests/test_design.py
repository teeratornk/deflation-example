"""Executable contracts for replaceable kernels and independent reporting."""

import json
from unittest.mock import patch
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
from deflation_example import reporting
from deflation_example.backends import cpu_kernel
from deflation_example.problems import build_problem, reference_modes
from deflation_example.runner import run_demo
from deflation_example.solvers import LinearResult


@pytest.mark.parametrize("correct", [True, False])
def test_injected_kernel_is_checked_independently(tmp_path, correct):
    results = []

    def solve(matrix, rhs, basis, diagonal, *, rtol, maxiter):
        state = spsolve(matrix, rhs) if correct else np.zeros_like(rhs)
        result = LinearResult(state, 0, 0.0, "converged")
        results.append(result)
        return result, {"total_seconds": 0.0}

    report = run_demo(tmp_path / "run", grid=6, rank=3, kernels={"injected": solve})
    assert report["success"] is correct
    assert len(results) == 6
    assert all(r.residual == 0.0 for r in results)  # Caller-owned results are unchanged.
    for case in report["cases"]:
        assert set(case["kernels"]) == {"jacobi_injected", "deflated_injected"}
        assert all(
            r["status"] == ("converged" if correct else "residual_failed")
            for r in case["kernels"].values()
        )


def test_cpu_adapter_preserves_all_inputs():
    problem = build_problem(n=5)
    b, basis = problem.target(0.0), reference_modes(problem, 3)
    diagonal = problem.H.diagonal()
    inputs = [problem.H.data, b, basis, diagonal]
    before = [a.copy() for a in inputs]
    result, timing = cpu_kernel(problem.H, b, basis, diagonal, rtol=1e-10, maxiter=2000)
    assert result.status == "converged" and timing["total_seconds"] >= 0
    for a, copy in zip(inputs, before):
        np.testing.assert_array_equal(a, copy)


def test_atomic_reports_preserve_previous_record_on_failure(tmp_path):
    path = tmp_path / "results.json"
    reporting.write_report(path, {"success": False})
    previous = path.read_bytes()
    with patch.object(reporting.os, "replace", side_effect=OSError("simulated write failure")):
        with pytest.raises(OSError):
            reporting.write_report(path, {"success": True})
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
    reporting.write_report(path, {"finite": np.float64(2), "nonfinite": np.inf})
    assert json.loads(path.read_text()) == {"finite": 2.0, "nonfinite": None}


def test_provenance_uses_an_allowlist():
    with patch.object(
        reporting,
        "threadpool_info",
        return_value=[{"version": "1.0", "unexpected_private_field": "not for publication"}],
    ):
        assert reporting.environment()["blas"] == [{"version": "1.0"}]


@pytest.mark.parametrize(
    "options",
    [
        {"kernels": {}},
        {"kernels": {"bad/path": cpu_kernel}},
        {"plot": "false"},
        {"rank": True},
        {"threads": False},
        {"maxiter": True},
    ],
)
def test_invalid_application_contract(tmp_path, options):
    with pytest.raises(ValueError):
        run_demo(tmp_path / "invalid", **options)
