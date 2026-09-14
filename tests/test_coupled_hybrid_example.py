"""The portable backend check preserves caps and independently checked results."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deflation_example.study_solvers import ArrayReference
from test_coupled_hybrid_solver import system


def driver():
    path = Path(__file__).parents[1] / "examples/coupled_optimization/benchmark_hybrid.py"
    spec = importlib.util.spec_from_file_location("hybrid_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cap", [1, 1000])
@pytest.mark.parametrize("method", ["jacobi", "reference", "recycling"])
def test_kernel_summary_keeps_every_system_and_iteration_cap(cap, method):
    problem, H, diagonal = system()
    rng = np.random.default_rng(16)
    reference = ArrayReference(rng.normal(size=(problem.size, 3)), {"construction": "test"})
    systems = []
    for indices in (np.arange(0, problem.size, 2), np.arange(problem.size)):
        exact = rng.normal(size=len(indices))
        systems.append((indices, exact, H.restrict(indices) @ exact))
    result = driver().check_policy(H, diagonal, systems, reference, "cpu", method, 3, cap)
    assert len(result["systems"]) == 2
    assert result["seconds"] >= sum(
        row["solve_seconds"] + row["additional_verification_seconds"] for row in result["systems"]
    )
    for row in result["systems"]:
        assert row["verified"] == (
            row["status"] == "converged" and row["independent_relative_residual"] <= 1e-10
        )
        assert row["iterations"] <= cap
    if cap == 1000:
        assert all(row["verified"] for row in result["systems"])
    else:
        assert not all(row["verified"] for row in result["systems"])
