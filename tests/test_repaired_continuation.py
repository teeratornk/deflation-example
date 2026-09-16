"""A continued step preserves source data and verifies the returned pair."""

import importlib.util
from pathlib import Path
import sys

import numpy as np

from test_coupled_newton_replay import data

directory = Path(__file__).parents[1] / "examples/coupled_optimization/fixed_point"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location(
    "continue_repaired", directory / "continue_repaired.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.pop(0)


def test_unchanged_trajectory_continues_from_verified_previous_state():
    problem, expected, evaluation, _ = data()
    previous = problem.full_temperature(expected.reshape(2, -1)[0])
    flow = evaluation.flows[0]
    source = np.zeros(len(problem.mesh.nodes))
    source[problem.free] = evaluation.control.reshape(2, -1)[1]
    originals = [a.copy() for a in (source, previous, flow.velocity, flow.pressure)]
    row, arrays = module.advance(problem, source, previous, flow, 1)
    assert row["verified"] and row["status"] == "converged"
    np.testing.assert_allclose(arrays["state"], expected.reshape(2, -1)[1], atol=1e-10)
    for a, b in zip(originals, (source, previous, flow.velocity, flow.pressure), strict=True):
        np.testing.assert_array_equal(a, b)
