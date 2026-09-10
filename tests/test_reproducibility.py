"""Repeat the default examples and independently check their saved solutions."""

import json
import numpy as np
import pytest

from deflation_example.problems import build_problem
from deflation_example.runner import run_demo
from deflation_example.solvers import kkt_metrics


@pytest.mark.parametrize("problem", ["diffusion", "thermal", "cht"])
def test_default_example_repeats_and_saved_fields_satisfy_equations(tmp_path, problem):
    outputs = [tmp_path / "first", tmp_path / "second"]
    reports = [run_demo(output, problem=problem) for output in outputs]
    model = build_problem(problem)
    assert all(report["success"] for report in reports)
    assert reports[0]["configuration"] == reports[1]["configuration"]
    assert reports[0]["bound"] == pytest.approx(reports[1]["bound"], rel=1e-12)
    for index, theta in enumerate(reports[0]["configuration"]["angles"]):
        with (
            np.load(outputs[0] / f"fields-{index}.npz", allow_pickle=False) as first,
            np.load(outputs[1] / f"fields-{index}.npz", allow_pickle=False) as second,
        ):
            for key in ("desired", "state", "deflated_state", "multiplier", "control"):
                np.testing.assert_allclose(first[key], second[key], rtol=1e-8, atol=1e-10)
            np.testing.assert_array_equal(first["active"], second["active"])
            np.testing.assert_allclose(first["desired"], model.target(theta), atol=1e-14)
            np.testing.assert_allclose(first["control"], model.A @ first["state"], atol=1e-12)
            metrics = kkt_metrics(
                model.H, first["desired"], first["bound"], first["state"], first["multiplier"]
            )
            assert max(metrics.values()) <= reports[0]["configuration"]["kkt_tolerance"]
        for output, report in zip(outputs, reports, strict=True):
            saved = json.loads((output / "results.json").read_text())
            assert saved["success"] and len(saved["cases"]) == 3
            for method in ("direct", "deflated"):
                row = report["cases"][index][method]
                assert row["status"] == "converged"
                assert max(row["kkt"].values()) <= report["configuration"]["kkt_tolerance"]
            assert all(
                row["status"] == "converged"
                and row["residual"] <= report["configuration"]["kernel_rtol"]
                for row in report["cases"][index]["kernels"].values()
            )
