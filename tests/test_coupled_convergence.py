"""Convergence figures preserve returned residuals, failed attempts and all methods."""

from copy import deepcopy

import pytest

from deflation_example.coupled_convergence import convergence_data, plot_convergence


def kkt(value):
    return dict(
        primal_absolute=0.0,
        stationarity=value,
        dual_feasibility=0.0,
        lower_complementarity=0.0,
        upper_complementarity=0.0,
    )


def record(method):
    return {
        "schema": "coupled-optimization-pilot-v1",
        "status": "complete",
        "configuration": {"methods": [method], "nonlinear_tolerance": 1e-8, "target_startup_s": 60},
        "baseline_sha256": "same baseline",
        "environment": {"source_sha256": {}, "cpu_model": "test", "numpy": "test", "scipy": "test"},
        "results": [
            {
                "method": method,
                "status": "converged",
                "optimizer_status": "converged",
                "kkt": kkt(1e-9),
                "history": [
                    {
                        "iteration": 0,
                        "kkt": kkt(1),
                        "attempts": [
                            {"qp_history": [{"linear_iterations": 2}, {"linear_iterations": 3}]},
                            {"qp_history": [{"linear_iterations": 4}]},
                        ],
                    }
                ],
            }
        ],
    }


def population():
    return [record(m) for m in ("jacobi", "reference", "recycling")]


def test_convergence_includes_all_inner_attempts_and_the_returned_residual():
    data = population()
    actual = convergence_data(data)
    for row in actual["methods"]:
        assert row["iteration"] == [0, 1]
        assert row["cumulative_inner_iterations"] == [0, 9]
        assert row["maximum_kkt"] == [1, 1e-9]
    data[0]["results"][0].update(
        status="globalization_failed", optimizer_status="globalization_failed"
    )
    data[0]["results"][0]["kkt"] = kkt(0.01)
    data[2].update(status="flow_newton_cap", results=[])
    actual = convergence_data(data)
    assert actual["methods"][0]["maximum_kkt"][-1] == 0.01
    assert actual["methods"][0]["status"] == "globalization_failed"
    assert actual["methods"][2]["status"] == "flow_newton_cap"
    assert not actual["methods"][2]["history_available"]


def test_mismatched_targets_sources_and_missing_methods_are_rejected():
    for section, key, value in (
        ("configuration", "target_startup_s", 0),
        ("environment", "source_sha256", {"a.py": "changed"}),
    ):
        data = population()
        data[0][section][key] = value
        with pytest.raises(ValueError, match="share"):
            convergence_data(data)
    with pytest.raises(ValueError, match="all three"):
        convergence_data(population()[:2])
    with pytest.raises(ValueError, match="distinct"):
        convergence_data(population() + [record("jacobi")])


def test_missing_optimality_components_and_invalid_iteration_work_are_rejected():
    data = population()
    del data[0]["results"][0]["kkt"]["stationarity"]
    with pytest.raises(ValueError, match="five"):
        convergence_data(data)
    data = population()
    data[0]["results"][0]["history"][0]["attempts"][0]["qp_history"][0]["linear_iterations"] = -1
    with pytest.raises(ValueError, match="nonnegative integers"):
        convergence_data(data)
    data = population()
    data[0]["results"][0]["kkt"] = kkt(1e-5)
    with pytest.raises(ValueError, match="declared optimality"):
        convergence_data(data)


def test_plot_preserves_zero_residual_values_and_does_not_overwrite(tmp_path):
    pytest.importorskip("matplotlib")
    data = population()
    data[0]["results"][0]["kkt"] = kkt(0)
    report = convergence_data(data)
    original = deepcopy(report)
    plot_convergence(report, tmp_path / "figures")
    assert report == original
    assert (tmp_path / "figures/convergence.pdf").is_file()
    assert (tmp_path / "figures/convergence.png").is_file()
    with pytest.raises(FileExistsError):
        plot_convergence(report, tmp_path / "figures")
