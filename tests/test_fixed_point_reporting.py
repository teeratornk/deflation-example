"""Reported timings and roots retain their complete-population definitions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deflation_example.reporting import file_sha256, write_report


def module():
    path = Path(__file__).parents[1] / "examples/coupled_optimization/fixed_point/audit.py"
    spec = importlib.util.spec_from_file_location("fixed_point_audit", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_empty_study_keeps_every_missing_screen(tmp_path):
    audit = module()
    summary = audit.audit(tmp_path / "inputs", tmp_path / "summary", plots=False)
    assert len(summary["rows"]) == 60
    assert all(r["status"] == "missing" and r["seconds"] is None for r in summary["rows"])
    assert not summary["selection_complete"]
    assert all(g["median_seconds"] is None for g in summary["groups"])


def test_declared_complete_population_has_correct_counts():
    audit = module()
    protocol = audit.read(audit.HERE / "protocol.json")
    selection = {"families": {key: {"selected": "anderson3"} for key in ("forward", "momentum")}}
    rows = list(audit.declared_records(protocol, selection))
    assert len(rows) == 60 + 12 + 2 + 18
    assert len({p for p, _ in rows}) == len(rows)


def test_failed_or_unfinished_repetition_has_no_median():
    audit = module()
    rows = [
        {
            "phase": "optimize",
            "group": "jacobi",
            "policy": "newton",
            "status": s,
            "verified": ok,
            "seconds": t,
        }
        for s, ok, t in (
            ("complete", True, 10),
            ("iteration_cap", False, 1),
            ("missing", False, None),
        )
    ]
    group = audit.groups(rows)[0]
    assert group["verified_seconds"] == [10]
    assert group["median_seconds"] is None
    assert group["statuses"] == ["complete", "iteration_cap", "missing"]


def equation_checks():
    return {
        "momentum_relative_residual": 5e-13,
        "continuity_relative_residual": 1e-16,
        "thermal_relative_residual": 2e-14,
        "mass_relative_imbalance": 1e-14,
        "energy_relative_defect": 1e-14,
    }


def optimizer_record():
    return {
        "status": "complete",
        "all_problems_verified": True,
        "configuration": {
            "queries": [{"target": 7}],
            "slabs": 1,
            "nonlinear_tolerance": 1e-8,
            "inner_tolerance": 1e-10,
        },
        "cases": [
            {
                "status": "converged",
                "verified": True,
                "equations": [equation_checks()],
                "kkt": dict.fromkeys(
                    (
                        "primal_absolute",
                        "stationarity",
                        "dual_feasibility",
                        "lower_complementarity",
                        "upper_complementarity",
                    ),
                    1e-9,
                ),
                "adjoint": {"maximum_momentum_adjoint_relative_residual": 5e-13},
            }
        ],
        "sequence_seconds": 12,
        "components_seconds": {"setup": 3, "solve": 9},
    }


@pytest.mark.parametrize("metric", list(equation_checks()))
def test_verified_screen_requires_its_recorded_equation_values(tmp_path, metric):
    path = tmp_path / "record.json"
    record = {
        "status": "complete",
        "family": "forward",
        "row": {"status": "converged", "verified": True, "checks": equation_checks()},
    }
    write_report(path, record)
    row, _ = module().outcome(path, {"phase": "screen"})
    assert row["maximum_equation_relative_residual"] == 5e-13
    record["row"]["checks"][metric] = 0.01
    write_report(path, record)
    with pytest.raises(ValueError, match="Verified record violates"):
        module().outcome(path, {"phase": "screen"})
    record["row"].update(verified=False, status="iteration_cap")
    write_report(path, record)
    row, _ = module().outcome(path, {"phase": "screen"})
    assert not row["verified"] and row["status"] == "iteration_cap"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, 1e-7])
def test_verified_optimizer_requires_each_kkt_component(tmp_path, value):
    path = tmp_path / "record.json"
    record = optimizer_record()
    record["cases"][0]["kkt"]["stationarity"] = value
    write_report(path, record)
    with pytest.raises(ValueError, match="stationarity"):
        module().outcome(path, {"phase": "optimize"})


def test_verified_optimizer_requires_every_time_slab(tmp_path):
    path = tmp_path / "record.json"
    record = optimizer_record()
    record["configuration"]["slabs"] = 2
    write_report(path, record)
    with pytest.raises(ValueError, match="time slab"):
        module().outcome(path, {"phase": "optimize"})


def test_verified_derivative_gate_requires_recorded_values(tmp_path):
    path = tmp_path / "record.json"
    record = {
        "status": "verified",
        "derivatives": {
            "relative_dot_product_error": 1e-12,
            "taylor_orders": [2.0, 2.0],
            "equations": [equation_checks()],
        },
    }
    write_report(path, record)
    assert module().outcome(path, {"phase": "derivatives"})[0]["verified"]
    record["derivatives"]["taylor_orders"][0] = 1.2
    write_report(path, record)
    with pytest.raises(ValueError, match="Taylor-order"):
        module().outcome(path, {"phase": "derivatives"})


def test_optimizer_checks_component_sum_and_resumed_cost(tmp_path):
    audit = module()
    path = tmp_path / "record.json"
    record = optimizer_record()
    write_report(path, record)
    row, _ = audit.outcome(path, {"phase": "optimize"})
    assert row["verified"] and row["seconds"] == 12
    record["stage"] = {"resume": {"prior_seconds": 4}}
    write_report(path, record)
    row, _ = audit.outcome(path, {"phase": "optimize"})
    assert row["verified"] and row["seconds"] is None
    assert row["timing_status"] == "requires_complete_attempt_accounting"
    record["components_seconds"]["setup"] = 4
    write_report(path, record)
    with pytest.raises(ValueError, match="component timers"):
        audit.outcome(path, {"phase": "optimize"})


def test_trajectory_cannot_be_verified_from_partial_steps(tmp_path):
    audit = module()
    path = tmp_path / "record.json"
    write_report(
        path,
        {
            "status": "converged",
            "slabs": 2,
            "steps": [{"verified": True, "seconds": 1}],
            "sum_step_seconds": 1,
            "all_attempt_process_seconds": None,
        },
    )
    row, _ = audit.outcome(path, {"phase": "trajectory"})
    assert not row["verified"] and row["seconds"] is None


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_duration_is_not_a_speedup(value):
    with pytest.raises(ValueError, match="duration"):
        module().duration(value)


def test_negative_component_cannot_cancel_a_positive_interval(tmp_path):
    path = tmp_path / "record.json"
    record = optimizer_record()
    record["components_seconds"] = {"setup": -5, "solve": 17}
    write_report(path, record)
    with pytest.raises(ValueError, match="negative interval"):
        module().outcome(path, {"phase": "optimize"})


def test_root_agreement_checks_both_archives_and_matching_inputs(tmp_path):
    audit = module()
    case = tmp_path / "screen/forward/ordinary"
    identity = {
        "environment": {"source_sha256": {"module": "abc"}},
        "optimization_field_sha256": "control",
        "baseline_sha256": "base",
        "fine_baseline_sha256": "fine",
        "replay_fields_sha256": "replay",
        "time_s": 1,
        "initial": "previous",
        "time_scheme": "backward_euler",
        "temperature_scale_K": 10,
        "temperature_offset_K": 300,
        "row": {"verified": True},
    }
    for policy, value in (("newton", 0.5), ("anderson3", 0.6)):
        folder = case / policy
        folder.mkdir(parents=True)
        np.savez(folder / "fields.npz", state=[value], velocity=[[1, 2]], pressure=[0])
        write_report(
            folder / "record.json",
            {**identity, "policy": policy, "field_sha256": file_sha256(folder / "fields.npz")},
        )
    assert audit.root_agreement(tmp_path)[0]["maximum_temperature_difference_K"] == pytest.approx(1)
    alternative = audit.root_agreement(tmp_path, "anderson3")[0]
    assert alternative["policy"] == "newton" and alternative["reference_policy"] == "anderson3"
    assert alternative["maximum_temperature_difference_K"] == pytest.approx(1)
    path = case / "anderson3/record.json"
    record = audit.read(path)
    write_report(path, {**record, "time_s": 2})
    with pytest.raises(ValueError, match="matched inputs"):
        audit.root_agreement(tmp_path)
    write_report(path, record)
    np.savez(case / "newton/fields.npz", state=[99], velocity=[[1, 2]], pressure=[0])
    with pytest.raises(ValueError, match="checksum"):
        audit.root_agreement(tmp_path)


def test_all_complete_repetitions_have_median_and_range():
    rows = [
        {
            "phase": "trajectory",
            "group": "fine",
            "policy": "newton",
            "status": "converged",
            "verified": True,
            "seconds": t,
        }
        for t in (12, 10, 11)
    ]
    result = module().groups(rows)[0]
    assert result["median_seconds"] == 11
    assert result["minimum_seconds"] == 10 and result["maximum_seconds"] == 12


def test_complete_time_figure_retains_failures(tmp_path):
    pytest.importorskip("matplotlib")
    audit = module()
    rows = [
        {
            "phase": "optimize",
            "group": "reference",
            "policy": "anderson3",
            "status": "iteration_cap",
            "verified": False,
            "seconds": 2,
        }
    ]
    audit.plot_comparisons({"groups": audit.groups(rows)}, tmp_path)
    assert (tmp_path / "optimize-complete-times.pdf").stat().st_size > 100


def example(name):
    path = Path(__file__).parents[1] / "examples/coupled_optimization/fixed_point" / (name + ".py")
    spec = importlib.util.spec_from_file_location("fixed_point_" + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_partial_summary_skips_empty_family_and_plots_failure(tmp_path):
    pytest.importorskip("matplotlib")
    folder = tmp_path / "input/screen/forward/ordinary/newton"
    folder.mkdir(parents=True)
    write_report(
        folder / "record.json",
        {
            "schema": "coupled-fixed-point-study-v1",
            "status": "complete",
            "family": "forward",
            "policy": "newton",
            "time_s": 1,
            "row": {
                "status": "iteration_cap",
                "verified": False,
                "history": [{"momentum_relative_residual": 1e-4}],
            },
        },
    )
    example("summarize").summarize(tmp_path / "input", tmp_path / "figure")
    assert (tmp_path / "figure/forward-residuals.pdf").exists()
    assert (tmp_path / "figure/forward-residuals.png").exists()
    assert not (tmp_path / "figure/momentum-residuals.pdf").exists()


def test_trajectory_field_reader_requires_complete_verified_hashes(tmp_path):
    reader = example("trajectory_fields")
    path = tmp_path / "step.npz"
    np.savez(path, state=[0.1], velocity=[[1, 2]], pressure=[0])
    record = {
        "status": "converged",
        "slabs": 1,
        "steps": [
            {
                "slab_zero_based": 0,
                "verified": True,
                "fields": path.name,
                "field_sha256": file_sha256(path),
            }
        ],
    }
    assert reader.verified_arrays(record, tmp_path)["velocity"].shape == (1, 1, 2)
    record["status"] = "running"
    with pytest.raises(ValueError, match="complete"):
        reader.verified_arrays(record, tmp_path)
    record["status"] = "converged"
    record["steps"][0]["verified"] = False
    with pytest.raises(ValueError, match="verified"):
        reader.verified_arrays(record, tmp_path)
    record["steps"][0]["verified"] = True
    record["steps"][0]["field_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        reader.verified_arrays(record, tmp_path)


def test_residual_curve_uses_recorded_iterations_without_counting_verification():
    iterations, values = example("summarize").residual_trace(
        [
            {"iteration": 1, "momentum_relative_residual": 0.5},
            {"iteration": 2, "flow_status": "iteration_cap"},
            {"procedure": "returned_state_verification", "momentum_relative_residual": 0.1},
        ]
    )
    assert iterations == [1, 2] and values == [0.5, 0.1]


def test_oseen_precision_diagnostic_preserves_matrix_boundary_and_initial_state():
    from scipy.sparse import csr_matrix

    matrix = csr_matrix([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]])
    exact = np.array([2.0, 3.0, 4.0])
    initial = np.array([2.0, 0.0, 0.0])
    saved = initial.copy()
    fixed = np.array([0])

    def verify(state):
        assert state[0] == 2.0
        return {"error": float(np.linalg.norm(state - exact))}

    rows = example("momentum_precision").fixed_matrix_comparison(
        matrix, matrix @ exact, initial, fixed, verify, corrections=2
    )
    assert np.array_equal(initial, saved)
    for route in ("state", "correction"):
        assert len(rows[route]) == 3
        assert max(r["original_nonlinear_equations"]["error"] for r in rows[route]) < 1e-14
    sequences = example("momentum_precision").nonlinear_comparison(
        lambda x: matrix, matrix @ exact, initial, fixed, verify
    )
    for route in ("state", "correction"):
        assert sequences[route]["status"] == "converged"
        assert len(sequences[route]["history"]) == 2
    assert np.array_equal(initial, saved)
