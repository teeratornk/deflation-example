"""Trajectory agreement requires matched problems and every verified field."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deflation_example.reporting import file_sha256, write_arrays, write_report


def module():
    path = (
        Path(__file__).parents[1]
        / "examples/coupled_optimization/fixed_point/trajectory_agreement.py"
    )
    spec = importlib.util.spec_from_file_location("trajectory_agreement", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(folder, shift=0.0):
    folder.mkdir()
    record = {
        "schema": "coupled-fixed-point-study-v1",
        "family": "forward",
        "status": "converged",
        "policy": "newton" if not shift else "anderson5",
        "repetition": 0,
        "configuration": {"horizon_s": 20, "upper_K": 350},
        "environment": {"source_sha256": {"solver.py": "source"}},
        "optimization_field_sha256": "control",
        "baseline_sha256": "baseline",
        "fine_baseline_sha256": "baseline",
        "control_policy": "fixed",
        "slabs": 2,
        "spatial_state_dofs": 3,
        "subdivision": 1,
        "time_scheme": "backward_euler",
        "temperature_offset_K": 340,
        "temperature_scale_K": 20,
        "tolerance": 1e-12,
        "mass_tolerance": 1e-6,
        "energy_tolerance": 1e-6,
        "steps": [],
    }
    for i in range(2):
        field = folder / f"step-{i}.npz"
        write_arrays(field, state=np.array([0.0, 0.1, i + shift]), velocity=np.full((4, 2), shift))
        record["steps"].append(
            {
                "verified": True,
                "slab_zero_based": i,
                "time_s": 10.0 * (i + 1),
                "checks": {key: 1e-14 for key in module().CRITERIA},
                "fields": field.name,
                "field_sha256": file_sha256(field),
            }
        )
    write_report(folder / "record.json", record)
    return record


def test_physical_temperature_and_vector_velocity_differences(tmp_path):
    fixture(tmp_path / "a")
    fixture(tmp_path / "b", 0.01)
    row = module().compare(tmp_path / "a", tmp_path / "b")
    assert row["maximum_temperature_difference_K"] == pytest.approx(0.2)
    assert row["maximum_velocity_vector_difference_m_s"] == pytest.approx(np.sqrt(2) * 0.01)
    assert len(row["steps"]) == 2
    assert len(row["candidate_record_sha256"]) == 64


def test_identical_fields_have_exact_zero_difference(tmp_path):
    fixture(tmp_path / "a")
    row = module().compare(tmp_path / "a", tmp_path / "a")
    assert row["maximum_temperature_difference_K"] == 0
    assert row["maximum_velocity_vector_difference_m_s"] == 0


@pytest.mark.parametrize(
    "key", ["optimization_field_sha256", "fine_baseline_sha256", "time_scheme"]
)
def test_different_problem_rejected(tmp_path, key):
    fixture(tmp_path / "a")
    record = fixture(tmp_path / "b")
    record[key] = "different"
    write_report(tmp_path / "b/record.json", record)
    with pytest.raises(ValueError, match="problems differ"):
        module().compare(tmp_path / "a", tmp_path / "b")


@pytest.mark.parametrize("status", ["running", "numerical_failure", "execution_failure"])
def test_unfinished_or_failed_trajectory_rejected(tmp_path, status):
    record = fixture(tmp_path / "a")
    record["status"] = status
    write_report(tmp_path / "a/record.json", record)
    with pytest.raises(ValueError, match="complete, converged"):
        module().read_complete(tmp_path / "a")


@pytest.mark.parametrize("failure", ["missing", "large", "nan", "negative"])
def test_verified_label_does_not_override_equation_check(tmp_path, failure):
    record = fixture(tmp_path / "a")
    checks = record["steps"][0]["checks"]
    if failure == "missing":
        del checks["momentum_relative_residual"]
    else:
        checks["momentum_relative_residual"] = {"large": 1e-8, "nan": None, "negative": -1}[failure]
    write_report(tmp_path / "a/record.json", record)
    with pytest.raises(ValueError, match="criterion"):
        module().read_complete(tmp_path / "a")


def test_checksum_and_directory_escape_rejected(tmp_path):
    record = fixture(tmp_path / "a")
    row = record["steps"][0]
    row["field_sha256"] = "changed"
    with pytest.raises(ValueError, match="checksum or location"):
        module().load_fields(tmp_path / "a", row, 3)
    row["fields"] = "../outside.npz"
    with pytest.raises(ValueError, match="checksum or location"):
        module().load_fields(tmp_path / "a", row, 3)


@pytest.mark.parametrize("field", ["state", "velocity"])
def test_nonfinite_field_rejected(tmp_path, field):
    record = fixture(tmp_path / "a")
    arrays = {"state": np.zeros(3), "velocity": np.zeros((4, 2))}
    arrays[field].flat[0] = np.nan
    path = tmp_path / "a/bad.npz"
    write_arrays(path, **arrays)
    row = {**record["steps"][0], "fields": path.name, "field_sha256": file_sha256(path)}
    with pytest.raises(ValueError, match="finite real"):
        module().load_fields(tmp_path / "a", row, 3)


def test_time_levels_and_source_must_match(tmp_path):
    fixture(tmp_path / "a")
    record = fixture(tmp_path / "b")
    record["steps"][0]["time_s"] = 9
    write_report(tmp_path / "b/record.json", record)
    with pytest.raises(ValueError, match="time levels differ"):
        module().compare(tmp_path / "a", tmp_path / "b")
    record["environment"]["source_sha256"]["solver.py"] = "different"
    write_report(tmp_path / "b/record.json", record)
    with pytest.raises(ValueError, match="numerical sources differ"):
        module().compare(tmp_path / "a", tmp_path / "b")


def test_completed_flag_does_not_hide_missing_final_step(tmp_path):
    record = fixture(tmp_path / "a")
    record["steps"].pop()
    write_report(tmp_path / "a/record.json", record)
    with pytest.raises(ValueError, match="complete, converged"):
        module().read_complete(tmp_path / "a")


@pytest.mark.parametrize("failure", ["order", "horizon", "scale", "empty_source"])
def test_invalid_time_grid_or_metadata_rejected(tmp_path, failure):
    record = fixture(tmp_path / "a")
    if failure == "order":
        record["steps"][1]["time_s"] = 10
    elif failure == "horizon":
        record["configuration"]["horizon_s"] = 30
    elif failure == "scale":
        record["temperature_scale_K"] = 0
    else:
        record["environment"]["source_sha256"] = {}
    write_report(tmp_path / "a/record.json", record)
    with pytest.raises(ValueError):
        module().read_complete(tmp_path / "a")


def test_figure_limits_include_small_differences_after_zero_curve(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    fixture(tmp_path / "a")
    fixture(tmp_path / "b", 1e-8)
    audit = module()
    same = audit.compare(tmp_path / "a", tmp_path / "a")
    shifted = audit.compare(tmp_path / "a", tmp_path / "b")
    limits = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, *args, **kwargs: limits.append([axis.get_ylim() for axis in figure.axes]),
    )
    audit.plot([same, shifted], tmp_path)
    assert len(limits) == 2
    for bounds in limits:
        for interval, key in zip(
            bounds,
            ("maximum_temperature_difference_K", "maximum_velocity_vector_difference_m_s"),
        ):
            assert interval[0] == 0
            assert interval[1] == pytest.approx(1.15 * shifted[key], abs=1e-20)
