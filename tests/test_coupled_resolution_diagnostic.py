"""Physical-time interpolation and exact identification of diagnostic inputs."""

import json
import numpy as np
import pytest

from deflation_example.coupled_resolution_diagnostic import interpolate_snapshot, diagnose
from deflation_example.coupled_saved import file_digest
from deflation_example.reporting import write_fields, write_report
from test_coupled_time_resolution_report import case


def test_snapshot_interpolation_includes_initial_state_and_exact_endpoints():
    values = np.array([[2.0, 4.0], [5.0, 10.0]])
    times = np.array([1.0, 3.0])
    initial = np.array([1.0, 2.0])
    np.testing.assert_array_equal(interpolate_snapshot(values, times, 0, initial), initial)
    np.testing.assert_array_equal(interpolate_snapshot(values, times, 3, initial), values[-1])
    np.testing.assert_allclose(interpolate_snapshot(values, times, 2, initial), [3.5, 7])
    snapshot = interpolate_snapshot(values, times, 1, initial)
    snapshot[:] = 0
    assert values[0, 0] == 2


@pytest.mark.parametrize("time", [-1, 3.01, np.nan])
def test_snapshot_extrapolation_is_rejected(time):
    with pytest.raises(ValueError):
        interpolate_snapshot(np.ones((2, 3)), [1, 3], time, np.zeros(3))


def test_location_report_checks_mesh_identity_and_has_runnable_plots(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from deflation_example.coupled_resolution_diagnostic import plot_diagnostic

    mesh = SimpleNamespace(
        nodes=np.array([[0.1, 0.0], [0.2, 0.0], [0.1, 0.5]]),
        cells=np.array([[0, 1, 2]]),
        free=np.array([1, 2]),
        materials=np.array([3]),
    )
    inputs = {"mesh": "test-mesh"}
    parameters = {"physical": {"temperature_scale_K": 20.0, "inlet_temperature_K": 341.3}}
    monkeypatch.setattr(
        "deflation_example.coupled_resolution_diagnostic.transformer_inputs",
        lambda level: (mesh, parameters, None, None, None, inputs),
    )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    write_fields(baseline / "baseline.npz", velocity=np.zeros((3, 2)))
    digest = file_digest(baseline / "baseline.npz")
    write_report(
        baseline / "record.json",
        {
            "status": "converged",
            "baseline_sha256": digest,
            "configuration": {"level": 0},
            "input_sha256": inputs,
        },
    )
    runs = [case(tmp_path, subdivision) for subdivision in (1, 2)]
    for run in runs:
        record = json.loads((run / "record.json").read_text())
        record["baseline_sha256"] = digest
        write_report(run / "record.json", record)
        with np.load(run / "states.npz") as arrays:
            state, times = arrays["state"], arrays["times_s"]
        write_fields(
            run / "states.npz", state=state, times_s=times, velocity=np.zeros((len(times), 3, 2))
        )
    report, geometry, snapshots = diagnose(runs, baseline)
    peak = report["pairs"][0]["peak"]
    assert peak["full_node_index"] == 2
    assert peak["signed_fine_minus_coarse_K"] == pytest.approx(-5)
    assert report["pairs"][0]["maximum_velocity_difference_at_temperature_peak_m_s"] == 0
    plot_diagnostic(report, geometry, snapshots, tmp_path)
    assert (tmp_path / "resolution_location_2_4.pdf").stat().st_size > 0
    record["baseline_sha256"] = "different"
    write_report(runs[-1] / "record.json", record)
    with pytest.raises(ValueError, match="same saved source"):
        diagnose(runs, baseline)
