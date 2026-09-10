"""Time-transfer and weighted-measure checks for the forward assessment."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from deflation_example.assess_transformer import assessment_times, evaluate_tracking
from deflation_example.assessment_controls import source_at_time


@pytest.mark.parametrize("dt", [37.5, 18.75, 9.375])
def test_assessment_grid_resolves_source_endpoints(dt):
    ends = np.array([150.0, 300.0, 450.0, 600.0])
    times, steps = assessment_times(ends, dt)
    assert len(times) == round(600 / dt)
    np.testing.assert_array_equal(steps, dt)
    assert all(t in times for t in ends)
    controls = np.arange(4.0)[:, None]
    integral = sum(h * source_at_time(controls, ends, t) for h, t in zip(steps, times))
    np.testing.assert_array_equal(integral, 150 * controls.sum(axis=0))


@pytest.mark.parametrize(
    "ends,dt", [([150, 600], 200), ([300, 150], 37.5), ([600], 37), ([np.nan], 1), ([[1]], 1)]
)
def test_invalid_assessment_grid(ends, dt):
    with pytest.raises(ValueError):
        assessment_times(ends, dt)


def test_steady_assessment_has_unit_time_weight():
    times, steps = assessment_times([], 37.5)
    np.testing.assert_array_equal(times, [0])
    np.testing.assert_array_equal(steps, [1])


def test_tracking_keeps_physical_weight_and_fixed_penalty():
    mass = np.array([2.0, 3.0])
    states = np.array([[1.0, 2.0], [3.0, 4.0]])
    desired = np.ones_like(states)
    controls = np.full_like(states, 2.0)
    steps = np.array([0.5, 1.5])
    result = evaluate_tracking(states, desired, controls, mass, steps, 20.0, 0.01)
    squared = 400 * (0.5 * 3 + 1.5 * (2 * 4 + 3 * 9))
    assert result["tracking_integral_half"] == 0.5 * squared
    assert result["tracking_weighted_rms_K"] == np.sqrt(squared / 10)
    assert result["fixed_control_penalty"] == 80
    assert result["total_objective"] == 0.5 * squared + 80


def test_cli_reports_numerical_failure_with_nonzero_exit(monkeypatch):
    from deflation_example import assess_transformer

    monkeypatch.setattr(assess_transformer, "run", lambda config: {"status": "baseline_failed"})
    with pytest.raises(SystemExit, match="2"):
        assess_transformer.main.__wrapped__(OmegaConf.create({"threads": 1}))


def test_execution_exception_preserves_failure_record(tmp_path, monkeypatch):
    import json
    from deflation_example import assess_transformer

    config = OmegaConf.load(
        assess_transformer.Path(assess_transformer.__file__).parent / "conf" / "assessment.yaml"
    )
    config.controls_directory = str(tmp_path / "controls")
    config.output = str(tmp_path / "result")

    def fail(config, output):
        raise RuntimeError("test numerical failure")

    monkeypatch.setattr(assess_transformer, "_execute", fail)
    with pytest.raises(RuntimeError, match="test numerical failure"):
        assess_transformer.run(config)
    report = json.loads((tmp_path / "result" / "record.json").read_text())
    assert report == {"status": "execution_failed", "exception_type": "RuntimeError"}
    # A second request cannot overwrite the existing attempt.
    with pytest.raises(FileExistsError):
        assess_transformer.run(config)


@pytest.mark.parametrize("alteration", [None, "properties", "field_hash", "equation"])
def test_baseline_reuse_checks_inputs_hashes_and_current_equations(tmp_path, alteration):
    import hashlib
    from deflation_example.assess_transformer import checked_baseline
    from deflation_example.axisymmetric_flow import AxisymmetricFlow
    from deflation_example.reporting import write_fields, write_report
    from test_axisymmetric_flow import annular_rectangle

    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    fixed = flow.boundary[flow.points[flow.boundary, 1] < 1 - 1e-12]
    values = np.zeros((len(fixed), 2))
    field = tmp_path / "baseline.npz"
    pressure = np.zeros(flow.np)
    if alteration == "equation":
        pressure = flow.mesh.nodes[flow.vertices, 1].copy()
    write_fields(field, velocity=np.zeros((flow.nv, 2)), pressure=pressure)
    record = {
        "baseline_status": "converged",
        "momentum_properties": {"viscosity": 0.1},
        "source_input_sha256": {"mesh": "example"},
        "configuration": {"level": 0},
        "baseline_seconds": 2.0,
        "baseline_fields_sha256": hashlib.sha256(field.read_bytes()).hexdigest(),
    }
    if alteration == "properties":
        record["momentum_properties"]["viscosity"] = 0.2
    elif alteration == "field_hash":
        record["baseline_fields_sha256"] = "incorrect"
    write_report(tmp_path / "record.json", record)
    args = (tmp_path, flow, fixed, values, {"viscosity": 0.1}, {"mesh": "example"}, 0, 1e-9, 1e-6)
    if alteration is None:
        result, metadata = checked_baseline(*args)
        assert result.status == "converged"
        assert metadata["baseline_construction_seconds"] == 2.0
    else:
        with pytest.raises(ValueError):
            checked_baseline(*args)


@pytest.mark.parametrize("valid", [True, False])
def test_continuation_restart_rechecks_equations_and_restores_viscosity(tmp_path, valid):
    from deflation_example.assess_transformer import checked_restart
    from deflation_example.axisymmetric_flow import AxisymmetricFlow
    from deflation_example.reporting import write_fields, write_report
    from test_axisymmetric_flow import annular_rectangle

    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    fixed = flow.boundary[flow.points[flow.boundary, 1] < 1 - 1e-12]
    values = np.zeros((len(fixed), 2))
    pressure = np.zeros(flow.np) if valid else flow.mesh.nodes[flow.vertices, 1].copy()
    write_fields(
        tmp_path / "baseline-checkpoint.npz", velocity=np.zeros((flow.nv, 2)), pressure=pressure
    )
    write_report(
        tmp_path / "baseline-checkpoint.json",
        {
            "stage": {"status": "converged", "viscosity_factor": 16.0},
            "momentum_properties": {"viscosity": 0.1},
            "input_sha256": {"mesh": "example"},
            "level": 0,
        },
    )
    args = (tmp_path, flow, fixed, values, {"viscosity": 0.1}, {"mesh": "example"}, 0, 1e-9)
    if valid:
        result, factor, metadata = checked_restart(*args)
        assert result.status == "initial"
        assert factor == 16.0
        assert metadata["preceding_initialization_seconds"] is None
    else:
        with pytest.raises(ValueError, match="declared steady equations"):
            checked_restart(*args)
    assert flow.viscosity == 0.1


@pytest.mark.parametrize("alteration", [None, "field_hash", "convection", "grad_div", "stage"])
def test_checkpoint_guess_keeps_physical_steady_acceptance_separate(tmp_path, alteration):
    import hashlib
    from deflation_example.assess_transformer import checked_initial_guess
    from deflation_example.axisymmetric_flow import AxisymmetricFlow
    from deflation_example.reporting import write_fields, write_report
    from test_axisymmetric_flow import annular_rectangle

    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    fixed = flow.boundary[flow.points[flow.boundary, 1] < 1 - 1e-12]
    values = np.zeros((len(fixed), 2))
    field = tmp_path / "stage.npz"
    write_fields(
        field,
        velocity=np.zeros((flow.nv, 2)),
        pressure=flow.mesh.nodes[flow.vertices, 1],
    )
    metadata = {
        "stage": {"status": "converged", "pseudo_time_step_s": 0.1},
        "momentum_properties": {"viscosity": 0.1},
        "input_sha256": {"mesh": "example"},
        "level": 0,
        "file": field.name,
        "field_sha256": hashlib.sha256(field.read_bytes()).hexdigest(),
    }
    record = {"configuration": {"convection_form": "advective"}, "grad_div_coefficient_m2_s": 0.0}
    if alteration == "field_hash":
        metadata["field_sha256"] = "different"
    elif alteration == "convection":
        record["configuration"]["convection_form"] = "skew"
    elif alteration == "grad_div":
        record["grad_div_coefficient_m2_s"] = 1.0
    elif alteration == "stage":
        metadata["stage"]["status"] = "iteration_cap"
    write_report(tmp_path / "baseline-checkpoint.json", metadata)
    write_report(tmp_path / "record.json", record)
    args = (tmp_path, flow, fixed, values, {"viscosity": 0.1}, {"mesh": "example"}, 0)
    if alteration is None:
        guess, origin = checked_initial_guess(*args)
        assert guess.status == "initial"
        assert (
            origin["initialization_physical_steady_residuals"]["momentum_relative_residual"] > 1e-9
        )
        solution = flow.solve(
            np.zeros_like(flow.quadrature_points), fixed, values, initial=guess, method="newton"
        )
        assert solution.status == "converged"
        assert (
            max(
                flow.verify(solution, np.zeros_like(flow.quadrature_points), fixed, values).values()
            )
            <= 1e-9
        )
    else:
        with pytest.raises(ValueError):
            checked_initial_guess(*args)
