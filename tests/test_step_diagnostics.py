"""Analytical conservation checks and immutable field-identity safeguards."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.axisymmetric_flow import AxisymmetricFlow
from deflation_example.coupled_saved import file_digest
from deflation_example.reporting import write_arrays
from test_axisymmetric_flow import annular_rectangle

path = Path(__file__).parents[1] / "examples/coupled_optimization/fixed_point/step_diagnostics.py"
spec = importlib.util.spec_from_file_location("step_diagnostics", path)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_cylindrical_mass_identity_and_exact_solenoidal_field():
    flow = AxisymmetricFlow(annular_rectangle(4), 0.1)
    r, z = flow.points.T
    velocity = np.column_stack((0.02 * r, -0.04 * z))
    report, arrays = diagnostic.cell_mass_diagnostics(flow, velocity)
    assert report["global_boundary_relative_imbalance"] < 1e-13
    assert np.max(np.abs(arrays["cell_net_flux_m3_s"])) < 1e-14
    assert np.max(arrays["cell_divergence_rms_s_inverse"]) < 1e-13
    divergent = np.column_stack((r, z))
    report, arrays = diagnostic.cell_mass_diagnostics(flow, divergent)
    np.testing.assert_allclose(arrays["cell_divergence_rms_s_inverse"], 3, atol=1e-13)
    np.testing.assert_allclose(
        arrays["cell_net_flux_m3_s"], 3 * flow.measure.sum(axis=1), atol=1e-13
    )
    assert report["divergence_theorem_absolute_defect_m3_s"] < 1e-12


def test_zero_boundary_flux_can_hide_interior_divergence():
    flow = AxisymmetricFlow(annular_rectangle(4), 0.1)
    velocity = np.zeros((flow.nv, 2))
    interior = np.setdiff1d(np.arange(flow.nv), flow.boundary)
    velocity[interior[0], 0] = 1
    report, arrays = diagnostic.cell_mass_diagnostics(flow, velocity)
    assert report["global_boundary_relative_imbalance"] == 0
    assert arrays["cell_divergence_rms_s_inverse"].max() > 0
    assert np.abs(arrays["cell_net_flux_m3_s"]).sum() > 0


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf]])
def test_summary_rejects_invalid_arrays(values):
    with pytest.raises(ValueError):
        diagnostic.summary(values)


def test_field_hash_shape_and_path_validation(tmp_path):
    flow = SimpleNamespace(nv=3, np=2)
    problem = SimpleNamespace(spatial_size=4, flow=flow, full_temperature=lambda x: x.copy())
    path = tmp_path / "fields.npz"
    write_arrays(path, state=np.zeros(4), velocity=np.zeros((3, 2)), pressure=np.zeros(2))
    row = {"fields": path.name, "field_sha256": file_digest(path)}
    state, result = diagnostic.checked_field(tmp_path, row, problem)
    assert state.shape == (4,) and result.velocity.shape == (3, 2)
    for bad in ({**row, "fields": "../fields.npz"}, {**row, "field_sha256": "incorrect"}):
        with pytest.raises(ValueError):
            diagnostic.checked_field(tmp_path, bad, problem)
    problem.spatial_size = 5
    with pytest.raises(ValueError, match="dimensions"):
        diagnostic.checked_field(tmp_path, row, problem)


def test_thermal_peclet_scaling_and_finite_fields():
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2], uniform_capacity=True)
    state = problem.full_temperature(problem.initial)
    report, arrays = diagnostic.thermal_diagnostics(problem, state, problem.initial_flow)
    assert np.isfinite(arrays["cell_peclet"]).all()
    assert 0 <= report["positive_off_diagonal_fraction"] <= 1
    before = arrays["cell_peclet"].copy()
    problem.conductivity *= 2
    _, arrays = diagnostic.thermal_diagnostics(problem, state, problem.initial_flow)
    np.testing.assert_allclose(arrays["cell_peclet"], before / 2)
