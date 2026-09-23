"""Face integration, weak continuity and elementwise balance are distinct checks."""

import numpy as np
from scipy.linalg import null_space

from deflation_example.axisymmetric_flow import AxisymmetricFlow
from deflation_example.flow_conservation import cell_fluxes, mass_diagnostics
from test_axisymmetric_flow import annular_rectangle


def test_cell_fluxes_satisfy_cylindrical_divergence_theorem_and_face_cancellation():
    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    velocity = np.random.default_rng(173).normal(size=(flow.nv, 2))
    faces = cell_fluxes(flow, velocity)
    divergence = np.einsum("eia,eqia->eq", velocity[flow.p2], flow.div_basis)
    np.testing.assert_allclose(
        faces.sum(axis=1), (flow.measure * divergence).sum(axis=1), atol=2e-14
    )
    np.testing.assert_allclose(faces.sum(), flow.boundary_flux(velocity).sum(), atol=2e-14)


def test_exact_axisymmetric_solenoidal_polynomial_has_zero_local_and_global_defect():
    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    r, z = flow.points.T
    velocity = np.column_stack((0.02 * r * z, -0.02 * z * z))
    report = mass_diagnostics(flow, velocity)
    assert report["maximum_cell_net_volume_flux_m3_s"] < 1e-14
    assert report["global_relative_imbalance"] < 1e-13
    assert report["divergence_rms_s_inverse"] < 1e-13


def test_weak_continuity_does_not_imply_elementwise_conservation():
    flow = AxisymmetricFlow(annular_rectangle(3), 0.1)
    interior = np.setdiff1d(np.arange(flow.nv), flow.boundary)
    B = np.column_stack([matrix[:, interior].toarray() for matrix in flow.divergence])
    null = null_space(B)
    vector = null @ np.random.default_rng(883).normal(size=null.shape[1])
    velocity = np.zeros((flow.nv, 2))
    velocity[interior] = vector.reshape(2, -1).T
    assert np.linalg.norm(B @ vector) < 1e-13
    report = mass_diagnostics(flow, velocity)
    assert report["global_relative_imbalance"] == 0
    assert report["maximum_cell_net_volume_flux_m3_s"] > 1e-4
    assert report["cell_imbalance_over_total_face_flux"] > 1e-4
    assert report["maximum_cell_divergence_theorem_defect_m3_s"] < 2e-14


def test_zero_velocity_has_finite_zero_diagnostics():
    flow = AxisymmetricFlow(annular_rectangle(2), 0.1)
    report = mass_diagnostics(flow, np.zeros((flow.nv, 2)))
    for key, value in report.items():
        if isinstance(value, float):
            assert value == 0, key


def test_baseline_cli_recomputes_mass_without_a_thermal_replay(monkeypatch, tmp_path):
    import json
    import sys
    from deflation_example import coupled_optimize, flow_conservation
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem(uniform_capacity=True)
    baseline = {"baseline_sha256": "baseline", "configuration": {"level": 0}}
    (tmp_path / "record.json").write_text(json.dumps(baseline))
    monkeypatch.setattr(coupled_optimize, "load_problem", lambda cfg: (problem, baseline))
    output = tmp_path / "mass"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mass",
            "--baseline",
            str(tmp_path),
            "--baseline-only",
            "--output",
            str(output),
            "--threads",
            "1",
        ],
    )
    flow_conservation.main()
    record = json.loads((output / "record.json").read_text())
    assert record["schema"] == "coupled-baseline-mass-diagnostic-v1"
    assert record["mass"]["maximum_cell_net_volume_flux_m3_s"] < 1e-13
    assert max(record["original_equations"].values()) < 1e-10
    assert "replay_source" not in record
