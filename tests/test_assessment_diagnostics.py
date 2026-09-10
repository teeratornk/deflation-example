"""Formula-based checks for fixed-source refinement diagnostics."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse.linalg import spsolve

from deflation_example.mesh_showcases import _assembly
from test_mesh_control import rectangle


def test_projected_thermal_residual_decomposition():
    path = Path(__file__).parents[1] / "examples/coupled_assessment/diagnose_thermal.py"
    spec = importlib.util.spec_from_file_location("thermal_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mesh = rectangle(True)
    parameters = {
        "geometry": "transformer_2d",
        "physical": {
            "oil_conductivity_W_m_K": 1.0,
            "winding_conductivity_rz_W_m_K": [1.0, 1.0],
            "baffle_conductivity_W_m_K": 1.0,
            "capacity_J_m3_K": [1.0, 1.0, 1.0],
            "length_scale_m": 1.0,
            "time_scale_s": 1.0,
            "inlet_temperature_K": 300.0,
            "temperature_scale_K": 1.0,
        },
    }
    velocity = np.tile([0.1, 0.2], (len(mesh.cells), 6, 1))
    background = np.full(len(mesh.cells), 0.7)
    control = np.array([0.0, 0.0, 1.0, 3.0])
    assembly = _assembly(mesh, parameters, velocity, background)
    state = np.zeros(len(mesh.nodes))
    state[mesh.free] = spsolve(
        assembly.stiffness[mesh.free][:, mesh.free],
        (assembly.mass * control + assembly.load)[mesh.free],
    )
    inputs = SimpleNamespace(
        ends=[],
        mesh=mesh,
        velocity=velocity,
        background=background,
        control=control[None, :],
        state=state[None, :],
        parameters=parameters,
        manifest={"bound_K": 310.0},
    )
    rows = module.diagnose(inputs, 2)
    assert len(rows) == 3
    for row in rows:
        assert row["decomposition_identity_relative_error"] < 1e-13
        assert row["thermal_original_relative_residual"] < 1e-12
        assert abs(row["source_integral_difference_W"]) < 1e-13
        assert row["galerkin_matrix_relative_changes"]["diffusion"] < 1e-13
        assert row["galerkin_matrix_relative_changes"]["transport"] < 1e-13
    assert rows[1]["galerkin_matrix_relative_changes"]["stabilization"] > 0.1
    assert rows[1]["projected_residual_components_relative_norm"]["source_loading"] > 0.01


def test_collected_checks_keep_populations_outcomes_and_derived_values():
    path = Path(__file__).parents[1] / "examples/coupled_assessment/checks-v1.json"
    data = json.loads(path.read_text())
    records = data["forward_records"]
    assert len(records) == 14
    assert len({r["record_sha256"] for r in records}) == 14
    assert sum(r["trajectory_completed"] for r in records) == 6
    assert sum(r["status"] == "property_temperature_domain_exceeded" for r in records) == 8
    for record in records:
        assert record["configuration"]["mode"] == "prescribed"
        assert record["source_tree_clean"]
        assert record["recorded_time_steps"] == len(record["steps"])
        assert record["observed_peak_temperature_K"] == max(
            s["peak_temperature_K"] for s in record["steps"]
        )
        assert record["observed_bound_violation_K"] == max(
            max(0, s["peak_temperature_K"] - record["bound_K"]) for s in record["steps"]
        )
        if not record["trajectory_completed"]:
            assert record["metrics"] is None
    assert len(data["thermal_refinement_diagnostics"]) == 2
    for item in data["thermal_refinement_diagnostics"]:
        for row in item["rows"]:
            assert row["decomposition_identity_relative_error"] < 1e-12
            assert row["galerkin_matrix_relative_changes"]["diffusion"] < 1e-12
            assert row["galerkin_matrix_relative_changes"]["transport"] < 1e-12
