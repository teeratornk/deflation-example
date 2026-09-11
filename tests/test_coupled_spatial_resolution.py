"""Spatial replay preserves the signed source function and temporal intervals."""

from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_spatial_resolution import transfer_source
from deflation_example.mesh_refinement import refine
from test_axisymmetric_flow import annular_rectangle


def test_nested_source_transfer_keeps_signed_values_without_clipping():
    mesh = annular_rectangle(3)
    fine, P, _ = refine(mesh)
    common = dict(
        physical_steps=np.array([0.2, 0.3]),
        slabs=2,
        temperature_scale=20.0,
        temperature_offset=341.3,
        time_scale=10.0,
        velocity_scale=10.0,
        expansion=0.002,
        buoyancy_reference=341.3,
    )
    coarse_problem = SimpleNamespace(mesh=mesh, spatial_size=len(mesh.free), **common)
    fine_problem = SimpleNamespace(mesh=fine, spatial_size=len(fine.free), **common)
    source = np.linspace(-4, 3, 2 * len(mesh.free)).reshape(2, -1)
    transferred, actual_P = transfer_source(coarse_problem, fine_problem, source)
    expected_P = P[fine.free][:, mesh.free]
    np.testing.assert_array_equal(actual_P.toarray(), expected_P.toarray())
    np.testing.assert_array_equal(transferred, np.asarray(expected_P @ source.T).T)
    assert transferred.min() < 0 and transferred.max() > 0
    fine_problem.physical_steps = np.array([0.25, 0.25])
    with pytest.raises(ValueError, match="time intervals"):
        transfer_source(coarse_problem, fine_problem, source)


def test_spatial_source_transfer_rejects_changed_physical_scaling():
    mesh = annular_rectangle(3)
    coarse = SimpleNamespace(physical_steps=np.ones(2), temperature_scale=20)
    fine = SimpleNamespace(physical_steps=np.ones(2), temperature_scale=10)
    with pytest.raises(ValueError, match="physical scales"):
        transfer_source(coarse, fine, np.zeros(2 * len(mesh.free)))
