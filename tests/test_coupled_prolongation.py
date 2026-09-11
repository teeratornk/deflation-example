"""Nested-flow initialization preserves its declared finite-element fields."""

import numpy as np

from deflation_example.axisymmetric_flow import AxisymmetricFlow, FlowResult
from deflation_example.coupled_pilot import prolong_flow
from deflation_example.mesh_refinement import refine
from test_axisymmetric_flow import annular_rectangle


def test_flow_prolongation_preserves_quadratic_velocity_and_linear_pressure():
    coarse = annular_rectangle(3)
    fine, P, parent = refine(coarse)
    a, b = AxisymmetricFlow(coarse, 0.1), AxisymmetricFlow(fine, 0.1)

    def velocity(x):
        r, z = x.T
        return np.column_stack((r * z + r**2, 2 * z**2 - r))

    def pressure(x):
        return 3 * x[:, 0] + 2 * x[:, 1] - 4

    original = FlowResult(velocity(a.points), pressure(a.mesh.nodes[a.vertices]), "test", [])
    actual = prolong_flow(a, b, P, parent, original)
    np.testing.assert_allclose(actual.velocity, velocity(b.points), atol=2e-14)
    np.testing.assert_allclose(actual.pressure, pressure(b.mesh.nodes[b.vertices]), atol=2e-14)
