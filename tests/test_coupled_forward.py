"""Coupled manufactured solution, feedback disablement, energy and acceptance."""

import numpy as np
import pytest

from deflation_example.axisymmetric_flow import AxisymmetricFlow, FlowResult
from deflation_example.coupled_forward import CoupledForward
from deflation_example.meshes import ThermalMesh, assemble_thermal, triangle_quadrature
from test_axisymmetric_flow import annular_rectangle


def manufactured(transient=False, feedback=True):
    base = annular_rectangle(4)
    x = base.nodes
    pins = np.flatnonzero(
        np.isclose(x[:, 0], 1)
        | np.isclose(x[:, 0], 2)
        | np.isclose(x[:, 1], 0)
        | np.isclose(x[:, 1], 1)
    )
    mesh = ThermalMesh(x, base.cells, base.materials, pins, True)
    flow = AxisymmetricFlow(mesh, 0.3)
    r, z = flow.points.T
    velocity = np.column_stack((0.02 * r, -0.04 * z))
    temperature = 0.5 * x[:, 1] + (0.01 if transient else 0)
    rq, zq = flow.quadrature_points.transpose(2, 0, 1)
    beta = 0.001 if feedback else 0.0
    force = np.stack((0.02**2 * rq + 0.03, 0.04**2 * zq + 0.07), axis=2)
    force -= flow.buoyancy(300 + temperature, 300, beta)
    bary, weights = triangle_quadrature()
    points = x[mesh.cells]
    area = np.linalg.det(points[:, 1:] - points[:, :1]) / 2
    quadrature = np.einsum("qi,eid->eqd", bary, points)
    measure = 2 * np.pi * area[:, None] * weights * quadrature[:, :, 0]
    # A fixed analytical source balances advection and optional storage.
    heat = -0.02 * quadrature[:, :, 1] + (0.05 if transient else 0)
    fixed_load = np.bincount(
        mesh.cells.ravel(),
        weights=np.einsum("eq,eq,qi->ei", measure, heat, bary).ravel(),
        minlength=len(x),
    )

    def build(v):
        a = assemble_thermal(
            mesh, np.tile(np.eye(2), (len(mesh.cells), 1, 1)), np.ones(len(mesh.cells)), v
        )
        # Include a nonzero saved nodal control and its fixed compensating load.
        a.load[:] = fixed_load - a.mass * 2
        return a

    model = CoupledForward(
        flow,
        build,
        flow.boundary,
        velocity[flow.boundary],
        expansion=beta,
        temperature_offset=300,
        buoyancy_reference=300,
        thermal_boundary=temperature[pins],
        acceleration=force,
        pressure_gauge=(0, 0.03),
    )
    initial = FlowResult(np.zeros_like(velocity), np.zeros(flow.np), "initial", [])
    args = {}
    if transient:
        args = {
            "time_step": 0.2,
            "previous_state": temperature - 0.01,
            "previous_velocity": velocity,
        }
    return model, temperature, velocity, initial, args


@pytest.mark.parametrize("transient", [False, True])
@pytest.mark.parametrize("feedback", [False, True])
@pytest.mark.parametrize("method", ["picard", "newton"])
@pytest.mark.parametrize("convection_form", ["advective", "skew"])
def test_complete_coupled_manufactured_solution(transient, feedback, method, convection_form):
    model, temperature, velocity, initial, args = manufactured(transient, feedback)
    model.flow_method = method
    model.flow.convection_form = convection_form
    control = np.full_like(temperature, 2.0)
    result = model.solve(
        control, np.zeros_like(temperature), initial, feedback=feedback, tolerance=1e-10, **args
    )
    assert result.status == "converged"
    np.testing.assert_array_equal(control, 2.0)
    np.testing.assert_allclose(result.state, temperature, atol=2e-9)
    np.testing.assert_allclose(result.flow.velocity, velocity, atol=2e-10)
    last = result.history[-1]
    assert last["momentum_relative_residual"] <= 1e-10
    assert last["thermal_relative_residual"] <= 1e-10
    assert last["mass_relative_imbalance"] < 1e-12
    assert last["energy"]["relative_defect"] < 1e-10


def test_one_staggered_update_does_not_claim_convergence():
    model, temperature, _, initial, args = manufactured()
    result = model.solve(
        np.full_like(temperature, 2.0),
        np.zeros_like(temperature),
        initial,
        max_iterations=1,
        **args,
    )
    assert result.status == "coupling_iteration_cap"
    assert result.history[-1]["momentum_relative_residual"] > 1e-8


def test_feedback_disabled_ignores_expansion_coefficient():
    model, temperature, _, initial, args = manufactured(feedback=False)
    model.expansion = 0.7
    result = model.solve(
        np.full_like(temperature, 2.0), np.zeros_like(temperature), initial, feedback=False, **args
    )
    assert result.status == "converged"
    np.testing.assert_allclose(result.state, temperature, atol=1e-9)


def test_solid_fluid_interface_conduction_and_energy_under_refinement():
    errors = []
    for n in (4, 8, 16):
        base = annular_rectangle(n)
        radius = base.nodes[:, 0]
        pins = np.flatnonzero(np.isclose(radius, 1) | np.isclose(radius, 2))
        materials = (radius[base.cells].mean(axis=1) > 1.5).astype(int)
        mesh = ThermalMesh(base.nodes, base.cells, materials, pins, True)
        flow = AxisymmetricFlow(mesh, 0.1)
        denominator = np.log(1.5) + np.log(2 / 1.5) / 4
        exact = (
            np.where(radius <= 1.5, np.log(radius), np.log(1.5) + np.log(radius / 1.5) / 4)
            / denominator
        )
        conductivity = (
            np.tile(np.eye(2), (len(mesh.cells), 1, 1)) * (1 + 3 * materials)[:, None, None]
        )

        def build(v):
            return assemble_thermal(mesh, conductivity, 1.0 + materials, v)

        model = CoupledForward(
            flow,
            build,
            flow.boundary,
            np.zeros((len(flow.boundary), 2)),
            expansion=0.0,
            thermal_boundary=exact[pins],
            pressure_gauge=(0, 0),
            flow_method="newton",
        )
        initial = FlowResult(np.zeros((flow.nv, 2)), np.zeros(flow.np), "initial", [])
        result = model.solve(np.zeros(len(radius)), np.zeros(len(radius)), initial, tolerance=1e-10)
        assert result.status == "converged"
        assert result.history[-1]["energy"]["relative_defect"] < 1e-12
        assert result.history[-1]["energy"]["boundary_reaction_absolute_sum"] > 1
        errors.append(np.max(np.abs(result.state - exact)))
    assert errors[0] / errors[1] > 3
    assert errors[1] / errors[2] > 3
