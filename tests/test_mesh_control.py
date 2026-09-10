"""Independent integration, transpose, refinement, and optimality checks."""

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import lsq_linear

from deflation_example.meshes import (
    ThermalMesh,
    assemble_thermal,
    simplex_geometry,
    orient_cells,
    add_facet_load,
)
from deflation_example.mesh_control import build_mesh_control
from deflation_example.mesh_refinement import refine, refine_quadratic_flow
from deflation_example.mesh_reference import build_mesh_reference
from deflation_example.solvers import pdas, deflated_cg, kkt_metrics
from deflation_example.study_solvers import StudySolver


def rectangle(axisymmetric=False):
    x = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    return ThermalMesh(
        x, np.array([[0, 1, 3], [0, 3, 2]]), np.array([0, 1]), np.array([0, 1]), axisymmetric
    )


def assembly(mesh):
    nc, d = len(mesh.cells), mesh.dimension
    k = np.tile(np.eye(d), (nc, 1, 1))
    k[mesh.materials == 1] *= 3
    return assemble_thermal(mesh, k, 1 + mesh.materials.astype(float))


@pytest.mark.parametrize("axisymmetric", [False, True])
def test_exact_affine_field_integrals(axisymmetric):
    mesh = rectangle(axisymmetric)
    grad, _ = simplex_geometry(mesh)
    affine = mesh.nodes @ np.array([2.0, -3.0]) + 4
    np.testing.assert_allclose(
        np.einsum("ei,eid->ed", affine[mesh.cells], grad), np.tile([2.0, -3.0], (2, 1)), atol=1e-13
    )
    a = assembly(mesh)
    np.testing.assert_allclose(a.mass.sum(), 3 * np.pi if axisymmetric else 1)
    np.testing.assert_allclose(a.diffusion @ np.ones(4), 0, atol=2e-14)
    if axisymmetric:
        # Integral r over the annular cross-section: 2*pi*int_1^2 r^2 dr.
        np.testing.assert_allclose(a.mass @ mesh.nodes[:, 0], 14 * np.pi / 3)
    add_facet_load(a, np.array([[0, 1]]), 2.0)
    np.testing.assert_allclose(a.load.sum(), 6 * np.pi if axisymmetric else 2.0)


@pytest.mark.parametrize("dim", [2, 3])
def test_nested_refinement_preserves_geometry_and_linear_fields(dim):
    coarse = (
        rectangle(True)
        if dim == 2
        else ThermalMesh(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0, 1, 2, 3]]),
            np.array([1]),
            np.array([0, 1, 2]),
        )
    )
    fine, P, parent = refine(coarse)
    np.testing.assert_allclose(P @ coarse.nodes, fine.nodes, atol=1e-14)
    np.testing.assert_allclose(assembly(fine).mass.sum(), assembly(coarse).mass.sum())
    np.testing.assert_array_equal(fine.materials, coarse.materials[parent])
    assert len(fine.cells) == len(coarse.cells) * (4 if dim == 2 else 8)
    # No interior midpoint becomes a temperature boundary condition.
    np.testing.assert_allclose(fine.nodes[fine.dirichlet, -1], 0)


def test_quadratic_velocity_integral_is_preserved_under_refinement():
    mesh = rectangle(True)
    x = mesh.nodes[mesh.cells]
    points = np.concatenate(
        (
            x,
            ((x[:, 0] + x[:, 1]) / 2)[:, None],
            ((x[:, 0] + x[:, 2]) / 2)[:, None],
            ((x[:, 1] + x[:, 2]) / 2)[:, None],
        ),
        axis=1,
    )
    v = np.stack((points[:, :, 0] ** 2, points[:, :, 1] ** 2), axis=2)
    fine, P, parent = refine(mesh)
    vf = refine_quadratic_flow(mesh, fine, parent, v)
    a = assemble_thermal(mesh, np.tile(np.eye(2), (2, 1, 1)), np.ones(2), velocity=v)
    af = assemble_thermal(fine, np.tile(np.eye(2), (8, 1, 1)), np.ones(8), velocity=vf)
    np.testing.assert_allclose(
        (P.T @ af.transport @ P).toarray(), a.transport.toarray(), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("steps", [None, [0.02, 0.03, 0.05]])
@pytest.mark.parametrize("matrix_free", [False, True])
def test_complete_optimization_matches_independent_bvls(steps, matrix_free):
    mesh, P, _ = refine(rectangle(True))
    a = assembly(mesh)
    problem = build_mesh_control(
        a,
        time_steps=steps,
        initial=np.full(len(mesh.free), 0.07),
        boundary_value=0.1,
        assembled_restriction=not matrix_free,
    )
    rng = np.random.default_rng(11)
    x, p = rng.normal(size=(2, problem.size))
    np.testing.assert_allclose(problem.apply_transpose(p), problem.A.T @ p, atol=1e-11)
    np.testing.assert_allclose(problem.forward(problem.recover(x)[0]), x, atol=1e-12)
    np.testing.assert_allclose(problem.H @ x, problem.H.assembled() @ x, atol=1e-11)
    np.testing.assert_allclose(problem.H.diagonal(), problem.H.assembled().diagonal(), atol=1e-11)
    I = np.arange(0, problem.size, 2)
    op = problem.H.restrict(I)
    np.testing.assert_allclose(op @ x[I], problem.H.assembled()[I][:, I] @ x[I], atol=1e-11)
    desired = rng.uniform(-0.5, 1.0, problem.size)
    bound = 0.12
    root = np.sqrt(problem.weights)
    factor = np.vstack(
        (np.diag(root), np.sqrt(problem.alpha) * root[:, None] * problem.A.toarray())
    )
    load = np.r_[root * desired, np.sqrt(problem.alpha) * root * problem.forcing]
    independent = lsq_linear(factor, load, bounds=(-np.inf, bound), method="bvls", tol=1e-13)
    assert independent.success
    reference = build_mesh_reference(
        problem, assembly(rectangle(True)), P[mesh.free][:, rectangle(True).free], 1
    )
    adapter = StudySolver("reference", reference=reference, rank=1, cg_factor=0.1)
    result = pdas(
        problem.H,
        problem.load(desired),
        bound,
        initial_active=np.ones(problem.size, dtype=bool),
        tolerance=1e-9,
        linear_solver=lambda B, b, I: adapter.solve(B, b, I)[0],
    )
    assert result["status"] == "converged"
    assert sum(row["left"] for row in result["history"]) > 0
    np.testing.assert_allclose(result["y"], independent.x, atol=1e-10)
    assert max(result["kkt"].values()) < 1e-9
    g = problem.H @ x - problem.load(desired)
    h = 1e-5
    difference = (problem.objective(x + h * p, desired) - problem.objective(x - h * p, desired)) / (
        2 * h
    )
    np.testing.assert_allclose(difference, problem.objective_scale * g @ p, rtol=1e-9)
    multiplier = np.where(
        independent.active_mask == 1, problem.load(desired) - problem.H @ independent.x, 0.0
    )
    assert (
        max(
            kkt_metrics(problem.H, problem.load(desired), bound, independent.x, multiplier).values()
        )
        < 1e-9
    )


def test_restriction_keeps_active_state_equation_rows():
    problem = build_mesh_control(assembly(refine(rectangle())[0]))
    I = np.arange(2)
    correct = problem.H.restrict(I).toarray()
    wrong = (
        sparse.diags(problem.weights[I])
        + problem.alpha
        * problem.A[I][:, I].T
        @ sparse.diags(problem.weights[I])
        @ problem.A[I][:, I]
    ).toarray()
    assert np.linalg.norm(correct - wrong) > 1e-3


def test_invalid_mesh_and_physics_fail_explicitly():
    m = rectangle()
    with pytest.raises(ValueError, match="orientation"):
        ThermalMesh(m.nodes, m.cells[:, [1, 0, 2]], m.materials, m.dirichlet)
    np.testing.assert_array_equal(orient_cells(m.nodes, m.cells[:, [1, 0, 2]]), m.cells)
    with pytest.raises(ValueError, match="positive definite"):
        assemble_thermal(m, np.zeros((2, 2, 2)), np.ones(2))
    with pytest.raises(ValueError, match="once"):
        add_facet_load(assembly(m), np.array([[0, 1], [0, 1]]), 1)
    with pytest.raises(ValueError, match="Time steps"):
        build_mesh_control(assembly(m), time_steps=[0.1, 0])


def test_projected_cg_accepts_matrix_free_weighted_operator():
    problem = build_mesh_control(assembly(refine(rectangle())[0]), assembled_restriction=False)
    I = np.arange(problem.size)
    B = problem.H.restrict(I)
    x = np.arange(problem.size, dtype=float)
    result = deflated_cg(B, B @ x, diagonal=B.diagonal(), rtol=1e-11)
    assert result.status == "converged"
    np.testing.assert_allclose(result.x, x, atol=1e-9)
