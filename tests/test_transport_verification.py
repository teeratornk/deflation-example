"""Verify the transport forms: convergence, the outflow energy identity and stability.

The symmetric-mode LU used below performs no row interchanges (checked), so by
Sylvester's law of inertia its pivot signs count the negative, zero and positive
eigenvalues of the symmetric part. A spatial operator whose symmetric part is
positive definite has no growing backward-Euler mode at any time step.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu

from deflation_example.mesh_showcases import build_showcase
from deflation_example.meshes import ThermalMesh, assemble_thermal

DATA = Path(__file__).resolve().parents[1] / "src/deflation_example/data/transformer_2d"


def inertia(matrix):
    """(negative, zero, positive) eigenvalue counts of a symmetric sparse matrix."""
    lu = splu(sparse.csc_matrix(matrix), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
              options={"SymmetricMode": True})
    if not np.array_equal(lu.perm_r, lu.perm_c):
        raise AssertionError("row interchanges invalidate the inertia count")
    pivots = lu.U.diagonal()
    return int((pivots < 0).sum()), int((pivots == 0).sum()), int((pivots > 0).sum())


def symmetric_free(matrix, free):
    block = matrix[free][:, free]
    return 0.5 * (block + block.T)


# ----------------------------------------------------------------------------- convergence
def annulus(n, r0=0.5, r1=1.5):
    r, z = np.meshgrid(np.linspace(r0, r1, n + 1), np.linspace(0, 1, n + 1), indexing="ij")
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    cells = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    boundary = np.unique(np.concatenate([idx[0], idx[-1], idx[:, 0], idx[:, -1]]))
    nodes = np.column_stack([r.ravel(), z.ravel()])
    return ThermalMesh(nodes, cells, np.zeros(len(cells), dtype=int), boundary, True)


def swirl_free_velocity(points, amplitude):
    # From the stream function psi = amplitude r^2 sin(pi z): divergence-free in (r, z).
    r, z = points[..., 0], points[..., 1]
    return np.stack((-amplitude * np.pi * r * np.cos(np.pi * z),
                     2 * amplitude * np.sin(np.pi * z)), axis=-1)


def exact(points):
    return np.sin(np.pi * points[..., 0]) * np.cos(np.pi * points[..., 1])


def source(points, amplitude):
    r, z = points[..., 0], points[..., 1]
    v = swirl_free_velocity(points, amplitude)
    dr = np.pi * np.cos(np.pi * r) * np.cos(np.pi * z)
    dz = -np.pi * np.sin(np.pi * r) * np.sin(np.pi * z)
    return (v[..., 0] * dr + v[..., 1] * dz + 2 * np.pi**2 * exact(points)
            - np.pi / r * np.cos(np.pi * r) * np.cos(np.pi * z))


def relative_error(n, amplitude, form, streamline):
    mesh = annulus(n)
    x = mesh.nodes[mesh.cells]
    p2 = np.concatenate((x, (x[:, :1] + x[:, 1:2]) / 2, (x[:, :1] + x[:, 2:3]) / 2,
                         (x[:, 1:2] + x[:, 2:3]) / 2), axis=1)
    a = assemble_thermal(mesh, np.tile(np.eye(2), (len(mesh.cells), 1, 1)),
                         np.ones(len(mesh.cells)), swirl_free_velocity(p2, amplitude),
                         source(x.mean(axis=1), amplitude), streamline=streamline,
                         transport_form=form)
    K = a.stiffness.tocsr()
    free, fixed = mesh.free, mesh.dirichlet
    reference = exact(mesh.nodes)
    state = reference.copy()
    state[free] = spsolve(K[free][:, free].tocsc(), a.load[free] - K[free][:, fixed] @ reference[fixed])
    return np.sqrt(np.sum(a.mass * (state - reference) ** 2) / np.sum(a.mass * reference**2))


@pytest.mark.parametrize("streamline", [False, True])
def test_both_forms_converge_at_second_order_for_resolved_axisymmetric_flow(streamline):
    errors = {form: [relative_error(n, 1.0, form, streamline) for n in (8, 16, 32)]
              for form in ("advective", "skew")}
    for values in errors.values():
        orders = np.log2(np.array(values[:-1]) / np.array(values[1:]))
        assert np.all(orders > 1.95)
    np.testing.assert_allclose(errors["skew"], errors["advective"], rtol=1e-2)


def test_streamline_diffusion_is_first_order_consistent_in_advection_dominated_flow():
    errors = [relative_error(n, 60.0, "skew", True) for n in (8, 16, 32, 64)]
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 0.9)


# ----------------------------------------------------------------------------- transformer
def transformer_inputs():
    mesh = ThermalMesh.load(DATA / "mesh.npz")
    parameters = json.loads((DATA / "parameters.json").read_text())
    p = parameters["physical"]
    with np.load(DATA / "inputs.npz", allow_pickle=False) as inputs:
        velocity = inputs["velocity_P2_m_s"] * p["time_scale_s"] / p["length_scale_m"]
    capacity = np.array(p["capacity_J_m3_K"])[mesh.materials] / p["capacity_J_m3_K"][0]
    return mesh, capacity, velocity


def boundary_energy_matrix(mesh, capacity, velocity):
    """Assemble (1/2) sum over boundary edges of c 2 pi r (v.n) N_i N_j independently."""
    edges = {}
    for cell, triangle in enumerate(mesh.cells):
        for local, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            key = tuple(sorted((triangle[i], triangle[j])))
            edges.setdefault(key, []).append((cell, i, j, local))
    s, weights = np.polynomial.legendre.leggauss(4)
    s, weights = (s + 1) / 2, weights / 2
    rows, cols, values = [], [], []
    outward = []
    for owners in edges.values():
        if len(owners) != 1:
            continue
        cell, i, j, local = owners[0]
        triangle = mesh.cells[cell]
        a, b = mesh.nodes[triangle[i]], mesh.nodes[triangle[j]]
        opposite = mesh.nodes[triangle[3 - i - j]]
        tangent = b - a
        normal = np.array([tangent[1], -tangent[0]])
        if normal @ (a - opposite) < 0:
            normal = -normal
        length = np.linalg.norm(tangent)
        normal = normal / length
        midpoint = velocity[cell, 3 + local]
        v = (velocity[cell, i][None] * ((1 - s) * (1 - 2 * s))[:, None]
             + velocity[cell, j][None] * (s * (2 * s - 1))[:, None]
             + midpoint[None] * (4 * s * (1 - s))[:, None])
        flux = v @ normal
        radius = (1 - s) * a[0] + s * b[0]
        shape = np.column_stack((1 - s, s))
        local_matrix = 0.5 * capacity[cell] * length * np.einsum(
            "q,q,qa,qb->ab", weights, 2 * np.pi * radius * flux, shape, shape)
        nodes = (triangle[i], triangle[j])
        for x in range(2):
            for y in range(2):
                rows.append(nodes[x])
                cols.append(nodes[y])
                values.append(local_matrix[x, y])
        outward.append((nodes, flux))
    size = len(mesh.nodes)
    return sparse.coo_matrix((values, (rows, cols)), shape=(size, size)).tocsr(), outward


def test_skew_transport_energy_is_the_transformer_outflow_flux():
    mesh, capacity, velocity = transformer_inputs()
    k = np.tile(np.eye(2), (len(mesh.cells), 1, 1))
    skew = assemble_thermal(mesh, k, capacity, velocity, transport_form="skew")
    boundary, edges = boundary_energy_matrix(mesh, capacity, velocity)
    symmetric = 0.5 * (skew.transport + skew.transport.T)
    difference = abs(symmetric - boundary).max()
    assert difference <= 1e-10 * abs(symmetric).max()
    dirichlet = set(mesh.dirichlet.tolist())
    scale = max(abs(flux).max() for _, flux in edges)
    for nodes, flux in edges:
        # Every boundary edge with a free node carries outflow or no flow.
        if not set(nodes) <= dirichlet:
            assert flux.min() >= -1e-9 * scale


def test_skew_transformer_operator_is_energy_stable_and_the_original_is_not():
    for form, stable in (("skew", True), ("advective", False)):
        a = build_showcase("transformer_2d", transport_form=form).assembly
        negative, zero, positive = inertia(symmetric_free(a.stiffness, a.mesh.free))
        assert zero == 0 and positive > 0
        assert (negative == 0) is stable


def test_bore_in_block_operator_is_energy_stable_on_the_first_refinement():
    a = build_showcase("engine_3d", level=1).assembly
    negative, zero, _ = inertia(symmetric_free(a.stiffness, a.mesh.free))
    assert negative == 0 and zero == 0
