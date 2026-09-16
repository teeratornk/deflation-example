"""The stabilised thermal operator must converge at the rate the elements allow.

Streamline diffusion adds an artificial diffusivity proportional to the element
size. Unless the same weighting is applied to the rest of the equation residual,
that addition does not vanish for the exact solution, and it leaves a first-order
error floor that refinement cannot remove. On this problem the added diffusivity
is hundreds of times the physical conductivity, so the floor dominates.

The test uses a divergence-free axial flow, so the two transport forms coincide
and the only thing under test is the stabilisation.
"""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from deflation_example.meshes import ThermalMesh, assemble_thermal, triangle_quadrature
from test_axisymmetric_flow import annular_rectangle

SPEED = 40.0
CONDUCTIVITY = 0.01


def exact_temperature(nodes):
    """Quadratic in the flow direction, so P1 cannot represent it exactly."""
    return nodes[:, 1] ** 2


def solve_manufactured(n, streamline):
    """One refinement level; returns the largest nodal error against the exact field."""
    base = annular_rectangle(n)
    nodes = base.nodes
    boundary = np.flatnonzero(
        np.isclose(nodes[:, 0], 1)
        | np.isclose(nodes[:, 0], 2)
        | np.isclose(nodes[:, 1], 0)
        | np.isclose(nodes[:, 1], 1)
    )
    mesh = ThermalMesh(nodes, base.cells, base.materials, boundary, True)
    cells = len(mesh.cells)
    conductivity = CONDUCTIVITY * np.tile(np.eye(2), (cells, 1, 1))
    capacity = np.ones(cells)
    # Purely axial and constant, so the axisymmetric divergence vanishes exactly.
    velocity = np.zeros((cells, 6, 2))
    velocity[:, :, 1] = SPEED

    # The source that makes the quadratic field exact, integrated against the P1
    # basis rather than lumped, so the load carries no error of its own.
    bary, weights = triangle_quadrature()
    points = nodes[mesh.cells]
    area = np.linalg.det(points[:, 1:] - points[:, :1]) / 2
    quadrature = np.einsum("qi,eid->eqd", bary, points)
    measure = 2 * np.pi * area[:, None] * weights * quadrature[:, :, 0]
    heat = SPEED * 2 * quadrature[:, :, 1] - 2 * CONDUCTIVITY
    load = np.bincount(
        mesh.cells.ravel(),
        weights=np.einsum("eq,eq,qi->ei", measure, heat, bary).ravel(),
        minlength=len(nodes),
    )

    assembly = assemble_thermal(mesh, conductivity, capacity, velocity, streamline=streamline)
    exact = exact_temperature(nodes)
    free = np.setdiff1d(np.arange(len(nodes)), mesh.dirichlet)
    K = assembly.stiffness.tocsr()
    rhs = load[free] - K[free][:, mesh.dirichlet] @ exact[mesh.dirichlet]
    state = spsolve(sparse.csr_matrix(K[free][:, free]), rhs)
    return float(np.max(np.abs(state - exact[free])))


def rates(streamline):
    errors = [solve_manufactured(n, streamline) for n in (4, 8, 16)]
    return errors, [errors[0] / errors[1], errors[1] / errors[2]]


def test_the_unstabilised_operator_reproduces_the_manufactured_field():
    """Without the artificial term this problem is solved essentially exactly.

    The quadratic field, the one-point diffusion rule and the quadrature load line
    up on this mesh, so the unstabilised error sits at roundoff. That makes the
    comparison below unusually clean: every bit of the stabilised error is the
    artificial term, not the discretisation.
    """
    errors, _ = rates(streamline=False)
    assert all(np.isfinite(errors))
    assert max(errors) < 1e-10, f"unstabilised errors {errors}"


@pytest.mark.xfail(
    reason="The streamline term is not residual weighted, so it leaves a first-order floor",
    strict=True,
)
def test_the_stabilised_operator_converges_at_the_same_rate():
    """This is the acceptance gate for a consistent stabilisation.

    It fails today, and the size of the failure is the point: the ratios sit near
    two rather than four, and the errors themselves are far larger than the
    unstabilised ones, because the added diffusivity dominates the physical one.
    """
    errors, ratio = rates(streamline=True)
    assert min(ratio) > 3, f"stabilised ratios {ratio} from errors {errors}"


def test_the_stabilisation_is_what_costs_the_accuracy():
    """Quantify the floor, so the gate above has a measured size rather than a claim."""
    plain, _ = rates(streamline=False)
    stabilised, ratio = rates(streamline=True)
    assert stabilised[-1] > 10 * plain[-1], (
        f"stabilised {stabilised} should be far worse than unstabilised {plain}"
    )
    assert max(ratio) < 3, f"stabilised convergence should be first order, got {ratio}"
