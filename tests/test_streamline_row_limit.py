"""The consistent weighting must be bounded, and the bound must be differentiated.

Without a bound a cell can take more out of a node's row than that node's share of
the cell mass, and the assembled storage and source action lose their positive row
sums. On the transformer mesh that happens in thirty-five rows of ten thousand eight
hundred and thirty, which is enough to leave the step operator nearly singular: the
forward Newton's best step then reduces the residual by a few parts in ten million.

The bound is part of the model, so it is part of the derivative too. These tests pin
that it binds once advection sets the parameter, that it restores the row sums, and
that its own derivative is exact. The small problem's own flow is slow enough that
the diffusive limit is the smaller one and the bound is slack, so the tests that need
it to bind drive it with a fast flow, and the coupled derivative test carries a
faster inlet for the same reason.
"""

import numpy as np
import pytest

from deflation_example.coupled_derivatives import streamline_parameter
from deflation_example.meshes import assemble_thermal, simplex_geometry
from test_coupled_derivatives import small_coupled_problem


def assemblies(problem):
    velocity = problem.velocity_scale * problem.flow.thermal_velocity(
        problem.initial_flow.velocity
    )
    built = {}
    for consistent in (False, True):
        built[consistent] = assemble_thermal(
            problem.mesh,
            problem.conductivity,
            problem.capacity,
            velocity,
            problem.source,
            streamline=True,
            consistent=consistent,
        )
    return built


def test_the_bound_is_slack_where_diffusion_sets_the_parameter():
    """The small problem is slow enough that the diffusive limit is the smaller one.

    So the bound changes nothing there, which is why the tests below drive it with a
    fast flow instead of relying on that problem to reach it.
    """
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    limit = assemblies(problem)[True].streamline_limit[problem.flow.fluid_cells]
    assert limit is not None and np.all(limit == 1.0)


def test_the_bound_restores_positive_rows():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    built = assemblies(problem)
    free = problem.free
    bounded = built[True].source_action[free][:, free]
    sums = np.asarray(bounded.sum(axis=1)).ravel()
    assert sums.min() > 0, f"the bounded source action still has a row summing to {sums.min()}"
    storage = built[True].storage[free][:, free]
    assert np.asarray(storage.sum(axis=1)).ravel().min() > 0
    # The lumped assembly carries no bound and no weighting at all.
    assert built[False].streamline_limit is None


def geometry(problem):
    grad, lump = simplex_geometry(problem.mesh)
    fluid = problem.flow.fluid_cells
    cells = problem.mesh.cells[fluid]
    c = np.asarray(problem.capacity)[fluid]
    kmin = np.linalg.eigvalsh(np.asarray(problem.conductivity)[fluid])[:, 0]
    return grad[fluid], lump[fluid], c, kmin, problem.mesh.nodes[cells]


def parameter_at(problem, velocity, limit_rows):
    grad, lump, c, kmin, vertices = geometry(problem)
    problem.flow.sampled_velocity(velocity)
    center = np.array([-1, -1, -1, 4, 4, 4]) / 9
    v = problem.velocity_scale * np.einsum("a,ead->ed", center, velocity[problem.flow.p2])
    return streamline_parameter(grad, lump, c, kmin, v, vertices, limit_rows)


def fast_flow(problem, speed=600.0, seed=3121):
    """Cell velocities fast enough that the advective limit sets the parameter.

    Whether the bound binds does not depend on how fast the flow is, because in the
    advective branch the parameter falls exactly as fast as the streamline derivative
    rises. It depends on which branch is active, and the small problem's own flow is
    slow enough that the diffusive one is.
    """
    grad, lump, c, kmin, vertices = geometry(problem)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(len(grad), 2))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return grad, lump, c, kmin, vertices, speed * direction


def test_the_bound_binds_once_advection_sets_the_parameter():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    grad, lump, c, kmin, vertices, v = fast_flow(problem)
    free_tau, _ = streamline_parameter(grad, lump, c, kmin, v, vertices, False)
    bounded_tau, _ = streamline_parameter(grad, lump, c, kmin, v, vertices, True)
    assert np.all(bounded_tau <= free_tau * (1 + 1e-12))
    binding = bounded_tau < free_tau * (1 - 1e-12)
    assert binding.mean() > 0.5, f"the bound binds in only {binding.mean():.1%} of cells"
    # What the bound is for: no cell may take more out of a node's row than that
    # node's own share of the cell mass. The bound is written with the Euclidean norm
    # of the streamline derivatives rather than their largest entry, so it holds with
    # room to spare rather than exactly.
    measure = lump.sum(axis=1)
    share = lump / measure[:, None]
    g = c[:, None] * np.einsum("eid,ed->ei", grad, v)
    taken = np.max(np.abs(bounded_tau[:, None] * g) / share, axis=1)
    assert np.all(taken <= 1 + 1e-12), f"a cell still takes {taken.max()} of a node's share"
    assert taken[binding].max() > 0.5, "the bound should be tight enough to be doing work"


def test_the_bounded_parameter_is_the_derivative_of_itself():
    """Centered differences of the bounded parameter against its own derivative."""
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    grad, lump, c, kmin, vertices, v = fast_flow(problem)
    tau, dtau = streamline_parameter(grad, lump, c, kmin, v, vertices, True)
    assert np.all(tau > 0) and np.any(np.abs(dtau) > 0)
    rng = np.random.default_rng(8802)
    step = rng.normal(size=v.shape)
    step /= np.max(np.abs(step))
    exact = np.einsum("ed,ed->e", dtau, step)
    errors = []
    for epsilon in (1e-4, 1e-5, 1e-6):
        plus = streamline_parameter(grad, lump, c, kmin, v + epsilon * step, vertices, True)[0]
        minus = streamline_parameter(grad, lump, c, kmin, v - epsilon * step, vertices, True)[0]
        numerical = (plus - minus) / (2 * epsilon)
        errors.append(
            float(np.max(np.abs(numerical - exact)) / max(np.max(np.abs(exact)), 1e-30))
        )
    assert min(errors) < 1e-7, f"relative errors {errors}"


def test_at_rest_the_bound_is_slack_and_nothing_moves():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    grad, lump, c, kmin, vertices = geometry(problem)
    v = np.zeros((len(grad), 2))
    tau, dtau = streamline_parameter(grad, lump, c, kmin, v, vertices, True)
    assert np.all(np.isfinite(tau)) and np.all(tau > 0)
    assert np.all(dtau == 0.0)


def test_the_switch_onto_the_bound_is_refused():
    """The bound meeting the declared parameter is a branch switch like any other.

    One unit triangle, unit capacity and unit conductivity, with the flow chosen so
    the diffusive limit and the bound coincide exactly. The longest edge is the root
    of two, so that limit is one sixth; a flow of root two along the first axis makes
    the streamline derivatives root two in norm and the bound a third over two, which
    is one sixth as well.
    """
    from deflation_example.coupled_derivatives import StabilizationBranchError

    grad = np.array([[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]])
    lump = np.full((1, 3), 1 / 6)
    vertices = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    c, kmin = np.array([1.0]), np.array([1.0])
    tau, _ = streamline_parameter(
        grad, lump, c, kmin, np.array([[np.sqrt(2), 0.0]]), vertices, False
    )
    np.testing.assert_allclose(tau, 1 / 6, rtol=1e-14)
    with pytest.raises(StabilizationBranchError, match="row limit"):
        streamline_parameter(
            grad, lump, c, kmin, np.array([[np.sqrt(2), 0.0]]), vertices, True
        )
