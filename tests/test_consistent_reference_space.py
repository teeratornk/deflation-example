"""The coarse space is built from the lumped assembly, on purpose.

The construction eliminates the control by dividing by the lumped mass and forms its
transient operator as a Kronecker sum with a scalar capacity per node. A streamline
weighted storage is neither, and the construction refuses one rather than returning a
number for a different problem. So a consistent model hands it the lumped assembly
deliberately: the coarse space is a preconditioner ingredient, and every solve is
still accepted on the true original residual.

Without this the corrected model's reference arm would not start at all, which is how
the dependency was found.
"""

import numpy as np
import pytest

from deflation_example.coupled_reference import build_reference
from test_coupled_derivatives import small_coupled_problem


def consistent_problem(steps=None):
    problem = small_coupled_problem(steps, uniform_capacity=True, inlet=0.3)
    problem.consistent_stabilization = True
    problem.assembly = problem.assemble(problem.initial_flow.velocity)
    problem.reference_assembly = problem.assemble(problem.initial_flow.velocity, consistent=False)
    return problem


def test_the_reference_assembly_is_the_lumped_one():
    problem = consistent_problem([0.2, 0.35])
    assert problem.assembly.consistent
    assert not problem.reference_assembly.consistent
    assert problem.reference_assembly.stabilized_storage is None
    assert problem.reference_assembly.streamline_limit is None
    # Everything the weighting does not touch is the same operator either way.
    np.testing.assert_array_equal(problem.assembly.mass, problem.reference_assembly.mass)
    np.testing.assert_array_equal(problem.assembly.capacity, problem.reference_assembly.capacity)


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_a_consistent_problem_can_build_its_reference_space(steps):
    problem = consistent_problem(steps)
    reference = build_reference(problem, 2)
    size = problem.spatial_size * max(1, len(problem.steps))
    restricted = reference.restrict(np.arange(size))
    assert restricted.shape[0] == size
    assert np.isfinite(restricted).all()


def test_a_lumped_problem_is_untouched():
    """Its two assemblies are the same object, so nothing about it changed."""
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    assert not problem.assembly.consistent
    assert problem.reference_assembly is problem.assembly


def test_the_construction_still_refuses_a_weighted_assembly_handed_to_it_directly():
    """The guard is the real protection; the lumped choice above is the intent."""
    from types import SimpleNamespace

    from deflation_example.mesh_control import build_mesh_control

    problem = consistent_problem([0.2, 0.35])
    with pytest.raises(ValueError, match="lumped thermal assembly"):
        build_mesh_control(problem.assembly, alpha=problem.alpha)
    mistaken = SimpleNamespace(
        mesh=problem.mesh,
        assembly=problem.assembly,
        reference_assembly=problem.assembly,
        alpha=problem.alpha,
        steps=problem.steps,
        spatial_size=problem.spatial_size,
    )
    with pytest.raises(ValueError, match="lumped thermal assembly"):
        build_reference(mistaken, 2)
