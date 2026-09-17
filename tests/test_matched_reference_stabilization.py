"""The coarse space should be built from the operator it deflates, as far as it can be.

The reference construction needs lumped storage and source, because it eliminates
the control by dividing by the lumped mass and forms a Kronecker sum. Asking for the
lumped assembly used to revert the streamline parameter to its unbounded value as
well, and on the corrected model that is about three times the artificial diffusion
of the operator being solved. The corrected-operator screen measured the consequence:
a rank-200 coarse space removing 2.6% of the iterations it removed 44% of before.

"matched" keeps the bound, so only what the construction cannot represent is lumped.
"""

import numpy as np
import pytest

from deflation_example.coupled_reference import build_reference
from deflation_example.meshes import assemble_thermal
from test_coupled_derivatives import small_coupled_problem

STEPS = [0.2, 0.35]


def fro(matrix):
    return float(np.sqrt(matrix.multiply(matrix).sum()))


def test_the_matched_reference_has_the_corrected_stiffness_exactly():
    problem = small_coupled_problem(
        STEPS, uniform_capacity=True, inlet=0.3, consistent=True, reference_stabilization="matched"
    )
    free = problem.free
    difference = problem.reference_assembly.stiffness - problem.assembly.stiffness
    assert abs(difference[free][:, free]).max() == 0.0


def test_the_matched_reference_lumps_only_what_the_construction_needs():
    problem = small_coupled_problem(
        STEPS, uniform_capacity=True, inlet=0.3, consistent=True, reference_stabilization="matched"
    )
    reference = problem.reference_assembly
    assert not reference.consistent
    assert reference.stabilized_storage is None and reference.stabilized_source is None
    np.testing.assert_array_equal(reference.mass, problem.assembly.mass)
    np.testing.assert_array_equal(reference.capacity, problem.assembly.capacity)
    # The bound is what it keeps, so it records the same factor the operator does.
    np.testing.assert_array_equal(reference.streamline_limit, problem.assembly.streamline_limit)


def test_the_shipped_reference_is_what_every_earlier_record_used():
    """Unbounded, which is the three-fold mismatch the matched option exists to remove."""
    problem = small_coupled_problem(STEPS, uniform_capacity=True, inlet=0.3, consistent=True)
    assert problem.reference_stabilization == "shipped"
    free = problem.free
    unbounded = assemble_thermal(
        problem.mesh,
        problem.conductivity,
        problem.capacity,
        problem.velocity_scale * problem.flow.thermal_velocity(problem.initial_flow.velocity),
        problem.source,
        streamline=True,
    )
    difference = problem.reference_assembly.stabilization - unbounded.stabilization
    assert abs(difference[free][:, free]).max() == 0.0
    ratio = fro(problem.reference_assembly.stabilization[free][:, free]) / fro(
        problem.assembly.stabilization[free][:, free]
    )
    assert ratio > 2.0, f"the shipped reference is only {ratio:.2f} times as diffusive"


@pytest.mark.parametrize("steps", [None, STEPS])
def test_a_matched_reference_space_can_be_built(steps):
    problem = small_coupled_problem(
        steps, uniform_capacity=True, inlet=0.3, consistent=True, reference_stabilization="matched"
    )
    reference = build_reference(problem, 2)
    size = problem.spatial_size * max(1, len(problem.steps))
    assert np.isfinite(reference.restrict(np.arange(size))).all()


def test_the_two_references_give_different_spaces():
    """Otherwise the option would be a name for nothing."""
    shipped = small_coupled_problem(STEPS, uniform_capacity=True, inlet=0.3, consistent=True)
    matched = small_coupled_problem(
        STEPS, uniform_capacity=True, inlet=0.3, consistent=True, reference_stabilization="matched"
    )
    size = shipped.spatial_size * len(shipped.steps)
    a = build_reference(shipped, 2).restrict(np.arange(size))
    b = build_reference(matched, 2).restrict(np.arange(size))
    assert a.shape == b.shape
    assert not np.allclose(a, b)


def test_the_lumped_model_is_untouched_by_the_option():
    for choice in ("shipped", "matched"):
        problem = small_coupled_problem(STEPS, uniform_capacity=True, reference_stabilization=choice)
        assert problem.reference_assembly is problem.assembly


def test_unknown_choices_and_an_unstabilised_bound_are_refused():
    with pytest.raises(ValueError, match="shipped or matched"):
        small_coupled_problem(STEPS, reference_stabilization="spectral")
    problem = small_coupled_problem(STEPS)
    with pytest.raises(ValueError, match="streamline bound needs"):
        assemble_thermal(
            problem.mesh,
            problem.conductivity,
            problem.capacity,
            problem.velocity_scale * problem.flow.thermal_velocity(problem.initial_flow.velocity),
            problem.source,
            streamline=False,
            bound_streamline=True,
        )
