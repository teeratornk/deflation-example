"""The control is recovered through a factored action, so the transpose must match.

With the consistent stabilisation the source is weighted by the streamline test
function, so the control is no longer recovered by dividing by the lumped mass.
The thermal and velocity blocks are left unnormalised and the action is applied
inside the Jacobian, which is exact only if the transpose applies the transposed
factor first. These tests pin that, and pin that the tangent really is the
derivative of the recovered control.
"""

import numpy as np
import pytest

from test_coupled_derivatives import small_coupled_problem


def consistent_problem(steps):
    problem = small_coupled_problem(steps, uniform_capacity=True)
    problem.consistent_stabilization = True
    problem.assembly = problem.assemble(problem.initial_flow.velocity)
    return problem


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_the_consistent_control_jacobian_is_an_exact_transpose(steps):
    problem = consistent_problem(steps)
    assert problem.assembly.consistent
    state = np.linspace(0.03, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    J = evaluation.jacobian
    assert J.source_factors is not None and len(J.source_factors) == problem.slabs

    rng = np.random.default_rng(4113)
    direction, dual = rng.normal(size=(2, problem.size))
    direction /= np.linalg.norm(direction)
    dual /= np.linalg.norm(dual)
    forward = float(dual @ (J @ direction))
    backward = float(direction @ (J.T @ dual))
    assert abs(forward - backward) <= 1e-10 * max(abs(forward), abs(backward), 1e-30)


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_the_consistent_tangent_is_the_derivative_of_the_recovered_control(steps):
    problem = consistent_problem(steps)
    state = np.linspace(0.03, 0.1, problem.size)
    evaluation = problem.evaluate(state)
    J = evaluation.jacobian
    rng = np.random.default_rng(991)
    direction = rng.normal(size=problem.size)
    direction /= np.linalg.norm(direction)
    tangent = J @ direction
    remainders = []
    for step in (0.01, 0.005, 0.0025):
        trial = problem.evaluate(state + step * direction)
        remainders.append(
            float(np.linalg.norm(trial.control - evaluation.control - step * tangent))
        )
    orders = np.log2(np.array(remainders[:-1]) / np.array(remainders[1:]))
    assert min(orders) > 1.9, f"orders {orders} from remainders {remainders}"


def test_the_lumped_default_is_untouched():
    """Without the consistent option nothing is factored and nothing changes."""
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    assert not problem.assembly.consistent
    evaluation = problem.evaluate(np.linspace(0.03, 0.1, problem.size))
    assert evaluation.jacobian.source_factors is None


@pytest.mark.parametrize("steps", [None, [0.2, 0.35]])
def test_the_consistent_equations_are_verified_against_their_own_forms(steps):
    """The verification must check the equation that was solved, not a lumped cousin.

    With the consistent stabilisation the storage and the source carry the streamline
    weight, and the energy identity still has to close: the weighted parts of both
    operators have vanishing column sums on a simplex, because the gradients of the
    linear shape functions sum to zero, so they move heat between nodes and create
    none. That is asserted here rather than assumed.
    """
    problem = small_coupled_problem(steps, uniform_capacity=True)
    problem.consistent_stabilization = True
    problem.assembly = problem.assemble(problem.initial_flow.velocity)
    assert problem.assembly.consistent
    evaluation = problem.evaluate(np.linspace(0.03, 0.15, problem.size))
    checks = problem.verify(evaluation)
    assert max(row["thermal_relative_residual"] for row in checks) < 1e-12
    assert max(row["energy"]["relative_defect"] for row in checks) < 1e-9
    assert max(row["mass_relative_imbalance"] for row in checks) < 1e-10

    assembly = problem.assembly
    rng = np.random.default_rng(6067)
    trial = rng.normal(size=len(problem.mesh.nodes))
    for weighted, lumped in (
        (assembly.stabilized_source, assembly.mass),
        (assembly.stabilized_storage, assembly.capacity),
    ):
        assert abs(np.asarray(weighted @ trial).sum()) <= 1e-10 * abs(lumped @ trial)


@pytest.mark.parametrize("steps", [[0.2, 0.35]])
def test_checking_the_consistent_model_against_lumped_forms_would_fail(steps):
    """The gate above is only worth anything if the wrong forms are caught by it.

    This reproduces what the verification did before it was rederived and pins that
    it reports a residual far above every tolerance the study uses, so a silent
    certification of the wrong equation is impossible.
    """
    problem = small_coupled_problem(steps, uniform_capacity=True)
    problem.consistent_stabilization = True
    problem.assembly = problem.assemble(problem.initial_flow.velocity)
    evaluation = problem.evaluate(np.linspace(0.03, 0.15, problem.size))
    worst = 0.0
    previous = problem.full_temperature(problem.initial)
    for n, (yn, un, flow) in enumerate(
        zip(
            evaluation.state.reshape(problem.slabs, -1),
            evaluation.control.reshape(problem.slabs, -1),
            evaluation.flows,
            strict=True,
        )
    ):
        y = problem.full_temperature(yn)
        u = np.zeros(len(problem.mesh.nodes))
        u[problem.free] = un
        assembly = problem.assemble(flow.velocity)
        lumped_storage = assembly.capacity * (y - previous) / problem.steps[n]
        lumped_rhs = assembly.load + assembly.mass * u
        reaction = assembly.stiffness @ y + lumped_storage - lumped_rhs
        worst = max(
            worst,
            float(
                np.linalg.norm(reaction[problem.free])
                / max(np.linalg.norm(lumped_rhs[problem.free]), np.finfo(float).tiny)
            ),
        )
        previous = y
    assert worst > 1e-6, f"the lumped check passes at {worst}, so it certifies nothing"
