"""Verify the coupled current/history Jacobian and the amplification pencil."""

import numpy as np
import pytest
from scipy import sparse

from deflation_example.coupled_step_spectrum import step_linearization, amplification_spectrum
from deflation_example.coupled_resolution import forward_model
from test_coupled_derivatives import small_coupled_problem


@pytest.mark.parametrize(
    "consistent,inlet",
    [(False, 0.02), (True, 0.02), (True, 0.3)],
    ids=["lumped", "consistent-slack-bound", "consistent-binding-bound"],
)
def test_current_and_previous_derivatives_against_complete_equations(consistent, inlet):
    """The step Jacobian is the derivative of the step the runner actually solves.

    With a consistent stabilisation the storage, the source and the control's action
    are all weighted by the streamline test function, so all three move with the
    velocity and all three belong in this derivative. The control and the previous
    state are handed over as complete nodal fields, the way the runner holds them.
    """
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True, inlet=inlet)
    if consistent:
        problem.consistent_stabilization = True
        problem.assembly = problem.assemble(problem.initial_flow.velocity)
    built = problem.assemble(problem.initial_flow.velocity)
    assert built.consistent is consistent
    if consistent:
        # The bound on the streamline parameter is part of the model, so it is part of
        # this derivative. At the faster inlet it binds in every fluid cell, which is
        # the regime the transformer mesh is in; at the default it is slack.
        binding = (built.streamline_limit[problem.flow.fluid_cells] < 1.0).mean()
        assert binding == (1.0 if inlet > 0.02 else 0.0)
    Y = np.linspace(0.04, 0.08, problem.size)
    evaluation = problem.evaluate(Y)
    model = forward_model(problem)
    n = 1
    flow = evaluation.flows[n]
    prior = evaluation.flows[n - 1]
    state = problem.full_temperature(Y.reshape(2, -1)[n])
    previous = problem.full_temperature(Y.reshape(2, -1)[n - 1])
    control = np.zeros_like(state)
    control[problem.free] = evaluation.control.reshape(2, -1)[n]
    H, C, _, _ = step_linearization(
        problem, state[problem.free], flow.velocity, n, control=control, previous=previous
    )
    if consistent:
        lumped = small_coupled_problem([0.2, 0.35], uniform_capacity=True, inlet=inlet)
        L = step_linearization(lumped, state[lumped.free], flow.velocity, n)[0]
        share = abs(H - L).max() / abs(L).max()
        assert share > 1e-3, f"the weighting moves the step by only {share}, so this proves nothing"
    rng = np.random.default_rng(842)
    direction = rng.normal(size=H.shape[0])
    nv = len(problem.flow_free)
    dfull = np.zeros(problem.flow.size)
    dfull[problem.flow_free] = direction[:nv]
    dv = np.column_stack((dfull[: problem.flow.nv], dfull[problem.flow.nv : 2 * problem.flow.nv]))
    dy = np.zeros_like(state)
    dy[problem.free] = direction[nv:]
    x = np.r_[flow.velocity[:, 0], flow.velocity[:, 1], flow.pressure]
    dt = float(problem.physical_steps[n])

    def residual(epsilon, history=False):
        v = flow.velocity if history else flow.velocity + epsilon * dv
        y = state if history else state + epsilon * dy
        old_v = prior.velocity + epsilon * dv if history else prior.velocity
        old_y = previous + epsilon * dy if history else previous
        force = problem.flow.load(model._force(y, True)) + problem.flow.mass @ old_v / dt
        rhs = np.r_[force[:, 0], force[:, 1], np.zeros(problem.flow.np)]
        xf = x if history else x + epsilon * dfull
        momentum = (problem.flow.operator(v, time_step=dt) @ xf - rhs)[problem.flow_free]
        _, K, thermal_rhs = model._thermal_system(v, control, old_y, dt)
        return np.r_[momentum, (K @ y - thermal_rhs)[problem.free]]

    epsilon = 1e-6
    current = (residual(epsilon) - residual(-epsilon)) / (2 * epsilon)
    history = (residual(epsilon, True) - residual(-epsilon, True)) / (2 * epsilon)
    np.testing.assert_allclose(current, H @ direction, rtol=2e-7, atol=1e-9)
    np.testing.assert_allclose(history, -C @ direction, rtol=2e-7, atol=1e-9)


def test_diagonal_pencil_reports_amplification_and_residual():
    H = sparse.diags(np.arange(1, 9, dtype=float), format="csc")
    C = sparse.eye(8, format="csr")
    report = amplification_spectrum(H, C, modes=2)
    assert report["status"] == "converged"
    np.testing.assert_allclose(
        sorted(row["amplification_modulus"] for row in report["modes"]), [0.5, 1]
    )
    assert max(row["pencil_relative_residual"] for row in report["modes"]) < 1e-8


def test_the_trust_region_scales_a_runaway_step_and_leaves_a_small_one_alone():
    """A near-singular step Jacobian returns an exact direction that is enormous.

    Halving from one then spends every trial outside the region where the linear
    model means anything, which is exactly how the corrected model stagnated: the
    residual grew quadratically through twenty-one halvings and the direction was
    never tested. The region is declared, not inferred, and it must cost nothing
    when the step is already inside it.
    """
    from deflation_example.coupled_newton_replay import trust_scale

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    flow = problem.initial_flow
    nv = len(problem.flow_free)
    speed = max(float(np.max(np.abs(flow.velocity))), np.finfo(float).tiny)

    small = np.zeros(nv + problem.spatial_size)
    small[nv:] = 0.01
    small[:nv] = 0.01 * speed
    assert trust_scale(problem, flow, small, 0.05) == 1.0

    runaway = np.zeros_like(small)
    runaway[nv:] = 5.0
    assert trust_scale(problem, flow, runaway, 0.05) == pytest.approx(0.01)

    fast = np.zeros_like(small)
    fast[:nv] = 5.0 * speed
    assert trust_scale(problem, flow, fast, 0.05) == pytest.approx(0.01)

    with pytest.raises(ValueError):
        trust_scale(problem, flow, small, 0.0)


def test_the_unrestricted_search_is_what_it_always_was():
    """Without the option every line search still starts at a full Newton step."""
    import inspect

    from deflation_example.coupled_newton_replay import newton_step, newton_trajectory

    for function in (newton_step, newton_trajectory):
        assert inspect.signature(function).parameters["trust_region"].default is None


def test_pressure_is_not_used_as_a_velocity_scale():
    from deflation_example.coupled_newton_replay import trust_scale

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    count = len(problem.flow_free)
    update = np.zeros(count + problem.spatial_size)
    update[:count][problem.flow_free >= 2 * problem.flow.nv] = 1e6
    assert trust_scale(problem, problem.initial_flow, update, 0.05) == 1
    update[count:] = 1.0
    assert trust_scale(problem, problem.initial_flow, update, 0.05) == pytest.approx(0.05)
