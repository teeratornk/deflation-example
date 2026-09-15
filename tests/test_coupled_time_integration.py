"""Storage coefficients, momentum history and independently evolved BDF2 fields."""

import numpy as np
import pytest

from deflation_example.coupled_time_integration import derivative_coefficients, effective_step
from deflation_example.coupled_newton_replay import newton_trajectory
from test_coupled_derivatives import small_coupled_problem


def test_nonuniform_bdf2_differentiates_quadratics_and_restarts_at_source_jumps():
    steps = np.array([0.2, 0.3, 0.15])
    times = np.r_[0, np.cumsum(steps)]
    for n in (1, 2):
        coefficients = derivative_coefficients(steps, n, "bdf2")
        past = times[n + 1 - np.arange(3)]
        assert coefficients @ np.ones(3) == pytest.approx(0, abs=1e-14)
        assert coefficients @ past == pytest.approx(1)
        assert coefficients @ past**2 == pytest.approx(2 * times[n + 1])
    np.testing.assert_array_equal(derivative_coefficients(steps, 0, "bdf2"), [5, -5, 0])
    np.testing.assert_array_equal(
        derivative_coefficients(steps, 1, "bdf2", restart=True), np.array([1, -1, 0]) / 0.3
    )


@pytest.mark.parametrize(
    "steps,slab,scheme",
    [([1, 3], 1, "bdf2"), ([0], 0, "bdf2"), ([1], 1, "bdf2"), ([1], 0, "unknown")],
)
def test_invalid_time_grids_or_schemes_are_rejected(steps, slab, scheme):
    with pytest.raises(ValueError):
        derivative_coefficients(steps, slab, scheme)


def test_effective_history_preserves_both_storage_terms_and_original_problem():
    from deflation_example.axisymmetric_flow import FlowResult

    problem = small_coupled_problem([0.2, 0.3], uniform_capacity=True)
    state = problem.full_temperature(np.full(problem.spatial_size, 0.1))
    older = problem.full_temperature(np.full(problem.spatial_size, 0.04))
    flow = problem.initial_flow
    older_flow = FlowResult(flow.velocity * 0.6, flow.pressure.copy(), "test", [])
    before = problem.physical_steps.copy()
    effective, previous, momentum, c = effective_step(
        problem, 1, state, flow, older, older_flow, "bdf2"
    )
    np.testing.assert_array_equal(problem.physical_steps, before)
    assert effective.physical_steps[1] == 1 / c[0]
    trial = state + 0.02
    np.testing.assert_allclose(
        (trial - previous) / effective.physical_steps[1], c[0] * trial + c[1] * state + c[2] * older
    )
    np.testing.assert_allclose(
        (flow.velocity - momentum.velocity) / effective.physical_steps[1],
        (c[0] + c[1]) * flow.velocity + c[2] * older_flow.velocity,
        atol=1e-16,
    )


def test_bdf2_replay_retains_backward_euler_when_every_source_interval_restarts():
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    state = np.linspace(0.04, 0.1, problem.size)
    control = problem.evaluate(state).control
    replay = newton_trajectory(problem, control, time_scheme="bdf2", restart_interval=1)
    assert replay["status"] == "converged"
    np.testing.assert_allclose(replay["states"].ravel(), state, atol=2e-10)
    assert all(row["storage_derivative_coefficients_s_inverse"][2] == 0 for row in replay["steps"])


def test_manufactured_thermal_trajectory_converges_at_second_order():
    errors = []
    for count in (8, 16, 32):
        problem = small_coupled_problem(
            np.full(count, 0.4 / count), feedback=0, uniform_capacity=True
        )
        problem.initial[:] = 0
        problem.thermal_boundary[:] = 0
        times = np.cumsum(problem.physical_steps)
        shape = np.linspace(0.04, 0.08, problem.spatial_size)
        full_shape = problem.full_temperature(shape)
        assembly = problem.assemble(problem.initial_flow.velocity)
        stiffness = (assembly.stiffness @ full_shape)[problem.free] / assembly.mass[problem.free]
        capacity = (
            assembly.capacity[problem.free] / assembly.mass[problem.free] * problem.time_scale
        )
        background = assembly.load[problem.free] / assembly.mass[problem.free]
        # Manufactured semi-discrete solution y(t) = sin(t) * shape. The source
        # uses its analytic derivative, independently of the time integrator.
        control = (
            np.sin(times[:, None]) * stiffness
            + np.cos(times[:, None]) * capacity * shape
            - background
        )
        replay = newton_trajectory(problem, control, time_scheme="bdf2", tolerance=1e-11)
        assert replay["status"] == "converged"
        expected = np.sin(times[:, None]) * shape
        errors.append(np.max(np.abs(replay["states"] - expected)))
    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5


def test_saved_bdf2_histories_reproduce_both_equations_at_each_recorded_step():
    from deflation_example.axisymmetric_flow import FlowResult
    from deflation_example.coupled_time_integration import saved_history
    from deflation_example.coupled_newton_replay import criteria_met, step_equations
    from deflation_example.coupled_resolution import forward_model

    problem = small_coupled_problem([0.1, 0.15, 0.08, 0.08], uniform_capacity=True)
    controls = problem.evaluate(np.linspace(0.04, 0.1, problem.size)).control.reshape(4, -1)
    replay = newton_trajectory(
        problem, controls, time_scheme="bdf2", restart_interval=2, tolerance=1e-11
    )
    assert replay["status"] == "converged"
    steps = problem.physical_steps.copy()
    for n in range(problem.slabs):
        effective, previous, previous_flow, coefficients = saved_history(
            problem,
            replay["states"],
            replay["velocities"],
            replay["pressures"],
            n,
            "bdf2",
            restart=n % 2 == 0,
        )
        np.testing.assert_array_equal(
            coefficients, replay["steps"][n]["storage_derivative_coefficients_s_inverse"]
        )
        current = problem.full_temperature(replay["states"][n])
        flow = FlowResult(replay["velocities"][n], replay["pressures"][n], "saved", [])
        source = np.zeros(len(problem.mesh.nodes))
        source[problem.free] = controls[n]
        _, metrics = step_equations(
            effective, forward_model(effective), current, flow, source, previous, previous_flow, n
        )
        assert criteria_met(metrics, 1e-11)
    np.testing.assert_array_equal(problem.physical_steps, steps)
    with pytest.raises(ValueError, match="histories"):
        saved_history(
            problem, replay["states"][:-1], replay["velocities"], replay["pressures"], 0, "bdf2"
        )


def test_saved_time_policy_rejects_unknown_scheme_or_undeclared_bdf2_restarts():
    from deflation_example.coupled_time_integration import replay_time_scheme

    assert replay_time_scheme({}) == "backward_euler"
    for scheme in ("bdf2", "unknown"):
        with pytest.raises(ValueError):
            replay_time_scheme({"forward_solver": {"time_scheme": scheme}})
