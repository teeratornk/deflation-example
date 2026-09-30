"""Separate time-grid studies retain physics, source identity and final accuracy."""

import copy
import json

import numpy as np
import pytest

from deflation_example.coupled_discretization import PROTOCOL, configuration, verify_returned
from deflation_example.coupled_interval_refinement import refine_interval


def source():
    return {
        "status": "trust_radius_exhausted",
        "configuration": dict(
            transient=True,
            slabs=16,
            horizon_s=600.0,
            target_startup_s=60.0,
            alpha=1e-14,
            queries=[{"target": n, "upper_K": 357.3} for n in (7, 8, 9)],
            lower_K=337.3,
            inner_preconditioner="frozen",
            frozen_sweeps=3,
            transport_form="advective",
            initial_state_snapshot="old-snapshot",
            initial_state_assessment="old-assessment",
        ),
    }


@pytest.mark.parametrize("form", ["advective", "skew"])
@pytest.mark.parametrize("slabs", [16, 32, 64])
def test_six_cases_preserve_physics_and_start_without_optimization_history(form, slabs):
    record = source()
    before = copy.deepcopy(record)
    cfg = configuration(record, form, slabs)
    assert record == before
    assert cfg["slabs"] == slabs and cfg["transport_form"] == form
    assert cfg["initial_state_policy"] == "physical_initial"
    assert "initial_state_snapshot" not in cfg
    for key in ("queries", "alpha", "lower_K", "horizon_s", "target_startup_s"):
        assert cfg[key] == before["configuration"][key]
    assert (cfg["inner_tolerance"], cfg["nonlinear_tolerance"], cfg["flow_tolerance"]) == (
        1e-10,
        1e-8,
        1e-12,
    )
    assert cfg["nonlinear_cap"] == 200 and cfg["minimum_radius_K"] == 1e-10


@pytest.mark.parametrize(
    "key,value",
    [
        ("alpha", 1e-12),
        ("horizon_s", 3600),
        ("target_startup_s", 0),
        ("slabs", 32),
        ("frozen_sweeps", 1),
    ],
)
def test_changed_source_protocol_fails(key, value):
    record = source()
    record["configuration"][key] = value
    with pytest.raises(ValueError, match="source"):
        configuration(record, "skew", 32)


def test_failed_optimization_cannot_trigger_derivative_or_gpu_gate(tmp_path):
    result = verify_returned({"all_problems_verified": False}, tmp_path)
    assert result["status"] == "optimization_gate_failed" and not result["all_verified"]
    assert PROTOCOL["temperature_sensitivity_threshold_K"] == 0.05


@pytest.mark.parametrize("subdivision", [1, 2, 4])
def test_interval_substeps_carry_temporal_sensitivity_and_preserve_inputs(subdivision):
    from test_coupled_derivatives import small_coupled_problem

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    state = np.linspace(0.04, 0.12, problem.size)
    evaluation = problem.evaluate(state)
    fields = dict(
        state=state,
        velocity=np.stack([f.velocity for f in evaluation.flows]),
        pressure=np.stack([f.pressure for f in evaluation.flows]),
    )
    original = {k: v.copy() for k, v in fields.items()}
    direction = np.zeros(len(problem.mesh.nodes))
    direction[problem.free] = np.linspace(0.2, 1, problem.spatial_size)
    center, derivative, report = refine_interval(problem, fields, direction, 1, subdivision, 0)
    assert report["verified"] and len(report["steps"]) == subdivision
    assert all(r["time_step_s"] == pytest.approx(0.35 / subdivision) for r in report["steps"])
    plus, _, rp = refine_interval(problem, fields, direction, 1, subdivision, 1e-6)
    minus, _, rm = refine_interval(problem, fields, direction, 1, subdivision, -1e-6)
    assert rp["verified"] and rm["verified"]
    numerical = (plus.velocity - minus.velocity) / 2e-6
    np.testing.assert_allclose(derivative, numerical, rtol=2e-4, atol=2e-8)
    if subdivision == 1:
        np.testing.assert_allclose(center.velocity, evaluation.flows[1].velocity, atol=1e-10)
    for key in fields:
        np.testing.assert_array_equal(fields[key], original[key])


def test_physical_initialization_and_explicit_radii_in_complete_runner(tmp_path, monkeypatch):
    from deflation_example import coupled_trust_run as runner
    from test_coupled_derivatives import small_coupled_problem

    class Memory:
        def __init__(self, *args):
            pass

        def start(self):
            pass

        def finish(self):
            return {"scope": "unit-test stub"}

    monkeypatch.setattr(runner, "ProcessMemory", Memory)
    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    cfg = dict(
        threads=1,
        device="cpu",
        method="jacobi",
        rank=0,
        recycle_window=16,
        inner_tolerance=1e-10,
        inner_cap=50000,
        inner_refresh=50,
        nonlinear_tolerance=1e-8,
        qp_tolerance=1e-10,
        qp_cap=100,
        secant_memory=10,
        inner_preconditioner="frozen",
        frozen_sweeps=3,
        trust_accuracy="adaptive_projected",
        trial_policy="backtrack",
        qp_solver="projected",
        nonlinear_cap=1,
        lower_K=299.4,
        queries=[{"target": 7, "upper_K": 300.6}],
        target_count=16,
        target_startup_s=60,
        initial_state_policy="physical_initial",
        initial_radius_K=0.25,
        minimum_radius_K=1e-10,
        linear_heartbeat_seconds=None,
    )
    monkeypatch.setattr(runner, "desired_temperature", lambda *args: np.full(problem.size, 0.1))
    report = runner.run(
        cfg,
        tmp_path / "run",
        problem_loader=lambda cfg: (problem, {"baseline_sha256": "fixture", "seconds": 0}),
        budget_seconds=120,
    )
    assert report["initial_state"]["policy"] == "physical_initial"
    assert report["cases"][0]["upper_K"] == 300.6
    assert report["cases"][0]["status"] in {"converged", "nonlinear_iteration_cap"}
    from deflation_example.coupled_recovery import RecoveryStore, identity

    stored = json.loads((tmp_path / "run/record.json").read_text())
    recovered = RecoveryStore(
        tmp_path / "run/recovery",
        identity(stored["configuration"], stored["environment"]["source_sha256"]),
    ).load()
    if recovered["optimizer"] is not None:
        assert recovered["optimizer"]["minimum_radius_K"] == 1e-10
    assert report["cumulative_attempt_seconds"] > 0


@pytest.mark.parametrize("form", ["advective", "skew"])
@pytest.mark.parametrize("consistent", [False, True])
def test_selected_transport_has_exact_thermal_and_coupled_derivatives(form, consistent):
    from test_coupled_derivatives import small_coupled_problem
    from deflation_example.coupled_derivatives import thermal_velocity_jacobian
    from deflation_example.coupled_trust_check import check

    problem = small_coupled_problem(
        [0.2, 0.35],
        uniform_capacity=True,
        consistent=consistent,
        streamline_rule="smooth_p8",
        transport_form=form,
    )
    rng = np.random.default_rng(834)
    state = rng.uniform(0.04, 0.12, problem.size)
    velocity = problem.initial_flow.velocity + 0.003 * rng.normal(size=(problem.flow.nv, 2))
    direction = rng.normal(size=velocity.shape)
    full = problem.full_temperature(state[: problem.spatial_size])
    derivative = thermal_velocity_jacobian(
        problem.flow,
        velocity,
        full,
        problem.capacity,
        problem.conductivity,
        problem.velocity_scale,
        limit_rows=consistent,
        residual_weighted=consistent,
        streamline_rule="smooth_p8",
        transport_form=form,
    )
    h = 1e-6
    numerical = (
        problem.assemble(velocity + h * direction).stiffness @ full
        - problem.assemble(velocity - h * direction).stiffness @ full
    ) / (2 * h)
    analytic = derivative @ np.r_[direction[:, 0], direction[:, 1], np.zeros(problem.flow.np)]
    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)
    evaluation = problem.evaluate(state)
    report = check(problem, state, evaluation, np.full(problem.size, 0.15))
    assert report["derivatives_passed"]
    assert report["adjoint"]["gradient_relative_difference"] < 1e-10
