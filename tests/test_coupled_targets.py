"""The declared startup preserves physical time, initial values and later targets."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from deflation_example.coupled_targets import (
    desired_temperature,
    refined_desired_temperature,
    smooth_startup,
)
from deflation_example.mesh_showcases import desired_temperature as original_target


def problem(steps=(15.0, 15.0, 30.0, 60.0, 480.0), time_scale=12385293.093):
    nodes = np.array([[0.1, 0.0], [0.2, 0.5], [0.3, 1.0]])
    return SimpleNamespace(
        assembly=SimpleNamespace(mesh=SimpleNamespace(nodes=nodes)),
        free=np.arange(3),
        initial=np.array([0.02, 0.04, 0.06]),
        physical_steps=np.asarray(steps),
        steps=np.asarray(steps) / time_scale,
        time_scale=time_scale,
    )


def test_smoothstep_values_and_endpoint_slopes():
    times = np.array([0.0, 15.0, 30.0, 60.0, 600.0])
    np.testing.assert_array_equal(smooth_startup(times, 60), [0, 0.15625, 0.5, 1, 1])
    h = 1e-4
    assert smooth_startup(h, 60) / h < 1e-6
    assert (1 - smooth_startup(60 - h, 60)) / h < 1e-6
    np.testing.assert_array_equal(smooth_startup(times, 0), np.ones(5))


@pytest.mark.parametrize("query", [0, 7, 15])
def test_startup_uses_the_initial_field_and_preserves_later_targets_exactly(query):
    p = problem()
    original = original_target(p, query, 16).reshape(-1, 3)
    actual = desired_temperature(p, query, 16, 60).reshape(-1, 3)
    expected = p.initial + np.array([0.15625, 0.5])[:, None] * (original[:2] - p.initial)
    np.testing.assert_array_equal(actual[:2], expected)
    np.testing.assert_array_equal(actual[2:], original[2:])
    np.testing.assert_array_equal(desired_temperature(p, query, 16), original.ravel())
    np.testing.assert_array_equal(
        desired_temperature(problem(time_scale=1), query, 16, 60), actual.ravel()
    )
    assert p.physical_steps.sum() == 600


@pytest.mark.parametrize("startup", [0.0, 60.0])
def test_refined_replay_samples_the_same_continuous_target(startup):
    p = problem()
    coarse = desired_temperature(p, 7, 16, startup).reshape(-1, 3)
    fine = refined_desired_temperature(p, 7, 16, 2, startup).reshape(-1, 3)
    np.testing.assert_allclose(fine[1::2], coarse, rtol=1e-15, atol=1e-16)
    subdivided = problem(steps=np.repeat(p.physical_steps / 2, 2))
    np.testing.assert_array_equal(fine.ravel(), desired_temperature(subdivided, 7, 16, startup))


@pytest.mark.parametrize("invalid", [-1, np.nan, np.inf, True, "60"])
def test_invalid_startup_is_rejected(invalid):
    with pytest.raises(ValueError):
        desired_temperature(problem(), 7, 16, invalid)


def test_steady_and_initial_field_contracts():
    p = problem(steps=())
    np.testing.assert_array_equal(desired_temperature(p, 7, 16), original_target(p, 7, 16))
    with pytest.raises(ValueError, match="physical time steps"):
        desired_temperature(p, 7, 16, 60)
    p = problem()
    p.initial = np.zeros(2)
    with pytest.raises(ValueError, match="spatial unknowns"):
        desired_temperature(p, 7, 16, 60)
    with pytest.raises(ValueError, match="nonnegative"):
        smooth_startup([-1, 0], 60)


def test_hydra_defaults_preserve_original_pilots_and_declare_the_new_sequence():
    from hydra import compose, initialize_config_dir
    import deflation_example

    directory = Path(deflation_example.__file__).parent / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        pilot = compose(config_name="coupled_optimize")
        sequence = compose(config_name="coupled_sequence")
        assert pilot.target_startup_s == 0
        assert sequence.target_startup_s == 60
        assert sequence.horizon_s == pilot.horizon_s == 600
        assert sequence.transient
