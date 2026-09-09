"""Complete-sequence contracts, including warm starts and failures."""

from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from omegaconf import OmegaConf
from importlib.resources import files

from deflation_example.amgx import AmgxSession, amgx_cg
from deflation_example.benchmark_extended import (
    complete_sequence,
    protocol,
    target_parameters,
    target_values,
)
from deflation_example.problems import build_problem
from deflation_example.solvers import pdas
from test_gpu_benchmark import fake_amgx


def small_protocol():
    config = OmegaConf.load(files("deflation_example.conf") / "benchmark_extended.yaml")
    config.pop("defaults")
    config.pop("hydra")
    config.grids, config.targets, config.rank, config.repeats = [4], 4, 2, 1
    config.calibration_grid = 4
    config.output = "unused"
    return protocol(config)


def test_targets_are_distinct_and_have_different_weight_sets():
    problem = build_problem("cht", 4)
    parameters = [target_parameters(i, 16) for i in range(16)]
    assert len({tuple(sorted(p["weights"])) for p in parameters}) == 16
    targets = [target_values(problem.coordinates, p) for p in parameters]
    assert all(
        np.linalg.norm(a - b) > 1e-5 for i, a in enumerate(targets) for b in targets[i + 1 :]
    )


@pytest.mark.parametrize("factor", [0, -0.1, 2, float("nan")])
def test_invalid_internal_margin_is_rejected(factor):
    controls = small_protocol()["controls"]
    controls["inner_stopping_factor"] = factor
    with pytest.raises(ValueError):
        protocol(controls)


@pytest.mark.parametrize("warm", ["cold", "outer", "outer_inner"])
def test_complete_sequence_warm_start_and_timing(warm):
    api, created, destroyed, _ = fake_amgx()
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    result = complete_sequence(
        4, 0.0001, "amgx_persistent_resources", warm, small_protocol(), torch, api
    )
    assert result["success"]
    assert len(result["cases"]) == 4
    assert created == destroyed
    assert created["Resources"] == 1
    assert all(
        c["status"] == "converged" and max(c["kkt"].values()) <= 1e-8 for c in result["cases"]
    )
    assert sum(result["components_seconds"].values()) == pytest.approx(result["total_seconds"])
    assert all(v >= 0 for v in result["components_seconds"].values())
    assert np.all(np.diff([c["cumulative_seconds"] for c in result["cases"]]) > 0)
    assert result["total_seconds"] >= result["cases"][-1]["cumulative_seconds"]
    if warm == "cold":
        assert all(c["initial_active_count"] == 0 for c in result["cases"])
    else:
        assert all(
            c["initial_active_count"] == p["final_active_count"]
            for p, c in zip(result["cases"], result["cases"][1:])
        )


def test_failed_targets_remain_visible_and_reset_history():
    api, created, destroyed, _ = fake_amgx(fail_setup=True)
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    result = complete_sequence(
        4, 0.0001, "amgx_persistent_resources", "outer_inner", small_protocol(), torch, api
    )
    assert not result["success"]
    assert len(result["cases"]) == 4
    assert all(c["status"] == "exception" for c in result["cases"])
    assert created == destroyed


def test_amgx_rhs_threshold_and_nonzero_initial_guess():
    api, created, destroyed, _ = fake_amgx()
    session = AmgxSession(api, rhs_relative=True).open()
    A, b, x0 = sparse.diags([2.0, 4.0, 8.0]), np.array([3.0, 5.0, 7.0]), np.ones(3)
    result, metrics = amgx_cg(
        A, b, api=api, synchronize=lambda: None, session=session, rhs_relative=True, x0=x0
    )
    session.close()
    np.testing.assert_allclose(result.x, b / A.diagonal())
    np.testing.assert_array_equal(x0, np.ones(3))
    assert session.configuration["solver"]["convergence"] == "ABSOLUTE"
    assert metrics["rhs_relative"] and result.residual <= 1e-10
    assert metrics["native_absolute_tolerance"] == pytest.approx(1e-10 * np.linalg.norm(b))
    assert metrics["solver_configuration_per_call"]
    assert created == destroyed


def test_pdas_timing_covers_accepted_outer_steps():
    result = pdas(sparse.diags([2.0, 4.0, 8.0]), np.array([3.0, 5.0, 7.0]), 0.1)
    assert result["status"] == "converged"
    assert sum(result["timing"]["components_seconds"].values()) == pytest.approx(
        result["timing"]["total_seconds"]
    )


@pytest.mark.gpu
def test_amgx_real_warm_start_uses_rhs_tolerance():
    torch = pytest.importorskip("torch")
    api = pytest.importorskip("pyamgx")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    api.initialize()
    session = AmgxSession(api, rtol=1e-11, rhs_relative=True).open()
    try:
        A = sparse.diags([-np.ones(19), 3 * np.ones(20), -np.ones(19)], [-1, 0, 1], format="csr")
        b = np.linspace(0.1, 2.0, 20)
        for initial in (np.zeros(20), np.linspace(0.05, 1.0, 20)):
            result, metrics = amgx_cg(
                A,
                b,
                api=api,
                synchronize=torch.cuda.synchronize,
                session=session,
                rtol=1e-11,
                acceptance_rtol=1e-10,
                rhs_relative=True,
                x0=initial,
            )
            assert result.status == "converged" and result.residual <= 1e-10
    finally:
        session.close()
        api.finalize()
