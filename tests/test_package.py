"""Portable API, Hydra configuration, fixture parity and failure behavior."""

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
from scipy import sparse
from deflation_example import build_problem, reference_modes
from deflation_example.problems import laplacian, sine_modes, thermal_convection, cht_operator
from deflation_example.runner import environment, run_demo
from deflation_example.solvers import deflated_cg, pdas


@pytest.mark.parametrize("dim", [2, 3])
def test_sine_eigenpairs(dim):
    L = laplacian(6, dim)
    Z = sine_modes(6, dim, 20)
    np.testing.assert_allclose(Z.T @ Z, np.eye(20), atol=1e-12)
    values = np.diag(Z.T @ (L @ Z))
    np.testing.assert_allclose(L @ Z, Z * values, atol=1e-11)


def test_operator_and_target_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/presets.json").read_text())
    for case in fixture["cases"]:
        problem = build_problem(case["problem"], case["n"])
        vectors = np.random.default_rng(case["seed"]).normal(size=(problem.A.shape[0], 2))
        np.testing.assert_allclose(
            problem.A @ vectors, case["operator_actions"], rtol=1e-13, atol=1e-10
        )
        for angle, target in zip([0, np.pi / 4, np.pi / 2], case["targets"]):
            np.testing.assert_allclose(problem.target(angle), target, rtol=1e-13, atol=1e-14)


def test_transport_coordinate_and_interface():
    p = build_problem("thermal", 8)
    x1, x2 = p.coordinates
    interior = (x1 > 0.2) & (x1 < 0.8) & (x2 > 0.2) & (x2 < 0.8)
    np.testing.assert_allclose(
        (thermal_convection(8, p.coordinates) @ x1)[interior],
        (-100 * np.cos(np.pi * x1) * np.sin(np.pi * x2))[interior],
        atol=1e-12,
    )
    p = build_problem("cht", 6)
    D, C = cht_operator(6, p.coordinates)
    assert np.linalg.norm((D - D.T).data) < 1e-12
    assert np.linalg.norm(C.data) > 0


@pytest.mark.parametrize("problem,grid", [("diffusion", 8), ("thermal", 8), ("cht", 6)])
def test_complete_small_sequence(tmp_path, problem, grid):
    result = run_demo(tmp_path / problem, problem=problem, grid=grid, rank=5)
    assert result["success"]
    assert len(result["cases"]) == 3
    assert "torch" not in sys.modules
    for row in result["cases"]:
        assert row["optimization_passed"]
        assert all(k["residual"] <= 1e-10 for k in row["kernels"].values())
        assert row["state_relative_error"] < 1e-7


def test_output_is_preserved_and_failure_is_recorded(tmp_path):
    output = tmp_path / "failed"
    result = run_demo(output, grid=8, rank=5, maxiter=0)
    assert not result["success"]
    original = (output / "results.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_demo(output, grid=8, rank=5)
    assert (output / "results.json").read_bytes() == original
    broken = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_demo(broken, grid=2, rank=20)
    assert json.loads((broken / "results.json").read_text())["success"] is False


@pytest.mark.parametrize(
    "options",
    [dict(alpha=0), dict(alpha=float("nan")), dict(n=1), dict(n=2.5), dict(name="unknown")],
)
def test_invalid_problem(options):
    with pytest.raises(ValueError):
        build_problem(**options)


@pytest.mark.parametrize(
    "options",
    [
        dict(b=[1.0, np.nan]),
        dict(b=[[1.0], [2.0]]),
        dict(diagonal=[1.0]),
        dict(x0=[0.0]),
        dict(basis=np.ones((3, 1))),
        dict(rtol=float("nan")),
        dict(maxiter=-1),
        dict(refresh=0),
        dict(condition_limit=0.5),
    ],
)
def test_invalid_linear_input(options):
    controls = dict(b=np.ones(2))
    controls.update(options)
    with pytest.raises(ValueError):
        deflated_cg(np.eye(2), **controls)


def test_coarse_failure_and_all_active():
    assert deflated_cg(-np.eye(2), np.ones(2), np.eye(2)).status == "breakdown"
    result = deflated_cg(np.diag([1.0, 100.0]), np.ones(2), np.eye(2), condition_limit=10)
    assert result.fallback_reason == "coarse_condition_limit"
    result = pdas(sparse.eye(3), np.ones(3), 0.0, initial_active=np.ones(3, dtype=bool))
    assert result["status"] == "converged" and np.all(result["active"])
    assert result["history"][0]["linear_iterations"] == 0


def test_no_cuda_is_an_error():
    from deflation_example.gpu import require_cuda

    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(RuntimeError, match="install"):
            require_cuda()


def test_portable_metadata():
    record = environment()
    from deflation_example import __version__

    assert record["source_package"] == __version__
    assert not {"hostname", "slurm_job_id", "username"} & record.keys()
    assert all("filepath" not in entry for entry in record["blas"])
    assert all("/" not in name for name in record["source_sha256"])


def cli(tmp_path, *options):
    return subprocess.run(
        [sys.executable, "-m", "deflation_example", *options],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )


def test_hydra_packaged_configuration_and_portable_run(tmp_path):
    config = cli(tmp_path, "--cfg", "job")
    assert config.returncode == 0 and "rank: 20" in config.stdout
    run = cli(tmp_path, "grid=8", "rank=5", "output=example")
    assert run.returncode == 0, run.stderr + run.stdout
    assert (tmp_path / "example/results.json").is_file()
    assert not (tmp_path / ".hydra").exists()
    assert not list(tmp_path.glob("*.log"))
    invalid = cli(tmp_path, "grid=bad", "output=invalid")
    assert invalid.returncode != 0
    unknown = cli(tmp_path, "misspelled=1", "output=unknown")
    assert unknown.returncode != 0
    failed = cli(tmp_path, "grid=8", "rank=5", "maxiter=0", "output=failed")
    assert failed.returncode == 1


def test_reference_validation():
    with pytest.raises(ValueError):
        reference_modes(build_problem(n=4), 16)


@pytest.mark.parametrize("flag", ["-m", "--multirun", "--multi"])
def test_sweep_mode_cannot_write_machine_metadata(tmp_path, flag):
    result = cli(tmp_path, flag, "grid=4,5", "rank=2")
    assert result.returncode == 2
    assert "not enabled" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.gpu
def test_cuda_complete_demo(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device")
    report = run_demo(tmp_path / "cuda", grid=8, rank=5, device="cuda")
    assert report["success"]
    for case in report["cases"]:
        assert set(case["kernels"]) == {
            "jacobi_cpu",
            "deflated_cpu",
            "jacobi_cuda",
            "deflated_cuda",
        }
        assert all(row["residual"] <= 1e-10 for row in case["kernels"].values())


@pytest.mark.gpu
def test_cuda_core_parity():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device")
    from deflation_example.gpu import gpu_deflated_cg

    rng = np.random.default_rng(3)
    M = rng.normal(size=(12, 12))
    A = sparse.csr_matrix(M.T @ M + np.eye(12))
    b = rng.normal(size=12)
    Z = np.linalg.eigh(A.toarray())[1][:, :4]
    for basis in [None, Z, np.column_stack([Z, Z]), np.eye(12)]:
        result, timing = gpu_deflated_cg(A, b, basis, A.diagonal(), refresh=3)
        assert result.status == "converged" and result.residual <= 1e-10
        np.testing.assert_allclose(result.x, np.linalg.solve(A.toarray(), b), rtol=1e-8, atol=1e-10)
        assert timing["total_seconds"] >= timing["setup_seconds"] + timing["solve_seconds"]
    result, _ = gpu_deflated_cg(sparse.eye(3), np.zeros(3), x0=np.ones(3))
    assert result.status == "converged" and result.residual <= 1e-10
    result, _ = gpu_deflated_cg(sparse.eye(3), np.ones(3), maxiter=0)
    assert result.status == "maxiter"
    result, _ = gpu_deflated_cg(-sparse.eye(3), np.ones(3), np.eye(3))
    assert result.status == "breakdown"
    result, _ = gpu_deflated_cg(
        sparse.diags([1.0, 100.0]), np.ones(2), np.eye(2), condition_limit=10
    )
    assert result.status == "converged"
    assert result.fallback_reason == "coarse_condition_limit" and result.rank == 0
    for scale in [1e-200, 1e200, 1e308]:
        result, _ = gpu_deflated_cg(sparse.eye(4), np.ones(4) * scale)
        if result.status == "converged":
            np.testing.assert_allclose(result.x / scale, np.ones(4), rtol=1e-10)
        else:
            assert result.residual > 1e-10
