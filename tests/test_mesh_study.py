import json
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
from threadpoolctl import threadpool_limits

from deflation_example import benchmark_mesh as study
from deflation_example.mesh_presentation import tetrahedral_slice, summarize
from deflation_example.mesh_reference import build_mesh_reference
from deflation_example.mesh_control import build_mesh_control
from deflation_example.mesh_showcases import build_showcase, engine_velocity


@pytest.fixture(autouse=True)
def bounded_blas_threads():
    with threadpool_limits(2):
        yield


def test_packaged_meshes_have_materials_and_expected_dimensions():
    for name, d, materials in (("engine_3d", 3, [0, 1]), ("transformer_2d", 2, [0, 1, 2])):
        data = build_showcase(name)
        assert data.assembly.mesh.dimension == d
        assert np.unique(data.assembly.mesh.materials).tolist() == materials
        assert np.all(data.assembly.mass > 0)
        assert "mesh.npz" in data.preparation["input_sha256"]
        assert data.preparation["weighted_volume"] > 0


def test_analytic_engine_velocity_is_divergence_free_and_zero_on_bore_walls():
    points = np.array([[0.6, 0.55, 0.4], [0.4, 0.42, 0.6], [0.5, 0.6, 0.3]])
    h = 1e-6
    divergence = np.zeros(len(points))
    for d in range(3):
        offset = np.eye(3)[d] * h
        divergence += (
            engine_velocity(points + offset)[:, d] - engine_velocity(points - offset)[:, d]
        ) / (2 * h)
    np.testing.assert_allclose(divergence, 0, atol=1e-7)
    boundary = np.array([[0.8, 0.5, 0.4], [0.5, 0.5, 0.12], [0.55, 0.6, 0.78]])
    np.testing.assert_allclose(engine_velocity(boundary), 0, atol=1e-12)


def test_slice_interpolates_a_three_dimensional_affine_field():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    values = (x @ np.array([2.0, 3.0, 4.0]))[:, None]
    xy, triangles, sampled, owners = tetrahedral_slice(x, np.array([[0, 1, 2, 3]]), values, 0.25)
    np.testing.assert_allclose(sampled[:, 0], 2 * xy[:, 0] + 0.75 + 4 * xy[:, 1])
    assert triangles.shape == (1, 3)
    assert owners.tolist() == [0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"methods": ["amgx"], "device": "cpu"},
        {"rank": 0},
        {"phase": "final", "repeats": 1},
        {"cg_factor": 2},
        {"matrix_free_inner": True, "device": "cuda"},
    ],
)
def test_invalid_protocol_is_rejected(overrides):
    with pytest.raises(ValueError):
        study.controls(OmegaConf.create(overrides))


def test_all_gpu_workers_initialize_and_finalize_amgx(tmp_path, monkeypatch):
    calls = []
    fake_api = SimpleNamespace(
        initialize=lambda: calls.append("initialize"), finalize=lambda: calls.append("finalize")
    )
    fake_torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="test"),
        set_num_threads=lambda n: None,
        cuda=SimpleNamespace(synchronize=lambda: None, get_device_name=lambda: "test"),
    )
    monkeypatch.setitem(sys.modules, "pyamgx", fake_api)
    monkeypatch.setattr("deflation_example.gpu.require_cuda", lambda: fake_torch)
    monkeypatch.setattr(
        "deflation_example.benchmark_cht.warmup", lambda t, a: calls.append("warmup")
    )
    monkeypatch.setattr(study, "sequence", lambda *args: ({"success": True}, [], None))
    c = study.controls(OmegaConf.create({"device": "cuda", "methods": ["jacobi", "amgx"]}))
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps(c))
    assert study.worker(source, tmp_path / "worker", "jacobi", 0)
    assert calls == ["initialize", "warmup", "finalize"]


def test_worker_preserves_failure_and_finalizes_resources(tmp_path, monkeypatch):
    calls = []
    api = SimpleNamespace(initialize=lambda: None, finalize=lambda: calls.append("finalize"))
    torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="test"),
        set_num_threads=lambda n: None,
        cuda=SimpleNamespace(synchronize=lambda: None, get_device_name=lambda: "test"),
    )
    monkeypatch.setitem(sys.modules, "pyamgx", api)
    monkeypatch.setattr("deflation_example.gpu.require_cuda", lambda: torch)
    monkeypatch.setattr("deflation_example.benchmark_cht.warmup", lambda *args: None)

    def fail(*args):
        raise RuntimeError("failure")

    monkeypatch.setattr(study, "sequence", fail)
    source = tmp_path / "protocol.json"
    c = study.controls(OmegaConf.create({"device": "cuda", "methods": ["amgx"]}))
    source.write_text(json.dumps(c))
    assert not study.worker(source, tmp_path / "worker", "amgx", 0)
    assert calls == ["finalize"]
    assert (
        json.loads((tmp_path / "worker" / "record.json").read_text())["failure"]["error_type"]
        == "RuntimeError"
    )


@pytest.mark.parametrize("construction", ["mode_dependent", "tensor"])
def test_mesh_temporal_reference_has_requested_total_rank(construction):
    data = build_showcase("engine_3d")
    model = build_mesh_control(data.assembly, alpha=1e-6, time_steps=[0.01, 0.04])
    reference = build_mesh_reference(
        model, data.coarse_assembly, data.prolongation, 5, construction, "scaled_schur", "jacobi"
    )
    basis = reference.restrict(np.arange(model.size))
    assert basis.shape == (model.size, 5)
    assert np.linalg.matrix_rank(basis) == 5
    assert reference.description["temporal_selection_metric"] == "jacobi"


def test_summary_retains_missing_and_failed_attempts(tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps({"methods": ["jacobi"], "repeats": 1}))
    (tmp_path / "attempts.json").write_text(
        json.dumps(
            [{"method": "jacobi", "repetition": 0, "status": "timeout", "record": "missing.json"}]
        )
    )
    report = summarize(tmp_path)
    assert report["methods"][0]["accepted"] == 0
    assert report["methods"][0]["median_seconds"] is None
    assert report["methods"][0]["failures"] == ["timeout"]


def test_rank_zero_path_does_not_construct_reference(monkeypatch):
    c = study.controls(
        OmegaConf.create({"targets": 1, "methods": ["jacobi"], "save_fields": False})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Jacobi must skip reference construction")

    monkeypatch.setattr(study, "build_mesh_reference", forbidden)
    record, _, _ = study.sequence(c, "jacobi")
    assert record["success"]
    assert all(i["deployed_rank"] == 0 for case in record["cases"] for i in case["inner"])
    np.testing.assert_allclose(sum(record["components_seconds"].values()), record["seconds"])
    assert (
        0
        <= record["initial_setup_seconds"]
        <= record["cases"][0]["cumulative_seconds"]
        <= record["seconds"]
    )


def test_recycling_budget_defaults_to_reference_rank_and_accepts_separate_budget():
    assert study.controls(OmegaConf.create({"rank": 40}))["recycle_rank"] == 40
    assert study.controls(OmegaConf.create({"rank": 40, "recycle_rank": 60}))["recycle_rank"] == 60
    with pytest.raises(ValueError):
        study.controls(OmegaConf.create({"recycle_rank": 0}))
