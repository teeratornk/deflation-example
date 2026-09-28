"""Per-mode space-time references keep every spatial mode across all time slabs."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from deflation_example import benchmark_cht, benchmark_mesh
from deflation_example.mesh_control import build_mesh_control
from deflation_example.mesh_reference import build_mesh_reference
from deflation_example.mesh_showcases import build_showcase
from deflation_example.spacetime_reference import build_space_time_reference


def spans(basis, vector):
    coefficients, *_ = np.linalg.lstsq(basis, vector, rcond=None)
    return np.linalg.norm(basis @ coefficients - vector) <= 1e-8 * np.linalg.norm(vector)


@pytest.fixture(scope="module")
def engine():
    data = build_showcase("engine_3d")
    model = build_mesh_control(data.assembly, alpha=1e-6, time_steps=[0.01] * 5)
    return data, model


def test_mesh_per_mode_reference_spans_each_mode_in_every_slab(engine):
    data, model = engine
    reference = build_mesh_reference(
        model,
        data.coarse_assembly,
        data.prolongation,
        15,
        "mode_dependent",
        "scaled_schur",
        "jacobi",
        spatial_modes=3,
    )
    assert reference.spatial.shape[1] == 3
    assert sorted(np.bincount(reference.spatial_columns)) == [5, 5, 5]
    basis = reference.restrict(np.arange(model.size))
    assert np.linalg.matrix_rank(basis) == 15
    n = model.spatial_size
    for j in range(3):
        for t in range(5):
            indicator = np.zeros(model.size)
            indicator[t * n : (t + 1) * n] = reference.spatial[:, j]
            assert spans(basis, indicator)
    assert reference.description["selection_policy"] == "3 spatial modes x 5 temporal factors"


def test_mesh_global_selection_is_unchanged_and_mismatches_are_refused(engine):
    data, model = engine
    default = build_mesh_reference(
        model,
        data.coarse_assembly,
        data.prolongation,
        15,
        "mode_dependent",
        "scaled_schur",
        "jacobi",
    )
    assert default.description["selection_policy"] == "global total rank"
    with pytest.raises(ValueError):
        build_mesh_reference(
            model,
            data.coarse_assembly,
            data.prolongation,
            14,
            "mode_dependent",
            "scaled_schur",
            "jacobi",
            spatial_modes=3,
        )
    with pytest.raises(ValueError):
        build_mesh_reference(
            model,
            data.coarse_assembly,
            data.prolongation,
            15,
            "mode_dependent",
            "scaled_schur",
            "jacobi",
            "tridiagonal",
            spatial_modes=3,
        )


def test_cartesian_per_mode_reference_keeps_the_declared_modes():
    config = OmegaConf.structured(benchmark_cht.StudyConfig)
    config.device, config.problem = "cpu", "transient"
    config.methods = ["jacobi", "reference"]
    config.n = config.calibration_grid = 3
    config.slabs, config.rank, config.spatial_modes = 4, 8, 2
    config.bound = 0.0001
    controls = benchmark_cht.specification(config)["controls"]
    trajectory = benchmark_cht.build_model(controls)
    reference = benchmark_cht.build_reference(trajectory, controls)
    assert reference.spatial.shape[1] == 2
    assert sorted(np.bincount(reference.spatial_columns)) == [4, 4]
    basis = reference.restrict(np.arange(trajectory.size))
    n = trajectory.spatial.n**3
    for j in range(2):
        for t in range(4):
            indicator = np.zeros(trajectory.size)
            indicator[t * n : (t + 1) * n] = reference.spatial[:, j]
            assert spans(basis, indicator)
    default = build_space_time_reference(trajectory, 8)
    assert default.description["selection_policy"] == "global total rank"


@pytest.mark.parametrize(
    "overrides",
    [
        {"transient": True, "slabs": 4, "rank": 12, "spatial_modes": 4},
        {"transient": False, "rank": 12, "spatial_modes": 3},
        {"transient": True, "slabs": 4, "rank": 12, "temporal_modes": 3},
        {
            "transient": True,
            "slabs": 4,
            "rank": 12,
            "spatial_modes": 3,
            "temporal_solver": "tridiagonal",
        },
    ],
)
def test_mesh_controls_refuse_inconsistent_per_mode_ranks(overrides):
    with pytest.raises(ValueError):
        benchmark_mesh.controls(OmegaConf.create(overrides))


def test_mesh_controls_accept_a_consistent_per_mode_rank():
    c = benchmark_mesh.controls(
        OmegaConf.create({"transient": True, "slabs": 4, "rank": 12, "spatial_modes": 3})
    )
    assert c["spatial_modes"] == 3 and c["temporal_modes"] is None
