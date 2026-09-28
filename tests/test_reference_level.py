"""A reference coarse space built on a refined mesh level."""

import numpy as np
import pytest
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from deflation_example import benchmark_mesh as study
from deflation_example.mesh_control import build_mesh_control
from deflation_example.mesh_reference import build_mesh_reference
from deflation_example.mesh_showcases import build_showcase


@pytest.fixture(autouse=True)
def bounded_blas_threads():
    with threadpool_limits(2):
        yield


def test_level_zero_reference_is_the_default():
    default = build_showcase("engine_3d", 1)
    explicit = build_showcase("engine_3d", 1, reference_level=0)
    assert abs(default.prolongation - explicit.prolongation).max() == 0
    assert len(default.coarse_assembly.mesh.nodes) == len(explicit.coarse_assembly.mesh.nodes)
    assert explicit.preparation["reference_level"] == 0


def test_a_level_one_reference_prolongs_to_level_two_and_spans_its_rank():
    one = build_showcase("engine_3d", 1)
    two = build_showcase("engine_3d", 2, reference_level=1)
    assert len(two.coarse_assembly.mesh.nodes) == len(one.assembly.mesh.nodes)
    assert two.prolongation.shape == (
        len(two.assembly.mesh.free),
        len(one.assembly.mesh.free),
    )
    assert two.prolongation.min() >= 0 and two.prolongation.max() <= 1 + 1e-12
    model = build_mesh_control(two.assembly, alpha=1e-6)
    reference = build_mesh_reference(
        model, two.coarse_assembly, two.prolongation, 400, spatial_policy="scaled_schur"
    )
    basis = reference.restrict(np.arange(model.size))
    assert basis.shape == (model.size, 400)
    assert np.linalg.matrix_rank(basis) == 400


def test_the_reference_level_cannot_exceed_the_target_level():
    with pytest.raises(ValueError):
        build_showcase("engine_3d", 1, reference_level=2)
    with pytest.raises(ValueError):
        study.controls(OmegaConf.create({"level": 1, "reference_level": 2}))
    assert (
        study.controls(OmegaConf.create({"level": 2, "reference_level": 1}))["reference_level"] == 1
    )
