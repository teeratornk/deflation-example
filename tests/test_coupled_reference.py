"""Nested construction must preserve the declared full-domain transfer."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_reference import build_reference, nested_prolongation
from deflation_example.mesh_refinement import refine
from deflation_example.meshes import assemble_thermal
from test_axisymmetric_flow import annular_rectangle


def problem(mesh, steps):
    n = len(mesh.cells)
    assembly = assemble_thermal(
        mesh,
        np.tile(np.eye(2), (n, 1, 1)),
        np.ones(n),
        np.zeros((n, 2)),
        np.zeros(n),
        streamline=True,
    )
    return SimpleNamespace(
        mesh=mesh,
        assembly=assembly,
        alpha=0.001,
        steps=np.asarray(steps),
        spatial_size=len(mesh.free),
    )


@pytest.mark.parametrize("steps", [[], [0.2, 0.3]])
def test_nested_reference_restricts_the_same_full_space_and_records_approximation(steps):
    coarse = annular_rectangle(3)
    fine, P, _ = refine(coarse)
    np.testing.assert_array_equal(
        nested_prolongation(coarse, fine).toarray(), P[fine.free][:, coarse.free].toarray()
    )
    reference = build_reference(problem(fine, steps), 2, problem(coarse, steps))
    size = len(fine.free) * max(1, len(steps))
    full = reference.restrict(np.arange(size))
    indices = np.arange(0, size, 2)
    np.testing.assert_array_equal(reference.restrict(indices), full[indices])
    assert reference.description["coarse_nodes"] == len(coarse.nodes)
    assert reference.description["fine_nodes"] == len(fine.nodes)
    assert np.linalg.matrix_rank(full) == 2


def test_nested_reference_rejects_material_boundary_and_coordinate_changes():
    coarse = annular_rectangle(3)
    fine, _, _ = refine(coarse)
    for name in ("materials", "nodes", "dirichlet"):
        changed = copy.deepcopy(fine)
        values = getattr(changed, name)
        values.flat[0] += 1
        with pytest.raises(ValueError, match="nested parent"):
            nested_prolongation(coarse, changed)


def test_reference_requires_matching_temporal_model():
    coarse = annular_rectangle(3)
    fine, _, _ = refine(coarse)
    with pytest.raises(ValueError, match="temporal discretization"):
        build_reference(problem(fine, [0.2, 0.3]), 2, problem(coarse, [0.25, 0.25]))
