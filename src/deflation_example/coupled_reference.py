"""Spatial and temporal references from verified computed-flow baselines."""

import numpy as np
from scipy import sparse

from .mesh_control import build_mesh_control
from .mesh_reference import build_mesh_reference
from .mesh_refinement import refine
from .recycling import inactive_indices, transfer_basis
from .validation import integer


class SequentialReference:
    """Ablation: retain only the preceding restricted reference, without learning.

    The first inactive system receives the same reference as direct restriction.
    Each later system receives its zero-extension transfer. Removed entries stay
    zero after release; no replacement directions or Ritz updates are introduced.
    """

    def __init__(self, reference, size):
        self.rank = reference.rank
        self.size = integer(size, "Full reference dimension", 1)
        self.description = {
            **reference.description,
            "transfer_policy": "sequential_zero_extension_without_learning",
        }
        self.reference = reference
        self.indices = None
        self.basis = None

    def begin_system(self, indices):
        indices = inactive_indices(indices)
        if len(indices) and indices.max() >= self.size:
            raise ValueError("Inactive indices exceed the full reference dimension")
        if self.indices is not None and np.array_equal(indices, self.indices):
            return
        self.basis = (
            self.reference.restrict(indices).copy()
            if self.indices is None
            else transfer_basis(self.basis, self.indices, indices)
        )
        self.indices = indices
        self.reference = None

    def restrict(self, indices):
        if self.indices is None or not np.array_equal(indices, self.indices):
            raise ValueError("Begin the inactive system before using sequential transfer")
        return self.basis.copy()

    def storage(self):
        return (
            self.reference.storage()
            if self.reference is not None
            else {
                "full_reference_bytes": 0,
                "compact_reference_bytes": 0,
                "previous_restricted_reference_bytes": self.basis.nbytes,
            }
        )


def nested_prolongation(coarse, fine):
    """Require one exact nested refinement, including materials and boundary nodes."""
    expected, transfer, _ = refine(coarse)
    for name in ("nodes", "cells", "materials", "dirichlet"):
        if not np.array_equal(getattr(expected, name), getattr(fine, name)):
            raise ValueError("The reference mesh must be the immediate nested parent")
    if expected.axisymmetric != fine.axisymmetric:
        raise ValueError("The reference and fine meshes must use the same spatial measure")
    return transfer[fine.free][:, coarse.free]


def build_reference(
    problem,
    rank,
    coarse_problem=None,
    *,
    construction="mode_dependent",
    spatial_policy="scaled_schur",
    temporal_metric="jacobi",
):
    """Charge construction on the fine or a declared nested reference mesh.

    Spatial eigenvectors use the coarse computed-flow thermal operator. Their
    prolongations are approximate fine-grid directions. Temporal factors use
    one-mode compressions of the fine frozen-baseline trajectory Hessian.
    """
    coarse = problem if coarse_problem is None else coarse_problem
    if coarse.alpha != problem.alpha or not np.array_equal(coarse.steps, problem.steps):
        raise ValueError("Reference regularization and temporal discretization must match")
    P = (
        sparse.eye(problem.spatial_size, format="csr")
        if coarse is problem
        else nested_prolongation(coarse.mesh, problem.mesh)
    )
    frozen = build_mesh_control(
        problem.assembly,
        alpha=problem.alpha,
        time_steps=problem.steps if len(problem.steps) else None,
    )
    return build_mesh_reference(
        frozen,
        coarse.assembly,
        P,
        rank,
        construction=construction,
        spatial_policy=spatial_policy,
        temporal_metric=temporal_metric,
    )


def configured_reference(problem, configuration, baseline):
    """Load a compatible coarse baseline only when explicitly requested."""
    from .coupled_optimize import load_problem

    directory = configuration.get("reference_baseline_directory")
    transfer = configuration.get("reference_transfer", "full")
    if transfer not in {"full", "sequential"}:
        raise ValueError("Choose full or sequential reference transfer")
    coarse, record = None, None
    if directory:
        coarse, record = load_problem({**configuration, "baseline_directory": directory})
        for name in (
            "properties",
            "inlet_velocity_m_s",
            "grad_div_coefficient_m2_s",
            "input_sha256",
        ):
            if record.get(name) != baseline.get(name):
                raise ValueError("Coarse and fine baselines must have matching physical inputs")
        if (
            record["configuration"]["convection_form"]
            != baseline["configuration"]["convection_form"]
        ):
            raise ValueError("Coarse and fine momentum forms must match")
    reference = build_reference(
        problem,
        configuration["rank"],
        coarse,
        construction=configuration.get("reference_construction", "mode_dependent"),
        spatial_policy=configuration.get("reference_spatial_policy", "scaled_schur"),
        temporal_metric=configuration.get("reference_temporal_metric", "jacobi"),
    )
    if record is not None:
        reference.description.update(
            coarse_baseline_sha256=record["baseline_sha256"],
            additional_coarse_calibration_seconds=record["seconds"],
        )
    return SequentialReference(reference, problem.size) if transfer == "sequential" else reference
