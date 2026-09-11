"""Spatial and temporal references from verified computed-flow baselines."""

import numpy as np
from scipy import sparse

from .mesh_control import build_mesh_control
from .mesh_reference import build_mesh_reference
from .mesh_refinement import refine


def nested_prolongation(coarse, fine):
    """Require one exact nested refinement, including materials and boundary nodes."""
    expected, transfer, _ = refine(coarse)
    for name in ("nodes", "cells", "materials", "dirichlet"):
        if not np.array_equal(getattr(expected, name), getattr(fine, name)):
            raise ValueError("The reference mesh must be the immediate nested parent")
    if expected.axisymmetric != fine.axisymmetric:
        raise ValueError("The reference and fine meshes must use the same spatial measure")
    return transfer[fine.free][:, coarse.free]


def build_reference(problem, rank, coarse_problem=None):
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
        spatial_policy="scaled_schur",
        temporal_metric="jacobi",
    )


def configured_reference(problem, configuration, baseline):
    """Load a compatible coarse baseline only when explicitly requested."""
    from .coupled_optimize import load_problem

    directory = configuration.get("reference_baseline_directory")
    if not directory:
        return build_reference(problem, configuration["rank"])
    coarse, record = load_problem({**configuration, "baseline_directory": directory})
    for name in ("properties", "inlet_velocity_m_s", "grad_div_coefficient_m2_s", "input_sha256"):
        if record.get(name) != baseline.get(name):
            raise ValueError("Coarse and fine baselines must have matching physical inputs")
    if record["configuration"]["convection_form"] != baseline["configuration"]["convection_form"]:
        raise ValueError("Coarse and fine momentum forms must match")
    reference = build_reference(problem, configuration["rank"], coarse)
    reference.description.update(
        coarse_baseline_sha256=record["baseline_sha256"],
        additional_coarse_calibration_seconds=record["seconds"],
    )
    return reference
