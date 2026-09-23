"""Locate streamline-parameter branch proximity at saved velocity fields."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_optimize import load_problem
from .coupled_review import read_snapshot
from .meshes import simplex_geometry
from .reporting import environment, write_report


def branch_distances(problem, velocities):
    grad, lump = simplex_geometry(problem.mesh)
    fluid = problem.flow.fluid_cells
    cells = problem.mesh.cells[fluid]
    grad, lump = grad[fluid], lump[fluid]
    vertices = problem.mesh.nodes[cells]
    h = np.max(np.linalg.norm(vertices[:, :, None] - vertices[:, None, :], axis=3), axis=(1, 2))
    c = np.asarray(problem.capacity)[fluid]
    k = np.linalg.eigvalsh(np.asarray(problem.conductivity)[fluid])[:, 0]
    diffusion = h**2 / (12 * k)
    share = (lump / lump.sum(axis=1)[:, None]).min(axis=1)
    center = np.array([-1, -1, -1, 4, 4, 4]) / 9
    rows = []
    for slab, velocity in enumerate(velocities):
        v = problem.velocity_scale * np.einsum("a,ead->ed", center, velocity[problem.flow.p2])
        norm = np.linalg.norm(v, axis=1)
        advection = np.full_like(norm, np.inf)
        np.divide(h, 2 * c * norm, out=advection, where=norm > 0)
        free = np.minimum(advection, diffusion)
        reach = np.linalg.norm(np.einsum("eid,ed->ei", grad, v), axis=1)
        bound = np.full_like(norm, np.inf)
        np.divide(share, c * reach, out=bound, where=reach > 0)
        # Only finite branch pairs can approach a switch.
        distances = np.full((len(fluid), 2), np.inf)
        moving = norm > 0
        distances[moving, 0] = np.abs(advection[moving] - diffusion[moving]) / np.maximum(
            advection[moving], diffusion[moving]
        )
        moving = reach > 0
        distances[moving, 1] = np.abs(bound[moving] - free[moving]) / np.maximum(
            bound[moving], free[moving]
        )
        nearest = np.argsort(distances.min(axis=1), kind="stable")[:10]
        rows.append(
            {
                "slab": slab,
                "limited_cells": int(np.count_nonzero(bound < free)),
                "nearest_switches": [
                    {
                        "fluid_cell": int(i),
                        "mesh_cell": int(fluid[i]),
                        "nodes": cells[i],
                        "centroid": vertices[i].mean(axis=0),
                        "advection_diffusion_relative_distance": float(distances[i, 0]),
                        "row_limit_relative_distance": float(distances[i, 1]),
                        "diffusion_tau": diffusion[i],
                        "advection_tau": advection[i],
                        "row_bound_tau": bound[i],
                    }
                    for i in nearest
                ],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record, _, arrays, manifest = read_snapshot(args.snapshot)
    with threadpool_limits(2):
        problem, baseline = load_problem(
            {**record["configuration"], "baseline_directory": str(args.baseline)}
        )
        report = {
            "schema": "coupled-streamline-branch-review-v1",
            "snapshot": manifest,
            "environment": environment(),
            "baseline_sha256": baseline["baseline_sha256"],
            "scope": "Distances between stabilization branches, without changing the discretization or asserting nonsmooth stationarity.",
            "slabs": branch_distances(problem, arrays["velocity"]),
        }
        write_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
