"""Independent element-flux diagnostics for axisymmetric P2 velocity.

Continuous P1 pressure tests impose weak continuity. They do not contain each
element indicator, so neither a small weak residual nor balanced external flux
certifies elementwise conservation. These diagnostics measure that distinction
without modifying the velocity or the equations.
"""

import numpy as np


def cell_fluxes(flow, velocity):
    """Outward volume flux in m^3/s on each fluid triangle's three faces.

    Three-point edge quadrature integrates the cubic product r*v_P2 exactly.
    Normals are oriented from each cell centroid, independently of stored winding.
    """
    flow.sampled_velocity(velocity)  # Validate shape and finite values.
    cells = flow.mesh.cells[flow.fluid_cells]
    center = flow.mesh.nodes[cells].mean(axis=1)
    t, weights = np.polynomial.legendre.leggauss(3)
    t, weights = (t + 1) / 2, weights / 2
    shape = np.stack(((1 - t) * (1 - 2 * t), 4 * t * (1 - t), t * (2 * t - 1)), axis=1)
    result = []
    for a, b, midpoint in ((0, 1, 3), (0, 2, 4), (1, 2, 5)):
        x = flow.mesh.nodes[cells[:, [a, b]]]
        tangent = x[:, 1] - x[:, 0]
        length = np.linalg.norm(tangent, axis=1)
        normal = np.column_stack((tangent[:, 1], -tangent[:, 0])) / length[:, None]
        inward = np.einsum("ed,ed->e", center - x.mean(axis=1), normal) > 0
        normal[inward] *= -1
        values = velocity[flow.p2[:, [a, midpoint, b]]]
        sample = np.einsum("qi,eid->eqd", shape, values)
        radius = x[:, 0, 0, None] * (1 - t) + x[:, 1, 0, None] * t
        flux = np.einsum("eqd,ed->eq", sample, normal)
        result.append(np.sum(flux * radius * weights, axis=1) * length * (2 * np.pi))
    return np.column_stack(result)


def mass_diagnostics(flow, velocity):
    """Global, elementwise and pointwise diagnostics with distinct units/scales."""
    faces = cell_fluxes(flow, velocity)
    net = faces.sum(axis=1)
    divergence = np.einsum("eia,eqia->eq", velocity[flow.p2], flow.div_basis)
    volume_integral = np.sum(flow.measure * divergence, axis=1)
    exterior = flow.boundary_flux(velocity)
    total_face_flux = max(float(np.abs(faces).sum()), 1e-30)
    external_scale = max(float(np.abs(exterior).sum() / 2), 1e-30)
    worst = int(np.argmax(np.abs(net)))
    return {
        "continuity_space": "continuous P1 pressure tests; not elementwise constraints",
        "global_relative_imbalance": float(abs(exterior.sum()) / external_scale),
        "maximum_cell_net_volume_flux_m3_s": float(np.max(np.abs(net))),
        "sum_absolute_cell_net_volume_flux_m3_s": float(np.abs(net).sum()),
        "cell_imbalance_over_total_face_flux": float(np.abs(net).sum() / total_face_flux),
        "divergence_rms_s_inverse": float(
            np.sqrt(np.sum(flow.measure * divergence**2) / flow.measure.sum())
        ),
        "maximum_cell_divergence_theorem_defect_m3_s": float(np.max(np.abs(net - volume_integral))),
        "interior_flux_cancellation_defect_m3_s": float(abs(net.sum() - exterior.sum())),
        "worst_cell_global_index": int(flow.fluid_cells[worst]),
        "worst_cell_centroid_r_z_m": flow.mesh.nodes[flow.mesh.cells[flow.fluid_cells[worst]]]
        .mean(axis=0)
        .tolist(),
    }


def main():
    """Recompute mass diagnostics at declared verified replay times and baseline."""
    import argparse
    from pathlib import Path
    from threadpoolctl import threadpool_limits
    from .coupled_late_step import checkpoint_fields
    from .coupled_optimize import load_problem
    from .coupled_prefix_diagnostics import read_forward
    from .reporting import environment, file_sha256, write_report
    from .validation import integer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--replay", type=Path)
    group.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--times", type=float, nargs="+", default=[])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.times) == args.baseline_only:
        raise ValueError("Declare replay times, or choose a baseline-only diagnostic without times")
    if not np.isfinite(args.times).all() or len(set(args.times)) != len(args.times):
        raise ValueError("Declare distinct finite physical times")
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        if args.baseline_only:
            from hydra import compose, initialize_config_module
            from omegaconf import OmegaConf

            with initialize_config_module(
                config_module="deflation_example.conf", version_base=None
            ):
                cfg = OmegaConf.to_container(compose(config_name="coupled_optimize"), resolve=True)
            cfg["baseline_directory"] = str(args.baseline)
            problem, baseline = load_problem(cfg)
            equations = problem.flow.verify(
                problem.initial_flow,
                np.zeros_like(problem.flow.quadrature_points),
                problem.boundary_indices,
                problem.boundary_values,
                pressure_gauge=problem.pressure_gauge,
            )
            write_report(
                args.output / "record.json",
                {
                    "schema": "coupled-baseline-mass-diagnostic-v1",
                    "status": "complete",
                    "environment": environment(),
                    "baseline_sha256": baseline["baseline_sha256"],
                    "baseline_configuration": baseline["configuration"],
                    "baseline_record_sha256": file_sha256(args.baseline / "record.json"),
                    "thermal_nodes": len(problem.mesh.nodes),
                    "fluid_cells": len(problem.flow.fluid_cells),
                    "original_equations": equations,
                    "mass": mass_diagnostics(problem.flow, problem.initial_flow.velocity),
                    "scope": "Saved isothermal baseline; independent integration of external and element fluxes. No thermal trajectory, optimization, or local-conservation guarantee.",
                },
            )
            return
        record, fields, count = read_forward(args.replay)
        problem, baseline = load_problem(
            {
                **record["configuration"],
                "baseline_directory": str(args.baseline),
                "slabs": record["forward_slabs"],
                "consistent_stabilization": record["forward_formulation"][
                    "consistent_stabilization"
                ],
            }
        )
        if baseline["baseline_sha256"] != record["baseline_sha256"]:
            raise ValueError("Baseline differs from the saved replay")
        rows = [
            {
                "time_s": 0.0,
                "source": "baseline",
                **mass_diagnostics(problem.flow, problem.initial_flow.velocity),
            }
        ]
        for physical_time in args.times:
            found = np.flatnonzero(np.isclose(fields["times_s"], physical_time, rtol=0, atol=1e-10))
            if len(found) != 1 or found[0] >= count:
                raise ValueError("Requested time has no independently verified replay state")
            step = int(found[0])
            checkpoint_fields(args.replay, record, fields, step, problem)
            rows.append(
                {
                    "time_s": physical_time,
                    "source": "saved_replay",
                    **mass_diagnostics(problem.flow, fields["velocity"][step]),
                }
            )
        write_report(
            args.output / "record.json",
            {
                "schema": "coupled-mass-diagnostic-v1",
                "status": "complete",
                "environment": environment(),
                "replay_source": record["environment"]["git_head"],
                "replay_record_sha256": file_sha256(args.replay / "record.json"),
                "replay_fields_sha256": file_sha256(args.replay / "states.npz"),
                "baseline_sha256": baseline["baseline_sha256"],
                "rows": rows,
                "scope": "Independent face/volume integration at saved fields; no velocity reconstruction, new trajectory, or elementwise-conservation claim.",
            },
        )


if __name__ == "__main__":
    main()
