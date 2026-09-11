"""Replay a fixed coupled source on the immediate nested spatial refinement."""

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_optimize import load_problem
from .coupled_reference import nested_prolongation
from .coupled_resolution import (
    add_forward_options,
    forward_options,
    forward_protocol,
    replay_controls,
)
from .coupled_saved import load_saved_solution, require_matching_baseline
from .coupled_targets import desired_temperature
from .reporting import environment, write_fields, write_report
from .validation import integer


def transfer_source(coarse, fine, controls):
    """Preserve the original P1 source function and physical time intervals."""
    if not np.array_equal(coarse.physical_steps, fine.physical_steps):
        raise ValueError("Spatial refinement requires unchanged physical time intervals")
    for name in (
        "temperature_scale",
        "temperature_offset",
        "time_scale",
        "velocity_scale",
        "expansion",
        "buoyancy_reference",
    ):
        if getattr(coarse, name) != getattr(fine, name):
            raise ValueError("Spatial refinement requires identical physical scales and properties")
    P = nested_prolongation(coarse.mesh, fine.mesh)
    source = np.asarray(controls, dtype=float).reshape(coarse.slabs, coarse.spatial_size)
    if not np.isfinite(source).all():
        raise ValueError("The saved source must be finite")
    return np.asarray(P @ source.T).T, P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fine-baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("jacobi", "reference", "recycling"), default="reference"
    )
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    add_forward_options(parser)
    args = parser.parse_args()
    options = forward_options(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        record, cfg, fields, digest = load_saved_solution(
            args.optimization, args.method, args.target_position
        )
        coarse, baseline = load_problem({**cfg, "baseline_directory": str(args.baseline)})
        fine, fine_baseline = load_problem({**cfg, "baseline_directory": str(args.fine_baseline)})
        require_matching_baseline(record, baseline)
        for key in (
            "properties",
            "inlet_velocity_m_s",
            "grad_div_coefficient_m2_s",
            "input_sha256",
        ):
            if baseline.get(key) != fine_baseline.get(key):
                raise ValueError("Baseline physical inputs must match across refinement")
        if (
            baseline["configuration"]["convection_form"]
            != fine_baseline["configuration"]["convection_form"]
        ):
            raise ValueError("The momentum discretization must match across refinement")
        source, P = transfer_source(coarse, fine, fields["control"])
        original = fields["state"].reshape(coarse.slabs, coarse.spatial_size)
        interpolated = np.asarray(P @ original.T).T
        args.output.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema": "coupled-fixed-control-spatial-resolution-v1",
            "environment": environment(),
            "configuration": {
                k: v
                for k, v in cfg.items()
                if k not in {"output", "baseline_directory", "reference_baseline_directory"}
            },
            "target_position": args.target_position,
            "optimization_field_sha256": digest,
            "baseline_sha256": baseline["baseline_sha256"],
            "fine_baseline_sha256": fine_baseline["baseline_sha256"],
            "coarse_state_dofs": coarse.size,
            "fine_state_dofs": fine.size,
            "forward_solver": forward_protocol(fine, options),
            "control_transfer": "Nested P1 interpolation with zero source at prescribed-temperature nodes; identical piecewise-constant temporal source intervals.",
            "scope": "Fixed-source spatial resolution with unchanged physical time steps; no reoptimization or temperature clipping.",
        }
        write_report(args.output / "record.json", {**metadata, "status": "running"})
        result = replay_controls(
            fine,
            source,
            subdivision=1,
            **options,
            callback=lambda row: write_report(args.output / "progress.json", row),
        )
        states = result.pop("states")
        write_fields(
            args.output / "states.npz",
            state=states,
            times_s=np.array([r["time_s"] for r in result["steps"]]),
        )
        report = {**metadata, **result}
        if result["status"] == "converged":
            report["maximum_temperature_difference_K"] = float(
                np.max(np.abs(states - interpolated)) * fine.temperature_scale
            )
            coarse_target = fields["desired"].reshape(original.shape)
            fine_target = desired_temperature(
                fine, cfg["query"], cfg["target_count"], cfg.get("target_startup_s", 0.0)
            ).reshape(states.shape)
            coarse_tracking = float(
                np.sum(
                    coarse.physical_steps[:, None]
                    * coarse.assembly.mass[coarse.free]
                    * (coarse.temperature_scale * (original - coarse_target)) ** 2
                )
            )
            fine_tracking = float(
                np.sum(
                    fine.physical_steps[:, None]
                    * fine.assembly.mass[fine.free]
                    * (fine.temperature_scale * (states - fine_target)) ** 2
                )
            )
            report["tracking_integral_original_K2_m3_s"] = coarse_tracking
            report["tracking_integral_refined_K2_m3_s"] = fine_tracking
            report["tracking_relative_change"] = abs(fine_tracking - coarse_tracking) / max(
                coarse_tracking, np.finfo(float).tiny
            )
            temperature = fine.temperature_offset + fine.temperature_scale * states
            excess = temperature - cfg["upper_K"]
            step, node = np.unravel_index(np.argmax(excess), excess.shape)
            report["maximum_upper_violation_K"] = float(max(0, excess[step, node]))
            report["maximum_lower_violation_K"] = float(max(0, cfg["lower_K"] - temperature.min()))
            report["maximum_upper_violation_location_rz_m"] = fine.mesh.nodes[
                fine.free[node]
            ].tolist()
            report["maximum_upper_violation_time_s"] = result["steps"][step]["time_s"]
            report["resolution_thresholds_met"] = (
                report["maximum_temperature_difference_K"] <= 0.05
                and report["tracking_relative_change"] <= 0.01
            )
        write_report(args.output / "record.json", report)


if __name__ == "__main__":
    main()
