"""Replay a fixed coupled source on the immediate nested spatial refinement."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_optimize import load_problem
from .coupled_newton_replay import newton_trajectory
from .coupled_reference import nested_prolongation
from .coupled_resolution import (
    add_forward_options,
    forward_options,
    forward_protocol,
    replay_controls,
)
from .coupled_saved import file_digest, load_saved_solution, require_matching_baseline
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


def coarse_replay_states(directory, problem, baseline_digest, source_digest, configuration):
    """Require a complete same-source coarse trajectory on the comparison time grid."""
    directory = Path(directory)
    record = json.loads((directory / "record.json").read_text())
    if record["forward_solver"].get("time_scheme", "backward_euler") != "backward_euler":
        raise ValueError("This spatial replay requires backward-Euler trajectories on both meshes")
    clean_cfg = {
        k: v
        for k, v in configuration.items()
        if k not in {"output", "baseline_directory", "reference_baseline_directory"}
    }
    if (
        record["optimization_field_sha256"] != source_digest
        or record["baseline_sha256"] != baseline_digest
        or record["configuration"] != clean_cfg
    ):
        raise ValueError("The coarse replay must use the same saved source and physical inputs")
    if record["status"] != "converged" or not all(
        r["status"] == "converged" for r in record["steps"]
    ):
        raise ValueError("The coarse replay must be a complete converged trajectory")
    with np.load(directory / "states.npz", allow_pickle=False) as data:
        states, times = data["state"].copy(), data["times_s"].copy()
    expected_times = np.cumsum(problem.physical_steps)
    if (
        states.shape != (problem.slabs, problem.spatial_size)
        or not np.isfinite(states).all()
        or times.shape != expected_times.shape
        or not np.allclose(times, expected_times, rtol=1e-12, atol=1e-12)
        or len(record["steps"]) != problem.slabs
        or not np.allclose([r["time_s"] for r in record["steps"]], times, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("The coarse replay must match every comparison time level")
    return states, {
        "record_sha256": file_digest(directory / "record.json"),
        "field_sha256": file_digest(directory / "states.npz"),
        "forward_solver": record["forward_solver"],
    }


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
    parser.add_argument(
        "--procedure", choices=("segregated", "monolithic_newton"), default="segregated"
    )
    parser.add_argument("--newton-cap", type=int, default=30)
    parser.add_argument("--subdivision", type=int, default=1)
    parser.add_argument("--coarse-replay", type=Path)
    add_forward_options(parser)
    args = parser.parse_args()
    subdivision = integer(args.subdivision, "Time subdivision", 1)
    if subdivision > 1 and args.coarse_replay is None:
        parser.error("Temporal subdivision requires a matching --coarse-replay")
    options = forward_options(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        record, cfg, fields, digest = load_saved_solution(
            args.optimization, args.method, args.target_position
        )
        if not cfg["transient"]:
            raise ValueError("Spatial trajectory replay requires a transient optimization")
        original_slabs = integer(cfg["slabs"], "Original time slabs", 1)
        comparison_cfg = {**cfg, "slabs": original_slabs * subdivision}
        coarse, baseline = load_problem(
            {**comparison_cfg, "baseline_directory": str(args.baseline)}
        )
        fine, fine_baseline = load_problem(
            {**comparison_cfg, "baseline_directory": str(args.fine_baseline)}
        )
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
        controls = np.repeat(
            fields["control"].reshape(original_slabs, coarse.spatial_size), subdivision, axis=0
        )
        source, P = transfer_source(coarse, fine, controls)
        comparison = None
        if args.coarse_replay is None:
            original = fields["state"].reshape(coarse.slabs, coarse.spatial_size)
        else:
            original, comparison = coarse_replay_states(
                args.coarse_replay, coarse, baseline["baseline_sha256"], digest, cfg
            )
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
            "subdivision": subdivision,
            "coarse_replay": comparison,
            "forward_solver": forward_protocol(fine, options),
            "control_transfer": "Nested P1 interpolation with zero source at prescribed-temperature nodes; each saved source is copied unchanged into its temporal subintervals.",
            "scope": "Fixed-source spatial comparison at matching physical time steps on both meshes; no reoptimization or temperature clipping.",
        }
        if args.procedure == "monolithic_newton":
            metadata["forward_solver"] = {
                "procedure": args.procedure,
                "tolerance": options["tolerance"],
                "newton_cap": integer(args.newton_cap, "Newton iteration cap", 0),
                "linear_internal_target": 1e-10,
                "linear_acceptance_target": 1e-8,
                "linear_correction_cap": 2,
                "linear_residual": "Row-equilibrated original Jacobian equation",
                "mass_tolerance": 1e-6,
                "energy_tolerance": 1e-6,
            }
        write_report(args.output / "record.json", {**metadata, "status": "running"})

        def callback(row):
            write_report(args.output / "progress.json", row)

        if args.procedure == "monolithic_newton":
            result = newton_trajectory(
                fine, source, tolerance=options["tolerance"], cap=args.newton_cap, callback=callback
            )
        else:
            result = replay_controls(fine, source, subdivision=1, **options, callback=callback)
        states = result.pop("states")
        flow_fields = {}
        if "velocities" in result:
            flow_fields = {
                "velocity": result.pop("velocities"),
                "pressure": result.pop("pressures"),
            }
        write_fields(
            args.output / "states.npz",
            state=states,
            times_s=np.array([r["time_s"] for r in result["steps"]]),
            **flow_fields,
        )
        report = {**metadata, **result}
        if result["status"] == "converged":
            report["maximum_temperature_difference_K"] = float(
                np.max(np.abs(states - interpolated)) * fine.temperature_scale
            )
            coarse_target = (
                fields["desired"].reshape(original.shape)
                if args.coarse_replay is None
                else desired_temperature(
                    coarse, cfg["query"], cfg["target_count"], cfg.get("target_startup_s", 0.0)
                ).reshape(original.shape)
            )
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
