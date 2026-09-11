"""Forward replay of a fixed optimized source on refined physical time steps.

Controls are constant on each original time interval and are copied unchanged
to its subdivisions. Temperature is never clipped and the source is never
recovered from the new response. This assesses resolution, not new optimality.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_forward import CoupledForward
from .coupled_optimize import load_problem
from .meshes import assemble_thermal
from .mesh_showcases import desired_temperature
from .reporting import environment, write_fields, write_report
from .validation import integer


def replay_controls(
    problem,
    controls,
    subdivision=1,
    *,
    tolerance=1e-8,
    coupling_cap=100,
    relaxation=0.5,
    callback=None,
):
    subdivision = integer(subdivision, "Time subdivision", 1)
    if not len(problem.physical_steps):
        raise ValueError("Temporal resolution requires a complete transient problem")
    controls = np.asarray(controls, dtype=float).reshape(problem.slabs, problem.spatial_size).copy()
    if not np.isfinite(controls).all():
        raise ValueError("The saved source must be finite")
    controls.flags.writeable = False
    fluid_capacity = problem.capacity[problem.flow.fluid_cells]
    if np.ptp(fluid_capacity) > 0:
        raise ValueError("The conservation check requires the declared constant fluid capacity")

    def thermal_builder(velocity):
        return assemble_thermal(
            problem.mesh,
            problem.conductivity,
            problem.capacity,
            problem.velocity_scale * velocity,
            problem.source,
            streamline=True,
        )

    model = CoupledForward(
        problem.flow,
        thermal_builder,
        problem.boundary_indices,
        problem.boundary_values,
        expansion=problem.expansion,
        temperature_offset=problem.temperature_offset,
        temperature_scale=problem.temperature_scale,
        time_scale=problem.time_scale,
        buoyancy_reference=problem.buoyancy_reference,
        thermal_boundary=problem.thermal_boundary,
        acceleration=problem.acceleration,
        pressure_gauge=problem.pressure_gauge,
        transport_factor=problem.velocity_scale * fluid_capacity[0],
        flow_method="newton",
    )
    state = problem.full_temperature(problem.initial)
    flow = problem.initial_flow
    states, rows, elapsed = [], [], 0.0
    status = "converged"
    start = time.perf_counter()
    for slab, source_values in enumerate(controls):
        source = np.zeros(len(problem.mesh.nodes))
        source[problem.free] = source_values
        dt = problem.physical_steps[slab] / subdivision
        for substep in range(subdivision):
            result = model.solve(
                source,
                state,
                flow,
                previous_state=state,
                previous_velocity=flow.velocity,
                time_step=dt,
                tolerance=tolerance,
                max_iterations=coupling_cap,
                flow_cap=problem.flow_cap,
                relaxation=relaxation,
                mass_tolerance=1e-6,
                energy_tolerance=1e-6,
            )
            elapsed += dt
            rows.append(
                {
                    "original_slab": slab,
                    "substep": substep,
                    "time_s": elapsed,
                    "status": result.status,
                    "history": result.history,
                    "seconds": result.seconds,
                }
            )
            states.append(result.state[problem.free].copy())
            if callback is not None:
                callback(rows[-1])
            if result.status != "converged":
                status = result.status
                break
            state, flow = result.state, result.flow
        if status != "converged":
            break
    return {
        "status": status,
        "steps": rows,
        "states": np.stack(states),
        "seconds": time.perf_counter() - start,
        "control_sha256": hashlib.sha256(controls.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument(
        "--method", default="reference", choices=("jacobi", "reference", "recycling")
    )
    parser.add_argument("--subdivision", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        record = json.loads((args.optimization / "record.json").read_text())
        results = [r for r in record.get("results", []) if r["method"] == args.method]
        if len(results) != 1 or results[0]["status"] != "converged":
            raise ValueError("Resolution checks require a verified optimized source")
        cfg = {**record["configuration"], "baseline_directory": str(args.baseline)}
        problem, _ = load_problem(cfg)
        field = args.optimization / (args.method + "-fields.npz")
        with np.load(field, allow_pickle=False) as data:
            source = data["control"].copy()
            optimized = data["state"].reshape(problem.slabs, problem.spatial_size).copy()
        args.output.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema": "coupled-fixed-control-temporal-resolution-v1",
            "environment": environment(),
            "optimization_field_sha256": hashlib.sha256(field.read_bytes()).hexdigest(),
            "subdivision": args.subdivision,
            "configuration": record["configuration"],
            "control_representation": "Piecewise constant on each original physical time interval; source at prescribed-temperature nodes is zero.",
            "scope": "Forward temporal-resolution assessment of the same source; temperature remains unconstrained during replay.",
        }
        write_report(args.output / "record.json", {**metadata, "status": "running"})
        result = replay_controls(
            problem,
            source,
            args.subdivision,
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
            endpoint = states[args.subdivision - 1 :: args.subdivision]
            report["maximum_endpoint_difference_K"] = float(
                np.max(np.abs(endpoint - optimized)) * problem.temperature_scale
            )
            starts = np.vstack((problem.initial[None, :], optimized[:-1]))
            fraction = np.arange(1, args.subdivision + 1) / args.subdivision
            interpolated = (
                starts[:, None, :] + fraction[None, :, None] * (optimized - starts)[:, None, :]
            ).reshape(states.shape)
            report["maximum_trajectory_difference_K"] = float(
                np.max(np.abs(states - interpolated)) * problem.temperature_scale
            )
            report["trajectory_comparison"] = (
                "Refined states versus piecewise-linear interpolation of original states, including the initial condition."
            )
            temperature = problem.temperature_offset + problem.temperature_scale * states
            violation = temperature - cfg["upper_K"]
            step, node = np.unravel_index(np.argmax(violation), violation.shape)
            report["maximum_upper_violation_K"] = float(max(0, violation[step, node]))
            report["maximum_lower_violation_K"] = float(max(0, cfg["lower_K"] - temperature.min()))
            report["maximum_upper_violation_location_rz_m"] = problem.mesh.nodes[
                problem.free[node]
            ].tolist()
            report["maximum_upper_violation_time_s"] = result["steps"][step]["time_s"]
            fine_steps = np.repeat(problem.physical_steps / args.subdivision, args.subdivision)
            fine_view = SimpleNamespace(
                assembly=problem.assembly, free=problem.free, steps=fine_steps / problem.time_scale
            )
            desired_fine = desired_temperature(
                fine_view, cfg["query"], cfg["target_count"]
            ).reshape(states.shape)
            desired_original = desired_temperature(
                problem, cfg["query"], cfg["target_count"]
            ).reshape(optimized.shape)
            mass = problem.assembly.mass[problem.free]
            coarse_tracking = float(
                np.sum(
                    problem.physical_steps[:, None]
                    * mass
                    * (problem.temperature_scale * (optimized - desired_original)) ** 2
                )
            )
            fine_tracking = float(
                np.sum(
                    fine_steps[:, None]
                    * mass
                    * (problem.temperature_scale * (states - desired_fine)) ** 2
                )
            )
            report["tracking_integral_original_K2_m3_s"] = coarse_tracking
            report["tracking_integral_refined_K2_m3_s"] = fine_tracking
            report["tracking_relative_change"] = abs(fine_tracking - coarse_tracking) / max(
                coarse_tracking, np.finfo(float).tiny
            )
            report["resolution_thresholds_met"] = (
                None
                if args.subdivision == 1
                else (
                    report["maximum_trajectory_difference_K"] <= 0.05
                    and report["tracking_relative_change"] <= 0.01
                )
            )
        write_report(args.output / "record.json", report)


if __name__ == "__main__":
    main()
