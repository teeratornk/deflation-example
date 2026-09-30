"""Independently verify and compare a fixed-control time-refinement replay."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_forward_verify import verify_fields
from .coupled_newton_replay import replay_configuration
from .coupled_optimize import load_problem
from .coupled_saved import load_saved_solution, require_matching_baseline
from .reporting import environment, file_sha256, write_report


def temperature_statistics(problem, original, refined, subdivision, lower_K, upper_K):
    """Compare every refined time with the unchanged coarse linear interpolant."""
    original = np.asarray(original).reshape(-1, problem.spatial_size)
    refined = np.asarray(refined)
    if (
        subdivision not in (2, 4)
        or refined.shape != (len(original) * subdivision, problem.spatial_size)
        or len(refined) != problem.slabs
        or not np.isfinite(original).all()
        or not np.isfinite(refined).all()
        or not np.isfinite([lower_K, upper_K]).all()
        or lower_K >= upper_K
    ):
        raise ValueError("Use complete finite nested trajectories and ordered physical bounds")
    starts = np.vstack((problem.initial, original[:-1]))
    fraction = np.arange(1, subdivision + 1) / subdivision
    interpolated = (
        starts[:, None, :] + fraction[None, :, None] * (original - starts)[:, None, :]
    ).reshape(refined.shape)
    delta = problem.temperature_scale * (refined - interpolated)
    mass = problem.assembly.mass[problem.free]
    weights = problem.physical_steps[:, None] * mass[None, :]
    maximum = float(np.abs(delta).max())
    peak = np.unravel_index(np.abs(delta).argmax(), delta.shape)
    times = np.cumsum(problem.physical_steps)
    temperature = problem.temperature_offset + problem.temperature_scale * refined
    bounds = {}
    for name, excess in (
        ("upper", np.maximum(temperature - upper_K, 0)),
        ("lower", np.maximum(lower_K - temperature, 0)),
    ):
        k, j = np.unravel_index(excess.argmax(), excess.shape)
        bounds[name] = {
            "maximum_K": float(excess[k, j]),
            "time_s": float(times[k]) if excess[k, j] > 0 else None,
            "mesh_node": int(problem.free[j]) if excess[k, j] > 0 else None,
        }
    return {
        "maximum_trajectory_difference_K": maximum,
        "maximum_endpoint_difference_K": float(
            np.max(np.abs(refined[subdivision - 1 :: subdivision] - original))
            * problem.temperature_scale
        ),
        "mass_weighted_space_time_rms_K": float(
            np.sqrt(np.sum(weights * delta**2) / weights.sum())
        ),
        "difference_peak": {
            "time_s": float(times[peak[0]]),
            "mesh_node": int(problem.free[peak[1]]),
            "coordinates_m": problem.mesh.nodes[problem.free[peak[1]]].tolist(),
        },
        "temperature_bound_excess": bounds,
        "temperature_threshold_K": 0.05,
        "temperature_sensitivity_met": maximum <= 0.05,
        "scope": "Discrete differences at all refined time levels; no spatial error estimate or bound-feasibility certificate.",
    }


def assess(optimization, forward, position):
    original, cfg, source, digest = load_saved_solution(optimization, "jacobi", position)
    record = json.loads((forward / "record.json").read_text())
    subdivision = record["subdivision"]
    public_cfg = {
        k: v
        for k, v in cfg.items()
        if k not in {"baseline_directory", "reference_baseline_directory", "output"}
    }
    if (
        record["status"] == "running"
        or record["optimization_field_sha256"] != digest
        or record["configuration"] != public_cfg
        or record["forward_formulation"]["consistent_stabilization"]
        != cfg["consistent_stabilization"]
        or record["forward_formulation"]["streamline_rule"] != cfg["streamline_rule"]
        or record["forward_solver"]["time_scheme"] != "backward_euler"
        or record["forward_solver"]["tolerance"] != 1e-12
        or subdivision not in (2, 4)
    ):
        raise ValueError("The terminated replay must preserve the selected optimization model")
    problem, baseline = load_problem(
        replay_configuration(cfg, cfg["baseline_directory"], subdivision)
    )
    require_matching_baseline(original, baseline)
    require_matching_baseline(record, baseline)
    controls = np.repeat(source["control"].reshape(cfg["slabs"], -1), subdivision, axis=0)
    with np.load(forward / "states.npz", allow_pickle=False) as data:
        fields = {k: data[k].copy() for k in ("state", "velocity", "pressure", "times_s")}
    checks = verify_fields(
        problem,
        controls,
        fields,
        record["steps"],
        time_scheme="backward_euler",
        subdivision=subdivision,
    )
    complete = checks["complete_trajectory_verified"] and record["status"] == "converged"
    metrics = (
        temperature_statistics(
            problem,
            source["state"],
            fields["state"],
            subdivision,
            cfg["lower_K"],
            cfg["upper_K"],
        )
        if complete
        else None
    )
    return {
        "schema": "coupled-discretization-replay-check-v1",
        "optimization_record_sha256": file_sha256(optimization / "record.json"),
        "optimization_field_sha256": digest,
        "forward_record_sha256": file_sha256(forward / "record.json"),
        "forward_fields_sha256": file_sha256(forward / "states.npz"),
        "baseline_sha256": baseline["baseline_sha256"],
        "position": position,
        "target": cfg["query"],
        "subdivision": subdivision,
        "forward_status": record["status"],
        "complete_trajectory_verified": bool(complete),
        "equations": checks,
        "temperature": metrics,
        "passed": bool(complete and metrics["temperature_sensitivity_met"]),
        "environment": environment(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--position", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous verification")
    with threadpool_limits(8):
        result = assess(args.optimization, args.forward, args.position)
    write_report(args.output, result)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
