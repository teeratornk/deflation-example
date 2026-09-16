"""Compare two saved trajectories of one source on nested spatial meshes.

The spatial question is what changes when the mesh is refined and nothing else is.
A trajectory that refines the mesh and the time grid together cannot answer it, and
the replay tool cannot help either, because it solves the fine trajectory itself
with the monolithic procedure that stalls on the finer mesh.

This module takes two trajectories that already exist, checks that they came from
the same saved source on the same physical time grid and differ only in the mesh,
lifts the coarse one onto the fine mesh by nested interpolation, and reports the
version two statistics of the difference.

It records each trajectory's forward procedure and states plainly whether they
match. Mixing procedures inside one refinement comparison is a real confound; the
rule is to report it, never to hide it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_optimize import load_problem
from .coupled_resolution_v2 import statistics_from_arrays
from .coupled_saved import file_digest, load_saved_solution, require_matching_baseline
from .coupled_spatial_resolution import transfer_source
from .reporting import environment, write_report
from .validation import integer

SCHEMA = "coupled-spatial-pair-assessment-v1"
# The temperature criteria of the original design, reported unchanged.
POINTWISE_K = 0.05
RMS_K = 0.05


def procedure_of(record):
    """How a trajectory was solved forward, however its own study named it."""
    solver = record.get("forward_solver")
    if isinstance(solver, dict):
        return {"source": "forward_solver", **solver}
    if solver is not None:
        return {"source": "forward_solver", "procedure": str(solver)}
    policy = record.get("policy")
    if policy is not None:
        return {
            "source": "fixed_point_policy",
            "procedure": str(policy),
            "settings": record.get("settings"),
            "family": record.get("family"),
        }
    return {"source": "unstated", "procedure": None}


def load_trajectory(directory):
    directory = Path(directory)
    record = json.loads((directory / "record.json").read_text())
    if record.get("status") != "converged":
        raise ValueError(f"{directory} is not a complete converged trajectory")
    steps = record.get("steps") or []
    if steps and not all(row.get("status") == "converged" for row in steps):
        raise ValueError(f"{directory} contains an unconverged step")
    with np.load(directory / "states.npz", allow_pickle=False) as data:
        state, times = data["state"].copy(), data["times_s"].copy()
    return {
        "directory": str(directory),
        "record": record,
        "record_sha256": file_digest(directory / "record.json"),
        "field_sha256": file_digest(directory / "states.npz"),
        "state": state,
        "times_s": times,
        "slabs": record.get("slabs", len(state)),
        "spatial_state_dofs": record.get("spatial_state_dofs"),
        "subdivision": record.get("subdivision", 1),
        "time_scheme": record.get("time_scheme", "backward_euler"),
        "procedure": procedure_of(record),
    }


def checked_pair(coarse, fine, source_digest):
    """Refuse anything that is not a pure spatial refinement of one saved source."""
    for side, name in ((coarse, "coarse"), (fine, "fine")):
        if side["record"].get("optimization_field_sha256") != source_digest:
            raise ValueError(f"The {name} trajectory replays a different saved source")
    if coarse["slabs"] != fine["slabs"] or coarse["subdivision"] != fine["subdivision"]:
        raise ValueError(
            "A spatial comparison needs one time grid; these trajectories refine time as well"
        )
    if coarse["time_scheme"] != fine["time_scheme"]:
        raise ValueError("A spatial comparison needs one time integration scheme")
    if not np.allclose(coarse["times_s"], fine["times_s"], rtol=1e-12, atol=1e-12):
        raise ValueError("The two trajectories are not on the same physical time levels")
    if coarse["spatial_state_dofs"] == fine["spatial_state_dofs"]:
        raise ValueError("The two trajectories are on the same mesh; there is nothing to compare")
    if (fine["spatial_state_dofs"] or 0) < (coarse["spatial_state_dofs"] or 0):
        raise ValueError("The fine trajectory must have more spatial degrees of freedom")


def assess(coarse_dir, fine_dir, baseline, fine_baseline, optimization, method, position, upper_K):
    coarse = load_trajectory(coarse_dir)
    fine = load_trajectory(fine_dir)
    record, cfg, fields, digest = load_saved_solution(optimization, method, position)
    if not cfg["transient"]:
        raise ValueError("A spatial trajectory comparison requires a transient optimization")
    checked_pair(coarse, fine, digest)
    slabs = integer(cfg["slabs"], "Original time slabs", 1)
    if coarse["slabs"] != slabs:
        raise ValueError("The trajectories must use the optimization's own time grid")
    coarse_problem, coarse_base = load_problem({**cfg, "baseline_directory": str(baseline)})
    fine_problem, fine_base = load_problem({**cfg, "baseline_directory": str(fine_baseline)})
    require_matching_baseline(record, coarse_base)
    for key in ("properties", "inlet_velocity_m_s", "grad_div_coefficient_m2_s"):
        if coarse_base.get(key) != fine_base.get(key):
            raise ValueError("Baseline physical inputs must match across refinement")
    controls = fields["control"].reshape(slabs, coarse_problem.spatial_size)
    _, P = transfer_source(coarse_problem, fine_problem, controls)
    lifted = np.asarray(P @ coarse["state"].T).T
    mass = fine_problem.assembly.mass[fine_problem.free]
    statistics = statistics_from_arrays(
        lifted,
        coarse["times_s"],
        fine["state"],
        fine["times_s"],
        fine_problem.mesh,
        mass,
        fine_problem.temperature_scale,
        0.0,
    )
    same = coarse["procedure"].get("procedure") == fine["procedure"].get("procedure")
    pointwise = statistics["pointwise_maximum_K"]
    rms = statistics["mass_weighted_space_time_rms_K"]
    return {
        "schema": SCHEMA,
        "environment": environment(),
        "configuration": {
            k: v
            for k, v in cfg.items()
            if k not in {"output", "baseline_directory", "reference_baseline_directory"}
        },
        "optimization_field_sha256": digest,
        "baseline_sha256": coarse_base["baseline_sha256"],
        "fine_baseline_sha256": fine_base["baseline_sha256"],
        "coarse": {
            k: coarse[k]
            for k in (
                "directory",
                "record_sha256",
                "field_sha256",
                "spatial_state_dofs",
                "procedure",
            )
        },
        "fine": {
            k: fine[k]
            for k in (
                "directory",
                "record_sha256",
                "field_sha256",
                "spatial_state_dofs",
                "procedure",
            )
        },
        "slabs": coarse["slabs"],
        "time_scheme": coarse["time_scheme"],
        "refinement": "one nested uniform refinement; the coarse trajectory is lifted by nested P1 interpolation",
        "forward_procedures_match": same,
        "procedure_note": (
            "Both trajectories were solved forward by the same procedure."
            if same
            else "The two trajectories used different forward procedures. The difference below "
            "therefore contains the procedure change as well as the mesh change, and is reported "
            "as such rather than as a mesh effect."
        ),
        "statistics": statistics,
        "criteria": {
            "pointwise_maximum_K": POINTWISE_K,
            "mass_weighted_space_time_rms_K": RMS_K,
            "pointwise_met": bool(pointwise <= POINTWISE_K),
            "rms_met": bool(rms <= RMS_K),
        },
        "scope": "Fixed-source spatial comparison at matching physical time levels. A small difference "
        "between two meshes is not a continuum solution certificate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fine-baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--method", default="reference")
    parser.add_argument("--target-position", type=int)
    parser.add_argument("--upper-K", type=float)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        report = assess(
            args.coarse,
            args.fine,
            args.baseline,
            args.fine_baseline,
            args.optimization,
            args.method,
            args.target_position,
            args.upper_K,
        )
    args.output.mkdir(parents=True)
    write_report(args.output / "record.json", report)
    print(
        json.dumps(
            {
                "pointwise_maximum_K": report["statistics"]["pointwise_maximum_K"],
                "mass_weighted_space_time_rms_K": report["statistics"][
                    "mass_weighted_space_time_rms_K"
                ],
                "criteria": report["criteria"],
                "forward_procedures_match": report["forward_procedures_match"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
