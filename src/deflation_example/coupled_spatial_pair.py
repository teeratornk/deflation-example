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
import hashlib
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
PROTOCOL_SCHEMA = "coupled-resolution-protocol-v3"


def load_protocol(path):
    """The declared horizon split and its digest.

    The split is read from the declared file rather than restated here, and the
    file's hash goes into the report, so an assessment can never quietly drift from
    the rule that was declared before the results were read.
    """
    raw = Path(path).read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"Expected a {PROTOCOL_SCHEMA} declaration")
    interval = protocol["split"]["pointwise_interval_s"]
    if len(interval) != 2 or not 0 <= interval[0] < interval[1]:
        raise ValueError("The declared pointwise interval must be a positive span from zero")
    return protocol, hashlib.sha256(raw).hexdigest()


def split_assessment(protocol, digest, lifted, coarse_times, fine_state, fine_times, mesh, mass,
                     scale, whole):
    """Apply the declared split: pointwise inside the interval, functional outside.

    Outside the interval the coupled linearisation amplifies a perturbation by about
    seven orders over this horizon, so a pointwise criterion there would demand early
    agreement finer than roundoff. The protocol therefore asks for pointwise
    agreement only where that amplification is bounded by ten, and for the
    mass-weighted norm everywhere. This function judges those two. The tracking
    integral and the bound excess are the protocol's other full-horizon criteria and
    are not formed here; they are named as such so their absence is visible.
    """
    boundary = float(protocol["split"]["pointwise_interval_s"][1])
    tolerance = 1e-9
    inside = fine_times <= boundary + tolerance
    reached = bool(fine_times[-1] >= boundary - tolerance) if len(fine_times) else False
    criteria = protocol["criteria"]
    interval = None
    if inside.any():
        interval = statistics_from_arrays(
            lifted[inside], coarse_times[inside], fine_state[inside], fine_times[inside],
            mesh, mass, scale, 0.0,
        )
    pointwise_limit = float(criteria["pointwise_interval"]["maximum_temperature_difference_K"])
    rms_limit = float(criteria["pointwise_interval"]["mass_weighted_rms_difference_K"])
    horizon_rms_limit = float(criteria["full_horizon"]["mass_weighted_rms_difference_K"])
    horizon_rms = statistics_from_arrays(
        lifted, coarse_times, fine_state, fine_times, mesh, mass, scale, 0.0
    )["mass_weighted_space_time_rms_K"]
    return {
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_sha256": digest,
        "pointwise_interval_s": [0.0, boundary],
        "levels_in_interval": int(inside.sum()),
        "interval_reached": reached,
        "interval_statistics": interval,
        "pointwise_met": (
            None
            if interval is None or not reached
            else bool(interval["pointwise_maximum_K"] <= pointwise_limit)
        ),
        "interval_rms_met": (
            None
            if interval is None or not reached
            else bool(interval["mass_weighted_space_time_rms_K"] <= rms_limit)
        ),
        "horizon_rms_K": horizon_rms,
        "horizon_rms_met": bool(horizon_rms <= horizon_rms_limit) if whole else None,
        "not_judged_here": [
            "tracking_relative_change",
            "maximum_upper_violation_change_K",
        ],
        "note": (
            "A verdict inside the interval needs a trajectory that reaches its end; a "
            "full-horizon verdict needs one that reaches the horizon. Where either is missing "
            "the numbers are reported and the verdict is withheld."
        ),
    }


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


def trajectory_states(directory, converged):
    """Temperatures and times, from the whole-trajectory archive or the per-step ones.

    A run writes one archive per step as it goes and the combined archive only on
    success, so a trajectory that stopped early has its fields in the per-step
    archives alone. Each step names its own file and carries its checksum, which is
    verified here, so reading them is not a weaker provenance than the combined one.
    """
    directory = Path(directory)
    combined = directory / "states.npz"
    if combined.exists():
        with np.load(combined, allow_pickle=False) as data:
            return data["state"].copy(), data["times_s"].copy()
    if not converged:
        raise ValueError(f"{directory} has neither a combined archive nor a converged step")
    states, times = [], []
    for step in converged:
        name = step.get("fields")
        if not name:
            raise ValueError(f"{directory} has a converged step with no field archive")
        path = directory / name
        if file_digest(path) != step.get("field_sha256"):
            raise ValueError(f"{path} does not match the checksum its step recorded")
        with np.load(path, allow_pickle=False) as data:
            states.append(data["state"].copy())
        times.append(float(step["time_s"]))
    return np.vstack(states), np.array(times)


def declared_levels(record, reached):
    """How long the trajectory was meant to be, which is not how far it got.

    The three records this reads name it three ways: a study record says slabs, a
    forward replay says forward_slabs, and a spatial-resolution record says neither
    and leaves it to be read off the optimization it replays, times the subdivision
    it declared. Falling back to what was reached would make a stalled run look like
    a complete short one, and two runs that stalled at different points would then
    look like two different time grids.
    """
    for key in ("slabs", "forward_slabs"):
        if record.get(key) is not None:
            return integer(record[key], "Declared time levels", 1)
    original = (record.get("configuration") or {}).get("slabs")
    if original is not None:
        subdivision = integer(record.get("subdivision", 1), "Time subdivision", 1)
        return integer(original, "Original time levels", 1) * subdivision
    return reached


def load_trajectory(directory, allow_partial=False):
    """One trajectory, with the levels it actually reached.

    No fine-mesh forward attempt of this problem has reached the horizon, so a
    comparison over the interval both meshes did reach is the only one available.
    That is a real result about the model, not a shortcut, and it is only allowed
    when the caller asks for it and it is recorded in full.
    """
    directory = Path(directory)
    record = json.loads((directory / "record.json").read_text())
    steps = record.get("steps") or []
    converged = [row for row in steps if row.get("status") == "converged"]
    complete = record.get("status") == "converged" and len(converged) == len(steps)
    if not complete and not allow_partial:
        raise ValueError(f"{directory} is not a complete converged trajectory")
    if converged != steps[: len(converged)]:
        raise ValueError(f"{directory} has a converged step after a failed one")
    state, times = trajectory_states(directory, converged)
    reached = min(len(converged), len(state)) if steps else len(state)
    if reached < 1:
        raise ValueError(f"{directory} reached no converged time level")
    state, times = state[:reached], times[:reached]
    declared = declared_levels(record, len(state))
    return {
        "complete": complete,
        "terminating_status": (
            steps[len(converged)]["status"] if len(steps) > len(converged) else record.get("status")
        ),
        "levels_reached": reached,
        "declared_levels": declared,
        "directory": str(directory),
        "record": record,
        "record_sha256": file_digest(directory / "record.json"),
        "field_sha256": (
            file_digest(directory / "states.npz")
            if (directory / "states.npz").exists()
            else [step.get("field_sha256") for step in converged]
        ),
        "state": state,
        "times_s": times,
        "slabs": declared,
        "spatial_state_dofs": record.get("spatial_state_dofs"),
        "subdivision": record.get("subdivision", 1),
        "time_scheme": record.get("time_scheme", "backward_euler"),
        "procedure": procedure_of(record),
    }


def checked_pair(coarse, fine, source_digest):
    """Refuse anything that is not a pure spatial refinement of one saved source.

    Returns the number of leading time levels both trajectories reached, which is
    the interval the comparison covers.
    """
    for side, name in ((coarse, "coarse"), (fine, "fine")):
        if side["record"].get("optimization_field_sha256") != source_digest:
            raise ValueError(f"The {name} trajectory replays a different saved source")
    if (
        coarse["declared_levels"] != fine["declared_levels"]
        or coarse["subdivision"] != fine["subdivision"]
    ):
        raise ValueError(
            "A spatial comparison needs one time grid; these trajectories refine time as well"
        )
    if coarse["time_scheme"] != fine["time_scheme"]:
        raise ValueError("A spatial comparison needs one time integration scheme")
    shared = min(coarse["levels_reached"], fine["levels_reached"])
    if not np.allclose(
        coarse["times_s"][:shared], fine["times_s"][:shared], rtol=1e-12, atol=1e-12
    ):
        raise ValueError("The two trajectories are not on the same physical time levels")
    if coarse["spatial_state_dofs"] == fine["spatial_state_dofs"]:
        raise ValueError("The two trajectories are on the same mesh; there is nothing to compare")
    if (fine["spatial_state_dofs"] or 0) < (coarse["spatial_state_dofs"] or 0):
        raise ValueError("The fine trajectory must have more spatial degrees of freedom")
    return shared


def assess(
    coarse_dir,
    fine_dir,
    baseline,
    fine_baseline,
    optimization,
    method,
    position,
    upper_K,
    allow_partial=False,
    protocol=None,
):
    coarse = load_trajectory(coarse_dir, allow_partial)
    fine = load_trajectory(fine_dir, allow_partial)
    record, cfg, fields, digest = load_saved_solution(optimization, method, position)
    if not cfg["transient"]:
        raise ValueError("A spatial trajectory comparison requires a transient optimization")
    shared = checked_pair(coarse, fine, digest)
    slabs = integer(cfg["slabs"], "Original time slabs", 1)
    subdivision = integer(coarse["subdivision"], "Time subdivision", 1)
    if coarse["declared_levels"] != slabs * subdivision:
        raise ValueError("The trajectories must subdivide the optimization's own time grid")
    comparison_cfg = {**cfg, "slabs": slabs * subdivision}
    coarse_problem, coarse_base = load_problem(
        {**comparison_cfg, "baseline_directory": str(baseline)}
    )
    fine_problem, fine_base = load_problem(
        {**comparison_cfg, "baseline_directory": str(fine_baseline)}
    )
    require_matching_baseline(record, coarse_base)
    for key in ("properties", "inlet_velocity_m_s", "grad_div_coefficient_m2_s"):
        if coarse_base.get(key) != fine_base.get(key):
            raise ValueError("Baseline physical inputs must match across refinement")
    # Each saved source is copied unchanged into its temporal subintervals.
    controls = np.repeat(
        fields["control"].reshape(slabs, coarse_problem.spatial_size), subdivision, axis=0
    )
    _, P = transfer_source(coarse_problem, fine_problem, controls)
    lifted = np.asarray(P @ coarse["state"][:shared].T).T
    mass = fine_problem.assembly.mass[fine_problem.free]
    statistics = statistics_from_arrays(
        lifted,
        coarse["times_s"][:shared],
        fine["state"][:shared],
        fine["times_s"][:shared],
        fine_problem.mesh,
        mass,
        fine_problem.temperature_scale,
        0.0,
    )
    same = coarse["procedure"].get("procedure") == fine["procedure"].get("procedure")
    pointwise = statistics["pointwise_maximum_K"]
    rms = statistics["mass_weighted_space_time_rms_K"]
    whole = shared == coarse["declared_levels"] and coarse["complete"] and fine["complete"]
    split = None
    if protocol is not None:
        declared, protocol_digest = load_protocol(protocol)
        split = split_assessment(
            declared,
            protocol_digest,
            lifted,
            coarse["times_s"][:shared],
            fine["state"][:shared],
            fine["times_s"][:shared],
            fine_problem.mesh,
            mass,
            fine_problem.temperature_scale,
            whole,
        )
    horizon = float(cfg["horizon_s"])
    assessed_to = float(fine["times_s"][shared - 1])
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
                "complete",
                "levels_reached",
                "terminating_status",
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
                "complete",
                "levels_reached",
                "terminating_status",
            )
        },
        "slabs": coarse["declared_levels"],
        "subdivision": subdivision,
        "time_scheme": coarse["time_scheme"],
        "assessed_interval": {
            "levels": shared,
            "declared_levels": coarse["declared_levels"],
            "to_time_s": assessed_to,
            "horizon_s": horizon,
            "fraction_of_horizon": assessed_to / horizon,
            "covers_the_horizon": whole,
            "note": (
                "The comparison covers the declared horizon."
                if whole
                else "No fine-mesh forward attempt reached the horizon. The comparison covers "
                "the leading interval both meshes reached, and the temperature criteria below "
                "are therefore reported for that interval and not as a verdict at the horizon."
            ),
        },
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
        "declared_split": split,
        "criteria": {
            "pointwise_maximum_K": POINTWISE_K,
            "mass_weighted_space_time_rms_K": RMS_K,
            # A criterion can only be met at the horizon. Over a truncated interval
            # the numbers are reported and the verdict is withheld.
            "pointwise_met": bool(pointwise <= POINTWISE_K) if whole else None,
            "rms_met": bool(rms <= RMS_K) if whole else None,
            "pointwise_met_over_assessed_interval": bool(pointwise <= POINTWISE_K),
            "rms_met_over_assessed_interval": bool(rms <= RMS_K),
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
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Compare the leading interval both trajectories reached, and withhold the verdict",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="A declared coupled-resolution-protocol-v3 file, applied as the horizon split",
    )
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
            args.allow_partial,
            args.protocol,
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
                "assessed_interval": report["assessed_interval"],
                "criteria": report["criteria"],
                "declared_split": (
                    None
                    if report["declared_split"] is None
                    else {
                        k: v
                        for k, v in report["declared_split"].items()
                        if k != "interval_statistics"
                    }
                    | {
                        "interval_pointwise_maximum_K": (
                            None
                            if report["declared_split"]["interval_statistics"] is None
                            else report["declared_split"]["interval_statistics"][
                                "pointwise_maximum_K"
                            ]
                        ),
                        "interval_rms_K": (
                            None
                            if report["declared_split"]["interval_statistics"] is None
                            else report["declared_split"]["interval_statistics"][
                                "mass_weighted_space_time_rms_K"
                            ]
                        ),
                    }
                ),
                "forward_procedures_match": report["forward_procedures_match"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
