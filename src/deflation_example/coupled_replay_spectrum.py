"""Local propagation diagnostics at verified states of a saved forward replay.

The fixed-source Jacobian is split into coupled, frozen-temperature momentum,
and frozen-velocity thermal pencils. Their eigenvalues describe a single
linearized backward-Euler step, not a product of maps or physical instability.
"""

import argparse
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_late_step import checkpoint_fields
from .coupled_newton_replay import criteria_met, step_equations
from .coupled_optimize import load_problem
from .coupled_prefix_diagnostics import read_forward
from .coupled_resolution import forward_model
from .coupled_saved import load_saved_solution, require_matching_baseline
from .coupled_step_spectrum import amplification_spectrum, step_linearization
from .reporting import environment, file_sha256, write_report
from .validation import integer


def verified_slab(times, verified_count, physical_time):
    indices = np.flatnonzero(np.isclose(times, physical_time, rtol=0, atol=1e-10))
    if len(indices) != 1 or indices[0] >= verified_count:
        raise ValueError("Select an existing independently verified physical time")
    return int(indices[0])


def pencils(problem, source, previous, state, velocity, slab):
    H, C, thermal, thermal_mass = step_linearization(
        problem, state, velocity, slab, control=source, previous=previous
    )
    n = len(problem.flow_free)
    return (
        ("coupled", H, C),
        ("frozen_temperature_momentum", H[:n, :n], C[:n, :n]),
        ("frozen_velocity_thermal", thermal, thermal_mass),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--time", type=float, required=True, help="An existing verified physical time in seconds"
    )
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modes = integer(args.modes, "Requested modes", 1)
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        saved, cfg, controls, digest = load_saved_solution(args.optimization, "reference", 0)
        record, fields, count = read_forward(args.replay)
        slab = verified_slab(fields["times_s"], count, args.time)
        public_cfg = {
            k: v
            for k, v in cfg.items()
            if k not in {"baseline_directory", "reference_baseline_directory", "output"}
        }
        if record["configuration"] != public_cfg or record["optimization_field_sha256"] != digest:
            raise ValueError("Saved control or configuration differs from the replay")
        if (
            record["forward_solver"].get("time_scheme") != "backward_euler"
            or not record["forward_formulation"]["consistent_stabilization"]
        ):
            raise ValueError("This diagnostic requires the corrected backward-Euler formulation")
        problem, baseline = load_problem(
            {
                **cfg,
                "slabs": record["forward_slabs"],
                "baseline_directory": str(args.baseline),
                "consistent_stabilization": True,
            }
        )
        require_matching_baseline(saved, baseline)
        if baseline["baseline_sha256"] != record["baseline_sha256"] or problem.slabs % cfg["slabs"]:
            raise ValueError("Baseline or source time grid differs")
        checkpoint_fields(args.replay, record, fields, slab, problem)
        ratio = problem.slabs // cfg["slabs"]
        source = np.zeros(len(problem.mesh.nodes))
        source[problem.free] = controls["control"].reshape(cfg["slabs"], -1)[slab // ratio]
        previous = problem.full_temperature(fields["state"][slab - 1] if slab else problem.initial)
        previous_flow = (
            FlowResult(fields["velocity"][slab - 1], fields["pressure"][slab - 1], "saved", [])
            if slab
            else problem.initial_flow
        )
        state = fields["state"][slab]
        flow = FlowResult(fields["velocity"][slab], fields["pressure"][slab], "saved", [])
        checks = step_equations(
            problem,
            forward_model(problem),
            problem.full_temperature(state),
            flow,
            source,
            previous,
            previous_flow,
            slab,
        )[1]
        if not criteria_met(checks, 1e-12):
            raise ValueError("Saved current state fails original-equation verification")
        report = {
            "schema": "coupled-replay-propagation-v1",
            "environment": environment(),
            "replay_source": record["environment"]["git_head"],
            "replay_record_sha256": file_sha256(args.replay / "record.json"),
            "replay_fields_sha256": file_sha256(args.replay / "states.npz"),
            "optimization_field_sha256": digest,
            "baseline_sha256": baseline["baseline_sha256"],
            "time_s": args.time,
            "slabs": problem.slabs,
            "slab_zero_based": slab,
            "time_step_s": float(problem.physical_steps[slab]),
            "independent_equations": checks,
            "requested_modes": modes,
            "blocks": {},
            "scope": "Local eigenvalues of H^{-1}C at the verified state, holding the source fixed. These do not certify physical instability, transient growth of products, or a resolved trajectory.",
        }
        write_report(args.output / "summary.json", {**report, "status": "running"})
        for name, H, C in pencils(problem, source, previous, state, flow.velocity, slab):
            start = time.perf_counter()
            block = amplification_spectrum(H, C, modes)
            block["seconds"] = time.perf_counter() - start
            block["all_reported_pencil_residuals_below_1e-6"] = bool(block["modes"]) and all(
                np.isfinite(r["pencil_relative_residual"]) and r["pencil_relative_residual"] < 1e-6
                for r in block["modes"]
            )
            report["blocks"][name] = block
            write_report(args.output / "summary.json", {**report, "status": "running"})
        write_report(args.output / "summary.json", {**report, "status": "complete"})


if __name__ == "__main__":
    main()
