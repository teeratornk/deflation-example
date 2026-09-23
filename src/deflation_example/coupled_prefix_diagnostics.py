"""Diagnose shared verified prefixes without certifying a complete trajectory."""

import argparse
from copy import copy
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_newton_replay import criteria_met
from .coupled_optimize import load_problem
from .coupled_saved import load_saved_solution, require_matching_baseline
from .coupled_targets import desired_temperature
from .reporting import environment, file_sha256, write_report
from .validation import integer


def verified_prefix(record):
    """Require both recorded convergence and the independent original equations."""
    count = 0
    for row in record["steps"]:
        if row["status"] != "converged":
            break
        checks = row["history"][-1]
        if checks.get("procedure") != "returned_state_verification":
            raise ValueError("Prefix diagnostics require independently verified returned states")
        names = (
            "momentum_relative_residual",
            "continuity_relative_residual",
            "thermal_relative_residual",
            "mass_relative_imbalance",
            "energy_relative_defect",
        )
        values = np.asarray([checks[k] for k in names])
        if not np.isfinite(values).all() or np.any(values < 0) or not criteria_met(checks, 1e-12):
            raise ValueError("A converged label conflicts with the recorded original equations")
        count += 1
    if any(row["status"] == "converged" for row in record["steps"][count:]):
        raise ValueError("A verified prefix cannot jump over a failed step")
    return count


def read_forward(directory):
    directory = Path(directory)
    record = json.loads((directory / "record.json").read_text())
    prefix = verified_prefix(record)
    with np.load(directory / "states.npz", allow_pickle=False) as stored:
        fields = {key: stored[key].copy() for key in ("state", "velocity", "pressure", "times_s")}
    count = len(record["steps"])
    slabs = integer(record["forward_slabs"], "Forward slabs", 1)
    if not 0 < count <= slabs:
        raise ValueError("Recorded steps must belong to the declared grid")
    if record["status"] == "converged" and prefix != slabs:
        raise ValueError("A complete trajectory requires every step to be verified")
    if (
        fields["state"].ndim != 2
        or fields["velocity"].ndim != 3
        or fields["pressure"].ndim != 2
        or fields["times_s"].ndim != 1
    ):
        raise ValueError("Stored trajectory arrays have invalid dimensions")
    if any(len(v) != count or not np.isfinite(v).all() for v in fields.values()):
        raise ValueError("Stored fields and verification rows must agree")
    if not np.allclose(
        fields["times_s"], [r["time_s"] for r in record["steps"]], rtol=0, atol=1e-10
    ):
        raise ValueError("Stored physical times differ from the verification record")
    expected = (
        np.arange(1, count + 1) * record["configuration"]["horizon_s"] / record["forward_slabs"]
    )
    if not np.allclose(fields["times_s"], expected, rtol=0, atol=1e-10):
        raise ValueError("These diagnostics require the declared uniform time grid")
    return record, fields, prefix


def aligned_counts(coarse_times, fine_times, coarse_prefix, fine_prefix, stride):
    """Exclude every unverified value and align the two end times exactly."""
    stride = integer(stride, "Nested temporal ratio", 2)
    if not 0 <= coarse_prefix <= len(coarse_times) or not 0 <= fine_prefix <= len(fine_times):
        raise ValueError("Verified counts exceed the stored trajectories")
    n = min(coarse_prefix, fine_prefix // stride)
    if n < 1 or not np.allclose(
        coarse_times[:n], fine_times[stride - 1 : n * stride : stride], rtol=0, atol=1e-10
    ):
        raise ValueError("A nonempty shared verified interval is required")
    return n, n * stride


def pair_diagnostics(problem, coarse, fine):
    cr, cf, cn, cd = coarse
    fr, ff, fn, fd = fine
    if fr["forward_slabs"] % cr["forward_slabs"]:
        raise ValueError("Time grids must be nested")
    stride = fr["forward_slabs"] // cr["forward_slabs"]
    nc, nf = aligned_counts(cf["times_s"], ff["times_s"], cn, fn, stride)
    cs, fs = cf["state"][:nc], ff["state"][:nf]
    ct, ft = cf["times_s"][:nc], ff["times_s"][:nf]
    scale = problem.temperature_scale
    mass = problem.assembly.mass[problem.free]
    error, rms, signed = (np.empty(nf) for _ in range(3))
    nodes = np.empty(nf, dtype=int)
    previous = np.vstack((problem.initial, cs[:-1]))
    for k in range(stride):
        delta = (fs[k::stride] - (previous + (k + 1) / stride * (cs - previous))) * scale
        selected = np.abs(delta).argmax(axis=1)
        values = delta[np.arange(nc), selected]
        error[k::stride], signed[k::stride], nodes[k::stride] = np.abs(values), values, selected
        rms[k::stride] = np.sqrt(np.einsum("ij,j,ij->i", delta, mass, delta) / mass.sum())
    peak = int(error.argmax())
    node = int(nodes[peak])
    mesh_node = int(problem.free[node])
    cells = np.flatnonzero(np.any(problem.mesh.cells == mesh_node, axis=1))
    tracking = []
    for record, fields, n in ((cr, cf, nc), (fr, ff, nf)):
        local = copy(problem)
        local.slabs = record["forward_slabs"]
        local.physical_steps = np.full(
            local.slabs, record["configuration"]["horizon_s"] / local.slabs
        )
        local.steps = local.physical_steps / local.time_scale
        local.size = local.slabs * local.spatial_size
        cfg = record["configuration"]
        target = desired_temperature(
            local, cfg["query"], cfg["target_count"], cfg.get("target_startup_s", 0.0)
        ).reshape(local.slabs, -1)[:n]
        difference = scale * (fields["state"][:n] - target)
        tracking.append(
            float(np.einsum("ij,j,ij->", difference, mass, difference) * local.physical_steps[0])
        )
    velocity_delta = ff["velocity"][stride - 1 : nf : stride] - cf["velocity"][:nc]
    velocity_max = np.linalg.norm(velocity_delta, axis=2).max(axis=1)
    return {
        "coarse_case": cd.name,
        "fine_case": fd.name,
        "coarse_slabs": cr["forward_slabs"],
        "fine_slabs": fr["forward_slabs"],
        "comparison_end_s": float(ct[-1]),
        "coarse_verified_steps_used": nc,
        "fine_verified_steps_used": nf,
        "complete_horizon": nc == cr["forward_slabs"] and nf == fr["forward_slabs"],
        "maximum_temperature_difference_K": float(error.max()),
        "maximum_weighted_temperature_rms_K": float(rms.max()),
        "tracking_integrals_common_prefix_K2_m3_s": tracking,
        "tracking_relative_change_common_prefix": (
            abs(tracking[1] - tracking[0]) / tracking[1]
            if tracking[1]
            else (0.0 if tracking[0] == 0 else None)
        ),
        "maximum_velocity_difference_at_shared_times_m_s": float(velocity_max.max()),
        "peak": {
            "time_s": float(ft[peak]),
            "free_node_index": node,
            "mesh_node_index": mesh_node,
            "mesh_coordinates": problem.mesh.nodes[mesh_node],
            "adjacent_material_ids": np.unique(problem.mesh.materials[cells]),
            "signed_fine_minus_coarse_K": float(signed[peak]),
        },
        "times_s": ft,
        "maximum_difference_K": error,
        "weighted_rms_K": rms,
        "shared_times_s": ct,
        "maximum_velocity_difference_m_s": velocity_max,
        "peak_node_coarse_times_s": np.r_[0, ct],
        "peak_node_fine_times_s": np.r_[0, ft],
        "peak_node_coarse_temperature_K": problem.temperature_offset
        + scale * np.r_[problem.initial[node], cs[:, node]],
        "peak_node_fine_temperature_K": problem.temperature_offset
        + scale * np.r_[problem.initial[node], fs[:, node]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if len(args.replays) < 2:
        parser.error("Supply at least two distinct forward grids")
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        saved, cfg, _, digest = load_saved_solution(args.optimization, "reference", 0)
        problem, baseline = load_problem(
            {**cfg, "baseline_directory": str(args.baseline), "consistent_stabilization": True}
        )
        require_matching_baseline(saved, baseline)
        runs, rows, identity = [], [], None
        for directory in args.replays:
            record, fields, count = read_forward(directory)
            key = {
                k: record[k]
                for k in (
                    "optimization_field_sha256",
                    "baseline_sha256",
                    "configuration",
                    "forward_solver",
                    "forward_formulation",
                )
            }
            key["source"] = record["environment"]["git_head"]
            if (
                (identity is not None and key != identity)
                or digest != key["optimization_field_sha256"]
                or key["baseline_sha256"] != baseline["baseline_sha256"]
            ):
                raise ValueError(
                    "Prefix comparison requires identical model, control and numerical sources"
                )
            identity = key
            if (
                fields["state"].shape[1:] != (problem.spatial_size,)
                or fields["velocity"].shape[1:] != (problem.flow.nv, 2)
                or fields["pressure"].shape[1:] != (problem.flow.np,)
            ):
                raise ValueError("Saved fields differ from the baseline dimensions")
            runs.append((record, fields, count, directory))
            rows.append(
                {
                    "case": directory.name,
                    "status": record["status"],
                    "slabs": record["forward_slabs"],
                    "verified_steps": count,
                    "last_verified_time_s": float(fields["times_s"][count - 1]) if count else 0,
                    "record_sha256": file_sha256(directory / "record.json"),
                    "fields_sha256": file_sha256(directory / "states.npz"),
                }
            )
        runs.sort(key=lambda row: row[0]["forward_slabs"])
        if len({r[0]["forward_slabs"] for r in runs}) != len(runs):
            raise ValueError("Supply distinct temporal grids")
        pairs = [pair_diagnostics(problem, a, b) for a, b in zip(runs[:-1], runs[1:])]
        report = {
            "schema": "coupled-verified-prefix-diagnostic-v1",
            "environment": environment(),
            "identity": identity,
            "scope": "Verified prefixes only; failed steps are excluded explicitly. This diagnostic does not grant a complete-trajectory resolution certificate.",
            "rows": rows,
            "pairs": pairs,
        }
        write_report(args.output / "summary.json", report)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
        for pair in pairs:
            label = f"{pair['coarse_slabs']}/{pair['fine_slabs']} steps"
            axes[0, 0].plot(pair["times_s"], pair["maximum_difference_K"], label=label)
            axes[0, 1].plot(pair["times_s"], pair["weighted_rms_K"], label=label)
            axes[1, 0].plot(
                pair["shared_times_s"], pair["maximum_velocity_difference_m_s"], label=label
            )
            axes[1, 1].plot(
                pair["peak_node_coarse_times_s"],
                pair["peak_node_coarse_temperature_K"],
                label=f"{label}: coarse",
            )
            axes[1, 1].plot(
                pair["peak_node_fine_times_s"],
                pair["peak_node_fine_temperature_K"],
                linestyle="--",
                label=f"{label}: fine",
            )
        for axis, ylabel in zip(
            axes.flat,
            (
                "Maximum temperature difference (K)",
                "Volume-weighted RMS difference (K)",
                "Maximum velocity difference (m/s)",
                "Temperature at pairwise peak node (K)",
            ),
            strict=True,
        ):
            axis.set(xlabel="Physical time (s)", ylabel=ylabel)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        axes[0, 0].axhline(0.05, color="black", linestyle=":", linewidth=1)
        for extension in ("pdf", "png"):
            fig.savefig(args.output / f"verified_prefixes.{extension}", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
