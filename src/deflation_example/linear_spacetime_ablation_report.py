"""Accuracy-gated time, rank and memory summaries for the fixed-flow ablation."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_confirmation_report import inner_evidence
from .coupled_optimize import adjoint_acceptance
from .coupled_report import validate_record
from .linear_spacetime_ablation import ARM_KEYS, RANKS, SELECTIONS, checked_ablation_settings
from .reporting import file_sha256, write_report
from .validation import integer


def matched_identity(record):
    cfg, env = record["configuration"], record["environment"]
    return {
        "configuration": {
            k: v
            for k, v in cfg.items()
            if k not in ARM_KEYS | {"repetition", "speedup_ablation_arm"}
        },
        "source": env["source_sha256"],
        "cpu": env["cpu_model"],
        "numpy": env["numpy"],
        "scipy": env["scipy"],
        "blas": env.get("blas"),
        "device": record["device"],
        "baseline": record["baseline_sha256"],
        "velocity": record.get("fixed_flow_preparation", {}).get("velocity_sha256"),
        "numerical_policy": record["numerical_policy"],
        "timing_boundary": record["timing_boundary"],
    }


def summarize(records, settings, agreements=None):
    """Keep missing/failed outcomes; ratios require every matched repetition."""
    groups = {name: {} for name in settings["arms"]}
    identity = None
    for record in records:
        cfg = record["configuration"]
        name = cfg["speedup_ablation_arm"]
        repeat = integer(cfg["repetition"], "Repetition", 0)
        if name not in groups or repeat >= settings["repetitions"] or repeat in groups[name]:
            raise ValueError("Unknown or duplicate ablation member")
        if {k: cfg[k] for k in ARM_KEYS} != settings["arms"][name]:
            raise ValueError("Solver choices differ from the declared arm")
        required = {
            "physics": "prescribed_flow",
            "feedback_multiplier": 0.0,
            "inner_tolerance": 1e-10,
            "nonlinear_tolerance": 1e-8,
            "equation_acceptance_tolerance": 1e-12,
            "conservation_tolerance": 1e-6,
            "alpha": settings["alpha"],
            "slabs": 64,
            "horizon_s": 600.0,
            "target_startup_s": 60.0,
            "lower_K": 337.3,
            "reference_krylov_steps": 96,
            "reference_krylov_seed": 20260923,
            "warm_start": True,
            "stage": None,
        }
        if any(cfg.get(k) != v for k, v in required.items()) or cfg["queries"] != [
            {"target": t, "upper_K": 357.3} for t in settings["targets"]
        ]:
            raise ValueError("Physical problem, accuracy or reference protocol differs")
        if record.get("status") == "running":
            groups[name][repeat] = {
                "repetition": repeat,
                "status": "unfinished_record",
                "verified": False,
                "recorded_status": "running",
                "scope": "Consult the scheduler outcome; a stale running record establishes no completed time.",
            }
            continue
        valid = validate_record(record)
        current = matched_identity(record)
        if record.get("assembly") or not current["velocity"]:
            raise ValueError("Use complete fresh sequences with a verified fixed velocity")
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError("Sources, hardware, timing or common settings differ")
        evidence = inner_evidence(record)
        valid &= evidence["complete_histories"] and all(
            adjoint_acceptance(c["adjoint"]) for c in record["cases"] if c["verified"]
        )
        agreement = None if agreements is None else agreements.get((name, repeat))
        if agreement is not None:
            errors = [agreement["state_absolute"], agreement["objective_relative"]]
            if not np.isfinite(errors).all() or min(errors) < 0:
                raise ValueError("Solution differences must be finite and nonnegative")
        groups[name][repeat] = {
            "repetition": repeat,
            "status": record["status"],
            "verified": bool(valid),
            "case_statuses": [c["status"] for c in record["cases"]],
            "seconds": record["sequence_seconds"],
            "preparation_inclusive_seconds": record["preparation_inclusive_seconds"],
            "components_seconds": record["components_seconds"],
            "inner_evidence": evidence,
            "agreement": agreement,
            "sampled_peak_host_rss_bytes": record["memory"].get("peak_host_rss_bytes")
            if record["memory"].get("complete")
            else None,
        }
    rows = {}
    for name, group in groups.items():
        outcomes = [
            group.get(rep, {"repetition": rep, "status": "missing_record", "verified": False})
            for rep in range(settings["repetitions"])
        ]
        valid = all(r["verified"] for r in outcomes)
        agreement = valid and all(
            r.get("agreement") is not None
            and r["agreement"]["state_absolute"] <= 1e-6
            and r["agreement"]["objective_relative"] <= 1e-6
            for r in outcomes
        )
        times = [r["seconds"] for r in outcomes if r["verified"]]
        memory = [r.get("sampled_peak_host_rss_bytes") for r in outcomes]
        rows[name] = {
            "arm": name,
            "configuration": settings["arms"][name],
            "outcomes": outcomes,
            "all_repetitions_verified": valid,
            "solution_agreement": bool(agreement),
            "median_seconds": float(np.median(times)) if valid else None,
            "range_seconds": [min(times), max(times)] if valid else None,
            "median_inner_iterations": float(
                np.median([r["inner_evidence"]["recorded_inner_iterations"] for r in outcomes])
            )
            if valid
            else None,
            "sampled_peak_host_rss_bytes": max(memory)
            if all(m is not None for m in memory)
            else None,
        }
    comparisons = []
    for rank in RANKS:
        controls = [rows[n] for n in ("jacobi", "frozen", f"recycling-r{rank}")]
        for label in SELECTIONS:
            ref, seq = (rows[f"{p}-{label}-r{rank}"] for p in ("reference", "sequential"))
            eligible = all(
                r["all_repetitions_verified"] and r["solution_agreement"] for r in [ref, *controls]
            )
            used = all(
                r.get("inner_evidence", {}).get("nonzero_coarse_space_used", False)
                for r in ref["outcomes"]
            )
            fastest = min(controls, key=lambda r: r["median_seconds"]) if eligible else None
            transfer = all(
                r["all_repetitions_verified"] and r["solution_agreement"] for r in (ref, seq)
            )
            comparisons.append(
                {
                    "reference": ref["arm"],
                    "fastest_tested_alternative": None if fastest is None else fastest["arm"],
                    "fastest_alternative_over_reference": fastest["median_seconds"]
                    / ref["median_seconds"]
                    if eligible and used
                    else None,
                    "reference_space_used_in_every_repetition": used,
                    "sequential_over_full": seq["median_seconds"] / ref["median_seconds"]
                    if transfer and used
                    else None,
                }
            )
    return {
        "schema": "linear-spacetime-speedup-ablation-summary-v1",
        "scope": "Complete prescribed-flow three-target development sequences. All compared repetitions must verify and agree. No fully coupled speedup or confirmation claim.",
        "rows": list(rows.values()),
        "comparisons": comparisons,
        "all_declared_sequences_verified": all(
            r["all_repetitions_verified"] for r in rows.values()
        ),
    }


def field_agreement(paths, records):
    """Compare each successful outcome to the fresh frozen repetition zero."""
    anchor = next(
        (
            i
            for i, r in enumerate(records)
            if r["configuration"]["speedup_ablation_arm"] == "frozen"
            and r["configuration"]["repetition"] == 0
            and r.get("status") != "running"
            and validate_record(r)
        ),
        None,
    )
    if anchor is None:
        return {}, []
    agreements, files = {}, []
    for i, record in enumerate(records):
        if record.get("status") == "running" or not validate_record(record):
            continue
        errors = {"state_absolute": 0.0, "objective_relative": 0.0}
        for position, case in enumerate(record["cases"]):
            filename = f"target-{position:02d}.npz"
            a_path, b_path = paths[anchor].parent / filename, paths[i].parent / filename
            with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
                x, y = a["state"], b["state"]
                if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError("Saved states must be finite and dimensionally matched")
                errors["state_absolute"] = max(
                    errors["state_absolute"], float(np.max(np.abs(x - y)))
                )
            oa, ob = records[anchor]["cases"][position]["objective"], case["objective"]
            if not np.isfinite([oa, ob]).all():
                raise ValueError("Saved objectives must be finite")
            errors["objective_relative"] = max(
                errors["objective_relative"], abs(oa - ob) / max(abs(oa), abs(ob), 1e-30)
            )
            files.append(
                {"file": b_path.parent.name + "/" + filename, "sha256": file_sha256(b_path)}
            )
        cfg = record["configuration"]
        agreements[cfg["speedup_ablation_arm"], cfg["repetition"]] = errors
    return agreements, files


def plot(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = {r["arm"]: r for r in summary["rows"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for prefix, label in (
        ("reference-low", "Full, low"),
        ("reference-ends", "Full, low/high"),
        ("sequential-low", "Sequential, low"),
        ("sequential-ends", "Sequential, low/high"),
        ("recycling", "Recycling"),
    ):
        good = [
            rows[f"{prefix}-r{r}"]
            for r in RANKS
            if rows[f"{prefix}-r{r}"]["all_repetitions_verified"]
        ]
        if not good:
            continue
        ranks = [r["configuration"]["rank"] for r in good]
        values = np.array([r["median_seconds"] for r in good])
        ranges = np.array([r["range_seconds"] for r in good]).T
        axes[0].errorbar(
            ranks,
            values,
            yerr=np.vstack((values - ranges[0], ranges[1] - values)),
            marker="o",
            capsize=3,
            label=label,
        )
        memory = [r for r in good if r["sampled_peak_host_rss_bytes"] is not None]
        axes[1].plot(
            [r["configuration"]["rank"] for r in memory],
            [r["sampled_peak_host_rss_bytes"] / 2**30 for r in memory],
            "o-",
            label=label,
        )
    for name in ("jacobi", "frozen"):
        if rows[name]["all_repetitions_verified"]:
            axes[0].axhline(rows[name]["median_seconds"], linestyle="--", label=name.capitalize())
    for ax in axes:
        ax.set(xlabel="Requested rank", xticks=RANKS)
        ax.grid(alpha=0.2)
    axes[0].set(ylabel="Median complete-sequence time (s)", yscale="log")
    axes[1].set(ylabel="Maximum sampled process RSS (GiB)")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8)
    failed = sum(not r["all_repetitions_verified"] for r in rows.values())
    fig.suptitle(
        f"Prescribed-flow space–time ablation; {failed} incomplete or failed configurations"
    )
    fig.savefig(output / "rank-time-memory.pdf")
    fig.savefig(output / "rank-time-memory.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    settings = checked_ablation_settings(args.screen, args.settings)
    paths = [p for p in args.records if p.is_file()]
    records = [json.loads(p.read_text()) for p in paths]
    digest = file_sha256(args.settings)
    if any(r["configuration"].get("speedup_ablation_settings_sha256") != digest for r in records):
        raise ValueError("Recorded settings checksum differs")
    agreements, files = field_agreement(paths, records)
    result = summarize(records, settings, agreements)
    result.update(
        settings_sha256=digest,
        field_files=files,
        records=[{"file": p.parent.name + "/" + p.name, "sha256": file_sha256(p)} for p in paths],
        missing_files=[p.parent.name + "/" + p.name for p in args.records if not p.is_file()],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", result)
    if args.plot:
        plot(result, args.output)


if __name__ == "__main__":
    main()
