"""Audit matched smooth-model optimizations from the same assessed initial field.

The report keeps initialization, discrete optimality, field agreement and elapsed
time separate. It does not establish physical resolution or repeated-sequence
performance. Nested coarse timers are subtracted from their enclosing interval.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_convergence import maximum_kkt, COLORS, LABELS
from .coupled_corrected_report import field_agreement, physical_temperature_scale
from .coupled_optimize import adjoint_acceptance
from .coupled_report import matched_identity, validate_record
from .coupled_sequence import COMPLETE_SCHEMA, STAGE_SCHEMA
from .reporting import environment, file_sha256, write_report

RANK_POLICY = {
    "jacobi": {"rank": 0, "recycle_window": 1},
    "reference": {"rank": 200, "recycle_window": 200},
}


def audit(record, source):
    cfg, stage = record["configuration"], record["stage"]
    env = record["environment"]
    if len(source) != 40 or env.get("git_head") != source or not env.get("source_tree_clean"):
        raise ValueError("Each numerical record must identify the declared clean source")
    if record["schema"] != STAGE_SCHEMA or stage["positions"] != [0]:
        raise ValueError("Supply the single-target initialized stage")
    if stage.get("resume") is not None or stage.get("restore") is not None:
        raise ValueError("Resumed work needs complete attempt accounting")
    if cfg["streamline_rule"] != "smooth_p8" or not cfg["consistent_stabilization"]:
        raise ValueError("The assessment requires smooth full-residual stabilization")
    if len(cfg["queries"]) != 1:
        raise ValueError("This assessment compares one declared target per method")
    initial = record["initial_state"]
    if (
        initial["policy"] != "fresh_optimization_from_assessed_checkpoint_temperature"
        or initial["evaluated_streamline_rule"] != "smooth_p8"
        or initial["retained_secant_pairs"] != 0
        or initial["retained_recycling_directions"] != 0
    ):
        raise ValueError("Fresh smooth initialization requires empty imported histories")
    verified = validate_record({**record, "schema": COMPLETE_SCHEMA})
    case = record["cases"][0]
    if case["verified"] and not adjoint_acceptance(case["adjoint"]):
        raise ValueError("Independent source, momentum and gradient checks are required")
    inner = [
        step
        for outer in case["history"]
        for attempt in outer.get("attempts", [])
        for step in attempt.get("qp_history", [])
        if "linear_status" in step
    ]
    residuals = [r["linear_residual"] for r in inner if r["linear_status"] == "converged"]
    if any(not np.isfinite(r) or not 0 <= r <= cfg["inner_tolerance"] for r in residuals):
        raise ValueError("Converged inner solve fails original-system residual verification")
    if sum(r["linear_iterations"] for r in inner) != case["inner_iterations"]:
        raise ValueError("Inner iteration totals differ from the retained attempts")
    inner_seconds, coarse_seconds = 0.0, 0.0
    for row in inner:
        timing = row["timing"]
        total = timing["total_seconds"]
        parts = np.asarray(list(timing["components_seconds"].values()))
        nested = timing.get("hybrid_coarse_correction", {}).get("seconds", 0.0)
        if (
            not np.isfinite(parts).all()
            or np.any(parts < 0)
            or not np.isfinite(nested)
            or not 0 <= nested <= total
            or not np.isclose(parts.sum(), total, rtol=1e-9, atol=1e-7)
        ):
            raise ValueError("Inner timers must sum and coarse work must stay nested")
        inner_seconds += total
        coarse_seconds += nested
    setup = record["components_seconds"]["reference_construction"]
    other = record["sequence_seconds"] - inner_seconds - setup
    if other < -1e-7:
        raise ValueError("Enclosed inner work exceeds complete optimization time")
    cost = {
        "reference_construction": setup,
        "coarse_processing_and_application": coarse_seconds,
        "remaining_inner_solve": inner_seconds - coarse_seconds,
        "remaining_optimization_and_verification": max(0.0, other),
    }
    history = case["history"]
    return {
        "method": cfg["method"],
        "status": case["status"],
        "verified": verified,
        "sequence_seconds": record["sequence_seconds"],
        "preparation_inclusive_seconds": record["preparation_inclusive_seconds"],
        "nonoverlapping_cost_seconds": cost,
        "objective": case["objective"],
        "kkt": case["kkt"],
        "outer_iterations": case["nonlinear_iterations"],
        "inner_iterations": case["inner_iterations"],
        "maximum_converged_inner_residual": max(residuals, default=None),
        "deployed_ranks": sorted({r["deployed_rank"] for r in inner}),
        "fallback_count": sum(bool(r.get("fallback")) for r in inner),
        "inner_statuses": sorted({r["linear_status"] for r in inner}),
        "iteration": [r["iteration"] for r in history] + [case["nonlinear_iterations"]],
        "maximum_kkt": [maximum_kkt(r["kkt"]) for r in history] + [maximum_kkt(case["kkt"])],
        "equation_maxima": {
            key: max(r[key] for r in case["equations"])
            for key in (
                "momentum_relative_residual",
                "continuity_relative_residual",
                "thermal_relative_residual",
                "mass_relative_imbalance",
            )
        },
        "maximum_energy_relative_defect": max(
            r["energy"]["relative_defect"] for r in case["equations"]
        ),
        "adjoint": {k: v for k, v in case["adjoint"].items() if k != "steps"},
        "memory": record["memory"],
    }


def summarize(directories, source):
    records, rows, fields = [], [], []
    for method in ("jacobi", "reference"):
        directory = Path(directories[method])
        record = json.loads((directory / "record.json").read_text())
        if record["configuration"]["method"] != method:
            raise ValueError("Method labels must agree with their records")
        row = audit(record, source)
        row["record_sha256"] = file_sha256(directory / "record.json")
        row["field_sha256"] = file_sha256(directory / "target-00.npz")
        with np.load(directory / "target-00.npz", allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in ("state", "control", "desired")}
        if arrays["state"].size != record["state_dofs_per_problem"]:
            raise ValueError("Saved fields differ from the recorded dimension")
        records.append(record)
        rows.append(row)
        fields.append(arrays)
    a, b = records
    if matched_identity(a, RANK_POLICY) != matched_identity(b, RANK_POLICY):
        raise ValueError("Compare matched physical, numerical and hardware settings")
    if a["initial_state"] != b["initial_state"]:
        raise ValueError("Both methods must receive the same assessed initial field")
    scales = [physical_temperature_scale(r) for r in records]
    if scales[0] != scales[1]:
        raise ValueError("Physical temperature scales differ")
    difference = field_agreement(*fields, temperature_scale=scales[0])
    objective = np.asarray([r["objective"] for r in rows])
    if not np.isfinite(objective).all():
        raise ValueError("Objectives must be finite")
    objective_difference = abs(objective[0] - objective[1]) / max(abs(objective).max(), 1e-30)
    comparable = (
        all(r["verified"] for r in rows) and difference <= 1e-3 and objective_difference <= 1e-6
    )
    control_scale = max(np.linalg.norm(f["control"]) for f in fields)
    return {
        "schema": "coupled-smooth-initialized-assessment-v1",
        "numerical_source": source,
        "report_environment": environment(),
        "rows": rows,
        "both_verified_and_field_agreement_met": bool(comparable),
        "maximum_temperature_difference_K": difference,
        "relative_objective_difference": float(objective_difference),
        "control_relative_difference": float(
            np.linalg.norm(fields[0]["control"] - fields[1]["control"]) / max(control_scale, 1e-30)
        ),
        "jacobi_over_reference_time": rows[0]["sequence_seconds"] / rows[1]["sequence_seconds"]
        if comparable
        else None,
        "inner_iteration_reduction_fraction": 1
        - rows[1]["inner_iterations"] / max(1, rows[0]["inner_iterations"])
        if comparable
        else None,
        "timing_boundary": a["timing_boundary"],
        "initialization_scope": a["initialization_scope"],
        "scope": "One full-trajectory optimization per method from the same assessed temperature field; no repeated timing, cold-start or physical-resolution claim. Global balance does not establish elementwise mass conservation.",
    }


def plot(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for row in report["rows"]:
        method = row["method"]
        axes[0].semilogy(
            row["iteration"],
            row["maximum_kkt"],
            "o-",
            color=COLORS[method],
            label=LABELS[method],
            markersize=4,
        )
    axes[0].axhline(1e-8, color="black", linestyle=":", label="Final criterion")
    axes[0].set(xlabel="Nonlinear update", ylabel="Maximum KKT component")
    axes[0].legend(fontsize=8)
    names = (
        "Reference construction",
        "Coarse processing/application",
        "Remaining inner solves",
        "Other optimization/verification",
    )
    bottom = np.zeros(2)
    for i, name in enumerate(names):
        values = (
            np.array([list(r["nonoverlapping_cost_seconds"].values())[i] for r in report["rows"]])
            / 60
        )
        axes[1].bar([0, 1], values, bottom=bottom, label=name)
        bottom += values
    axes[1].set(
        xticks=[0, 1],
        xticklabels=[LABELS[r["method"]] for r in report["rows"]],
        ylabel="Complete optimization time (min)",
    )
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.32))
    for i, total in enumerate(bottom):
        axes[1].text(i, total + 2, f"{total:.1f}", ha="center", fontsize=9)
    axes[1].set_ylim(0, max(bottom) * 1.12)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jacobi", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = summarize({"jacobi": args.jacobi, "reference": args.reference}, args.source)
    args.output.mkdir(parents=True)
    write_report(args.output / "report.json", report)
    plot(report, args.output / "convergence_cost.png")
    plot(report, args.output / "convergence_cost.pdf")


if __name__ == "__main__":
    main()
