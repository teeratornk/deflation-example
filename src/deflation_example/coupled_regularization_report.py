"""Verify the bounded regularization screen and apply its predeclared gate."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .coupled_regularization import ALPHAS, RANKS, TARGETS, configuration
from .coupled_regularization_resume import read_population
from .reporting import file_sha256, write_report


KKT_COMPONENTS = {
    "primal_absolute",
    "stationarity",
    "dual_feasibility",
    "lower_complementarity",
    "upper_complementarity",
}


def finite(value, *, positive=False):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def verified_case(row):
    """Check recorded independent tests and every inner termination, not a flag alone."""
    kkt = row.get("kkt", {})
    history = row.get("history", [])
    objective = row.get("quadratic_objective")
    return (
        row.get("verified") is True
        and row.get("status") == "converged"
        and set(kkt) == KKT_COMPONENTS
        and all(finite(v) and v <= 1e-10 for v in kkt.values())
        and finite(row.get("maximum_original_relative_residual"))
        and row["maximum_original_relative_residual"] <= 1e-10
        and row.get("outer_pdas_steps") == len(history)
        and all(
            item.get("linear_status") in {"converged", "empty"}
            and finite(item.get("linear_residual", 0.0))
            and item.get("linear_residual", 0.0) <= 1e-10
            and finite(item.get("linear_iterations"))
            for item in history
        )
        and row.get("inner_iterations") == sum(h["linear_iterations"] for h in history)
        and row.get("deployed_ranks") == [h.get("deployed_rank", 0) for h in history]
        and row.get("fallbacks") == [h.get("fallback") for h in history]
        and isinstance(objective, (int, float))
        and not isinstance(objective, bool)
        and np.isfinite(objective)
    )


def load_increments(directory, descriptor, dofs):
    name = descriptor["file"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("Increment archives must be local filenames")
    path = directory / name
    if file_sha256(path) != descriptor["sha256"]:
        raise ValueError("Increment archive checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        values = data["increments"].copy()
    if values.shape != (len(TARGETS), dofs) or not np.isfinite(values).all():
        raise ValueError("Increment archives must contain all finite target solutions")
    return values


def summarize_alpha(directory, continuations=()):
    directory = Path(directory)
    path = directory / "record.json"
    original_status = json.loads(path.read_text())["status"]
    record, origins, continuation_evidence = read_population(directory, continuations)
    if record.get("schema") != "coupled-regularization-screen-v1":
        raise ValueError("Unexpected regularization screen schema")
    cfg = configuration({"configuration": record["configuration"]}, record["alpha"])
    if cfg != record["configuration"]:
        raise ValueError("The recorded configuration differs from the frozen CPU protocol")
    if (
        record.get("ranks") != list(RANKS)
        or record.get("targets") != list(TARGETS)
        or record.get("repetitions") != 3
    ):
        raise ValueError("The recorded rank, target or repetition population changed")
    sequences = {}
    for row in record["sequences"]:
        key = (row["rank"], row["repetition"])
        if key in sequences or key[0] not in RANKS or key[1] not in range(3):
            raise ValueError("Unexpected or duplicate sequence")
        if [c.get("target") for c in row["cases"]] != list(TARGETS):
            raise ValueError("Each sequence must contain the three declared targets in order")
        if not finite(row.get("online_seconds"), positive=True):
            raise ValueError("Online timing must be finite and positive")
        sequences[key] = row
    rows = []
    construction = record.get("reference", {}).get("construction_seconds")
    for rank in RANKS:
        group = [sequences[rank, rep] for rep in range(3) if (rank, rep) in sequences]
        reasons = []
        if record["status"] != "complete" or len(group) != 3:
            reasons.append("incomplete_population")
        bad = [
            [seq["repetition"], case["target"], case["status"]]
            for seq in group
            for case in seq["cases"]
            if not verified_case(case)
        ]
        if bad or any(seq.get("verified") is not True for seq in group):
            reasons.append("residual_or_quadratic_optimality_failure")
        ranks = [r for seq in group for c in seq["cases"] for r in c["deployed_ranks"]]
        if any(not isinstance(r, int) or isinstance(r, bool) or r < 0 or r > rank for r in ranks):
            raise ValueError("Deployed ranks must lie between zero and the requested rank")
        if rank and not any(r > 0 for r in ranks):
            reasons.append("no_nonzero_reference_deployment")
        expected_construction = construction if rank else 0.0
        if not finite(expected_construction):
            reasons.append("missing_construction_cost")
        else:
            for seq in group:
                if seq.get("construction_seconds_once") != expected_construction or not np.isclose(
                    seq.get("setup_inclusive_model_seconds", np.nan),
                    seq["online_seconds"] + expected_construction,
                    rtol=1e-12,
                    atol=1e-9,
                ):
                    raise ValueError("Setup-inclusive cost must charge reference construction once")
        differences, objective_differences = [], []
        for seq in group:
            rep = seq["repetition"]
            baseline = sequences.get((0, rep))
            if baseline is None or not all(verified_case(c) for c in baseline["cases"]):
                reasons.append("unverified_rank_zero_control")
                continue
            values = load_increments(origins[rank, rep], seq["arrays"], record["state_dofs"])
            controls = (
                values
                if rank == 0
                else load_increments(origins[0, rep], baseline["arrays"], record["state_dofs"])
            )
            differences.append(float(np.max(np.abs(values - controls))))
            for case, control in zip(seq["cases"], baseline["cases"], strict=True):
                f, f0 = case["quadratic_objective"], control["quadratic_objective"]
                if f is None or f0 is None or not np.isfinite([f, f0]).all():
                    reasons.append("nonfinite_quadratic_objective")
                    continue
                objective_differences.append(abs(f - f0) / max(1.0, abs(f0)))
        if differences and max(differences) > 1e-6:
            reasons.append("increment_disagreement")
        if objective_differences and max(objective_differences) > 1e-6:
            reasons.append("quadratic_objective_disagreement")
        times = [seq["online_seconds"] for seq in group]
        rows.append(
            {
                "rank": rank,
                "eligible_accuracy": not reasons,
                "reasons": sorted(set(reasons)),
                "failed_cases": bad,
                "repetitions": len(group),
                "online_seconds": times,
                "median_online_seconds": float(np.median(times)) if times else None,
                "range_online_seconds": [min(times), max(times)] if times else None,
                "construction_seconds_once": expected_construction,
                "inner_iterations": [
                    sum(c["inner_iterations"] for c in seq["cases"]) for seq in group
                ],
                "deployed_ranks": sorted(set(ranks)),
                "fallback_count": sum(
                    bool(f) for seq in group for c in seq["cases"] for f in c["fallbacks"]
                ),
                "maximum_increment_difference": max(differences, default=None),
                "maximum_relative_quadratic_objective_difference": max(
                    objective_differences, default=None
                ),
            }
        )
    control = rows[0]
    for row in rows:
        row["gate_passed"] = False
        if row["rank"] and row["eligible_accuracy"] and control["eligible_accuracy"]:
            saving = control["median_online_seconds"] - row["median_online_seconds"]
            fraction = saving / control["median_online_seconds"]
            count = math.ceil(row["construction_seconds_once"] / saving) if saving > 0 else None
            row.update(
                online_saving_fraction=fraction,
                constant_cost_break_even_three_quadratic_sequences=count,
                predicted_saving_over_nine_sequences_seconds=9 * saving
                - row["construction_seconds_once"],
                gate_passed=bool(fraction >= 0.05 and count is not None and count <= 9),
            )
    return {
        "alpha": record["alpha"],
        "status": record["status"],
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
        "record_sha256": file_sha256(path),
        "source": record["environment"].get("git_head"),
        "source_scope": "Original timing source; separately identified continuation drivers preserve every original numerical module byte-for-byte.",
        "original_record_status": original_status,
        "continuations": continuation_evidence,
        "rows": rows,
    }, record


def summarize(directories, continuations=()):
    assignments = {file_sha256(Path(d) / "record.json"): [] for d in directories}
    for folder in continuations:
        addition = json.loads((Path(folder) / "record.json").read_text())
        original_digest = addition.get("original_record_sha256")
        if original_digest not in assignments:
            raise ValueError("Every continuation must identify a supplied original attempt")
        assignments[original_digest].append(folder)
    groups, records = [], []
    for directory in directories:
        group, record = summarize_alpha(
            directory, assignments[file_sha256(Path(directory) / "record.json")]
        )
        groups.append(group)
        records.append(record)
    alphas = [g["alpha"] for g in groups]
    if len(set(alphas)) != len(alphas):
        raise ValueError("Declare each alpha once, retaining failed attempts separately")
    keys = []
    for record in records:
        env = record["environment"]
        keys.append(
            json.dumps(
                {
                    "configuration": {
                        k: v for k, v in record["configuration"].items() if k != "alpha"
                    },
                    "trace_sha256": record["trace_sha256"],
                    "initial_trajectory_sha256": record["initial_trajectory_sha256"],
                    "environment": {
                        k: env.get(k)
                        for k in (
                            "cpu_model",
                            "python",
                            "numpy",
                            "scipy",
                            "blas",
                            "git_head",
                            "source_sha256",
                            "source_tree_clean",
                        )
                    },
                },
                sort_keys=True,
            )
        )
    if len(set(keys)) != 1 or any(
        r["environment"].get("source_tree_clean") is not True for r in records
    ):
        raise ValueError("The alpha comparison must share physics, source, hardware and runtime")
    complete_population = set(alphas) == set(ALPHAS) and all(
        g["status"] in {"complete", "screen_error", "time_limit"} for g in groups
    )
    candidates = [
        {"alpha": g["alpha"], **row} for g in groups for row in g["rows"] if row["gate_passed"]
    ]
    candidates.sort(
        key=lambda row: (
            -row["predicted_saving_over_nine_sequences_seconds"],
            row["rank"],
            row["alpha"],
        )
    )
    return {
        "schema": "coupled-regularization-summary-v1",
        "all_alpha_attempts_finished": complete_population,
        "groups": sorted(groups, key=lambda group: group["alpha"]),
        "selected": candidates[0] if complete_population and candidates else None,
        "decision": "await_declared_population"
        if not complete_population
        else (
            "evaluate_one_complete_optimization_comparison" if candidates else "stop_bounded_screen"
        ),
        "scope": "Constrained Gauss-Newton subproblem screen. Break-even counts repeat the same three-quadratic mix, not nonlinear optimization queries. Complete optimization must establish its own benefit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--continuations", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = summarize(args.runs, args.continuations)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", result)
    if args.plot:
        plot(result, args.output)
    print(json.dumps({k: result[k] for k in ("decision", "selected")}, indent=2))


def plot(result, output):
    """Show all ranks and repetitions, including ineligible comparisons."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    # Separate the scope title and legend explicitly; constrained layout can
    # allocate their outside-upper positions to the same strip.
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.85, hspace=0.40, wspace=0.25)
    for axis, alpha in zip(axes.flat, ALPHAS, strict=True):
        group = next((g for g in result["groups"] if g["alpha"] == alpha), None)
        axis.set_title(rf"$\alpha=10^{{{int(round(np.log10(alpha)))}}}$")
        axis.set(xlabel="Three-quadratic sequence time (s)", ylabel="Reference rank")
        axis.set_yticks(range(len(RANKS)), [str(r) for r in RANKS])
        if group is None:
            axis.text(0.5, 0.5, "No recorded attempt", transform=axis.transAxes, ha="center")
            continue
        for y, row in enumerate(group["rows"]):
            online = row["median_online_seconds"]
            if online is None:
                axis.text(0, y, "incomplete", va="center", fontsize=8)
                continue
            valid = row["eligible_accuracy"]
            axis.barh(y, online, height=0.65, color="#3b75af", hatch=None if valid else "//")
            cost = row["construction_seconds_once"]
            if cost is not None:
                axis.barh(
                    y,
                    cost,
                    left=online,
                    height=0.65,
                    color="#e1a349",
                    hatch=None if valid else "//",
                )
            axis.scatter(
                row["online_seconds"],
                [y] * len(row["online_seconds"]),
                s=18,
                color="black",
                zorder=3,
            )
            if not valid:
                axis.annotate(
                    "ineligible",
                    (online + (cost or 0), y),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                )
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.invert_yaxis()
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    figure.legend(
        handles=[
            Patch(color="#3b75af", label="Median online time"),
            Patch(color="#e1a349", label="Reference construction (once)"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="black",
                label="Individual online repetition",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        fontsize=9,
    )
    figure.suptitle(
        "Constrained quadratic screening; not complete nonlinear optimization", fontsize=11
    )
    for extension in ("png", "pdf"):
        figure.savefig(Path(output) / f"regularization_cost.{extension}", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
