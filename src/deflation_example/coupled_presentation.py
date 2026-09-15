"""Tables and figures of the coupled final study from assembled sequence records.

Every number comes from checked records: the campaign summary, the assembled
complete-sequence records of each chain, and optionally the screen stage
records. Rows are written as LaTeX table bodies with the same conventions as
the mesh-study tables, and a summary JSON carries the generator environment so
the manuscript importer can refuse artifacts from an unclean checkout.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_convergence import maximum_kkt
from .coupled_report import LABELS, METHODS
from .reporting import atomic_output, environment, write_report

SCHEMA = "coupled-final-presentation-v1"


def linear_split(record):
    """Inner linear-solve, flow-evaluation and remaining seconds of one sequence.

    The first evaluation of every problem and every accepted or rejected trial
    evaluation are counted once; later history rows repeat the accepted trial's
    evaluation, so they are not summed again.
    """
    linear = cg = evaluations = outer = solves = 0.0
    for case in record["cases"]:
        history = case.get("history") or []
        outer += len(history)
        if history:
            evaluations += history[0].get("evaluation_seconds") or 0.0
        for row in history:
            for attempt in row.get("attempts", []):
                for trial in attempt.get("trials", []):
                    evaluations += trial.get("evaluation_seconds") or 0.0
                for step in attempt.get("qp_history", []):
                    solves += 1
                    cg += step.get("linear_iterations") or 0
                    linear += (step.get("timing") or {}).get("total_seconds") or 0.0
    total = float(record["sequence_seconds"])
    return {
        "sequence_seconds": total,
        "linear_seconds": float(linear),
        "evaluation_seconds": float(evaluations),
        "other_seconds": float(max(total - linear - evaluations, 0.0)),
        "cg_iterations": int(cg),
        "linear_solves": int(solves),
        "outer_iterations": int(outer),
        "linear_fraction": float(linear / total) if total else None,
    }


def accuracy(record):
    kkt, residual, ranks, fallbacks = 0.0, 0.0, [], 0
    for case in record["cases"]:
        if case.get("verified"):
            kkt = max(kkt, maximum_kkt(case["kkt"]))
        for row in case.get("history") or []:
            for attempt in row.get("attempts", []):
                for step in attempt.get("qp_history", []):
                    timing = step.get("timing") or {}
                    for item in timing.get("refinement_attempts", []):
                        residual = max(residual, item.get("original_residual") or 0.0)
                    if step.get("deployed_rank") is not None:
                        ranks.append(int(step["deployed_rank"]))
                    if step.get("fallback"):
                        fallbacks += 1
    return {
        "maximum_kkt": kkt,
        "maximum_original_residual": residual,
        "deployed_rank_range": [min(ranks), max(ranks)] if ranks else None,
        "fallback_count": fallbacks,
    }


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def load_campaign(summary_directory):
    summary_directory = Path(summary_directory)
    campaign = json.loads((summary_directory / "summary.json").read_text())
    records = {method: [] for method in METHODS}
    for directory in sorted((summary_directory / "assembled").glob("*-rep-*")):
        record = json.loads((directory / "record.json").read_text())
        records[record["configuration"]["method"]].append(record)
    return campaign, records


def load_screen(directories):
    rows = []
    for directory in map(Path, directories):
        record = json.loads((directory / "record.json").read_text())
        cfg = record["configuration"]
        split = linear_split(record)
        rows.append(
            {
                "directory": str(directory),
                "method": cfg["method"],
                "rank": cfg["rank"],
                "recycle_window": cfg["recycle_window"],
                "device": cfg["device"],
                "coarse_device": cfg.get("hybrid_coarse_device"),
                "status": record["status"],
                "verified": bool(record.get("all_problems_verified")),
                "sequence_seconds": record["sequence_seconds"],
                "preparation_inclusive_seconds": record["preparation_inclusive_seconds"],
                **{k: split[k] for k in ("cg_iterations", "linear_seconds", "outer_iterations")},
            }
        )
    return rows


def build(summary_directory, output, screen_directories=(), plots=False):
    campaign, records = load_campaign(summary_directory)
    summary = campaign.get("summary") or {}
    population = summary.get("headline_population", ["jacobi", "reference"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    methods = {}
    for method in METHODS:
        verified = [r for r in records[method] if r["all_problems_verified"]]
        splits = [linear_split(r) for r in verified]
        checks = [accuracy(r) for r in verified]
        methods[method] = {
            "recorded": len(records[method]),
            "verified": len(verified),
            "median_sequence_seconds": median([r["sequence_seconds"] for r in verified]),
            "range_sequence_seconds": [
                min(r["sequence_seconds"] for r in verified),
                max(r["sequence_seconds"] for r in verified),
            ]
            if verified
            else None,
            "median_preparation_inclusive_seconds": median(
                [r["preparation_inclusive_seconds"] for r in verified]
            ),
            "median_stage_sum_plus_overheads_seconds": median(
                [
                    r["sequence_seconds"]
                    + (r.get("assembly") or {}).get("restart_overhead_seconds", 0.0)
                    for r in verified
                ]
            ),
            "median_linear_seconds": median([s["linear_seconds"] for s in splits]),
            "median_evaluation_seconds": median([s["evaluation_seconds"] for s in splits]),
            "median_other_seconds": median([s["other_seconds"] for s in splits]),
            "median_cg_iterations": median([s["cg_iterations"] for s in splits]),
            "median_outer_iterations": median([s["outer_iterations"] for s in splits]),
            "median_linear_fraction": median([s["linear_fraction"] for s in splits]),
            "maximum_kkt": max((c["maximum_kkt"] for c in checks), default=None),
            "maximum_original_residual": max(
                (c["maximum_original_residual"] for c in checks), default=None
            ),
            "deployed_rank_range": [
                min(c["deployed_rank_range"][0] for c in checks if c["deployed_rank_range"]),
                max(c["deployed_rank_range"][1] for c in checks if c["deployed_rank_range"]),
            ]
            if any(c["deployed_rank_range"] for c in checks)
            else None,
            "fallback_count": sum(c["fallback_count"] for c in checks),
            "peak_host_bytes": max(
                ((r["memory"].get("peak_host_rss_bytes") or 0) for r in verified), default=None
            ),
            "peak_gpu_bytes": max(
                ((r["memory"].get("peak_gpu_process_bytes") or 0) for r in verified), default=None
            ),
            "restart_counts": [(r.get("assembly") or {}).get("restart_count") for r in verified],
            "restart_overhead_seconds": [
                (r.get("assembly") or {}).get("restart_overhead_seconds") for r in verified
            ],
            "discarded_seconds": [
                (r.get("assembly") or {}).get("discarded_seconds") for r in verified
            ],
            "elapsed_wall_seconds": [_elapsed_wall((r.get("assembly") or {})) for r in verified],
            "per_position_median_seconds": [
                median([c["seconds"] for r in verified for c in r["cases"] if c["position"] == p])
                for p in range(len(verified[0]["cases"]))
            ]
            if verified
            else [],
        }
    ratios = None
    reference = methods["reference"]
    alternatives = [m for m in population if m != "reference"]
    if summary.get("all_declared_sequences_verified") and reference["median_sequence_seconds"]:
        fastest = min(alternatives, key=lambda m: methods[m]["median_sequence_seconds"])
        alt = methods[fastest]
        floor = reference["median_evaluation_seconds"] + reference["median_other_seconds"]
        ratios = {
            "fastest_tested_alternative": fastest,
            "wall_ratio": alt["median_sequence_seconds"] / reference["median_sequence_seconds"],
            "wall_ratio_with_restart_overheads": alt["median_stage_sum_plus_overheads_seconds"]
            / reference["median_stage_sum_plus_overheads_seconds"],
            "wall_ratio_preparation_inclusive": alt["median_preparation_inclusive_seconds"]
            / reference["median_preparation_inclusive_seconds"],
            "inner_solve_ratio": alt["median_linear_seconds"] / reference["median_linear_seconds"],
            "cg_iteration_ratio": alt["median_cg_iterations"] / reference["median_cg_iterations"],
            "deflation_immune_floor_seconds": floor,
            "amdahl_ceiling": alt["median_sequence_seconds"] / floor if floor else None,
            "scope": "Medians over verified complete sequences of the headline population. The floor is the reference method's evaluation and other time, which no inner solver changes.",
        }
    screen = load_screen(screen_directories)
    report = {
        "schema": SCHEMA,
        "generator_environment": environment(),
        "campaign_summary_schema": campaign.get("schema"),
        "headline_population": population,
        "all_declared_sequences_verified": summary.get("all_declared_sequences_verified"),
        "fastest_tested_alternative_over_reference": summary.get(
            "fastest_tested_alternative_over_reference"
        ),
        "methods": methods,
        "ratios": ratios,
        "screen": screen,
        "chains": campaign.get("chains"),
    }
    write_report(output / "summary.json", report)
    write_tables(report, output)
    if plots:
        plot_positions(report, output)
    return report


def _elapsed_wall(assembly):
    import datetime

    try:
        started = datetime.datetime.fromisoformat(assembly["started_utc"])
        finished = datetime.datetime.fromisoformat(assembly["finished_utc"])
    except (KeyError, ValueError, TypeError):
        return None
    return (finished - started).total_seconds()


def _fmt(value, spec=".1f", missing="---"):
    return missing if value is None else format(value, spec)


def write_tables(report, output):
    end = r" \\"
    complete, split, accuracy_rows, memory_rows, execution_rows = [], [], [], [], []
    for method in METHODS:
        m = report["methods"][method]
        span = m["range_sequence_seconds"]
        timing = (
            "---"
            if m["median_sequence_seconds"] is None
            else f"{m['median_sequence_seconds']:.0f} [{span[0]:.0f}, {span[1]:.0f}]"
        )
        complete.append(
            " & ".join(
                [
                    LABELS[method],
                    f"{m['verified']}/{m['recorded']}",
                    timing,
                    _fmt(m["median_preparation_inclusive_seconds"], ".0f"),
                    _fmt(m["median_cg_iterations"], ".0f"),
                    _fmt(m["median_outer_iterations"], ".0f"),
                ]
            )
            + end
        )
        split.append(
            " & ".join(
                [
                    LABELS[method],
                    _fmt(m["median_linear_seconds"], ".0f"),
                    _fmt(m["median_evaluation_seconds"], ".0f"),
                    _fmt(m["median_other_seconds"], ".0f"),
                    _fmt(
                        None
                        if m["median_linear_fraction"] is None
                        else 100 * m["median_linear_fraction"],
                        ".1f",
                    ),
                ]
            )
            + end
        )
        rank = m["deployed_rank_range"]
        accuracy_rows.append(
            " & ".join(
                [
                    LABELS[method],
                    _fmt(m["maximum_original_residual"], ".2e"),
                    _fmt(m["maximum_kkt"], ".2e"),
                    "---" if rank is None else "--".join(map(str, rank)),
                    str(m["fallback_count"]),
                ]
            )
            + end
        )
        memory_rows.append(
            " & ".join(
                [
                    LABELS[method],
                    _fmt(
                        None if m["peak_host_bytes"] is None else m["peak_host_bytes"] / 2**30,
                        ".2f",
                    ),
                    _fmt(
                        None if m["peak_gpu_bytes"] is None else m["peak_gpu_bytes"] / 2**30, ".2f"
                    ),
                ]
            )
            + end
        )
        restarts = [r for r in m["restart_counts"] if r is not None]
        overheads = [r for r in m["restart_overhead_seconds"] if r is not None]
        walls = [r for r in m["elapsed_wall_seconds"] if r is not None]
        execution_rows.append(
            " & ".join(
                [
                    LABELS[method],
                    "---" if not restarts else f"{min(restarts)}--{max(restarts)}",
                    _fmt(median(overheads), ".0f"),
                    _fmt(median(walls) / 3600 if walls else None, ".1f"),
                ]
            )
            + end
        )
    ratio_rows = []
    if report["ratios"]:
        r = report["ratios"]
        for label, key, spec in (
            ("Complete wall time", "wall_ratio", ".2f"),
            ("Wall time with restart overheads", "wall_ratio_with_restart_overheads", ".2f"),
            ("Preparation-inclusive time", "wall_ratio_preparation_inclusive", ".2f"),
            ("Inner linear-solve time", "inner_solve_ratio", ".2f"),
            ("CG iterations", "cg_iteration_ratio", ".2f"),
            ("Amdahl ceiling (floor-limited)", "amdahl_ceiling", ".2f"),
        ):
            ratio_rows.append(f"{label} & {_fmt(r[key], spec)}" + end)
    screen_rows = []
    for row in report["screen"]:
        screen_rows.append(
            " & ".join(
                [
                    LABELS[row["method"]],
                    str(row["rank"]),
                    row["device"]
                    + ("" if not row["coarse_device"] else f" ({row['coarse_device']} coarse)"),
                    "verified" if row["verified"] else row["status"],
                    _fmt(row["preparation_inclusive_seconds"], ".0f"),
                    _fmt(row["cg_iterations"], ".0f"),
                ]
            )
            + end
        )
    positions = []
    count = max(
        (len(m["per_position_median_seconds"]) for m in report["methods"].values()), default=0
    )
    for position in range(count):
        cells = [str(position)]
        for method in METHODS:
            values = report["methods"][method]["per_position_median_seconds"]
            cells.append(_fmt(values[position] if position < len(values) else None, ".0f"))
        positions.append(" & ".join(cells) + end)
    for name, rows in (
        ("complete_rows.tex", complete),
        ("split_rows.tex", split),
        ("ratio_rows.tex", ratio_rows),
        ("accuracy_rows.tex", accuracy_rows),
        ("memory_rows.tex", memory_rows),
        ("execution_rows.tex", execution_rows),
        ("screen_rows.tex", screen_rows),
        ("target_rows.tex", positions),
    ):
        with atomic_output(output / name) as stream:
            stream.write("\n".join(rows) + "\n")


def plot_positions(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    for method in METHODS:
        values = report["methods"][method]["per_position_median_seconds"]
        if values:
            cumulative = np.cumsum(values) / 3600
            axis.plot(range(1, len(values) + 1), cumulative, marker="o", label=LABELS[method])
    axis.set(xlabel="Completed problems", ylabel="Median cumulative time (h)")
    axis.grid(alpha=0.25)
    axis.legend()
    for extension in ("pdf", "png"):
        figure.savefig(output / f"cumulative.{extension}", dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, required=True, help="coupled_campaign report output"
    )
    parser.add_argument("--screen", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    build(args.summary, args.output, args.screen, args.plot)


if __name__ == "__main__":
    main()
