"""Summarize matched transfer replays without pooling them with optimization timings."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .benchmark_cht_presentation import number, save_figure
from .reporting import atomic_output, environment, write_report
from .tutorials import Records


METHODS = {"full_reference": "Full reference", "sequential_transfer": "Sequential transfer"}
COLORS = {"full_reference": "#0072B2", "sequential_transfer": "#D55E00"}


def summarize(result):
    specification = result["specification"]
    c = specification["source_protocol"]["controls"]
    rows = result["rows"]
    if not rows:
        raise ValueError("Transfer summary requires at least one recorded kernel")
    summary = []
    for method in METHODS:
        samples = [entry for row in rows for entry in row["methods"][method]["repetitions"]]
        accepted = [
            s for s in samples if s["status"] == "converged" and s["original_residual"] <= c["rtol"]
        ]
        diagnostics = [row["methods"][method]["diagnostics"] for row in rows]
        for row in rows:
            if row["one_step_identity_error"] > 1e-12:
                raise ValueError("Transfer identity failed its stated check")
            group = row["methods"][method]
            for sample in group["repetitions"]:
                if not np.isclose(
                    sample["transfer_seconds"] + sample["kernel_seconds"],
                    sample["total_seconds"],
                    rtol=1e-9,
                    atol=1e-9,
                ):
                    raise ValueError("Transfer timing components overlap")
            verified = [
                s
                for s in group["repetitions"]
                if s["status"] == "converged" and s["original_residual"] <= c["rtol"]
            ]
            if len(verified) != group["accepted_repetitions"]:
                raise ValueError("Transfer acceptance count differs from its raw repetitions")
            all_accepted = len(verified) == specification["repetitions_per_kernel"]
            if group["all_repetitions_accepted"] != all_accepted:
                raise ValueError("Transfer acceptance flag differs from its declared repetitions")
            if not verified and group["accepted_median_seconds"] is not None:
                raise ValueError("A failed kernel cannot have an accepted-solve median")
            if verified and not np.isclose(
                np.median([s["total_seconds"] for s in verified]),
                group["accepted_median_seconds"],
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ValueError("Transfer median differs from its timed repetitions")
        if any(d["status"] != "verified" for d in diagnostics):
            raise ValueError("Requested energy summary requires matched, verified diagnostic ranks")
        for diagnostic in diagnostics:
            if not np.isclose(
                diagnostic["coarse_removed_energy_fraction"]
                + diagnostic["remaining_energy_fraction"],
                1,
                rtol=1e-7,
                atol=1e-8,
            ):
                raise ValueError("Coarse energy fractions fail their decomposition")
        spectra = [d for d in diagnostics if d.get("effective_condition") is not None]
        later = [row for row in rows if row["query"] > rows[0]["query"]]
        complete = (
            result.get("complete", result.get("success", False))
            and len(samples) == len(rows) * specification["repetitions_per_kernel"]
            and len(accepted) == len(samples)
        )
        summary.append(
            {
                "problem": c["problem"],
                "n": c["n"],
                "slabs": c["slabs"] if c["problem"] == "transient" else 1,
                "device": specification["device"],
                "method": method,
                "kernels": len(rows),
                "accepted_repetitions": len(accepted),
                "declared_repetitions": len(rows) * specification["repetitions_per_kernel"],
                "all_accepted": complete,
                "replay_complete": result.get("complete", result.get("success", False)),
                "statuses": dict(Counter(s["status"] for s in samples)),
                "newly_active": sum(row["newly_active"] for row in rows),
                "newly_inactive": sum(row["newly_inactive"] for row in rows),
                "after_first_query_newly_active": sum(row["newly_active"] for row in later),
                "after_first_query_newly_inactive": sum(row["newly_inactive"] for row in later),
                "after_first_query_median_coarse_energy_removed": float(
                    np.median(
                        [
                            row["methods"][method]["diagnostics"]["coarse_removed_energy_fraction"]
                            for row in later
                        ]
                    )
                )
                if later
                else None,
                "requested_rank": c["rank"],
                "deployed_rank_min": min((s["deployed_rank"] for s in samples), default=None),
                "deployed_rank_max": max((s["deployed_rank"] for s in samples), default=None),
                "fallbacks": sum(s["fallback_reason"] is not None for s in samples),
                "sum_kernel_median_iterations": float(
                    sum(
                        np.median([s["iterations"] for s in row["methods"][method]["repetitions"]])
                        for row in rows
                    )
                )
                if complete
                else None,
                "sum_kernel_median_seconds": float(
                    sum(row["methods"][method]["accepted_median_seconds"] for row in rows)
                )
                if complete
                else None,
                "median_coarse_energy_removed": float(
                    np.median([d["coarse_removed_energy_fraction"] for d in diagnostics])
                ),
                "median_original_preconditioned_condition": float(
                    np.median([d["original_preconditioned_condition"] for d in spectra])
                )
                if spectra
                else None,
                "median_effective_condition": float(
                    np.median([d["effective_condition"] for d in spectra])
                )
                if spectra
                else None,
                "full_space_corrections": sum(
                    "effective_condition" in d and d["effective_condition"] is None
                    for d in diagnostics
                ),
                "maximum_original_residual": max(
                    (s["original_residual"] for s in accepted), default=None
                ),
            }
        )
    return summary


def figure(results, selected, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, len(selected), figsize=(3.4 * len(selected), 4.8), squeeze=False)
    for column, name in enumerate(selected):
        result = results[name]
        c = result["specification"]["source_protocol"]["controls"]
        rows, x = result["rows"], np.arange(1, len(result["rows"]) + 1)
        axes[0, column].plot(
            x, [r["newly_inactive"] for r in rows], color="#0072B2", label="Released"
        )
        axes[0, column].plot(
            x, [r["newly_active"] for r in rows], color="#D55E00", label="Activated"
        )
        axes[0, column].set_ylabel("Changed constraints")
        for method in METHODS:
            axes[1, column].plot(
                x,
                [
                    r["methods"][method]["diagnostics"]["coarse_removed_energy_fraction"]
                    for r in rows
                ],
                color=COLORS[method],
                label=METHODS[method],
                lw=1.3,
            )
        for axis in axes[:, column]:
            axis.set_xlabel("Inactive solve in the recorded sequence")
            axis.grid(alpha=0.2)
        axes[1, column].set(ylabel="Initial energy error removed", ylim=(0, 1.02))
        title = f"{c['problem'].capitalize()}, " + rf"$ {c['n']}^3$"
        if c["problem"] == "transient":
            title += f", {c['slabs']} slabs"
        axes[0, column].set_title(title)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output, "transfer_and_energy")
    plt.close(fig)


def run(records_roots, output, figure_studies):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results, rows, inputs = {}, [], {}
    for root in records_roots:
        name = Path(root).name
        if name in results:
            raise ValueError("Replay identifiers must be unique")
        records = Records(root)
        result, protocol = records.load("results.json"), records.load("protocol.json")
        if result["specification"] != protocol:
            raise ValueError("Replay result and declared protocol differ")
        results[name], inputs[name] = result, records.manifest
        rows.extend({"study": name, **row} for row in summarize(result))
    if set(figure_studies) - set(results):
        raise ValueError("Every plotted replay requires its complete records")
    with atomic_output(output / "transfer_rows.tex") as stream:
        for row in rows:
            grid = rf"$ {row['n']}^3$" + (rf"$\times {row['slabs']}$" if row["slabs"] > 1 else "")
            stream.write(
                f"{grid} & {row['kernels']} & {METHODS[row['method']]} & "
                f"{row['deployed_rank_min']}--{row['deployed_rank_max']} & "
                f"{number(row['median_coarse_energy_removed'])} & "
                f"{number(row['sum_kernel_median_iterations'], 0)} & "
                f"{number(row['sum_kernel_median_seconds'])}" + r" \\" + "\n"
            )
    if figure_studies:
        figure(results, figure_studies, output)
    report = {
        "protocol": "transfer-summary-v1",
        "generator_environment": environment(),
        "input_sha256": inputs,
        "rows": rows,
        "figure_studies": figure_studies,
        "scope": "matched kernels replayed from complete PDAS traces; iteration and time aggregates sum per-kernel medians, not independently repeated complete-sequence times; initial guesses are zero; activation and release counts compare consecutive recorded inactive systems and exclude the first shared restriction",
    }
    write_report(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figure-studies", nargs="*", default=[])
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
