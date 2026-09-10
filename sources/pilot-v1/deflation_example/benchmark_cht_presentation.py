"""Generate tables and figures from verified complete CHT records."""

import argparse
from pathlib import Path

import numpy as np

from .benchmark_cht import build_model, desired_and_load, unpack_mask
from .benchmark_cht_campaign import campaign_specification
from .benchmark_cht_report import load_study, summarize
from .benchmark_extended import digest
from .reporting import atomic_output, environment, write_report


LABELS = {"jacobi": "Jacobi-CG", "reference": "Reference", "recycling": "Recycling", "amgx": "AmgX"}
COLORS = {"jacobi": "#666666", "reference": "#0072B2", "recycling": "#CC79A7", "amgx": "#D55E00"}


def number(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def requested_rank(row):
    if row["method"] in {"jacobi", "amgx"}:
        return 0
    return row["recycle_rank"] if row["method"] == "recycling" else row["reference_rank"]


def save_figure(figure, output, name):
    figure.savefig(output / (name + ".pdf"), bbox_inches="tight")
    figure.savefig(output / (name + ".png"), bbox_inches="tight", dpi=180)


def median_sequence(sequences):
    accepted = sorted((s for s in sequences if s["success"]), key=lambda s: s["total_seconds"])
    if not accepted or len(accepted) % 2 == 0:
        return None
    return accepted[len(accepted) // 2]


def timing_groups(sequence):
    parts = sequence["components_seconds"]
    setup = {
        "assembly",
        "reference_construction",
        "solver_resources",
        "reference_and_recycling_transfer",
        "kernel_basis_processing",
        "kernel_coarse_or_hierarchy_setup",
    }
    iteration = {"kernel_iteration"}
    transfer = {"kernel_conversion", "kernel_upload", "kernel_download"}
    used = setup | iteration | transfer
    grouped = {
        "Construction and setup": sum(parts.get(k, 0.0) for k in setup),
        "Iteration": sum(parts.get(k, 0.0) for k in iteration),
        "Conversion and transfers": sum(parts.get(k, 0.0) for k in transfer),
        "Verification and remaining work": sum(v for k, v in parts.items() if k not in used),
    }
    if not np.isclose(sum(grouped.values()), sequence["total_seconds"], rtol=1e-9, atol=1e-9):
        raise ValueError("Presentation timing groups do not sum to the complete interval")
    return grouped


def memory_envelopes(rows):
    policy = campaign_specification()["memory_comparison"]
    groups = {}
    for row in rows:
        if row["warm_start"] == "outer_inner":
            key = (row["problem"], row["n"], row["slabs"], row["method"])
            groups.setdefault(key, []).append(row)
    result = []
    for (problem, n, slabs, method), candidates in groups.items():
        for gpu_budget in policy["gpu_budgets_bytes"]:
            feasible = [
                row
                for row in candidates
                if row["accepted_sequences"] == row["declared_sequences"]
                and row["verified_memory_sequences"] == row["declared_sequences"]
                and row["peak_host_rss_bytes"] <= policy["host_budget_bytes"]
                and row["peak_gpu_process_bytes"] <= gpu_budget
            ]
            best = min(feasible, key=lambda row: row["median_seconds"]) if feasible else None
            result.append(
                {
                    "problem": problem,
                    "n": n,
                    "slabs": slabs,
                    "method": method,
                    "gpu_budget_bytes": gpu_budget,
                    "host_budget_bytes": policy["host_budget_bytes"],
                    "measured_requested_ranks": sorted({requested_rank(row) for row in candidates}),
                    "feasible_choices": len(feasible),
                    "selected_rank": None if best is None else requested_rank(best),
                    "selected_deployed_rank_range": None
                    if best is None
                    else [best["deployed_rank_min"], best["deployed_rank_max"]],
                    "selected_fallbacks": None if best is None else best["fallbacks"],
                    "median_seconds": None if best is None else best["median_seconds"],
                    "scope": "finite measured-rank envelope under common sampled-memory screens",
                }
            )
    return result


def memory_budget_table(envelopes, output):
    """Show the measured rank-study envelope, including infeasible choices."""
    with atomic_output(output / "memory_budget_rows.tex") as stream:
        for row in envelopes:
            if (row["problem"], row["n"], row["slabs"]) not in {
                ("steady", 32, 1),
                ("transient", 24, 8),
            }:
                continue
            rank = "--" if row["selected_rank"] is None else str(row["selected_rank"])
            stream.write(
                f"{row['problem'].capitalize()} & {row['gpu_budget_bytes'] / 2**30:.0f} & "
                f"{LABELS[row['method']]} & {rank} & {number(row['median_seconds'])}"
                + r" \\"
                + "\n"
            )


def table_rows(rows, output, suffix=""):
    with atomic_output(output / f"complete_rows{suffix}.tex") as stream:
        for row in rows:
            grid = rf"$ {row['n']}^3$"
            if row["problem"] == "transient":
                grid += rf"$\times {row['slabs']}$"
            rank = requested_rank(row)
            warm = "Warm" if row["warm_start"] == "outer_inner" else "Cold"
            stream.write(
                f"{grid} & {warm} & {LABELS[row['method']]} & {rank} & "
                f"{row['accepted_sequences']}/{row['declared_sequences']} & "
                f"{number(row['median_seconds'])} & "
                f"{number(row['minimum_seconds'])}--{number(row['maximum_seconds'])} & "
                f"{number(row['median_outer_iterations'], 0)} & "
                f"{number(row['median_inner_iterations'], 0)}" + r" \\" + "\n"
            )
    with atomic_output(output / f"memory_rows{suffix}.tex") as stream:
        for row in rows:
            gpu, host = row["peak_gpu_process_bytes"], row["peak_host_rss_bytes"]
            warm = "Warm" if row["warm_start"] == "outer_inner" else "Cold"
            stream.write(
                f"$ {row['n']}^3$ & {row['slabs']} & {warm} & "
                f"{LABELS[row['method']]} & "
                f"{requested_rank(row)} & "
                f"{number(None if host is None else host / 2**30)} & "
                f"{number(None if gpu is None else gpu / 2**30)} & "
                f"{number(row['maximum_memory_sampling_gap_seconds'])}" + r" \\" + "\n"
            )
    with atomic_output(output / f"preparation_rows{suffix}.tex") as stream:
        for row in rows:
            grid = rf"$ {row['n']}^3$"
            if row["problem"] == "transient":
                grid += rf"$\times {row['slabs']}$"
            warm = "Warm" if row["warm_start"] == "outer_inner" else "Cold"
            stream.write(
                f"{grid} & {warm} & {LABELS[row['method']]} & {requested_rank(row)} & "
                f"{number(row['median_seconds'])} & "
                f"{number(row['median_preparation_inclusive_seconds'])}" + r" \\" + "\n"
            )
    groups = {}
    for row in rows:
        key = (row["study"], row["warm_start"])
        groups.setdefault(key, {})[row["method"]] = row
    with atomic_output(output / f"comparison_rows{suffix}.tex") as stream:
        for (_, warm), methods in groups.items():
            first = next(iter(methods.values()))
            grid = rf"$ {first['n']}^3$"
            if first["problem"] == "transient":
                grid += rf"$\times {first['slabs']}$"
            rank = next(
                (requested_rank(row) for row in methods.values() if row["method"] == "reference"),
                0,
            )
            entries = []
            for method in LABELS:
                row = methods.get(method)
                if row is None:
                    entries.append("--")
                    continue
                value = number(row["median_seconds"])
                if row["accepted_sequences"] != row["declared_sequences"]:
                    value += f" ({row['accepted_sequences']}/{row['declared_sequences']})"
                entries.append(value)
            stream.write(
                f"{first['problem'].capitalize()} & {grid} & {rank} & "
                f"{'Warm' if warm == 'outer_inner' else 'Cold'} & "
                + " & ".join(entries)
                + r" \\"
                + "\n"
            )
    if not suffix:
        for problem in ("steady", "transient"):
            table_rows([row for row in rows if row["problem"] == problem], output, "_" + problem)
        with atomic_output(output / "rank_status_rows.tex") as stream:
            for row in rows:
                if row["method"] not in {"reference", "recycling"}:
                    continue
                grid = rf"$ {row['n']}^3$"
                if row["problem"] == "transient":
                    grid += rf"$\times {row['slabs']}$"
                warm = "Warm" if row["warm_start"] == "outer_inner" else "Cold"
                stream.write(
                    f"{grid} & {warm} & {LABELS[row['method']]} & {requested_rank(row)} & "
                    f"{row['deployed_rank_min']}--{row['deployed_rank_max']} & "
                    f"{row['fallbacks']}" + r" \\" + "\n"
                )
        with atomic_output(output / "resolution_rows.tex") as stream:
            for (_, warm), methods in groups.items():
                first = next(iter(methods.values()))
                if (
                    warm != "outer_inner"
                    or first["reference_rank"] != 100
                    or first["recycle_rank"] != 100
                    or set(methods) != set(LABELS)
                ):
                    continue
                grid = rf"$ {first['n']}^3$"
                if first["problem"] == "transient":
                    grid += rf"$\times {first['slabs']}$"
                values = []
                for method in LABELS:
                    row = methods[method]
                    value = number(row["median_seconds"])
                    if row["accepted_sequences"] != row["declared_sequences"]:
                        value += f" ({row['accepted_sequences']}/{row['declared_sequences']})"
                    values.append(value)
                stream.write(
                    f"{first['problem'].capitalize()} & {grid} & "
                    + " & ".join(values)
                    + r" \\"
                    + "\n"
                )


def cumulative_figure(studies, selected, output, plt):
    columns = 2 if len(selected) > 1 else 1
    count = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(count, columns, figsize=(6.8, 2.6 * count), squeeze=False)
    for axis, name in zip(axes.ravel(), selected, strict=False):
        protocol, sequences = studies[name]
        c, medians = protocol["controls"], {}
        for method in c["methods"]:
            accepted = [
                s
                for s in sequences
                if s["method"] == method and s["warm_start"] == "outer_inner" and s["success"]
            ]
            label = f"{LABELS[method]} ({len(accepted)}/{c['repeats']})"
            if not accepted:
                axis.plot([], [], color=COLORS[method], label=label)
                continue
            curves = []
            for sequence in accepted:
                curve = [case["cumulative_seconds"] for case in sequence["cases"]]
                curve[-1] = sequence["total_seconds"]
                curves.append(curve)
                axis.plot(
                    range(1, c["targets"] + 1), curve, color=COLORS[method], alpha=0.18, lw=0.8
                )
            axis.plot(
                range(1, c["targets"] + 1),
                np.median(curves, axis=0),
                color=COLORS[method],
                lw=2 if method == "reference" else 1.4,
                label=label,
            )
            if len(accepted) == c["repeats"]:
                medians[method] = float(np.median([s["total_seconds"] for s in accepted]))
        title = f"{c['problem'].capitalize()}, " + rf"$ {c['n']}^3$"
        if c["problem"] == "transient":
            title += f", {c['slabs']} slabs"
        axis.set_title(title)
        axis.set(xlabel="Completed targets", ylabel="Complete cumulative time (s)")
        axis.set_xticks(
            np.unique(np.rint(np.linspace(1, c["targets"], min(5, c["targets"]))).astype(int))
        )
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="upper left")
        if "reference" in medians and "amgx" in medians:
            axis.text(
                0.98,
                0.03,
                f"AmgX/reference: {medians['amgx'] / medians['reference']:.2f}" + r"$\times$",
                transform=axis.transAxes,
                ha="right",
                fontsize=8,
            )
    for axis in axes.ravel()[len(selected) :]:
        axis.set_visible(False)
    figure.tight_layout()
    save_figure(figure, output, "complete_sequences")
    plt.close(figure)


def component_figure(studies, selected, output, plt):
    figure, axes = plt.subplots(1, len(selected), figsize=(3.4 * len(selected), 3.2), squeeze=False)
    records, colors = [], ["#56B4E9", "#0072B2", "#E69F00", "#999999"]
    for axis, name in zip(axes.ravel(), selected, strict=True):
        protocol, sequences = studies[name]
        c, methods = protocol["controls"], protocol["controls"]["methods"]
        representatives = [
            median_sequence(
                [s for s in sequences if s["method"] == m and s["warm_start"] == "outer_inner"]
            )
            for m in methods
        ]
        bottom = np.zeros(len(methods))
        groups = [None if s is None else timing_groups(s) for s in representatives]
        names = next((list(g) for g in groups if g is not None), [])
        for key, color in zip(names, colors, strict=False):
            values = [np.nan if g is None else g[key] for g in groups]
            axis.bar(range(len(methods)), values, bottom=bottom, color=color, label=key)
            bottom += values
        counts = [
            sum(
                s["success"] and s["method"] == m and s["warm_start"] == "outer_inner"
                for s in sequences
            )
            for m in methods
        ]
        axis.set_xticks(
            range(len(methods)),
            [f"{LABELS[m]} ({k}/{c['repeats']})" for m, k in zip(methods, counts, strict=True)],
            rotation=25,
            ha="right",
        )
        for index, group in enumerate(groups):
            if group is None:
                axis.text(index, 0, "n/a", ha="center", va="bottom")
        axis.set_ylabel("Complete sequence time (s)")
        title = f"{c['problem'].capitalize()}, " + rf"$ {c['n']}^3$"
        if c["problem"] == "transient":
            title += f", {c['slabs']} slabs"
        axis.set_title(title)
        for method, sequence, group in zip(methods, representatives, groups, strict=True):
            records.append(
                {
                    "study": name,
                    "method": method,
                    "representative_repetition": None
                    if sequence is None
                    else sequence["repetition"],
                    "complete_seconds": None if sequence is None else sequence["total_seconds"],
                    "groups_seconds": group,
                    "components_seconds": None
                    if sequence is None
                    else sequence["components_seconds"],
                }
            )
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, fontsize=7, loc="upper center", ncol=2)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(figure, output, "complete_components")
    plt.close(figure)
    return records


def active_sets_figure(studies, selected, output, plt):
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(len(selected), 3, figsize=(7.4, 3.0 * len(selected)), squeeze=False)
    records = []
    for axes_row, name in zip(axes, selected, strict=True):
        protocol, sequences = studies[name]
        c = protocol["controls"]
        sequence = next(
            s
            for s in sequences
            if s["method"] == "reference" and s["warm_start"] == "outer_inner" and s["success"]
        )
        model = build_model(c)
        n, slab_count = c["n"], c["slabs"] if c["problem"] == "transient" else 1
        slice_index = n // 2
        slice_coordinate = (slice_index + 1) / (n + 1)
        coordinates = np.arange(1, n + 1) / (n + 1)
        choices = (
            [(q, 0) for q in (0, (c["targets"] - 1) // 2, c["targets"] - 1)]
            if c["problem"] == "steady"
            else [((c["targets"] - 1) // 2, t) for t in (0, slab_count // 2 - 1, slab_count - 1)]
        )
        for axis, (query, slab) in zip(axes_row, choices, strict=True):
            case = sequence["cases"][query]
            desired, _ = desired_and_load(model, protocol["targets"][query])
            if digest(desired) != case["target_sha256"]:
                raise ValueError("Figure target differs from its optimization record")
            target = desired.reshape(slab_count, n, n, n)[slab, :, slice_index, :].T
            active = unpack_mask(case["final_active_bits"], model.H.shape[0])
            mask = active.reshape(slab_count, n, n, n)[slab, :, slice_index, :].T
            mesh = axis.pcolormesh(
                coordinates,
                coordinates,
                target,
                shading="nearest",
                cmap="viridis",
                vmin=0,
                vmax=1.05,
            )
            if mask.any() and not mask.all():
                axis.contour(
                    coordinates, coordinates, mask, levels=[0.5], colors="white", linewidths=1.0
                )
            axis.axhline(0.5, color="#D55E00", linestyle="--", lw=1)
            title = rf"$\theta={protocol['targets'][query]['theta']:.2f}$"
            if c["problem"] == "transient":
                title += rf", $t={model.times[slab]:.4f}$"
            if axis is axes_row[0]:
                title = (
                    f"{c['problem'].capitalize()}, "
                    + rf"$ {n}^3$, $x_2={slice_coordinate:.4f}$"
                    + "\n"
                    + title
                )
            axis.set(title=title, xlabel=r"$x_1$", ylabel=r"$x_3$", aspect="equal")
            records.append(
                {
                    "study": name,
                    "source_repetition": sequence["repetition"],
                    "query": query,
                    "time_slab_one_based": slab + 1,
                    "x2_index_zero_based": slice_index,
                    "x2_coordinate": slice_coordinate,
                    "n": n,
                    "target_sha256": case["target_sha256"],
                    "active_slice_count": int(mask.sum()),
                }
            )
    figure.legend(
        [
            Line2D([0], [0], color="white", markeredgecolor="black", marker="s"),
            Line2D([0], [0], color="#D55E00", linestyle="--"),
        ],
        ["Active-set boundary", "Solid–fluid interface"],
        loc="lower center",
        ncol=2,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.08, right=0.85, bottom=0.14, top=0.88, hspace=0.65, wspace=0.4)
    color_axis = figure.add_axes([0.90, 0.23, 0.025, 0.55])
    figure.colorbar(mesh, cax=color_axis, label="Desired temperature")
    save_figure(figure, output, "targets_and_constraints")
    plt.close(figure)
    return records


def scaling_figure(rows, output, plt):
    figure, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), squeeze=False)
    for column, problem in enumerate(("steady", "transient")):
        selected = [
            row
            for row in rows
            if row["problem"] == problem
            and row["reference_rank"] == 100
            and row["warm_start"] == "outer_inner"
            and (problem == "steady" or row["slabs"] == 8)
        ]
        for method in LABELS:
            samples = sorted(
                (row for row in selected if row["method"] == method), key=lambda row: row["n"]
            )
            accepted = [
                row for row in samples if row["accepted_sequences"] == row["declared_sequences"]
            ]
            grids = [row["n"] for row in accepted]
            seconds = [row["median_seconds"] for row in accepted]
            if not accepted:
                axes[0, column].plot([], [], color=COLORS[method], label=LABELS[method])
                continue
            axes[0, column].errorbar(
                grids,
                seconds,
                yerr=[
                    [row["median_seconds"] - row["minimum_seconds"] for row in accepted],
                    [row["maximum_seconds"] - row["median_seconds"] for row in accepted],
                ],
                color=COLORS[method],
                marker="o",
                ms=4,
                capsize=3,
                label=LABELS[method],
            )
            axes[1, column].plot(
                grids,
                [row["peak_gpu_process_bytes"] / 2**30 for row in accepted],
                color=COLORS[method],
                marker="o",
                ms=4,
            )
            for row in samples:
                if row["accepted_sequences"] != row["declared_sequences"]:
                    axes[0, column].annotate(
                        f"{LABELS[method]}: {row['accepted_sequences']}/{row['declared_sequences']}",
                        (row["n"], 0.95),
                        xycoords=("data", "axes fraction"),
                        color=COLORS[method],
                        ha="center",
                        va="top",
                        fontsize=6,
                    )
        axes[0, column].set_title(
            problem.capitalize() + (", 8 slabs" if problem == "transient" else "")
        )
        axes[0, column].set_ylabel("Complete sequence time (s)")
        axes[1, column].set_ylabel("Peak GPU process memory (GiB)")
        for axis in axes[:, column]:
            axis.set(xlabel="Interior nodes per spatial coordinate", yscale="log")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    save_figure(figure, output, "spatial_time_memory")
    plt.close(figure)


def rank_figure(rows, output, plt):
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for axis, (problem, n) in zip(axes, (("steady", 32), ("transient", 24)), strict=True):
        selected = [
            row
            for row in rows
            if row["problem"] == problem
            and row["n"] == n
            and row["warm_start"] == "outer_inner"
            and row["accepted_sequences"] == row["declared_sequences"]
            and (problem == "steady" or row["slabs"] == 8)
        ]
        for method in LABELS:
            samples = sorted(
                (row for row in selected if row["method"] == method), key=requested_rank
            )
            axis.plot(
                [row["peak_gpu_process_bytes"] / 2**30 for row in samples],
                [row["median_seconds"] for row in samples],
                "o-",
                color=COLORS[method],
                label=LABELS[method],
                ms=4,
            )
            for row in samples:
                if method in {"reference", "recycling"}:
                    axis.annotate(
                        str(requested_rank(row)),
                        (row["peak_gpu_process_bytes"] / 2**30, row["median_seconds"]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                    )
        axis.set(
            title=f"{problem.capitalize()}, "
            + rf"$ {n}^3$"
            + (", 8 slabs" if problem == "transient" else ""),
            xlabel="Sampled peak GPU process memory (GiB)",
            ylabel="Median complete time (s)",
            xscale="log",
        )
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    save_figure(figure, output, "rank_time_memory")
    plt.close(figure)


def accuracy_table(studies, output):
    rows = []
    with atomic_output(output / "accuracy_rows.tex") as stream:
        for method in LABELS:
            sequences = [
                s
                for _, population in studies.values()
                for s in population
                if s["method"] == method and s["success"]
            ]
            if not sequences:
                continue
            cases = [case for sequence in sequences for case in sequence["cases"]]
            components = ("stationarity", "primal", "dual", "complementarity", "projected_gradient")
            maxima = {key: max(case["kkt"][key] for case in cases) for key in components}
            residual = max(
                (inner["original_residual"] for case in cases for inner in case["inner"]),
                default=0.0,
            )
            rows.append(
                {
                    "method": method,
                    "accepted_sequences": len(sequences),
                    "accepted_targets": len(cases),
                    "maximum_kkt_components": maxima,
                    "maximum_original_residual": residual,
                }
            )
            stream.write(
                f"{LABELS[method]} & {len(sequences)} & "
                + " & ".join(f"{value:.2e}" for value in [residual, *maxima.values()])
                + r" \\"
                + "\n"
            )
    return rows


def run(records_roots, output, main_studies, physical_studies, component_studies=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    studies, rows, hashes = {}, [], {}
    for root in records_roots:
        name = Path(root).name
        if name in studies:
            raise ValueError("Study identifiers must be unique")
        protocol, sequences, source_hashes = load_study(root)
        studies[name] = (protocol, sequences)
        hashes[name] = source_hashes
        rows.extend({"study": name, **row} for row in summarize(protocol, sequences))
    rows.sort(
        key=lambda row: (
            row["problem"] != "steady",
            row["n"],
            row["slabs"],
            row["warm_start"] != "outer_inner",
            row["reference_rank"],
            row["recycle_rank"],
            list(LABELS).index(row["method"]),
        )
    )
    if component_studies is None:
        component_studies = main_studies[-2:]
    if not main_studies or set(main_studies + physical_studies + component_studies) - set(studies):
        raise ValueError("Every selected figure population must have verified records")
    table_rows(rows, output)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    cumulative_figure(studies, main_studies, output, plt)
    components = component_figure(studies, component_studies, output, plt)
    for name in component_studies:
        entries = [row for row in components if row["study"] == name]
        keys = sorted(
            {
                key
                for row in entries
                if row["components_seconds"] is not None
                for key in row["components_seconds"]
            }
        )
        with atomic_output(output / f"component_rows_{name}.tex") as stream:
            for key in keys:
                stream.write(
                    key.replace("kernel_", "kernel: ").replace("_", " ").capitalize()
                    + " & "
                    + " & ".join(
                        number(
                            None
                            if row["components_seconds"] is None
                            else row["components_seconds"].get(key, 0.0),
                            4,
                        )
                        for row in entries
                    )
                    + r" \\"
                    + "\n"
                )
            stream.write(
                r"\midrule Total & "
                + " & ".join(number(row["complete_seconds"], 4) for row in entries)
                + r" \\"
                + "\n"
            )
    slices = active_sets_figure(studies, physical_studies, output, plt) if physical_studies else []
    scaling_figure(rows, output, plt)
    rank_figure(rows, output, plt)
    envelopes = memory_envelopes(rows)
    memory_budget_table(envelopes, output)
    result = {
        "protocol": "complete-cht-presentation-v1",
        "generator_environment": environment(),
        "input_sha256": hashes,
        "rows": rows,
        "memory_envelopes": envelopes,
        "main_figure_studies": main_studies,
        "component_figure_studies": component_studies,
        "component_sequences": components,
        "physical_slices": slices,
        "accuracy": accuracy_table(studies, output),
        "scope": "all declared methods and outcomes in each loaded study; complete-sequence medians and ranges; component bars use actual median-total sequences",
    }
    write_report(output / "presentation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--main-studies", nargs="+", required=True)
    parser.add_argument("--physical-studies", nargs="*", default=[])
    parser.add_argument("--component-studies", nargs="+", default=None)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
