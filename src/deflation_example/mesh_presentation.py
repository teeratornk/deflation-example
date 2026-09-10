"""Regenerate mesh-study figures and acceptance summaries from saved records."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .reporting import write_report


def tetrahedral_slice(nodes, cells, values, coordinate=0.5, axis=1):
    """Intersect P1 tetrahedra with a coordinate plane and interpolate nodal values."""
    if axis not in (0, 1, 2) or not np.isfinite(coordinate):
        raise ValueError("Choose a finite coordinate and a spatial axis from 0 to 2")
    remaining = [i for i in range(3) if i != axis]
    points, triangles, samples, owners = [], [], [], []
    for owner, cell in enumerate(cells):
        xyz = nodes[cell]
        delta = xyz[:, axis] - coordinate
        if delta.min() > 1e-12 or delta.max() < -1e-12:
            continue
        vertices = []
        for i in range(4):
            if abs(delta[i]) <= 1e-12:
                vertices.append((xyz[i, remaining], values[cell[i]]))
        for i, j in combinations(range(4), 2):
            if delta[i] * delta[j] < -1e-24:
                t = delta[i] / (delta[i] - delta[j])
                vertices.append(
                    (
                        (1 - t) * xyz[i, remaining] + t * xyz[j, remaining],
                        (1 - t) * values[cell[i]] + t * values[cell[j]],
                    )
                )
        unique = {}
        for point, value in vertices:
            unique[tuple(np.round(point, 12))] = (point, value)
        vertices = list(unique.values())
        if len(vertices) < 3:
            continue
        xy = np.array([p for p, _ in vertices])
        center = xy.mean(axis=0)
        order = np.argsort(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
        offset = len(points)
        points.extend(xy[order])
        samples.extend([vertices[i][1] for i in order])
        for i in range(1, len(order) - 1):
            triangles.append([offset, offset + i, offset + i + 1])
            owners.append(owner)
    if not triangles:
        raise ValueError("The requested plane has no two-dimensional intersection with the mesh")
    return np.array(points), np.array(triangles), np.array(samples), np.array(owners)


def plot_fields(record_directory, output, query=0, slab=-1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import ListedColormap, BoundaryNorm

    directory = Path(record_directory)
    record = json.loads((directory / "record.json").read_text())
    if record["cases"][query]["status"] != "converged":
        raise ValueError("Optimization field figures require an accepted solve")
    with np.load(directory / "geometry.npz", allow_pickle=False) as mesh:
        nodes, cells, free, materials = (mesh[k] for k in ("nodes", "cells", "free", "materials"))
    values = np.zeros((len(nodes), 4))
    with np.load(directory / f"fields-{query}.npz", allow_pickle=False) as data:
        for j, name in enumerate(("desired", "state", "active", "control")):
            values[free, j] = data[name].reshape(-1, len(free))[slab]
    geometry = record["controls"]["geometry"]
    if nodes.shape[1] == 3:
        xy, triangles, values, owners = tetrahedral_slice(nodes, cells, values)
        label = "Engine: section at x2/L = 0.5; P1 interpolation on intersected tetrahedra"
        xlab, ylab = "x1/L", "x3/L"
        aspect = "equal"
    else:
        xy, triangles, owners = nodes[:, ::-1], cells[:, [0, 2, 1]], np.arange(len(cells))
        label = "Transformer: axisymmetric meridional section"
        xlab, ylab = "z (m)", "r (m)"
        aspect = "auto"
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    count = int(materials.max()) + 1
    palette = ListedColormap(["#66c2a5", "#a6d854", "#8da0cb"][:count])
    artist = axes[0, 0].tripcolor(
        triangulation,
        facecolors=materials[owners],
        cmap=palette,
        norm=BoundaryNorm(np.arange(count + 1) - 0.5, count),
        shading="flat",
    )
    axes[0, 0].set_title("Material regions")
    fig.colorbar(artist, ax=axes[0, 0], ticks=np.unique(materials), label="Material identifier")
    for axis, j, title, cmap in (
        (axes[0, 1], 0, "Desired temperature", "inferno"),
        (axes[0, 2], 1, "Optimized temperature", "inferno"),
        (axes[1, 0], 2, "Interpolated nodal active indicator", "Greys"),
        (axes[1, 1], 3, "Distributed control", "coolwarm"),
    ):
        limits = {"vmin": 0, "vmax": 1} if j == 2 else {}
        if j in (0, 1):
            limits = {"vmin": float(values[:, :2].min()), "vmax": float(values[:, :2].max())}
        if j == 3:
            maximum = max(float(np.max(np.abs(values[:, j]))), np.finfo(float).tiny)
            limits = {"vmin": -maximum, "vmax": maximum}
        artist = axis.tripcolor(
            triangulation,
            values[:, j],
            shading="gouraud",
            cmap=cmap,
            edgecolors="none",
            antialiased=False,
            **limits,
        )
        axis.set_title(title)
        fig.colorbar(artist, ax=axis, label="Dimensionless" if j != 2 else "0 to 1")
    axes[1, 2].triplot(triangulation, linewidth=0.15, color="0.25")
    axes[1, 2].set_title("Conforming mesh section")
    for axis in axes.flat:
        axis.set(xlabel=xlab, ylabel=ylab, aspect=aspect)
    label += f"; query {query + 1}"
    if record["controls"]["transient"]:
        n = record["controls"]["slabs"]
        index = slab % n
        label += f"; time level {index + 1}/{n}"
    fig.suptitle(label, fontsize=10)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return geometry


def summarize(directory):
    directory = Path(directory)
    controls = json.loads((directory / "protocol.json").read_text())
    attempts = json.loads((directory / "attempts.json").read_text())
    indexed = {}
    for attempt in attempts:
        key = (attempt["method"], attempt["repetition"])
        if (
            key in indexed
            or key[0] not in controls["methods"]
            or key[1] not in range(controls["repeats"])
        ):
            raise ValueError("Attempts contain a duplicate or undeclared method/repetition")
        indexed[key] = attempt
    target_hashes = None
    rows = []
    for method in controls["methods"]:
        records = []
        failures = []
        for repetition in range(controls["repeats"]):
            attempt = indexed.get((method, repetition))
            if attempt is None:
                failures.append("not_recorded")
                continue
            path = (directory / attempt["record"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("An attempt record must stay inside its comparison directory")
            if not path.exists():
                failures.append(attempt["status"])
                continue
            record = json.loads(path.read_text())
            if (
                record["method"] != method
                or record["repetition"] != repetition
                or record["controls"] != controls
            ):
                raise ValueError("Sequence labels or controls differ from the declared comparison")
            if record["success"]:
                if attempt["status"] != "completed":
                    failures.append(attempt["status"])
                    continue
                if len(record["cases"]) != controls["targets"]:
                    raise ValueError("An accepted sequence has an incorrect population")
                if len({c["target_sha256"] for c in record["cases"]}) != controls["targets"]:
                    raise ValueError("A sequence repeats a desired-temperature field")
                hashes = [c["target_sha256"] for c in record["cases"]]
                if target_hashes is not None and hashes != target_hashes:
                    raise ValueError("Methods or repetitions use different target sequences")
                target_hashes = hashes
                if [c["query"] for c in record["cases"]] != list(range(controls["targets"])):
                    raise ValueError("Query indices do not match the declared sequence")
                if any(
                    not np.isfinite(t) or t < -1e-9 for t in record["components_seconds"].values()
                ):
                    raise ValueError("Timing components must be finite and nonnegative")
                if not np.isclose(
                    sum(record["components_seconds"].values()), record["seconds"], rtol=1e-12
                ):
                    raise ValueError("Timing components do not sum to complete time")
                for case in record["cases"]:
                    if (
                        case["status"] != "converged"
                        or any(
                            v is None or not np.isfinite(v) or v > controls["kkt_tolerance"]
                            for v in case["kkt"].values()
                        )
                        or any(
                            row["status"] != "converged"
                            or row["original_residual"] is None
                            or not np.isfinite(row["original_residual"])
                            or row["original_residual"] > controls["rtol"]
                            for row in case["inner"]
                        )
                    ):
                        raise ValueError("A reported success fails the declared accuracy checks")
                if record.get("initial_setup_seconds") is not None:
                    timestamps = (
                        [record["initial_setup_seconds"]]
                        + [c["cumulative_seconds"] for c in record["cases"]]
                        + [record["seconds"]]
                    )
                    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
                        raise ValueError(
                            "Cumulative sequence timestamps must be finite and increasing"
                        )
                records.append(record)
            else:
                failures.append(
                    record.get("failure") or [c["status"] for c in record.get("cases", [])]
                )
        times = [r["seconds"] for r in records]
        rows.append(
            {
                "method": method,
                "accepted": len(records),
                "requested": controls["repeats"],
                "failures": failures,
                "complete_seconds": times,
                "median_seconds": float(np.median(times)) if times else None,
                "inner_iterations": [
                    sum(c["inner_iterations"] for c in r["cases"]) for r in records
                ],
                "outer_iterations": [
                    sum(c["outer_iterations"] for c in r["cases"]) for r in records
                ],
                "peak_host_bytes": [
                    r["memory"]["peak_host_rss_bytes"]
                    for r in records
                    if (r.get("memory") or {}).get("complete")
                ],
                "peak_gpu_bytes": [
                    r["memory"]["peak_gpu_process_bytes"]
                    for r in records
                    if (r.get("memory") or {}).get("complete")
                ],
                "complete_memory_measurements": sum(
                    bool((r.get("memory") or {}).get("complete")) for r in records
                ),
                "reference_seconds": [
                    r["components_seconds"]["reference_construction"] for r in records
                ],
            }
        )
    return {
        "protocol": "mesh-cht-summary-v1",
        "controls": controls,
        "methods": rows,
        "interpretation": "Each timing is an independently executed complete sequence. Accepted times and failed outcomes are reported separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fields", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", summarize(args.records))
    if args.fields:
        for directory in sorted(args.records.glob("reference-*")):
            if (directory / "geometry.npz").exists():
                plot_fields(directory, args.output / (directory.name + "-fields.pdf"))


if __name__ == "__main__":
    main()
