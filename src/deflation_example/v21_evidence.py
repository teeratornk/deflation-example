"""Evidence and manuscript artifacts of the prescribed-speedup-v21 corrections.

v21 reruns every transformer comparison with the skew-symmetric transport and
re-baselines the six primary body-fitted cases. ``bundle`` digests the frozen v20
run tree and the v21 run tree into one evidence tree; v21 population keys carry the
prefix ``v21-`` on their wave directory (``v21-wave2/C-S4b-transformer-x4-skew``),
so every v20 key is unchanged. Body-fitted digests also carry the disjoint cost
components of ``mesh_report.components``.

The manuscript rows substitute each advective transformer population by its skew
twin (``TWINS``); the advective originals form a separate diagnostic table. Main
rows report the speedup with the range over repetitions, the smallest and largest
ratio of the fastest alternative's times to the reference times, and name every
method with a repetition that does not converge together with its converged count.
"""

import argparse
import json
from pathlib import Path, PurePosixPath

from . import v20_evidence as v20
from .mesh_report import components

FORMAT = "prescribed-speedup-v21-evidence-v1"
CAMPAIGN_FILES = v20.CAMPAIGN_FILES + (
    "prescribed-speedup-v21-protocol.json",
    "prescribed-speedup-v21-operations/O3-transformer-steady-flows-skew.args",
    "prescribed-speedup-v21-operations/O5-shuffled-transformer-steady-skew.args",
    "prescribed-speedup-v21-operations/O5-sorted-transformer-steady-skew.args",
)
PREFIX = "v21-"

TWINS = {
    "wave9/O5-sorted-transformer-steady": "v21-wave2/O5-sorted-transformer-steady-skew",
    "wave9/O5-shuffled-transformer-steady": "v21-wave2/O5-shuffled-transformer-steady-skew",
    "wave9/O3-transformer-steady-flows": "v21-wave2/O3-transformer-steady-flows-skew",
    "wave3/C-S4b-transformer-x4": "v21-wave2/C-S4b-transformer-x4-skew",
    "wave3/C-S4b-transformer-x8": "v21-wave2/C-S4b-transformer-x8-skew",
    "wave6/A-tol-transformer-steady-rtol1e-8": "v21-wave2/A-tol-transformer-steady-rtol1e-8-skew",
    "wave6/A-tol-transformer-steady-rtol1e-6": "v21-wave2/A-tol-transformer-steady-rtol1e-6-skew",
    "wave6/A-alpha-transformer-steady-1e-13": "v21-wave2/A-alpha-transformer-steady-1e-13-skew",
    "wave6/A-alpha-transformer-steady-1e-12": "v21-wave2/A-alpha-transformer-steady-1e-12-skew",
    "wave3/C-S3a-transformer-steady-q64": "v21-wave2/C-S3a-transformer-steady-q64-skew",
    "wave3/C-S3a-transformer-x8-q64": "v21-wave2/C-S3a-transformer-x8-q64-skew",
    "wave6/A-q256-transformer-steady": "v21-wave2/A-q256-transformer-steady-skew",
    "wave3/C-S3b-rating-steady": "v21-wave2/C-S3b-rating-steady-skew",
    "wave3/C-S3b-rating-x8": "v21-wave2/C-S3b-rating-x8-skew",
    "wave1/S2a-gpu-transformer-x16-global": "v21-wave2/S2a-gpu-transformer-x16-global-skew",
    "wave1/S2a-gpu-transformer-x32-global": "v21-wave2/S2a-gpu-transformer-x32-global-skew",
    "wave4/S1d-transformer-L2-steady": "v21-wave2/S1d-transformer-L2-steady-skew",
    "wave4/S1d-transformer-L2-x4": "v21-wave2/S1d-transformer-L2-x4-skew",
    "wave12/D-transformer-steady": "v21-wave3/D-transformer-steady-skew",
    "wave12/D-transformer-x8": "v21-wave3/D-transformer-x8-skew",
}
AUGMENT = {
    "v21-wave2/O3-transformer-steady-flows-skew": [
        "v21-wave2/H-amgx001-O3-transformer-steady-flows-skew",
        "v21-wave2/H-amgx0001-O3-transformer-steady-flows-skew",
    ]
}
PAIRED = {
    "v21-wave2/S2a-gpu-transformer-x16-global-skew": [
        ("v21-wave2/S2a-gpu-transformer-x16-perslab4-skew", "reference", "reference-perslab4")
    ],
    "v21-wave2/S2a-gpu-transformer-x32-global-skew": [
        ("v21-wave2/S2a-gpu-transformer-x32-perslab4-skew", "reference", "reference-perslab4")
    ],
}


def twin(spec):
    """A v20 row specification with every advective transformer population replaced."""
    return [(row[0], row[1], TWINS.get(row[2], row[2]), *row[3:]) for row in spec]


PRIMARY_ROWS = [
    ("Transformer", "Steady", "v21-wave1/P-transformer-steady-skew"),
    ("Transformer", "4 slabs", "v21-wave1/P-transformer-x4-skew"),
    ("Bore 1", "Steady", "v21-wave1/P-engine-L1-steady"),
    ("Bore 1", "4 slabs", "v21-wave1/P-engine-L1-x4"),
    ("Bore 2", "Steady", "v21-wave1/P-engine-L2-steady"),
    ("Bore 2", "4 slabs", "v21-wave1/P-engine-L2-x4"),
]
SCALE_ROWS = twin(v20.SCALE_ROWS)
TRANSIENT_ROWS = twin(v20.TRANSIENT_ROWS)
OPERATION_ROWS = twin(v20.OPERATION_ROWS)
SI_ROWS = [row for row in twin(v20.SI_ROWS) if row[1] != "advective transport"]
SINGLE_ROWS = twin(v20.SINGLE_ROWS)
DIRECT_ROWS = [
    (label, TWINS.get(cpu, cpu), TWINS.get(gpu, gpu)) for label, cpu, gpu in v20.DIRECT_ROWS
]
# The original advective transport, kept as a labelled diagnostic.
ORIGINAL_ROWS = [
    ("Transformer", "steady", "wave9/O5-sorted-transformer-steady", "reference"),
    ("Transformer", "4 slabs", "wave3/C-S4b-transformer-x4", "reference"),
    ("Transformer", "8 slabs", "wave3/C-S4b-transformer-x8", "reference"),
    ("Transformer day", "24 hourly windows", "wave9/O1-transformer-day-x8", "reference"),
    ("Overloads", "transformer, 8 slabs", "wave9/O4-transformer-overload-x8", "reference"),
    ("Off-design flow", "transformer", "wave9/O3-transformer-steady-flows", "reference"),
    ("Shuffled order", "transformer", "wave9/O5-shuffled-transformer-steady", "reference"),
]
ALTERNATIVES = {"jacobi", "recycling", "amgx"}


# ---------------------------------------------------------------------- evidence


def _with_components(path, digest):
    """Add the disjoint cost components of an accepted body-fitted record."""
    if not digest.get("accepted") or not digest.get("source", "").endswith("record.json"):
        return digest
    record = json.loads((path / digest["source"]).read_text())
    return {**digest, "components": components(record)}


def bundle(roots, campaign, output, extra=()):
    """Digest every population of each (run tree, wave prefix) in ``roots``."""
    campaign, output = Path(campaign), Path(output)
    if output.exists():
        raise ValueError("The evidence destination must be new")
    written = 0
    for runs, prefix in roots:
        for wave in sorted(p for p in Path(runs).iterdir() if p.is_dir()):
            for path in sorted(p for p in wave.iterdir() if p.is_dir()):
                found = v20.population_digests(path)
                if found is None:
                    continue
                controls, digests = found
                base = output / "populations" / f"{prefix}{wave.name}" / path.name
                v20._write_json(base / "controls.json", controls)
                for digest in digests:
                    if (path / "protocol.json").is_file():
                        digest = _with_components(path, digest)
                    v20._write_json(
                        base / f"runs/{digest['method']}-{digest['repetition']}.json", digest
                    )
                written += 1
    if not written:
        raise ValueError("The run trees contain no populations")
    for name in CAMPAIGN_FILES:
        destination = output / "campaign" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((campaign / name).read_bytes())
    for target, source in extra:
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            if len(data) > v20.MAX_FILE_BYTES:
                raise ValueError(f"Evidence file exceeds the size limit: {path.name}")
            files[path.relative_to(output).as_posix()] = {
                "sha256": v20._sha(data),
                "bytes": len(data),
            }
    manifest = {"format": FORMAT, "populations": written, "files": files}
    v20._write_json(output / "manifest.json", manifest)
    return manifest


def verify(root):
    """Check that the tree holds exactly the manifest's files with their hashes."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or not manifest.get("files"):
        raise ValueError("Unsupported or empty v21 evidence manifest")
    observed = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).as_posix() != "manifest.json"
    }
    if observed != set(manifest["files"]):
        raise ValueError("The evidence tree and its manifest list different files")
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe evidence path")
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Evidence files must not be symbolic links")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or v20._sha(data) != expected["sha256"]:
            raise ValueError(f"Evidence hash mismatch: {name}")
    return manifest


def populations(root):
    found = v20.populations(root)
    for key, extras in AUGMENT.items():
        if key in found:
            for extra in extras:
                if extra in found:
                    label = extra.split("/")[1].split("-")[1]
                    found[key]["methods"][label] = found[extra]["methods"].get("amgx", [])
    for key, arms in PAIRED.items():
        if key in found:
            for extra, method, label in arms:
                if extra in found:
                    found[key]["methods"][label] = found[extra]["methods"].get(method, [])
    return found


def summarize(root):
    verify(root)
    out = {}
    for key, population in populations(root).items():
        rows, readings = v20.summarize_methods(population["methods"])
        for method, row in rows.items():
            row["attempted"] = row["declared"]
        sizes = [
            r.get("problem_size")
            for runs in population["methods"].values()
            for r in runs
            if r.get("problem_size")
        ]
        out[key] = {
            "methods": rows,
            "readings": readings,
            "unknowns": max(sizes) if sizes else None,
            "phase": population["controls"].get("phase"),
        }
    return out


# ---------------------------------------------------------------------- rows


def _range(population, arm, alternative):
    """Smallest and largest ratio of alternative times to reference times over repetitions."""
    reference = population["methods"][arm]["seconds"]
    other = population["methods"][alternative]["seconds"]
    return min(other) / max(reference), max(other) / min(reference)


def not_converged(methods):
    """Methods with a repetition that did not converge, with converged/attempted counts."""
    cells = sorted(
        f"{v20.NAMES.get(m, m)} {row['accepted']}/{row['declared']}"
        for m, row in methods.items()
        if row["accepted"] < row["declared"]
    )
    return ", ".join(cells) if cells else "--"


def coverage(methods):
    """True when some GPU iterative alternative was not run in the comparison."""
    present = {m if not m.startswith("amgx") else "amgx" for m in methods}
    return not ALTERNATIVES <= present


def main_rows(summary, spec, marker=r"$^\dagger$"):
    """Rows with the speedup, its range over repetitions and converged counts.

    A row whose reference arm misses a repetition reports the fastest converged
    alternative without a speedup; a row in which no method converges says so.
    The arm None of a v20 specification reads the reference arm.
    """
    lines, values = [], []
    for label, detail, key, arm in spec:
        population = summary.get(key)
        if population is None:
            raise ValueError(f"No population {key}")
        arm = arm or "reference"
        methods = population["methods"]
        mark = marker if coverage(methods) else ""
        head = [label + mark, detail, v20._count(population["unknowns"])]
        converged = {m: r for m, r in methods.items() if r["accepted"] == r["declared"]}
        if not any(r["accepted"] for r in methods.values()):
            lines.append(
                " & ".join(head) + r" & \multicolumn{5}{l}{no method converges at every query} \\"
            )
            values.append({"population": key, "converged": False})
            continue
        if arm not in converged or arm not in population["readings"]:
            others = {
                m: r["median_seconds"]
                for m, r in methods.items()
                if not v20._is_reference(m) and r["median_seconds"] is not None
            }
            fastest = min(others, key=others.get)
            cells = head + [
                "--",
                f"{v20.NAMES.get(fastest, fastest)} {v20._seconds(others[fastest])}",
                "--",
                "--",
                not_converged(methods),
            ]
            lines.append(" & ".join(cells) + r" \\")
            values.append({"population": key, "converged": False, "fastest": fastest})
            continue
        reading = population["readings"][arm]
        alternative = reading["fastest_alternative"]
        low, high = _range(population, arm, alternative)
        name = v20.NAMES.get(alternative, alternative)
        cells = head + [
            v20._seconds(reading["reference_seconds"]),
            f"{name} {v20._seconds(reading['alternative_seconds'])}",
            f"{v20._ratio(reading['ratio'])} [{v20._ratio(low)}, {v20._ratio(high)}]",
            v20._ratio(reading["jacobi_iteration_ratio"]),
            not_converged(methods),
        ]
        lines.append(" & ".join(cells) + r" \\")
        values.append({**reading, "range": [low, high], "population": key, "converged": True})
    return "\n".join(lines) + "\n", values


def diagnostic_rows(summary, spec):
    """SI rows use the main-table format."""
    return main_rows(summary, spec)


def primary_rows(summary):
    """Table 2: median seconds of every method, speedup and its range over repetitions."""
    lines, values = [], []
    order = ("jacobi", "reference", "recycling", "amgx")
    for geometry, form, key in PRIMARY_ROWS:
        population = summary[key]
        methods = population["methods"]
        cells = [geometry, form, str(population["unknowns"])]
        for method in order:
            row = methods[method]
            text = "--" if row["median_seconds"] is None else f"{row['median_seconds']:.3f}"
            if row["accepted"] < row["declared"]:
                text += f" ({row['accepted']}/{row['declared']})"
            cells.append(text)
        reading = population["readings"]["reference"]
        low, high = _range(population, "reference", reading["fastest_alternative"])
        cells.append(f"{v20._ratio(reading['ratio'])} [{v20._ratio(low)}, {v20._ratio(high)}]")
        lines.append(" & ".join(cells) + r" \\")
        values.append({**reading, "range": [low, high], "population": key})
    return "\n".join(lines) + "\n", values


def direct_rows(summary):
    saved = v20.DIRECT_ROWS
    try:
        v20.DIRECT_ROWS = DIRECT_ROWS
        return v20.direct_rows(summary)
    finally:
        v20.DIRECT_ROWS = saved


def primary_components(root):
    """Cost components of the median-time repetition per primary case and method."""
    found = populations(root)
    out = {}
    for geometry, form, key in PRIMARY_ROWS:
        case = {}
        for method, runs in found[key]["methods"].items():
            accepted = sorted((r for r in runs if r["accepted"]), key=lambda r: r["seconds"])
            if not accepted:
                continue
            median = accepted[(len(accepted) - 1) // 2] if len(accepted) % 2 else None
            if median is None:
                # Even counts: the lower of the two middle repetitions, as mesh_report does.
                median = accepted[len(accepted) // 2 - 1]
            case[method] = {
                "seconds": median["seconds"],
                "components": median["components"],
                "repetition": median["repetition"],
            }
        out[f"{geometry}|{form}"] = case
    return out


def cumulative_curves(root, key):
    """Per-method cumulative seconds of every converged repetition of one population."""
    found = populations(root)
    return {
        method: [[c["cumulative_seconds"] for c in r["cases"]] for r in runs if r["accepted"]]
        for method, runs in found[key]["methods"].items()
    }


def export(root, output):
    """Write rows, figure data and the population listing of a verified evidence tree."""
    output = Path(output)
    if output.exists():
        raise ValueError("The artifact destination must be new")
    summary = summarize(root)
    (output / "generated").mkdir(parents=True)
    written = {}
    for name, (text, values) in {
        "primary_rows": primary_rows(summary),
        "scale_rows": main_rows(summary, SCALE_ROWS),
        "transient_rows": main_rows(summary, TRANSIENT_ROWS),
        "operation_rows": main_rows(summary, OPERATION_ROWS),
        "si_rows": diagnostic_rows(summary, SI_ROWS),
        "single_rows": diagnostic_rows(summary, SINGLE_ROWS),
        "original_rows": diagnostic_rows(summary, ORIGINAL_ROWS),
        "direct_rows": direct_rows(summary),
    }.items():
        (output / f"generated/{name}.tex").write_text(text)
        written[name] = values
    figures = {
        "primary_components": primary_components(root),
        "primary_curves": cumulative_curves(root, "v21-wave1/P-engine-L2-x4"),
        "q256_curves": cumulative_curves(root, "wave6/A-q256-engine-L2-steady"),
    }
    v20._write_json(output / "figures.json", figures)
    v20._write_json(output / "summary.json", summary)
    index = {
        name: [dict(zip(("label", "detail", "population", "arm"), row)) for row in spec]
        for name, spec in (
            ("scale_rows", SCALE_ROWS),
            ("transient_rows", TRANSIENT_ROWS),
            ("operation_rows", OPERATION_ROWS),
            ("si_rows", SI_ROWS),
            ("single_rows", SINGLE_ROWS),
            ("original_rows", ORIGINAL_ROWS),
        )
    }
    index["primary_rows"] = [dict(zip(("geometry", "form", "population"), r)) for r in PRIMARY_ROWS]
    index["direct_rows"] = [dict(zip(("label", "cpu", "gpu"), r)) for r in DIRECT_ROWS]
    index["twins"], index["augment"] = TWINS, AUGMENT
    index["paired"] = {k: [list(a) for a in v] for k, v in PAIRED.items()}
    v20._write_json(output / "rows.json", index)
    (output / "populations.md").write_text(v20.listing(summary))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--v20-runs", type=Path, required=True)
    b.add_argument("--v21-runs", type=Path, required=True)
    b.add_argument("--campaign", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--extra", nargs="*", default=[], help="bundle_path=source_file")
    e = sub.add_parser("export")
    e.add_argument("--evidence", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "bundle":
        extra = [tuple(item.split("=", 1)) for item in args.extra]
        manifest = bundle(
            [(args.v20_runs, ""), (args.v21_runs, PREFIX)], args.campaign, args.output, extra
        )
        print(f"Bundled {manifest['populations']} populations, {len(manifest['files'])} files")
    else:
        export(args.evidence, args.output)
        print(f"Exported the v21 artifacts to {args.output}")


if __name__ == "__main__":
    main()
