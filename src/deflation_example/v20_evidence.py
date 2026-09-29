"""Frozen evidence and manuscript artifacts of the prescribed-speedup-v20 campaign.

``bundle`` reads every population under a campaign run tree and writes one compact
digest per declared run (method and repetition): acceptance, complete-sequence
seconds, inner and outer iterations, per-query status, cumulative time and largest
inner solve, peak memory, problem size, and the sha256 and size of the source record.
Raw records stay in the run tree; the digests carry their hashes. The campaign
protocol, the operating-scenario declaration and the job log are copied beside the
digests, and a manifest lists every file with its sha256.

``summarize`` reads a verified tree: per method the accepted and declared repetitions,
the median complete-sequence time over accepted repetitions and the median inner
iterations, and per reference arm the reading against the fastest completed
alternative (the campaign's pays rule). ``export`` writes the manuscript rows, macros
and figure from those summaries, and the complete population listing.

Two raw layouts are read. Body-fitted runs hold ``<method>-<repetition>/record.json``
with ``protocol.json`` and ``attempts.json``; Cartesian runs hold
``sequence-<repetition>-<warm>-<method>.json`` with ``declared_protocol.json``.
"""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path, PurePosixPath

FORMAT = "prescribed-speedup-v20-evidence-v1"
REFERENCE_METHODS = ("reference", "block_reference")
CAMPAIGN_FILES = (
    "prescribed-speedup-v20-protocol.json",
    "prescribed-speedup-v20-operations.json",
    "prescribed-speedup-v20-jobs.txt",
)
MAX_FILE_BYTES = 5 * 1024 * 1024
# Populations run as one method (and, for confirmations, one repetition) per job,
# read together under one name.
MERGED = {
    "G-pilot-engine-L4-steady-o400-": "G-pilot-engine-L4-steady-o400",
    "G-engine-L4-steady-o400-": "G-engine-L4-steady-o400",
}
# AmgX with a tighter stopping factor, added as an alternative of its paired population.
AUGMENT = {
    "wave8/G-steady96": ["wave10/H-amgx001-steady96"],
    "wave8/G-steady128-retry1": ["wave10/H-amgx001-steady128", "wave10/H-amgx0001-steady128"],
    "wave8/G-transient64-perslab12": ["wave10/H-amgx001-transient64-perslab12"],
    "wave9/O3-transformer-steady-flows": [
        "wave10/H-amgx001-O3-transformer-steady-flows",
        "wave10/H-amgx0001-O3-transformer-steady-flows",
    ],
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _is_reference(method):
    return method.split("-")[0] in REFERENCE_METHODS


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- digests


def _case(case):
    inner = case.get("inner") or []
    return {
        "status": case.get("status"),
        "inner_iterations": case.get("inner_iterations"),
        "outer_iterations": case.get("outer_iterations"),
        "cumulative_seconds": case.get("cumulative_seconds"),
        "max_inner_per_solve": max((row.get("iterations", 0) for row in inner), default=0),
    }


def _digest(method, repetition, source=None, content=None, record=None, failure=None):
    """One declared run: the fields the manuscript reads and the source record's hash."""
    if record is None:
        return {
            "method": method,
            "repetition": repetition,
            "accepted": False,
            "failure": failure,
            "source": None,
        }
    return {
        "method": method,
        "repetition": repetition,
        "source": source,
        "source_sha256": _sha(content),
        "source_bytes": len(content),
        **record,
    }


def _mesh_runs(path, controls):
    attempts = {}
    if (path / "attempts.json").is_file():
        for row in json.loads((path / "attempts.json").read_text()):
            attempts[(row["method"], row["repetition"])] = row["status"]
    for method in controls["methods"]:
        for repetition in range(controls["repeats"]):
            record_path = path / f"{method}-{repetition}" / "record.json"
            if not record_path.is_file():
                yield _digest(
                    method, repetition, failure=attempts.get((method, repetition), "not_recorded")
                )
                continue
            content = record_path.read_bytes()
            record = json.loads(content)
            cases = record.get("cases") or []
            accepted = bool(record.get("success")) and all(
                c.get("status") == "converged" for c in cases
            )
            failure = None
            if not accepted:
                failure = (record.get("failure") or {}).get("status") or sorted(
                    {c.get("status") for c in cases} - {"converged"}
                )
            yield _digest(
                method,
                repetition,
                f"{method}-{repetition}/record.json",
                content,
                {
                    "accepted": accepted,
                    "failure": failure,
                    "seconds": record.get("seconds"),
                    "inner": sum(c.get("inner_iterations", 0) for c in cases),
                    "outer": sum(c.get("outer_iterations", 0) for c in cases),
                    "peak_gpu_bytes": (record.get("memory") or {}).get("peak_gpu_process_bytes"),
                    "problem_size": record.get("problem_size"),
                    "cases": [_case(c) for c in cases],
                },
            )


def _cht_runs(path, controls):
    size = controls["n"] ** 3 * (controls["slabs"] if controls["problem"] == "transient" else 1)
    for method in controls["methods"]:
        for repetition in range(controls["repeats"]):
            found = sorted(path.glob(f"sequence-{repetition}-*-{method}.json"))
            if not found:
                yield _digest(method, repetition, failure="not_recorded")
                continue
            content = found[0].read_bytes()
            record = json.loads(content)
            accepted = bool(record.get("success"))
            yield _digest(
                method,
                repetition,
                found[0].name,
                content,
                {
                    "accepted": accepted,
                    "failure": None if accepted else record.get("status"),
                    "seconds": record.get("total_seconds"),
                    "inner": record.get("inner_iterations"),
                    "outer": record.get("outer_iterations"),
                    "peak_gpu_bytes": (record.get("memory") or {}).get("peak_gpu_process_bytes"),
                    "problem_size": size,
                    "cases": [_case(c) for c in record.get("cases") or []],
                },
            )


def population_digests(path):
    """(controls, digests) of one raw population directory, or None if it is not one."""
    path = Path(path)
    if (path / "declared_protocol.json").is_file():
        controls = json.loads((path / "declared_protocol.json").read_text())["controls"]
        return controls, list(_cht_runs(path, controls))
    if (path / "protocol.json").is_file() and "methods" in json.loads(
        (path / "protocol.json").read_text()
    ):
        controls = json.loads((path / "protocol.json").read_text())
        return controls, list(_mesh_runs(path, controls))
    return None


def bundle(runs, campaign, output, extra=()):
    """Write digests of every population under ``runs`` (``wave/name``) to ``output``.

    ``campaign`` holds CAMPAIGN_FILES. ``extra`` lists (bundle path, source file)
    pairs of small summaries copied as they are (for example temporal summaries).
    """
    runs, campaign, output = Path(runs), Path(campaign), Path(output)
    if output.exists():
        raise ValueError("The evidence destination must be new")
    written = 0
    for wave in sorted(p for p in runs.iterdir() if p.is_dir()):
        for path in sorted(p for p in wave.iterdir() if p.is_dir()):
            found = population_digests(path)
            if found is None:
                continue
            controls, digests = found
            base = output / "populations" / wave.name / path.name
            _write_json(base / "controls.json", controls)
            for digest in digests:
                _write_json(base / f"runs/{digest['method']}-{digest['repetition']}.json", digest)
            written += 1
    if not written:
        raise ValueError("The run tree contains no populations")
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
            if len(data) > MAX_FILE_BYTES:
                raise ValueError(f"Evidence file exceeds the size limit: {path.name}")
            files[path.relative_to(output).as_posix()] = {"sha256": _sha(data), "bytes": len(data)}
    manifest = {"format": FORMAT, "populations": written, "files": files}
    _write_json(output / "manifest.json", manifest)
    return manifest


def verify(root):
    """Check that the tree holds exactly the manifest's files with their hashes."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or not manifest.get("files"):
        raise ValueError("Unsupported or empty v20 evidence manifest")
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
        if len(data) != expected["bytes"] or _sha(data) != expected["sha256"]:
            raise ValueError(f"Evidence hash mismatch: {name}")
    return manifest


# ---------------------------------------------------------------------- population


def read_population(path, label_levels=False):
    """(methods, controls) of one bundled population: method -> list of run digests."""
    path = Path(path)
    controls = json.loads((path / "controls.json").read_text())
    level = controls.get("reference_level")
    methods = {}
    for run_path in sorted((path / "runs").glob("*.json")):
        run = json.loads(run_path.read_text())
        name = run["method"]
        if label_levels and _is_reference(name) and level:
            name = f"{name}-ref{level}"
        methods.setdefault(name, []).append(run)
    for runs in methods.values():
        runs.sort(key=lambda r: r["repetition"])
    return methods, controls


def summarize_methods(methods):
    """Per-method statistics and the reading of every reference arm."""
    rows = {}
    for method, runs in methods.items():
        accepted = [r for r in runs if r["accepted"]]
        rows[method] = {
            "declared": len(runs),
            "accepted": len(accepted),
            "median_seconds": statistics.median(r["seconds"] for r in accepted)
            if accepted
            else None,
            "seconds": [r["seconds"] for r in accepted],
            "median_inner": statistics.median(r["inner"] for r in accepted) if accepted else None,
            "failures": [r["failure"] for r in runs if not r["accepted"]],
            "peak_gpu_bytes": max((r.get("peak_gpu_bytes") or 0 for r in runs), default=0),
            "max_inner_per_solve": max(
                (c["max_inner_per_solve"] for r in runs for c in r.get("cases") or []),
                default=0,
            ),
        }
    readings = {}
    for method, row in rows.items():
        if not _is_reference(method) or row["median_seconds"] is None:
            continue
        others = {
            m: r["median_seconds"]
            for m, r in rows.items()
            if not _is_reference(m) and r["median_seconds"] is not None
        }
        if not others:
            continue
        fastest = min(others, key=others.get)
        jacobi = rows.get("jacobi", {}).get("median_inner")
        readings[method] = {
            "fastest_alternative": fastest,
            "alternative_seconds": others[fastest],
            "reference_seconds": row["median_seconds"],
            "ratio": others[fastest] / row["median_seconds"],
            "jacobi_iteration_ratio": jacobi / row["median_inner"]
            if jacobi and row["median_inner"]
            else None,
        }
    return rows, readings


def populations(root):
    """{"wave/name": {"methods": ..., "controls": ...}} of a bundled evidence tree."""
    base = Path(root) / "populations"
    found, groups = {}, {}
    for wave in sorted(p for p in base.iterdir() if p.is_dir()):
        for path in sorted(p for p in wave.iterdir() if p.is_dir()):
            prefix = next((p for p in MERGED if path.name.startswith(p)), None)
            if prefix is None:
                methods, controls = read_population(path)
                found[f"{wave.name}/{path.name}"] = {"methods": methods, "controls": controls}
                continue
            key = f"{wave.name}/{MERGED[prefix]}"
            methods, controls = read_population(path, label_levels=True)
            repetition = re.search(r"-rep(\d+)$", path.name)
            group = groups.setdefault(key, {"methods": {}, "controls": controls, "parts": []})
            group["parts"].append(path.name)
            for method, runs in methods.items():
                if repetition:
                    runs = [{**r, "repetition": int(repetition.group(1))} for r in runs]
                group["methods"].setdefault(method, []).extend(runs)
    found.update(groups)
    for key, extras in AUGMENT.items():
        if key not in found:
            continue
        for extra in extras:
            if extra in found:
                label = extra.split("/")[1].split("-")[1]
                found[key]["methods"][label] = found[extra]["methods"].get("amgx", [])
    return found


def summarize(root):
    verify(root)
    out = {}
    for key, population in populations(root).items():
        rows, readings = summarize_methods(population["methods"])
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


# ---------------------------------------------------------------------- manuscript

# (label, detail, population, reference arm) for each main-text row.
SCALE_ROWS = [
    ("Cartesian steady", r"$32^3$", "wave8/G-steady32-device", "reference"),
    ("Cartesian steady", r"$48^3$", "wave8/G-steady48-device", "reference"),
    ("Cartesian steady", r"$64^3$", "wave8/G-steady64-device", "reference"),
    ("Cartesian steady", r"$96^3$", "wave8/G-steady96", "reference"),
    ("Cartesian steady", r"$128^3$", "wave8/G-steady128-retry1", "reference"),
    ("Transformer steady", "level 0", "wave9/O5-sorted-transformer-steady", "reference"),
    ("Bore steady", "level 1", "wave12/D-gpu-engine-L1-steady", "reference"),
    ("Bore steady", "level 2", "wave14/E-engine-L2-steady-ref1-r200", "reference"),
    ("Bore steady", "level 3", "wave14/E-engine-L3-steady-ref1-r200", "reference"),
    ("Bore steady", "level 4", "wave8/G-engine-L4-steady-o400", "reference-ref2"),
]
TRANSIENT_ROWS = [
    ("Cartesian", r"$12^3$, 8 slabs", "wave3/C-S2b-transient12-perslab12", "reference"),
    ("Cartesian", r"$16^3$, 8 slabs", "wave3/C-S2b-transient16-perslab12", "reference"),
    ("Cartesian", r"$24^3$, 8 slabs", "wave3/C-S2b-transient24-perslab12", "reference"),
    ("Cartesian", r"$32^3$, 8 slabs", "wave3/C-S2b-transient32-perslab12", "reference"),
    ("Cartesian", r"$48^3$, 8 slabs", "wave8/G-transient48-perslab12", "reference"),
    ("Cartesian", r"$64^3$, 8 slabs", "wave8/G-transient64-perslab12", "reference"),
    ("Cartesian", r"$16^3$, 16 slabs", "wave3/C-S2b-time16-perslab6", "reference"),
    ("Cartesian", r"$16^3$, 32 slabs", "wave3/C-S2b-time32-perslab6", "reference"),
    ("Transformer", "4 slabs", "wave3/C-S4b-transformer-x4", "reference"),
    ("Transformer", "8 slabs", "wave3/C-S4b-transformer-x8", "reference"),
    ("Bore level 1", "4 slabs", "wave3/C-S4b-engine-L1-x4", "reference"),
    ("Bore level 1", "8 slabs", "wave3/C-S4b-engine-L1-x8", "reference"),
    ("Bore level 2", "4 slabs", "wave3/C-S4b-engine-L2-x4", "reference"),
    ("Bore level 2", "8 slabs", "wave3/C-S4b-engine-L2-x8", "reference"),
    ("Bore level 3", "4 slabs", "wave8/G-engine-L3-x4-ref1-perslab50", "reference"),
]
OPERATION_ROWS = [
    ("Transformer day", "24 chained hourly windows", "wave9/O1-transformer-day-x8", "reference"),
    ("Drive cycle", "16 chained windows", "wave9/O2-engine-L2-cycle-x4", "reference"),
    ("Off-design flow", "transformer", "wave9/O3-transformer-steady-flows", "reference"),
    ("Off-design flow", "bore level 2", "wave9/O3-engine-L2-steady-flows", "reference"),
    ("Off-design flow", "bore level 3", "wave9/O3-engine-L3-steady-ref1-flows", "reference"),
    ("Overloads", "transformer, 8 slabs", "wave9/O4-transformer-overload-x8", "reference"),
    ("Shuffled order", "transformer", "wave9/O5-shuffled-transformer-steady", "reference"),
    ("Shuffled order", "bore level 2", "wave9/O5-shuffled-engine-L2-steady", "reference"),
    ("Shuffled order", "bore level 2, 4 slabs", "wave9/O5-shuffled-engine-L2-x4", "reference"),
]
NAMES = {
    "jacobi": "Jacobi-CG",
    "reference": "reference",
    "recycling": "recycling",
    "amgx": "AmgX",
    "amgx001": "AmgX ($10^{-2}$)",
    "amgx0001": "AmgX ($10^{-3}$)",
    "direct": "direct",
    "block": "block",
}


def _seconds(value):
    return "--" if value is None else f"{value:.1f}"


def _ratio(value):
    return "--" if value is None else f"{value:.2f}"


def _count(value):
    return f"{int(round(value)):,}".replace(",", "{,}")


def table_rows(summary, spec):
    """LaTeX rows: label, detail, unknowns, reference s, alternative (s), ratio, iterations."""
    lines, values = [], []
    for label, detail, key, arm in spec:
        population = summary.get(key)
        if population is None or arm not in population["readings"]:
            raise ValueError(f"No reading for {key} ({arm})")
        reading = population["readings"][arm]
        alternative = NAMES.get(reading["fastest_alternative"], reading["fastest_alternative"])
        lines.append(
            " & ".join(
                [
                    label,
                    detail,
                    _count(population["unknowns"]),
                    _seconds(reading["reference_seconds"]),
                    f"{alternative} {_seconds(reading['alternative_seconds'])}",
                    _ratio(reading["ratio"]),
                    _ratio(reading["jacobi_iteration_ratio"]),
                ]
            )
            + r" \\"
        )
        values.append(reading["ratio"])
    return "\n".join(lines) + "\n", values


CARTESIAN_TRANSIENT = 8  # the first rows of TRANSIENT_ROWS are Cartesian


def macros(summary, scale, transient, operation):
    def reading(key, arm="reference"):
        return summary[key]["readings"][arm]

    cartesian_transient = transient[:CARTESIAN_TRANSIENT]
    body_transient = transient[CARTESIAN_TRANSIENT:]
    l4 = summary["wave8/G-engine-L4-steady-o400"]["methods"]
    values = {
        "psScaleMin": _ratio(min(scale)),
        "psScaleMax": _ratio(max(scale)),
        "psCartesianTransientMin": _ratio(min(cartesian_transient)),
        "psCartesianTransientMax": _ratio(max(cartesian_transient)),
        "psBodyTransientMin": _ratio(min(body_transient)),
        "psBodyTransientMax": _ratio(max(body_transient)),
        "psOperationMin": _ratio(min(operation)),
        "psOperationMax": _ratio(max(operation)),
        "psMaxUnknowns": _count(max(p["unknowns"] or 0 for p in summary.values())),
        "psDayRatio": _ratio(reading("wave9/O1-transformer-day-x8")["ratio"]),
        "psCycleRatio": _ratio(reading("wave9/O2-engine-L2-cycle-x4")["ratio"]),
        "psOffDesignTransformerRatio": _ratio(
            reading("wave9/O3-transformer-steady-flows")["ratio"]
        ),
        "psTightAmgxRatio": _ratio(reading("wave8/G-steady96")["ratio"]),
        "psLFourReferenceMaxInner": _count(l4["reference-ref2"]["max_inner_per_solve"]),
        "psLFourJacobiInnerCap": _count(l4["jacobi"]["max_inner_per_solve"]),
    }
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items()), values


def listing(summary):
    """Markdown listing of every population, pilots included."""
    lines = [
        "| Population | Phase | Method: accepted/declared, median s | Reading |",
        "|---|---|---|---|",
    ]
    for key in sorted(summary):
        population = summary[key]
        cells = "; ".join(
            f"{m} {r['accepted']}/{r['declared']}, {_seconds(r['median_seconds'])}"
            for m, r in sorted(population["methods"].items())
        )
        reading = "; ".join(
            f"{m}: {x['ratio']:.2f}x vs {x['fastest_alternative']}"
            for m, x in population["readings"].items()
        )
        lines.append(f"| {key} | {population['phase']} | {cells} | {reading or '--'} |")
    return "\n".join(lines) + "\n"


def figure(summary, runs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    series = [
        ("Cartesian steady", SCALE_ROWS[:5], "o"),
        ("Bore steady", SCALE_ROWS[6:], "s"),
        ("Cartesian, 8 slabs", TRANSIENT_ROWS[:6], "^"),
    ]
    for name, spec, marker in series:
        x = [summary[k]["unknowns"] for _, _, k, _ in spec]
        y = [summary[k]["readings"][a]["ratio"] for _, _, k, a in spec]
        left.plot(x, y, marker=marker, label=name)
    left.axhline(1.0, color="0.5", lw=0.8, ls="--")
    left.set_xscale("log")
    left.set_xlabel("Unknowns")
    left.set_ylabel("Fastest alternative / reference")
    left.legend(fontsize=7, frameon=False)
    for method, method_runs in sorted(runs["wave6/A-q256-engine-L2-steady"].items()):
        curves = [
            [c["cumulative_seconds"] for c in r["cases"]] for r in method_runs if r["accepted"]
        ]
        if not curves:
            continue
        median = [statistics.median(v) for v in zip(*curves)]
        right.plot(range(1, len(median) + 1), median, label=NAMES.get(method, method))
    right.set_xlabel("Query")
    right.set_ylabel("Cumulative time (s)")
    right.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export(root, output):
    """Write the manuscript artifacts of a verified evidence tree to ``output``."""
    output = Path(output)
    if output.exists():
        raise ValueError("The artifact destination must be new")
    summary = summarize(root)
    runs = {key: population["methods"] for key, population in populations(root).items()}
    (output / "generated").mkdir(parents=True)
    (output / "figures").mkdir()
    scale_text, scale = table_rows(summary, SCALE_ROWS)
    transient_text, transient = table_rows(summary, TRANSIENT_ROWS)
    operation_text, operation = table_rows(summary, OPERATION_ROWS)
    (output / "generated/scale_rows.tex").write_text(scale_text)
    (output / "generated/transient_rows.tex").write_text(transient_text)
    (output / "generated/operation_rows.tex").write_text(operation_text)
    text, values = macros(summary, scale, transient, operation)
    (output / "generated/macros.tex").write_text(text)
    (output / "populations.md").write_text(listing(summary))
    figure(summary, runs, output / "figures/speedup_scale.pdf")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("bundle")
    b.add_argument("--runs", required=True)
    b.add_argument("--campaign", required=True)
    b.add_argument("--output", required=True)
    e = commands.add_parser("export")
    e.add_argument("--evidence", required=True)
    e.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "bundle":
        manifest = bundle(args.runs, args.campaign, args.output)
        print(f"Bundled {manifest['populations']} populations, {len(manifest['files'])} files")
    else:
        print(json.dumps(export(args.evidence, args.output), indent=2))


if __name__ == "__main__":
    main()
