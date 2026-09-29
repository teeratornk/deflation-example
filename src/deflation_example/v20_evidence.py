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
    history = case.get("outer_history") or case.get("history") or []
    last = history[-1] if history else {}
    residuals = [
        row["original_residual"] for row in inner if row.get("original_residual") is not None
    ]
    return {
        "status": case.get("status"),
        "inner_iterations": case.get("inner_iterations"),
        "outer_iterations": case.get("outer_iterations"),
        "cumulative_seconds": case.get("cumulative_seconds"),
        "max_inner_per_solve": max((row.get("iterations", 0) for row in inner), default=0),
        "max_residual": max(residuals) if residuals else None,
        "final_stationarity": last.get("stationarity"),
        # The last outer iteration changed no constraint (the active set settled).
        "settled": bool(history) and last.get("entered") == 0 and last.get("left") == 0,
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
                    "git_head": (record.get("environment") or {}).get("git_head"),
                    "source_tree_clean": (record.get("environment") or {}).get("source_tree_clean"),
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
                    "git_head": (record.get("environment") or {}).get("git_head"),
                    "source_tree_clean": (record.get("environment") or {}).get("source_tree_clean"),
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
        if label_levels and _is_reference(name) and level is not None:
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
            phase = "pilot" if "pilot" in MERGED[prefix] else "final"
            group = groups.setdefault(
                key, {"methods": {}, "controls": {**controls, "phase": phase}, "parts": []}
            )
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
            "timeout_seconds": population["controls"].get("timeout_seconds"),
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
    (
        "Transformer day",
        "24 hourly windows",
        "wave15/O1s-transformer-day-x8-skew",
        "reference",
    ),
    ("Drive cycle", "16 windows", "wave9/O2-engine-L2-cycle-x4", "reference"),
    ("Off-design flow", "transformer", "wave9/O3-transformer-steady-flows", "reference"),
    ("Off-design flow", "bore level 2", "wave9/O3-engine-L2-steady-flows", "reference"),
    ("Off-design flow", "bore level 3", "wave9/O3-engine-L3-steady-ref1-flows", "reference"),
    ("Overloads", "transformer, 8 slabs", "wave15/O4s-transformer-overload-x8-skew", "reference"),
    ("Shuffled order", "transformer", "wave9/O5-shuffled-transformer-steady", "reference"),
    ("Shuffled order", "bore level 2", "wave9/O5-shuffled-engine-L2-steady", "reference"),
    ("Shuffled order", "bore level 2, 4 slabs", "wave9/O5-shuffled-engine-L2-x4", "reference"),
]
# Supplementary confirmations: tolerance, regularization, cold starts, long sequences,
# rank and coarse level, global-rank Cartesian transients and the level-0 bore L3 row.
SI_ROWS = [
    ("Transformer day", "advective transport", "wave9/O1-transformer-day-x8", "reference"),
    ("Overloads", "advective transport", "wave9/O4-transformer-overload-x8", "reference"),
    (
        "Transformer steady",
        r"rtol $10^{-8}$",
        "wave6/A-tol-transformer-steady-rtol1e-8",
        "reference",
    ),
    (
        "Transformer steady",
        r"rtol $10^{-6}$",
        "wave6/A-tol-transformer-steady-rtol1e-6",
        "reference",
    ),
    ("Bore L2 steady", r"rtol $10^{-8}$", "wave6/A-tol-engine-L2-steady-rtol1e-8", "reference"),
    ("Bore L2 steady", r"rtol $10^{-6}$", "wave6/A-tol-engine-L2-steady-rtol1e-6", "reference"),
    ("Bore L3 steady", r"rtol $10^{-8}$", "wave6/A-tol-engine-L3-steady-rtol1e-8", "reference"),
    ("Bore L3 steady", r"rtol $10^{-6}$", "wave6/A-tol-engine-L3-steady-rtol1e-6", "reference"),
    ("Bore L2 steady", r"$\alpha=10^{-5}$", "wave6/A-alpha-engine-L2-steady-1e-5", "reference"),
    ("Bore L2 steady", r"$\alpha=10^{-4}$", "wave6/A-alpha-engine-L2-steady-1e-4", "reference"),
    ("Transformer steady", r"$\alpha=10^{-13}$", "wave6/A-alpha-transformer-steady-1e-13", None),
    ("Transformer steady", r"$\alpha=10^{-12}$", "wave6/A-alpha-transformer-steady-1e-12", None),
    ("Bore L2 steady", "cold start", "wave6/A-cold-engine-L2-steady", "reference"),
    ("Bore L3 steady", "cold start", "wave6/A-cold-engine-L3-steady", "reference"),
    ("Transformer steady", "64 targets", "wave3/C-S3a-transformer-steady-q64", "reference"),
    ("Transformer, 8 slabs", "64 targets", "wave3/C-S3a-transformer-x8-q64", "reference"),
    ("Bore L2 steady", "64 targets", "wave3/C-S3a-engine-L2-steady-q64", "reference"),
    ("Bore L2, 4 slabs", "64 targets", "wave3/C-S3a-engine-L2-x4-q64", "reference"),
    ("Transformer steady", "256 targets", "wave6/A-q256-transformer-steady", "reference"),
    ("Bore L2 steady", "256 targets", "wave6/A-q256-engine-L2-steady", "reference"),
    ("Transformer steady", "4 loads x 4 bounds", "wave3/C-S3b-rating-steady", "reference"),
    ("Transformer, 8 slabs", "4 loads x 4 bounds", "wave3/C-S3b-rating-x8", "reference"),
    ("Bore L3 steady", "level 0, rank 50", "wave11/R-engine-L3-steady-ref0-r50", "reference"),
    ("Bore L3 steady", "level 0, rank 100", "wave11/R-engine-L3-steady-ref0-r100", "reference"),
    ("Bore L3 steady", "level 0, rank 200", "wave11/R-engine-L3-steady-ref0-r200", "reference"),
    ("Bore L3 steady", "level 0, rank 300", "wave11/R-engine-L3-steady-ref0-r300", "reference"),
    ("Bore L3 steady", "level 1, rank 50", "wave11/R-engine-L3-steady-ref1-r50", "reference"),
    ("Bore L3 steady", "level 1, rank 100", "wave11/R-engine-L3-steady-ref1-r100", "reference"),
    ("Bore L3 steady", "level 1, rank 400", "wave7/F-engine-L3-steady-ref1-r400", "reference"),
    ("Bore L3 steady", "level 2, rank 200", "wave8/G-engine-L3-steady-ref2-r200", "reference"),
    ("Bore L3 steady", "level 0, four methods", "wave3/C-S1a-engine-L3-steady", "reference"),
    ("Bore L4 steady", "level 3, rank 200", "wave8/G-engine-L4-steady-o400", "reference-ref3"),
    ("Cartesian", r"$12^3$, 8 slabs, global", "wave3/C-S4a-transient12-global", "reference"),
    ("Cartesian", r"$16^3$, 8 slabs, global", "wave3/C-S4a-transient16-global", "reference"),
    ("Cartesian", r"$24^3$, 8 slabs, global", "wave3/C-S4a-transient24-global", "reference"),
    ("Cartesian", r"$32^3$, 8 slabs, global", "wave3/C-S4a-transient32-global", "reference"),
]
# (label, CPU population, GPU population of the same problem and query plan).
DIRECT_ROWS = [
    ("Transformer steady", "wave12/D-transformer-steady", "wave9/O5-sorted-transformer-steady"),
    ("Bore L1 steady", "wave12/D-engine-L1-steady", "wave12/D-gpu-engine-L1-steady"),
    ("Bore L2 steady", "wave12/D-engine-L2-steady", "wave9/O5-sorted-engine-L2-steady"),
    ("Transformer, 8 slabs", "wave12/D-transformer-x8", "wave3/C-S4b-transformer-x8"),
    ("Bore L1, 8 slabs", "wave12/D-engine-L1-x8", "wave3/C-S4b-engine-L1-x8"),
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


def incomplete(methods):
    """Methods of a population with a repetition that did not meet the acceptance tests."""
    names = sorted(
        NAMES.get(m, m) for m, row in methods.items() if row["accepted"] < row["declared"]
    )
    return ", ".join(names) if names else "--"


def table_rows(summary, spec):
    """LaTeX rows: label, detail, unknowns, reference s, alternative (s), ratio, iterations,
    and the methods with an unaccepted repetition.

    The row's reference arm must meet the acceptance tests in every repetition. A row
    with arm None states a population in which no method completes.
    """
    lines, values = [], []
    for label, detail, key, arm in spec:
        population = summary.get(key)
        if population is None:
            raise ValueError(f"No population {key}")
        if arm is None:
            if any(row["accepted"] for row in population["methods"].values()):
                raise ValueError(f"A method completes in {key}")
            lines.append(
                " & ".join([label, detail, _count(population["unknowns"])])
                + r" & \multicolumn{5}{l}{no method completes} \\"
            )
            continue
        if arm not in population["readings"]:
            raise ValueError(f"No reading for {key} ({arm})")
        own = population["methods"][arm]
        if own["accepted"] != own["declared"]:
            raise ValueError(f"The reference arm of {key} has an unaccepted repetition")
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
                    incomplete(population["methods"]),
                ]
            )
            + r" \\"
        )
        values.append(reading)
    return "\n".join(lines) + "\n", values


def direct_rows(summary):
    """CPU direct and block medians beside the GPU reference and fastest GPU alternative."""
    lines, values = [], {}
    for label, cpu, gpu in DIRECT_ROWS:
        cpu_rows = summary[cpu]["methods"]
        reading = summary[gpu]["readings"]["reference"]
        for method, row in cpu_rows.items():
            if row["accepted"] != row["declared"]:
                raise ValueError(f"{cpu} {method} has unaccepted repetitions")
        alternative = NAMES.get(reading["fastest_alternative"], reading["fastest_alternative"])
        lines.append(
            " & ".join(
                [
                    label,
                    _seconds(cpu_rows.get("direct", {}).get("median_seconds")),
                    _seconds(cpu_rows.get("block", {}).get("median_seconds")),
                    _seconds(reading["reference_seconds"]),
                    f"{alternative} {_seconds(reading['alternative_seconds'])}",
                ]
            )
            + r" \\"
        )
        values[label] = {
            "direct": cpu_rows.get("direct", {}).get("median_seconds"),
            "block": cpu_rows.get("block", {}).get("median_seconds"),
            "reference": reading["reference_seconds"],
        }
    return "\n".join(lines) + "\n", values


def campaign_counts(runs, keys):
    """Populations of the manuscript tables, their accepted sequences and target solves."""
    populations_used = sorted(set(keys))
    sequences = solves = 0
    for key in populations_used:
        for method_runs in runs[key].values():
            for run in method_runs:
                if run["accepted"]:
                    sequences += 1
                    solves += len(run.get("cases") or [])
    return len(populations_used), sequences, solves


CARTESIAN_TRANSIENT = 8  # the first rows of TRANSIENT_ROWS are Cartesian
TEMPORAL = [
    ("600 s", "advective", "temporal/published-600s-advective"),
    ("600 s", "skew", "temporal/published-600s-skew"),
    ("1 h", "advective", "temporal/hour-advective"),
    ("1 h", "skew", "temporal/hour-skew"),
]
STABILITY = {"600 s": "temporal/published-600s-stability", "1 h": "temporal/hour-stability"}
# The operating trajectories use eight slabs per window.
TEMPORAL_SLABS = 8


def _sci(value):
    """A positive number as LaTeX math, for example 1.0\\times10^{-8}."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def temporal_rows(root):
    """Time-refinement checks: verified cases, differences and amplification moduli."""
    root = Path(root)
    lines, values = [], {}
    for horizon, form, folder in TEMPORAL:
        summary = json.loads((root / folder / "summary.json").read_text())
        stability = json.loads((root / STABILITY[horizon] / "stability.json").read_text())
        rows = summary["rows"]
        verified = sum(bool(r.get("verified")) for r in rows)
        # Difference of the eight-slab optimum from the finest verified optimum; the
        # summary omits it when the finest grid is unverified.
        differences = [
            r["temperature_max_difference_K"]
            for r in rows
            if r.get("verified")
            and r["slabs"] == TEMPORAL_SLABS
            and r.get("temperature_max_difference_K") is not None
        ]
        changes = [
            r["replay"]["temperature_change_max_K"]
            for r in rows
            if r.get("verified") and r.get("replay")
        ]
        moduli = [
            row["largest_computed_amplification_modulus"]
            for row in stability["rows"]
            if row["transport_form"] == form and row.get("status") == "eigenpair_verified"
        ]
        cells = [
            horizon,
            form,
            f"{verified}/{len(rows)}",
            f"{max(differences):.2f}" if differences else "--",
            f"${_sci(max(changes))}$"
            if changes and max(changes) >= 1e3
            else (f"{max(changes):.2f}" if changes else "--"),
            f"{max(moduli):.2f}" if moduli else "--",
        ]
        lines.append(" & ".join(cells) + r" \\")
        values[(horizon, form)] = {
            "verified": verified,
            "cases": len(rows),
            "difference": max(differences) if differences else None,
            "change": max(changes) if changes else None,
            "modulus": max(moduli) if moduli else None,
        }
    return "\n".join(lines) + "\n", values


def macros(summary, runs, controls, scale, transient, operation, counts, direct, temporal):
    def reading(key, arm="reference"):
        return summary[key]["readings"][arm]

    pilot_block = summary["wave5/W5-engine-L2-x4"]["methods"]["block"]
    pilot_direct = summary["wave5/W5-engine-L3-steady"]["methods"]["direct"]
    if pilot_direct["accepted"] or pilot_direct["failures"] != ["timeout"]:
        raise ValueError("The engine L3 direct pilot is cited as a timeout")
    # AmgX with its declared stopping factor on the 96^3 grid: every failed query
    # stops at a settled active set; report the final stationarity range.
    stops = [
        case
        for run in runs["wave8/G-steady96"]["amgx"]
        for case in run["cases"]
        if case["status"] != "converged"
    ]
    if not stops or any(not c["settled"] or c["status"] != "cycle" for c in stops):
        raise ValueError("The 96^3 AmgX stops are stated as settled active sets")
    stationarity = [c["final_stationarity"] for c in stops]
    # Jacobi-CG on Bore 4: every recorded repetition reaches the declared inner cap.
    cap = controls["wave8/G-engine-L4-steady-o400"]["inner_cap"]
    jacobi = [r for r in runs["wave8/G-engine-L4-steady-o400"]["jacobi"] if r.get("source")]
    capped = [
        r
        for r in jacobi
        if not r["accepted"] and any(c["max_inner_per_solve"] >= cap for c in r["cases"])
    ]
    l4 = summary["wave8/G-engine-L4-steady-o400"]["methods"]
    iterations = [r["jacobi_iteration_ratio"] for r in scale if r["jacobi_iteration_ratio"]]
    ratios = {
        name: [r["ratio"] for r in rows]
        for name, rows in (("scale", scale), ("transient", transient), ("operation", operation))
    }
    hour_skew, hour_advective = temporal[("1 h", "skew")], temporal[("1 h", "advective")]
    values = {
        "psScaleMin": _ratio(min(ratios["scale"])),
        "psScaleMax": _ratio(max(ratios["scale"])),
        "psIterationMin": _ratio(min(iterations)),
        "psIterationMax": _ratio(max(iterations)),
        "psCartesianTransientMin": _ratio(min(ratios["transient"][:CARTESIAN_TRANSIENT])),
        "psCartesianTransientMax": _ratio(max(ratios["transient"][:CARTESIAN_TRANSIENT])),
        "psBodyTransientMin": _ratio(min(ratios["transient"][CARTESIAN_TRANSIENT:])),
        "psBodyTransientMax": _ratio(max(ratios["transient"][CARTESIAN_TRANSIENT:])),
        "psOperationMin": _ratio(min(ratios["operation"])),
        "psOperationMax": _ratio(max(ratios["operation"])),
        "psMaxUnknowns": _count(max(summary[k]["unknowns"] for _, _, k, _ in SCALE_ROWS)),
        "psDayRatio": _ratio(reading("wave15/O1s-transformer-day-x8-skew")["ratio"]),
        "psCycleRatio": _ratio(reading("wave9/O2-engine-L2-cycle-x4")["ratio"]),
        "psOffDesignTransformerRatio": _ratio(
            reading("wave9/O3-transformer-steady-flows")["ratio"]
        ),
        "psTightAmgxRatio": _ratio(reading("wave8/G-steady96")["ratio"]),
        "psAmgxStopMin": _sci(min(stationarity)),
        "psAmgxStopMax": _sci(max(stationarity)),
        "psLFourReferenceMaxInner": _count(l4["reference-ref2"]["max_inner_per_solve"]),
        "psLFourJacobiInnerCap": _count(cap),
        "psLFourJacobiCapped": _count(len(capped)),
        "psLFourJacobiDeclared": _count(l4["jacobi"]["declared"]),
        "psPopulations": _count(counts["main"][0]),
        "psSequences": _count(counts["main"][1]),
        "psTargetSolves": _count(counts["main"][2]),
        "psAllPopulations": _count(counts["all"][0]),
        "psAllSequences": _count(counts["all"][1]),
        "psAllTargetSolves": _count(counts["all"][2]),
        "psDirectTransformer": _seconds(direct["Transformer steady"]["direct"]),
        "psReferenceTransformer": _seconds(direct["Transformer steady"]["reference"]),
        "psDirectBoreTwo": _seconds(direct["Bore L2 steady"]["direct"]),
        "psReferenceBoreTwo": _seconds(direct["Bore L2 steady"]["reference"]),
        "psPilotBlockBoreTwoSlabs": _seconds(pilot_block["median_seconds"]),
        "psPilotDirectBoreThreeTimeout": _count(
            summary["wave5/W5-engine-L3-steady"]["timeout_seconds"]
        ),
        "psHourSkewChange": f"{hour_skew['change']:.2f}",
        "psHourSkewDifference": f"{hour_skew['difference']:.2f}",
        "psHourSkewModulus": f"{hour_skew['modulus']:.2f}",
        "psHourAdvectiveModulus": f"{hour_advective['modulus']:.2f}",
        "psHourAdvectiveVerified": _count(hour_advective["verified"]),
        "psHourSkewVerified": _count(hour_skew["verified"]),
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
    found = populations(root)
    runs = {key: population["methods"] for key, population in found.items()}
    controls = {key: population["controls"] for key, population in found.items()}
    for base, extras in AUGMENT.items():
        for extra in extras:
            if extra not in runs or "amgx" not in runs[extra]:
                raise ValueError(f"The added alternative {extra} of {base} is missing")
    (output / "generated").mkdir(parents=True)
    (output / "figures").mkdir()
    scale_text, scale = table_rows(summary, SCALE_ROWS)
    transient_text, transient = table_rows(summary, TRANSIENT_ROWS)
    operation_text, operation = table_rows(summary, OPERATION_ROWS)
    si_text, _ = table_rows(summary, SI_ROWS)
    direct_text, direct = direct_rows(summary)
    temporal_text, temporal = temporal_rows(root)
    for name, text in (
        ("scale_rows", scale_text),
        ("transient_rows", transient_text),
        ("operation_rows", operation_text),
        ("si_rows", si_text),
        ("direct_rows", direct_text),
        ("temporal_rows", temporal_text),
    ):
        (output / f"generated/{name}.tex").write_text(text)
    index = {
        name: [
            {"label": label, "detail": detail, "population": key, "arm": arm}
            for label, detail, key, arm in spec
        ]
        for name, spec in (
            ("scale_rows", SCALE_ROWS),
            ("transient_rows", TRANSIENT_ROWS),
            ("operation_rows", OPERATION_ROWS),
            ("si_rows", SI_ROWS),
        )
    }
    index["direct_rows"] = [
        {"label": label, "cpu": cpu, "gpu": gpu} for label, cpu, gpu in DIRECT_ROWS
    ]
    index["temporal_rows"] = [
        {"horizon": h, "form": f, "folder": folder} for h, f, folder in TEMPORAL
    ]
    index["merged"] = MERGED
    index["augment"] = AUGMENT
    (output / "rows.json").write_text(json.dumps(index, indent=2) + "\n")
    # Augmented AmgX runs belong to their paired populations and count once there.
    main_keys = [k for spec in (SCALE_ROWS, TRANSIENT_ROWS, OPERATION_ROWS) for _, _, k, _ in spec]
    all_keys = main_keys + [k for _, _, k, _ in SI_ROWS]
    all_keys += [k for _, cpu, gpu in DIRECT_ROWS for k in (cpu, gpu)]
    counts = {"main": campaign_counts(runs, main_keys), "all": campaign_counts(runs, all_keys)}
    text, values = macros(
        summary, runs, controls, scale, transient, operation, counts, direct, temporal
    )
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
    b.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="bundle-path=source-file pairs copied as they are (temporal summaries)",
    )
    e = commands.add_parser("export")
    e.add_argument("--evidence", required=True)
    e.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "bundle":
        extra = []
        for item in args.extra:
            target, separator, source = item.partition("=")
            if not separator or PurePosixPath(target).is_absolute() or ".." in target:
                raise ValueError("Each extra file needs a relative bundle-path=source-file")
            extra.append((target, source))
        manifest = bundle(args.runs, args.campaign, args.output, extra)
        print(f"Bundled {manifest['populations']} populations, {len(manifest['files'])} files")
    else:
        print(json.dumps(export(args.evidence, args.output), indent=2))


if __name__ == "__main__":
    main()
