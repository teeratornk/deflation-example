"""Verify frozen mesh evidence and reproduce comparison artifacts without a GPU."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from .mesh_report import generate
from .mesh_figures import plot_showcases
from .mesh_support import run as support
from .reporting import atomic_output, write_report


PRIMARY = (
    "transformer-steady", "transformer-transient4", "engine-level1-steady",
    "engine-level1-transient4", "engine-level2-steady", "engine-level2-transient4",
)
CONTROLS = ("engine-level2-rank20", "engine-level2-rank200", "engine-level1-transient8", "transformer-transient8")


def memory_envelope(studies, output, budgets=(1.5, 2.5, 5.0)):
    """A retrospective envelope over measured ranks, using complete NVML samples.

    This comparison does not enforce an allocation cap during execution. Rank-zero
    methods use their rank-100 comparison records to avoid selecting among timing
    repetitions from different rank studies.
    """
    selected = [s for s in studies if s["controls"]["geometry"] == "engine_3d"
                and s["controls"]["level"] == 2 and not s["controls"]["transient"]]
    if sorted(s["controls"]["rank"] for s in selected) != [20, 100, 200]:
        raise ValueError("The memory comparison needs exactly ranks 20, 100 and 200")
    rows = []
    for budget in budgets:
        for method in ("jacobi", "reference", "recycling", "amgx"):
            candidates = []
            for study in selected:
                rank = study["controls"]["rank"]
                if method in {"jacobi", "amgx"} and rank != 100:
                    continue
                m = next(m for m in study["methods"] if m["method"] == method)
                if (m["accepted"] != m["requested"] or m["complete_memory_measurements"] != m["requested"]
                    or max(m["peak_gpu_bytes"])/2**30 > budget):
                    continue
                candidates.append((m["median_seconds"], rank, max(m["peak_gpu_bytes"])/2**30))
            best = min(candidates) if candidates else None
            rows.append({"gpu_budget_gib": budget, "method": method,
                         "median_seconds": best[0] if best else None,
                         "requested_rank": (best[1] if method in {"reference", "recycling"} else 0) if best else None,
                         "sampled_peak_gpu_gib": best[2] if best else None})
    write_report(Path(output) / "memory_envelope.json", {
        "scope": "Retrospective minimum median among accepted measured ranks under a common sampled GPU-memory budget; no allocation cap was imposed.",
        "rows": rows})
    with atomic_output(Path(output) / "memory_rows.tex") as stream:
        for row in rows:
            values = [f"{row['gpu_budget_gib']:.1f}", row["method"].title(),
                      str(row["requested_rank"]) if row["requested_rank"] is not None else "---"]
            values += [f"{row[k]:.3f}" if row[k] is not None else "---"
                       for k in ("median_seconds", "sampled_peak_gpu_gib")]
            stream.write(" & ".join(values) + r" \\" + "\n")
    return rows


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "mesh-cht-evidence-v1" or not manifest.get("files"):
        raise ValueError("Unsupported or empty mesh evidence manifest")
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*")
                if p.is_file() and ".git" not in p.relative_to(root).parts
                and p.relative_to(root).as_posix() not in {"manifest.json", "README.md", ".gitignore"}}
    declared = set(manifest["files"])
    if observed - declared:
        raise ValueError("The evidence tree contains unlisted files")
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Unsafe evidence path")
        path = root / name
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Evidence files must exist without symbolic links")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Evidence hash mismatch: {name}")
    return manifest


def run(root, output=None):
    root = Path(root)
    manifest = verify(root)
    if output is None:
        return manifest
    groups = {p.name for p in (root / "final").iterdir() if p.is_dir()}
    if groups != set(PRIMARY + CONTROLS):
        raise ValueError("The repeated mesh evidence requires all declared populations")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    primary = generate([root / "final" / name for name in PRIMARY], output / "primary")
    controls = generate([root / "final" / name for name in CONTROLS], output / "controls")
    for study in primary + controls:
        if any("not_recorded" in m["failures"] for m in study["methods"]):
            raise ValueError("A declared repeated comparison has missing attempts")
    memory_envelope(primary + controls, output)
    plot_showcases(root / "final/transformer-transient4/reference-0",
                    root / "final/engine-level2-transient4/reference-0",
                    output / "showcases.pdf")
    support(root / "diagnostic/independent-validation/validation.json",
            {"Engine steady": root / "diagnostic/engine-steady-transfer/transfer.json",
             "Engine transient": root / "diagnostic/engine-transient-transfer/transfer.json",
             "Transformer transient": root / "diagnostic/transformer-transient-transfer/transfer.json"},
            {"Steady": root / "pilot/finer-transformer-steady",
             "Transient": root / "pilot/finer-transformer-transient"},
            output / "support")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.records, args.output)
    print(f"Verified {len(report['files'])} frozen mesh evidence files")


if __name__ == "__main__":
    main()
