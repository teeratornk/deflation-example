"""Archive every guarded GPU repetition and regenerate its checked summaries."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from deflation_example.mesh_report import generate, load_study
from deflation_example.reporting import atomic_output, write_report

SOURCE = "0b3b3f06a535f515256932c0dbf27537c242744d"


def archive(steady, transient, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "guarded-refinement-evidence-v1", "numerical_source": SOURCE, "files": {}}
    targets = []
    for form, source in (("steady", Path(steady)), ("transient", Path(transient))):
        summary, _ = load_study(source)
        c = summary["controls"]
        expected = {
            "geometry": "transformer_2d",
            "level": 1,
            "targets": 4,
            "repeats": 5,
            "rank": 100,
            "recycle_rank": 100,
            "rtol": 1e-10,
            "kkt_tolerance": 1e-8,
            "residual_policy": "refine",
            "inner_cap": 50000,
            "warm_start": "outer_inner",
            "transient": form == "transient",
        }
        if any(c[k] != v for k, v in expected.items()):
            raise ValueError("The comparison differs from the declared guarded follow-up")
        target = output / "records" / form
        target.mkdir(parents=True)
        paths = [
            source / "protocol.json",
            source / "attempts.json",
            *sorted(source.glob("*/record.json")),
        ]
        for path in paths:
            if path.name == "record.json":
                r = json.loads(path.read_text())
                if (
                    r["environment"]["git_head"] != SOURCE
                    or not r["environment"]["source_tree_clean"]
                ):
                    raise ValueError("The guarded study contains a different numerical source")
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            manifest["files"][destination.relative_to(output).as_posix()] = hashlib.sha256(
                destination.read_bytes()
            ).hexdigest()
        targets.append(target)
    generate(targets, output / "summary", plots=False)
    write_report(output / "manifest.json", manifest)
    summaries = json.loads((output / "summary/summary.json").read_text())["studies"]
    lines = [
        "# Guarded residual correction: complete GPU comparisons",
        "",
        "These separate measurements use the recommended initial-residual guard on the finer transformer.",
        "Each method solves four targets per sequence, with five complete repetitions per form.",
        "The published primary and correction-study timings remain unchanged.",
        "",
        "| Form | Method | Completed / attempted | Median time [full range] (s) | Maximum original residual | Maximum KKT component |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        for m in s["methods"]:
            times = m["complete_seconds"]
            timing = (
                f"{m['median_seconds']:.3f} [{min(times):.3f}, {max(times):.3f}]" if times else "—"
            )
            residual = f"{m['max_original_residual']:.3e}" if times else "—"
            kkt = f"{m['max_kkt']:.3e}" if times else "—"
            lines.append(
                f"| {'Four slabs' if s['controls']['transient'] else 'Steady'} | {m['method']} | {m['accepted']} / {m['requested']} | {timing} | {residual} | {kkt} |"
            )
    lines.extend(
        [
            "",
            "Times include reference construction, all active-set steps, transfers, verification and solver cleanup.",
            "Library preparation and full process times remain separately identified in the records.",
            "KKT denotes the Karush–Kuhn–Tucker conditions; primal feasibility is absolute, and the other components follow the normalized definitions in the protocol.",
            "All methods use the same final residual and KKT criteria. The records retain each iteration, correction, rank and sampled process-memory measurement.",
            "",
            "The initial-residual guard is a numerical safeguard; these timings do not isolate its causal effect from changes in hardware load between studies.",
            "AmgX has the lowest steady median; Jacobi-CG has the lowest transient median, with overlapping reference/Jacobi timing ranges.",
            "",
            "## Reproduce",
            "",
            "The [GPU commands](../temporal_resolution/guarded_gpu.md) pin the measured numerical source and configurations.",
            "To regenerate the tables and figures from these raw records, use a checkout containing this archive:",
            "",
            "```bash",
            "uv sync --frozen --extra study --extra plot",
            "uv run --frozen python -m deflation_example.mesh_report \\",
            "  --studies examples/guarded_refinement/records/steady examples/guarded_refinement/records/transient \\",
            "  --output runs/guarded-refinement-summary",
            "```",
            "",
            "The SHA-256 manifest covers the complete comparison records. No temperature or control field is needed to regenerate the timing and accuracy summaries; the numerical reproduction commands regenerate those fields.",
        ]
    )
    with atomic_output(output / "README.md") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steady", required=True)
    parser.add_argument("--transient", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive(args.steady, args.transient, args.output)
