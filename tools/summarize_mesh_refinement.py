"""Generate finer-transformer diagnostic and correction tables from complete records."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from deflation_example.reporting import write_report, atomic_output
from deflation_example.mesh_report import load_study, NAMES


def diagnostics(directory, form):
    directory = Path(directory)
    trace = json.loads((directory / "diagnostics.json").read_text())
    sequence = json.loads((directory / "sequence/record.json").read_text())
    calls = trace[1:]  # The common library warmup solves the initial 32-row system.
    if trace[0]["size"] != 32:
        raise ValueError("Unexpected diagnostic warmup")
    mapping = []
    for case in sequence["cases"]:
        steps = [
            k for k, h in enumerate(case["outer_history"]) if h["active"] < sequence["problem_size"]
        ]
        if len(steps) != len(case["inner"]):
            raise ValueError("PDAS steps differ from the recorded inactive solves")
        mapping.extend((case["query"], step, inner) for step, inner in zip(steps, case["inner"]))
    if len(calls) != len(mapping):
        raise ValueError("Diagnostic calls differ from the complete optimization trace")
    rows = []
    for call, (query, step, inner) in zip(calls, mapping):
        if (
            call["iterations"] != inner["iterations"]
            or call["residual"] != inner["original_residual"]
        ):
            raise ValueError("A scalar diagnostic belongs to a different inactive solve")
        if call["status"] == "converged":
            continue
        event = next(
            (d for d in call["diagnostics"] if d["event"] == call["termination_test"]), None
        )
        if event is None:
            raise ValueError("The failing scalar was not recorded")
        rows.append(
            {
                "form": form,
                "query": query,
                "pdas_step": step,
                "full_dimension": sequence["problem_size"],
                "inactive_dimension": call["size"],
                "source": sequence["environment"]["git_head"],
                "source_sha256": sequence["environment"]["source_sha256"],
                "rhs_sha256": inner["rhs_sha256"],
                "initial_sha256": inner["initial_sha256"],
                **event,
                "events": call["diagnostics"],
            }
        )
    return rows


def generate(diagnostic_paths, repair_paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    failures, comparisons, inputs = [], [], {}
    for form, path in diagnostic_paths.items():
        failures.extend(diagnostics(path, form))
    rows = []
    for form, path in repair_paths.items():
        study, records = load_study(path)
        c = study["controls"]
        original = json.loads((Path(diagnostic_paths[form]) / "sequence/record.json").read_text())
        fixed = (
            "geometry",
            "level",
            "transient",
            "slabs",
            "horizon",
            "alpha",
            "bound",
            "initial_temperature",
            "targets",
            "rank",
            "construction",
            "spatial_reference",
            "temporal_metric",
            "warm_start",
            "inner_cap",
            "outer_cap",
            "residual_refresh",
            "rtol",
            "kkt_tolerance",
            "cg_factor",
            "amgx_factor",
        )
        if any(c[key] != original["controls"][key] for key in fixed):
            raise ValueError("Residual correction changed a physical problem or comparison control")
        if c.get("residual_policy") != "refine" or c["rtol"] != 1e-10 or c["kkt_tolerance"] != 1e-8:
            raise ValueError("The correction study changed the final accuracy requirement")
        if c["targets"] != 4 or c["level"] != 1 or c["geometry"] != "transformer_2d":
            raise ValueError("The correction study must preserve the four-target finer pilot")
        for method in study["methods"]:
            name = method["method"]
            attempts = [
                json.loads(p.read_text()) for p in sorted(Path(path).glob(f"{name}-*/record.json"))
            ]
            for record in attempts:
                if record["problem_size"] != original["problem_size"]:
                    raise ValueError("The correction study changed the state dimension")
                if [case["target_sha256"] for case in record["cases"]] != [
                    case["target_sha256"] for case in original["cases"]
                ]:
                    raise ValueError("The correction study changed the desired target functions")
                for module in (
                    "meshes.py",
                    "mesh_control.py",
                    "mesh_reference.py",
                    "mesh_refinement.py",
                    "mesh_showcases.py",
                    "solvers.py",
                ):
                    if (
                        record["environment"]["source_sha256"][module]
                        != original["environment"]["source_sha256"][module]
                    ):
                        raise ValueError("A physical, reference, or PDAS implementation changed")
            accepted_targets = sum(
                case["status"] == "converged" for r in attempts for case in r.get("cases", [])
            )
            corrections = sum(
                len(i.get("refinement_attempts", [])) - 1
                for r in attempts
                for case in r.get("cases", [])
                for i in case["inner"]
            )
            for r in attempts:
                for case in r.get("cases", []):
                    for inner in case["inner"]:
                        history = inner["refinement_attempts"]
                        if (
                            sum(a["iterations"] for a in history) != inner["iterations"]
                            or inner["iterations"] > c["inner_cap"]
                        ):
                            raise ValueError(
                                "Residual correction exceeded the shared iteration cap"
                            )
                        if len(history) > 5:
                            raise ValueError("The residual-correction budget changed")
            accepted = records[name]
            times = [r["seconds"] for r in accepted]
            success = len(times) == c["repeats"]
            maximum_residual = max(
                (
                    i["original_residual"]
                    for r in accepted
                    for case in r["cases"]
                    for i in case["inner"]
                ),
                default=None,
            )
            maximum_kkt = max(
                (max(case["kkt"].values()) for r in accepted for case in r["cases"]), default=None
            )
            if accepted and (maximum_residual > c["rtol"] or maximum_kkt > c["kkt_tolerance"]):
                raise ValueError("A completed correction sequence fails independent acceptance")
            row = {
                "form": form,
                "method": name,
                "accepted_sequences": len(times),
                "declared_sequences": c["repeats"],
                "accepted_targets": accepted_targets,
                "declared_targets": c["targets"] * c["repeats"],
                "complete_seconds": times,
                "max_original_residual": maximum_residual,
                "max_kkt": maximum_kkt,
                "correction_equations": corrections,
            }
            rows.append(row)
            values = [
                form,
                NAMES[name],
                f"{len(times)}/{c['repeats']}",
                f"{accepted_targets}/{c['targets'] * c['repeats']}",
                f"{np.median(times):.3f} [{min(times):.3f}, {max(times):.3f}]"
                if success
                else "---",
                f"{maximum_residual:.2e}" if accepted else "---",
                f"{maximum_kkt:.2e}" if accepted else "---",
                str(corrections),
            ]
            comparisons.append(" & ".join(values) + r" \\")
    for group, mapping in (("diagnostic", diagnostic_paths), ("correction", repair_paths)):
        for form, path in mapping.items():
            for file in sorted(Path(path).rglob("*.json")):
                inputs[f"{group}/{form}/{file.relative_to(path).as_posix()}"] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
    write_report(
        output / "summary.json",
        {
            "format": "mesh-refinement-summary-v1",
            "input_sha256": inputs,
            "diagnostics": failures,
            "comparisons": rows,
        },
    )
    with atomic_output(output / "diagnostic_rows.tex") as stream:
        for row in failures:
            values = [
                row["form"],
                str(row["query"] + 1),
                str(row["pdas_step"] + 1),
                str(row["inactive_dimension"]),
                str(row["iteration"]),
                f"{row['coarse_condition']:.2f}",
                f"{row['fresh_cpu_residual']:.2e}",
                f"{row['rz']:.2e}",
                f"{row['curvature']:.2e}",
            ]
            stream.write(" & ".join(values) + r" \\" + "\n")
    with atomic_output(output / "correction_rows.tex") as stream:
        stream.write("\n".join(comparisons) + "\n")
    return failures, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for form in ("steady", "transient"):
        parser.add_argument(f"--diagnostic-{form}", required=True, type=Path)
        parser.add_argument(f"--repair-{form}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate(
        {f: getattr(args, f"diagnostic_{f}") for f in ("steady", "transient")},
        {f: getattr(args, f"repair_{f}") for f in ("steady", "transient")},
        args.output,
    )
