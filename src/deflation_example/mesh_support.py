"""Generate mesh verification and transfer tables from frozen numerical records."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .reporting import atomic_output, environment, write_report
from .mesh_report import NAMES


def run(validation, transfers, pilots, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs, summaries = {}, {}

    def read(path, key):
        path = Path(path)
        data = path.read_bytes()
        if key in inputs:
            raise ValueError("Every supporting input needs a distinct identifier")
        inputs[key] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def table(name, rows):
        with atomic_output(output / name) as stream:
            stream.write("\n".join(" & ".join(map(str, row)) + r" \\" for row in rows) + "\n")

    checked = read(validation, "validation/validation.json")
    if not checked["success"] or len(checked["cases"]) != 12:
        raise ValueError("Mesh verification requires the complete 12-case population")
    rows = []
    for c in checked["cases"]:
        metrics = [
            *c["independent_kkt"].values(),
            *c["pdas_kkt"].values(),
            c["state_relative_difference"],
            c["objective_relative_difference"],
            c["forward_recovery_relative"],
            c["adjoint_action_relative"],
        ]
        if not np.isfinite(metrics).all() or min(metrics) < 0:
            raise ValueError("Verification metrics must be finite and nonnegative")
        if (
            not c["success"]
            or not c["independent_success"]
            or c["pdas_status"] != "converged"
            or max(c["independent_kkt"].values()) > 1e-8
            or max(c["pdas_kkt"].values()) > 1e-8
            or c["state_relative_difference"] > 1e-7
            or c["forward_recovery_relative"] > 1e-8
            or c["adjoint_action_relative"] > 1e-12
        ):
            raise ValueError("A small mesh verification failed independent optimality")
        name = "Engine" if c["geometry"] == "engine_3d" else "Transformer subproblem"
        steps = c["time_steps"]
        kind = "Steady" if steps is None else "Uniform" if len(set(steps)) == 1 else "Unequal"
        rows.append(
            [
                name,
                kind,
                c["query"] + 1,
                c["dimension"],
                f"{c['state_relative_difference']:.2e}",
                f"{c['objective_relative_difference']:.2e}",
                f"{max(c['pdas_kkt'].values()):.2e}",
            ]
        )
    table("validation_rows.tex", rows)
    summaries["validation"] = checked

    rows, summarized = [], []
    for name, path in transfers.items():
        source = read(path, f"transfer/{name}/transfer.json")
        if not source["success"]:
            raise ValueError("The transfer source contains a failed replay")
        selected = [r for r in source["rows"] if r["methods"]]
        if [r["trace_index"] for r in selected] != source["selection"]:
            raise ValueError("Transfer samples differ from their declared indices")
        for method, label in (
            ("full_reference", "Full reference"),
            ("sequential_transfer", "Sequential transfer"),
        ):
            data = [r["methods"][method] for r in selected]
            repetitions = [p for v in data for p in v["repetitions"]]
            for p in repetitions:
                if not np.isfinite(
                    [
                        p["original_residual"],
                        p["seconds"],
                        p["basis_transfer_seconds"],
                        p["kernel_seconds"],
                    ]
                ).all():
                    raise ValueError("Transfer metrics must be finite")
                if (
                    p["status"] != "converged"
                    or p["original_residual"] > source["source_controls"]["rtol"]
                ):
                    raise ValueError("A transfer repetition fails residual acceptance")
                if not np.isclose(
                    p["seconds"], p["basis_transfer_seconds"] + p["kernel_seconds"], rtol=1e-12
                ):
                    raise ValueError("Transfer timing components overlap")
            if any(v["correction"]["status"] != "verified" for v in data):
                raise ValueError("Coarse energy diagnostics must match the deployed rank")
            ranks = [p["deployed_rank"] for p in repetitions]
            record = {
                "study": name,
                "method": method,
                "trace_kernels": len(source["rows"]),
                "sampled_kernels": len(selected),
                "repetitions": len(repetitions),
                "rank_range": [min(ranks), max(ranks)],
                "median_energy_removed": float(
                    np.median([v["correction"]["coarse_removed_energy_fraction"] for v in data])
                ),
                "summed_median_iterations": sum(
                    float(np.median([p["iterations"] for p in v["repetitions"]])) for v in data
                ),
                "summed_median_seconds": sum(
                    float(np.median([p["seconds"] for p in v["repetitions"]])) for v in data
                ),
                "newly_active": sum(r["newly_active"] for r in source["rows"]),
                "newly_inactive": sum(r["newly_inactive"] for r in source["rows"]),
                "maximum_original_residual": max(p["original_residual"] for p in repetitions),
                "maximum_transfer_identity_error": max(
                    r["transfer_identity_error"] for r in source["rows"]
                ),
            }
            rank_label = (
                str(min(ranks))
                if min(ranks) == max(ranks)
                else "--".join(map(str, record["rank_range"]))
            )
            rows.append(
                [
                    name,
                    label,
                    len(selected),
                    rank_label,
                    f"{record['median_energy_removed']:.3f}",
                    f"{record['summed_median_iterations']:.0f}",
                    f"{record['summed_median_seconds']:.3f}",
                ]
            )
            summarized.append(record)
    table("transfer_rows.tex", rows)
    summaries["transfers"] = summarized

    rows, summarized = [], []
    for name, directory in pilots.items():
        directory = Path(directory)
        protocol = read(directory / "protocol.json", f"pilot/{name}/protocol.json")
        for method in protocol["methods"]:
            record = read(
                directory / f"{method}-0/record.json", f"pilot/{name}/{method}-0/record.json"
            )
            accepted = sum(c["status"] == "converged" for c in record.get("cases", []))
            statuses = sorted(
                {
                    c["status"].removeprefix("inner_")
                    for c in record.get("cases", [])
                    if c["status"] != "converged"
                }
            )
            if record.get("failure"):
                statuses.append(record["failure"]["status"])
            record_summary = {
                "study": name,
                "method": method,
                "accepted_targets": accepted,
                "declared_targets": protocol["targets"],
                "complete_success": record["success"],
                "attempt_seconds": record.get("seconds"),
                "failure_statuses": statuses,
            }
            rows.append(
                [
                    name,
                    NAMES[method],
                    f"{accepted}/{protocol['targets']}",
                    "Accepted"
                    if record["success"]
                    else ", ".join(s.replace("_", " ") for s in statuses),
                    f"{record['seconds']:.3f}" if record.get("seconds") is not None else "---",
                ]
            )
            summarized.append(record_summary)
    table("finer_pilot_rows.tex", rows)
    summaries["finer_pilots"] = summarized
    report = {
        "protocol": "mesh-support-tables-v1",
        "generator_environment": environment(),
        "input_sha256": inputs,
        **summaries,
    }
    write_report(output / "support.json", report)
    return report
