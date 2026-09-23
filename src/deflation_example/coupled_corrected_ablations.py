"""Run frozen-source construction and retention ablations; retain all outcomes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .coupled_corrected_report import (
    audit_record,
    field_agreement,
    physical_temperature_scale,
)
from .reporting import environment, file_sha256, write_report


def read_design(path):
    path = Path(path)
    design = json.loads(path.read_text())
    if design.get("schema") != "coupled-corrected-ablations-v1":
        raise ValueError("Use the corrected six-case ablation design")
    source = design["numerical_source"]
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("The design must identify the full numerical source")
    base_name = design["base_design"]
    if Path(base_name).name != base_name:
        raise ValueError("The base design must be a neighboring filename")
    base = json.loads(path.with_name(base_name).read_text())
    if base.get("schema") != "coupled-corrected-study-v4":
        raise ValueError("The base must be the corrected pilot design")
    expected = [
        (alpha, variant, method, overrides)
        for alpha in (1e-14, 1e-11)
        for variant, method, overrides in (
            ("recycling", "recycling", {}),
            ("sequential", "reference", {"reference_transfer": "sequential"}),
            ("tensor", "reference", {"reference_construction": "tensor"}),
        )
    ]
    cases = design["cases"]
    if len(cases) != 6 or design.get("replications") != 1:
        raise ValueError("The declared ablation has six cases and one repetition")
    for index, (row, (alpha, variant, method, overrides)) in enumerate(
        zip(cases, expected, strict=True)
    ):
        if row != dict(
            case=index, alpha=alpha, variant=variant, method=method, overrides=overrides, rank=200
        ):
            raise ValueError("Ablation order, rank or one-factor settings differ")
    return design, base


def launch_command(design, base, case_id, numerical_root, baseline, output):
    """Use the measured numerical checkout, not the reporting package's solver."""
    numerical_root, baseline, output = map(Path, (numerical_root, baseline, output))
    if isinstance(case_id, bool) or not isinstance(case_id, int) or not 0 <= case_id < 6:
        raise ValueError("Choose one of the six declared cases")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=numerical_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "tests", "examples"],
        cwd=numerical_root,
        text=True,
    ).strip()
    if head != design["numerical_source"] or dirty:
        raise ValueError("The numerical checkout must be the frozen clean source")
    saved = json.loads((baseline / "record.json").read_text())
    if any(
        saved[key] != base[key] for key in ("baseline_sha256", "thermal_nodes", "flow_unknowns")
    ):
        raise ValueError("The baseline differs from the corrected pilot")
    if output.exists():
        raise FileExistsError(output)
    case = design["cases"][case_id]
    overrides = {
        "method": case["method"],
        "alpha": case["alpha"],
        "rank": case["rank"],
        "recycle_window": 200,
        "device": "hybrid",
        **case["overrides"],
    }
    return [
        sys.executable,
        "-m",
        "deflation_example.coupled_sequence",
        "--config-name",
        "coupled_corrected",
        f"baseline_directory={baseline.resolve()}",
        f"output={output.resolve()}",
        *(f"{key}={value}" for key, value in overrides.items()),
    ]


def compare_pair(left, right, left_directory, right_directory):
    """Compare a completed variant with full-reference reuse at the same alpha."""
    for key in ("device", "timing_boundary", "input_sha256", "state_dofs_per_problem"):
        if left[key] != right[key]:
            raise ValueError(f"Paired {key} differs")
    for key in ("source_sha256", "cpu_model", "numpy", "scipy", "blas"):
        if left["environment"].get(key) != right["environment"].get(key):
            raise ValueError(f"Paired {key} differs")
    scale = physical_temperature_scale(left)
    if scale != physical_temperature_scale(right):
        raise ValueError("Physical temperature scales differ")
    fields, hashes = [], []
    for record, directory in ((left, left_directory), (right, right_directory)):
        path = Path(directory) / "target-00.npz"
        with np.load(path, allow_pickle=False) as data:
            fields.append({key: data[key].copy() for key in ("state", "control", "desired")})
        if fields[-1]["state"].size != record["state_dofs_per_problem"]:
            raise ValueError("Stored field size differs from the record")
        hashes.append(file_sha256(path))
    difference = field_agreement(*fields, temperature_scale=scale)
    objectives = np.array([record["cases"][0]["objective"] for record in (left, right)])
    if not np.isfinite(objectives).all():
        raise ValueError("Objectives must be finite")
    relative = float(abs(objectives[0] - objectives[1]) / max(abs(objectives).max(), 1e-30))
    same = difference <= 0.001 and relative <= 1e-6
    return {
        "same_solution": same,
        "maximum_temperature_difference_K": difference,
        "relative_objective_difference": relative,
        "field_sha256": hashes,
        "variant_seconds_over_full_reference_seconds": left["sequence_seconds"]
        / right["sequence_seconds"]
        if same
        else None,
    }


def summarize(baselines, ablations, design, base):
    if len(baselines) != 4 or len(ablations) != 6:
        raise ValueError("Supply all four original pilots and all six ablation slots")
    paths = [Path(path).resolve() for path in (*baselines, *ablations)]
    if len(set(paths)) != 10:
        raise ValueError("Every declared run needs a distinct directory")
    rows, records = [], {}
    declared = [
        {**case, "variant": "jacobi" if case["method"] == "jacobi" else "full_reference"}
        for case in base["pilots"]
    ] + design["cases"]
    for index, (case, path) in enumerate(zip(declared, paths, strict=True)):
        row = {
            **case,
            "population": "baseline" if index < 4 else "ablation",
            "status": "missing",
            "optimization_verified": False,
        }
        filename = path / "record.json"
        if filename.exists():
            row["record_sha256"] = file_sha256(filename)
            try:
                record = json.loads(filename.read_text())
                row["reported_status"] = record.get("status")
                row.update(audit_record(record, case, base, design["numerical_source"]))
                records[index] = record
            except (KeyError, ValueError, TypeError) as error:
                row.update(status="invalid_record", reason=str(error))
        rows.append(row)
    pairs = []
    for index in (0, 2, 4, 5, 6, 7, 8, 9):
        reference = 1 if rows[index]["alpha"] == 1e-14 else 3
        pair = {
            "alpha": rows[index]["alpha"],
            "variant": rows[index]["variant"],
            "both_optimization_verified": rows[index]["optimization_verified"]
            and rows[reference]["optimization_verified"],
            "same_solution": False,
            "variant_seconds_over_full_reference_seconds": None,
        }
        if pair["both_optimization_verified"]:
            try:
                pair.update(
                    compare_pair(records[index], records[reference], paths[index], paths[reference])
                )
            except (KeyError, ValueError, TypeError, OSError) as error:
                pair["comparison_error"] = (
                    type(error).__name__ if isinstance(error, OSError) else str(error)
                )
        pairs.append(pair)
    return {
        "schema": "coupled-corrected-ablation-summary-v1",
        "numerical_source": design["numerical_source"],
        "rows": rows,
        "pairs": pairs,
        "submission_ready": False,
        "scope": design["scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--case", type=int, choices=range(6), required=True)
    run.add_argument("--numerical-checkout", type=Path, required=True)
    run.add_argument("--baseline", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--baselines", type=Path, nargs=4, required=True)
    report.add_argument("--ablations", type=Path, nargs=6, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design, base = read_design(args.design)
    if args.command == "run":
        command = launch_command(
            design, base, args.case, args.numerical_checkout, args.baseline, args.output
        )
        if args.dry_run:
            print(" ".join(command))
            return
        env = {**os.environ, "PYTHONPATH": str(args.numerical_checkout.resolve() / "src")}
        subprocess.run(command, cwd=args.numerical_checkout, env=env, check=True)
    else:
        result = summarize(args.baselines, args.ablations, design, base)
        args.output.mkdir(parents=True, exist_ok=False)
        write_report(
            args.output / "record.json",
            {
                **result,
                "environment": environment(),
                "design_sha256": file_sha256(args.design),
                "base_design_sha256": file_sha256(args.design.with_name(design["base_design"])),
            },
        )


if __name__ == "__main__":
    main()
