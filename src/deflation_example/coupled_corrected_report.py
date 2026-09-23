"""Audit all four corrected pilots without treating them as a final campaign.

Optimization acceptance, paired solution agreement, physical resolution and
performance are separate quantities. Missing and unsuccessful runs stay visible.
This command reads saved fields; it never changes or reconstructs a control.
"""

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_module
import numpy as np
from omegaconf import OmegaConf

from .coupled_convergence import maximum_kkt
from .coupled_report import validate_record
from .coupled_sequence import recorded_configuration, STAGE_SCHEMA, COMPLETE_SCHEMA
from .reporting import environment, file_sha256, write_report


def pilot_configuration(case, device="hybrid"):
    with initialize_config_module(config_module="deflation_example.conf", version_base=None):
        config = compose(config_name="coupled_corrected")
    config.method = case["method"]
    config.alpha = case["alpha"]
    config.rank = case["rank"]
    config.recycle_window = max(1, case["rank"])
    config.device = device
    for key, value in case.get("overrides", {}).items():
        if key not in {"reference_transfer", "reference_construction"}:
            raise ValueError("Only the declared reference construction and transfer may vary")
        config[key] = value
    return recorded_configuration(OmegaConf.to_container(config, resolve=True))


def audit_record(record, case, design, source):
    """Recheck settings, stored independent criteria and nonoverlapping timing."""
    cfg = record["configuration"]
    if cfg.get("device") not in {"cpu", "hybrid"}:
        raise ValueError("The pilot requires CPU or the declared hybrid deployment")
    if cfg != pilot_configuration(case, cfg["device"]):
        raise ValueError("Configuration differs from the declared corrected pilot")
    env = record["environment"]
    if env.get("git_head") != source or env.get("source_tree_clean") is not True:
        raise ValueError("The numerical source must match the frozen clean checkout")
    if record.get("status") == "running":
        if record.get("baseline_sha256") not in {None, design["baseline_sha256"]}:
            raise ValueError("The baseline differs from the declared pilot")
        return {"status": "running", "optimization_verified": False}
    if record.get("baseline_sha256") != design["baseline_sha256"]:
        raise ValueError("The baseline differs from the declared pilot")
    if record.get("schema") != STAGE_SCHEMA or record.get("stage", {}).get("positions") != [0]:
        raise ValueError("This report requires the single target-7 pilot stage")
    if record["stage"].get("restore") is not None or record["stage"].get("resume") is not None:
        raise ValueError("Resumed pilots require complete attempt-cost assembly before comparison")
    # Reuse the accuracy/timer checks without changing the saved stage's identity.
    validation = {**record, "schema": COMPLETE_SCHEMA}
    complete = validate_record(validation)
    outcome = record["cases"][0]
    inner = [
        step
        for outer in outcome.get("history", [])
        for attempt in outer.get("attempts", [])
        for step in attempt.get("qp_history", [])
        if "linear_status" in step
    ]
    converged = [row for row in inner if row["linear_status"] == "converged"]
    residuals = [row["linear_residual"] for row in converged]
    if any(not np.isfinite(value) or not 0 <= value <= 1e-10 for value in residuals):
        raise ValueError("A converged inner solve fails the original-system residual criterion")
    if sum(row.get("linear_iterations", 0) for row in inner) != outcome.get("inner_iterations", 0):
        raise ValueError("The reported inner work differs from the retained history")
    return {
        "status": outcome["status"],
        "optimization_verified": complete,
        "sequence_seconds": record["sequence_seconds"],
        "preparation_inclusive_seconds": record["preparation_inclusive_seconds"],
        "maximum_kkt": maximum_kkt(outcome["kkt"]) if "kkt" in outcome else None,
        "maximum_converged_inner_residual": max(residuals, default=None),
        "objective": outcome.get("objective"),
        "outer_iterations": outcome.get("nonlinear_iterations"),
        "inner_iterations": outcome.get("inner_iterations"),
        "deployed_ranks": sorted({row["deployed_rank"] for row in inner if "deployed_rank" in row}),
        "fallback_count": sum(bool(row.get("fallback")) for row in inner),
        "inner_termination_statuses": sorted({row["linear_status"] for row in inner}),
        "memory": record.get("memory"),
    }


def field_agreement(left, right, *, temperature_scale):
    if set(left) != {"state", "control", "desired"} or set(right) != set(left):
        raise ValueError("Both compared fields require state, source and desired temperature")
    arrays = [np.asarray(side[key]) for side in (left, right) for key in left]
    if (
        any(array.shape != arrays[0].shape for array in arrays)
        or not arrays[0].size
        or any(not np.isfinite(array).all() for array in arrays)
    ):
        raise ValueError("Paired fields must be finite and have identical nonempty shapes")
    if not np.array_equal(left["desired"], right["desired"]):
        raise ValueError("The paired target fields differ")
    return float(np.max(np.abs(left["state"] - right["state"])) * temperature_scale)


def physical_temperature_scale(record):
    """Read physical units from the actual, checksum-bound model input."""
    path = Path(__file__).parent / "data/transformer_2d/parameters.json"
    if record["input_sha256"].get(path.name) != file_sha256(path):
        raise ValueError("Physical parameter checksum differs from the numerical record")
    value = float(json.loads(path.read_text())["physical"]["temperature_scale_K"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("The physical temperature scale must be finite and positive")
    if "thermal_scale_K" in record and record["thermal_scale_K"] != value:
        raise ValueError("The reported temperature scale differs from the model input")
    return value


def summarize(directories, design, source):
    if design.get("schema") != "coupled-corrected-study-v4" or len(directories) != 4:
        raise ValueError("Supply the four declared pilots in case order")
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("Supply the full numerical source commit")
    paths = [Path(path).resolve() for path in directories]
    if len(set(paths)) != 4:
        raise ValueError("Pilot directories must be distinct")
    rows, records = [], {}
    for case, directory in zip(design["pilots"], paths, strict=True):
        row = {**case, "status": "missing", "optimization_verified": False}
        filename = directory / "record.json"
        if filename.exists():
            row["record_sha256"] = file_sha256(filename)
            try:
                record = json.loads(filename.read_text())
                row["reported_sequence_status"] = record.get("status")
                row["reported_target_statuses"] = [
                    item.get("status") for item in record.get("cases", [])
                ]
                row.update(audit_record(record, case, design, source))
                records[case["case"]] = record
            except (ValueError, KeyError, TypeError) as error:
                row.update(status="invalid_record", reason=str(error), optimization_verified=False)
        rows.append(row)
    pairs = []
    for first, second in ((0, 1), (2, 3)):
        jacobi, reference = rows[first], rows[second]
        pair = {
            "alpha": jacobi["alpha"],
            "both_optimization_verified": False,
            "same_solution": False,
            "paired_complete_cost_ratio": None,
            "resolution_and_feasibility_qualified": False,
        }
        if jacobi["optimization_verified"] and reference["optimization_verified"]:
            try:
                a, b = records[first], records[second]
                for key in ("device", "timing_boundary"):
                    if a[key] != b[key]:
                        raise ValueError(f"Paired {key} differs")
                for key in ("source_sha256", "cpu_model", "numpy", "scipy", "blas"):
                    if a["environment"].get(key) != b["environment"].get(key):
                        raise ValueError(f"Paired {key} differs")
                scale = physical_temperature_scale(a)
                if scale != physical_temperature_scale(b):
                    raise ValueError("Temperature scales must match and be positive")
                fields = []
                for index in (first, second):
                    path = paths[index] / "target-00.npz"
                    with np.load(path, allow_pickle=False) as data:
                        fields.append(
                            {key: data[key].copy() for key in ("state", "control", "desired")}
                        )
                    if fields[-1]["state"].size != records[index]["state_dofs_per_problem"]:
                        raise ValueError(
                            "The saved state dimension differs from the numerical record"
                        )
                    rows[index]["field_sha256"] = file_sha256(path)
                maximum = field_agreement(*fields, temperature_scale=scale)
                values = np.array([jacobi["objective"], reference["objective"]], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError("Objectives must be finite")
                difference = float(abs(values[0] - values[1]) / max(np.abs(values).max(), 1e-30))
                same = maximum <= 1e-3 and difference <= 1e-6
                pair.update(
                    both_optimization_verified=True,
                    same_solution=same,
                    maximum_temperature_difference_K=maximum,
                    relative_objective_difference=difference,
                )
                if same:
                    pair["paired_complete_cost_ratio"] = (
                        jacobi["sequence_seconds"] / reference["sequence_seconds"]
                    )
            except (ValueError, KeyError, TypeError, OSError) as error:
                pair["comparison_error"] = (
                    type(error).__name__ if isinstance(error, OSError) else str(error)
                )
        pairs.append(pair)
    return {
        "schema": "coupled-corrected-pilot-summary-v1",
        "numerical_source": source,
        "rows": rows,
        "pairs": pairs,
        "submission_ready": False,
        "scope": "Single target-7 pilots. A paired ratio requires verified, matching solutions and matched timing/hardware. Independent spatial and temporal resolution, unchanged-source feasibility, rank selection, three methods and repeated complete sequences remain separate gates. Missing and unsuccessful outcomes are retained.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=4, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    report = summarize(args.runs, design, args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(
        args.output / "record.json",
        {**report, "environment": environment(), "design_sha256": file_sha256(args.design)},
    )


if __name__ == "__main__":
    main()
