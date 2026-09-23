"""Accuracy- and coverage-gated four-way complete optimization comparisons."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .coupled_report import validate_record
from .reporting import file_sha256, write_report
from .validation import integer


ARM_KEYS = {"method", "rank", "recycle_window", "inner_preconditioner", "reference_selection"}
ARMS = {"jacobi", "recycling", "frozen", "reference"}


def inner_evidence(record):
    """Audit actual inner work without confusing requested and deployed ranks."""
    ranks, statuses, fallbacks = Counter(), Counter(), Counter()
    iterations = 0
    used_space = False
    complete = True
    maximum_residual = None
    requested = integer(record["configuration"]["rank"], "Requested rank", 0)
    tolerance = record["configuration"]["inner_tolerance"]
    for case in record["cases"]:
        if "history" not in case:
            complete = False
            continue
        case_iterations = 0
        for outer in case["history"]:
            for attempt in outer["attempts"]:
                for step in attempt["qp_history"]:
                    if "linear_status" not in step:
                        continue
                    status = step["linear_status"]
                    count = integer(step["linear_iterations"], "Inner iterations", 0)
                    rank = integer(step["deployed_rank"], "Deployed rank", 0)
                    if rank > requested:
                        raise ValueError("Deployed rank exceeds the requested rank")
                    ranks[str(rank)] += 1
                    reason = step.get("fallback")
                    if reason is not None:
                        fallbacks[str(reason)] += 1
                    case_iterations += count
                    used_space |= rank > 0 and count > 0 and step.get("candidate_retained", False)
                    if status == "converged":
                        residual = step.get("linear_residual")
                        if residual is None or not np.isfinite(residual) or residual < 0:
                            raise ValueError("Converged inner solve fails original-residual check")
                        if residual > tolerance:
                            if (
                                step.get("candidate_retained", False)
                                or attempt.get("qp_status") != "linear_original_residual_failed"
                            ):
                                raise ValueError(
                                    "Converged inner solve fails original-residual check"
                                )
                            status = "original_residual_failed"
                        else:
                            maximum_residual = max(maximum_residual or 0.0, residual)
                    statuses[status] += 1
        if case_iterations != case.get("inner_iterations"):
            raise ValueError("Inner iteration total differs from the recorded solve histories")
        iterations += case_iterations
    return {
        "complete_histories": complete,
        "recorded_inner_iterations": iterations,
        "deployed_rank_counts": dict(sorted(ranks.items(), key=lambda item: int(item[0]))),
        "termination_counts": dict(sorted(statuses.items())),
        "fallback_counts": dict(sorted(fallbacks.items())),
        "maximum_converged_original_residual": maximum_residual,
        "nonzero_coarse_space_used": used_space,
        "scope": "All recorded inner attempts, including rejected outer attempts; final ranks describe returned solver states.",
    }


def summarize(records, settings, fields=None):
    """Require the declared arms and repetitions; preserve all unsuccessful attempts."""
    repetitions = integer(settings["repetitions"], "Repetitions", 1)
    targets = settings["targets"]
    arms = settings["arms"]
    if set(arms) != ARMS or len(targets) != 3 or len(set(targets)) != 3:
        raise ValueError("Declare four solver arms and three distinct targets")
    if any(set(row) != ARM_KEYS for row in arms.values()):
        raise ValueError("Arm settings may change only the declared solver choices")
    phase = settings.get("phase", "development")
    if phase not in {"development", "confirmation"} or (
        phase == "confirmation" and repetitions != 5
    ):
        raise ValueError("Confirmation requires five new complete repetitions")
    required = {
        "jacobi": ("jacobi", "jacobi"),
        "frozen": ("jacobi", "frozen"),
        "reference": ("reference", "frozen"),
        "recycling": ("recycling", "frozen"),
    }
    if any(
        (arms[name]["method"], arms[name]["inner_preconditioner"]) != value
        for name, value in required.items()
    ):
        raise ValueError("The four arms must include matched frozen preconditioning")
    if (
        arms["jacobi"]["rank"] != 0
        or arms["frozen"]["rank"] != 0
        or arms["reference"]["rank"] != arms["recycling"]["rank"]
        or arms["reference"]["rank"] <= 0
    ):
        raise ValueError("Rank-zero controls and rank-matched reference/recycling are required")
    groups = {name: [] for name in arms}
    identity = None
    seen = set()
    for index, record in enumerate(records):
        verified = validate_record(record)
        cfg = record["configuration"]
        for key, threshold in (
            ("inner_tolerance", 1e-10),
            ("equation_acceptance_tolerance", 1e-12),
            ("conservation_tolerance", 1e-6),
            ("nonlinear_tolerance", 1e-8),
        ):
            if cfg.get(key) != threshold:
                raise ValueError("The final accuracy requirements must remain unchanged")
        actual = {key: cfg[key] for key in ARM_KEYS}
        matching = [name for name, row in arms.items() if row == actual]
        if len(matching) != 1:
            raise ValueError("A record must match exactly one frozen solver configuration")
        name = matching[0]
        repeat = integer(cfg["repetition"], "Repetition", 0)
        if repeat >= repetitions or (name, repeat) in seen:
            raise ValueError("Repeated or out-of-range confirmation repetition")
        seen.add((name, repeat))
        if [q["target"] for q in cfg["queries"]] != targets:
            raise ValueError("The confirmation target order differs")
        env = record["environment"]
        current = {
            "configuration": {k: v for k, v in cfg.items() if k not in ARM_KEYS | {"repetition"}},
            "source": env["source_sha256"],
            "cpu": env["cpu_model"],
            "numpy": env["numpy"],
            "scipy": env["scipy"],
            "blas": env.get("blas"),
            "device": record["device"],
            "baseline": record["baseline_sha256"],
            "numerical_policy": record["numerical_policy"],
            "timing_boundary": record["timing_boundary"],
        }
        if record.get("assembly") or cfg.get("stage"):
            raise ValueError("Confirmation requires independently timed single-process sequences")
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError("Sources, deployment, timing and physical settings must match")
        groups[name].append(
            {
                "index": index,
                "repetition": repeat,
                "verified": verified,
                "seconds": record["sequence_seconds"],
                "status": record["status"],
                "case_statuses": [c["status"] for c in record["cases"]],
                "inner_iterations": sum(c.get("inner_iterations", 0) for c in record["cases"]),
                "outer_iterations": sum(c.get("nonlinear_iterations", 0) for c in record["cases"]),
                "sampled_gpu_peak_bytes": record["memory"].get("peak_gpu_process_bytes"),
                "inner_evidence": inner_evidence(record),
            }
        )
    rows = []
    for name, outcomes in groups.items():
        eligible = len(outcomes) == repetitions and all(r["verified"] for r in outcomes)
        times = [r["seconds"] for r in outcomes]
        rows.append(
            {
                "arm": name,
                "eligible": eligible,
                "outcomes": outcomes,
                "median_seconds": float(np.median(times)) if eligible else None,
                "range_seconds": [min(times), max(times)] if eligible else None,
            }
        )
    complete = all(r["eligible"] for r in rows)
    histories_complete = complete and all(
        item["inner_evidence"]["complete_histories"] for row in rows for item in row["outcomes"]
    )
    reference_used = bool(groups["reference"]) and all(
        item["inner_evidence"]["nonzero_coarse_space_used"] for item in groups["reference"]
    )
    agreement = None
    if complete and fields is not None:
        if len(fields) != len(records):
            raise ValueError("Every record must have its corresponding saved fields")
        state_error, objective_error = 0.0, 0.0
        for index, record in enumerate(records):
            for position in range(3):
                a, b = fields[0][position], fields[index][position]
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("Comparison states must be finite and dimensionally matched")
                state_error = max(state_error, float(np.max(np.abs(a - b))))
                oa, ob = (
                    records[0]["cases"][position]["objective"],
                    record["cases"][position]["objective"],
                )
                if not np.isfinite([oa, ob]).all():
                    raise ValueError("Objectives must be finite")
                objective_error = max(objective_error, abs(oa - ob) / max(abs(oa), abs(ob), 1e-30))
        agreement = {
            "maximum_absolute_state_difference": state_error,
            "maximum_relative_objective_difference": objective_error,
            "passed": state_error <= 1e-6 and objective_error <= 1e-6,
            "state_units": "Declared nondimensional temperature",
        }
    ratio = None
    ranges_separated = False
    if complete and agreement is not None and agreement["passed"]:
        by_name = {r["arm"]: r for r in rows}
        reference = by_name["reference"]
        fastest = min(
            (r for r in rows if r["arm"] != "reference"), key=lambda r: r["median_seconds"]
        )
        ratio = fastest["median_seconds"] / reference["median_seconds"]
        ranges_separated = reference["range_seconds"][1] < fastest["range_seconds"][0]
    return {
        "schema": "coupled-confirmation-summary-v1",
        "methods": rows,
        "all_sequences_verified": complete,
        "inner_histories_complete": histories_complete,
        "reference_used_in_every_reference_sequence": reference_used,
        "solution_agreement": agreement,
        "fastest_tested_alternative_over_reference": ratio,
        "ten_percent_time_saving": ratio is not None and ratio >= 1 / 0.9,
        "observed_timing_ranges_separated": ranges_separated,
        "phase": phase,
        "publication_gate_passed": phase == "confirmation"
        and histories_complete
        and reference_used
        and ratio is not None
        and ratio >= 1 / 0.9
        and ranges_separated,
        "scope": "Complete fixed-target sequences; field agreement and all repetitions are required for the speedup. Observed timing ranges describe these repetitions, not a statistical guarantee.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.records]
    fields = []
    for path, record in zip(args.records, records, strict=True):
        row = []
        if record.get("all_problems_verified"):
            for i in range(len(record["cases"])):
                with np.load(path.parent / f"target-{i:02d}.npz", allow_pickle=False) as data:
                    row.append(data["state"].copy())
        fields.append(row)
    result = summarize(records, json.loads(args.settings.read_text()), fields)
    result["settings_sha256"] = file_sha256(args.settings)
    result["records"] = [
        {"file": p.parent.name + "/" + p.name, "sha256": file_sha256(p)} for p in args.records
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(args.output / "summary.json", result)


if __name__ == "__main__":
    main()
