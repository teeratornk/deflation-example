"""Validate complete CHT records and summarize accepted sequences separately."""

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from .benchmark_cht import unpack_mask
from .reporting import atomic_output, write_report
from .tutorials import Records


def close(a, b, name):
    if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a, b, rtol=1e-8, atol=1e-8):
        raise ValueError(f"Inconsistent {name}")


def validate_sequence(sequence, protocol):
    c = protocol["controls"]
    if "total_seconds" not in sequence:
        if sequence["success"]:
            raise ValueError("An accepted sequence requires complete timing and accuracy records")
        return
    parts = sequence["components_seconds"]
    if any(not np.isfinite(v) or v < -1e-9 for v in parts.values()):
        raise ValueError("Invalid complete timing component")
    close(sum(parts.values()), sequence["total_seconds"], "complete timing sum")
    cases = sequence["cases"]
    close(
        sum(case["inner_iterations"] for case in cases), sequence["inner_iterations"], "inner work"
    )
    close(
        sum(case["outer_iterations"] for case in cases), sequence["outer_iterations"], "outer work"
    )
    dimension = c["n"] ** 3 * (c["slabs"] if c["problem"] == "transient" else 1)
    if sequence["success"] and (
        len(cases) != c["targets"] or sequence.get("error_type") is not None
    ):
        raise ValueError("Accepted sequence has missing targets or an error")
    last = 0.0
    for case in cases:
        if (
            case["cumulative_seconds"] < last
            or case["cumulative_seconds"] > sequence["total_seconds"]
        ):
            raise ValueError("Invalid cumulative sequence timing")
        last = case["cumulative_seconds"]
        if sequence["success"] and case["status"] != "converged":
            raise ValueError("Accepted sequence contains a failed target")
        if case["status"] == "converged":
            kkt = case["kkt"]
            if not kkt or any(
                not np.isfinite(v) or v > c["outer_tolerance"] or v < 0 for v in kkt.values()
            ):
                raise ValueError("Accepted target fails KKT criteria")
            active = unpack_mask(case["final_active_bits"], dimension)
            if active.sum() != case["final_active_count"]:
                raise ValueError("Final active count differs from its mask")
            if c["problem"] == "transient":
                if any(
                    not np.isfinite(v) or v > c["outer_tolerance"]
                    for v in case["recovery"].values()
                ):
                    raise ValueError("Accepted trajectory fails recovery checks")
                masks = active.reshape(c["slabs"], c["n"] ** 3)
                if (
                    case["physical_time_newly_active"]
                    != np.sum(masks[1:] & ~masks[:-1], axis=1).tolist()
                    or case["physical_time_newly_inactive"]
                    != np.sum(~masks[1:] & masks[:-1], axis=1).tolist()
                ):
                    raise ValueError("Physical-time activation or release count differs from masks")
        for inner in case["inner"]:
            close(
                sum(inner["components_seconds"].values()),
                inner["total_seconds"],
                "kernel timing sum",
            )
            if inner["status"] == "converged" and (
                not np.isfinite(inner["original_residual"])
                or inner["original_residual"] > c["rtol"]
            ):
                raise ValueError("Accepted inner solve fails the original residual check")
            if not np.isclose(inner["acceptance_rtol"], c["rtol"], rtol=1e-12, atol=0):
                raise ValueError("Final residual target differs from the protocol")
            expected = c["rtol"] * (c["amgx_factor"] if sequence["method"] == "amgx" else 1.0)
            if not np.isclose(inner["iteration_rtol"], expected, rtol=1e-12, atol=0):
                raise ValueError("Internal stopping target differs from the protocol")
            mask = unpack_mask(inner["inactive_mask_bits"], dimension)
            if mask.sum() != inner["n_inactive"]:
                raise ValueError("Inactive count differs from its mask")
            rank = inner["deployed_rank"]
            if not 0 <= rank <= inner["input_basis_columns"]:
                raise ValueError("Deployed rank exceeds supplied directions")
            if rank and inner["coarse_condition"] > protocol["coarse_condition_limit"] * (
                1 + 1e-10
            ):
                raise ValueError("Deployed space exceeds coarse conditioning limit")
            if inner["fallback_reason"] is not None and rank != 0:
                raise ValueError("Fallback did not deploy rank zero")
            selection = inner.get("completion")
            if selection and selection["status"] == "selected":
                if selection["selected_rank"] > min(c["recycle_rank"], selection["candidate_rank"]):
                    raise ValueError("Recycled rank exceeds its candidate or retained budget")
                if (
                    selection["candidate_columns"]
                    != selection["existing_columns"] + selection["new_directions_retained"]
                ):
                    raise ValueError("Recycling candidate pool differs from the declared policy")
    if "process_seconds" in sequence:
        measured = sum(
            sequence[key]
            for key in (
                "total_seconds",
                "initialization_seconds",
                "warmup_seconds",
                "finalization_seconds",
                "launch_import_report_and_exit_seconds",
            )
        )
        close(measured, sequence["process_seconds"], "process timing sum")


def load_study(root):
    records = Records(root)
    manifest, protocol = records.load("results.json"), records.load("protocol.json")
    if manifest["protocol_sha256"] != records.manifest["protocol.json"]:
        raise ValueError("Protocol hash differs from manifest")
    c = protocol["controls"]
    expected = {
        (m, w, r) for m in c["methods"] for w in c["warm_starts"] for r in range(c["repeats"])
    }
    seen, sequences = set(), []
    for entry in manifest["sequences"]:
        key = (entry["method"], entry["warm_start"], entry["repetition"])
        if key not in expected or key in seen:
            raise ValueError("Duplicate or undeclared sequence")
        seen.add(key)
        sequence = records.load(entry["path"])
        if (
            entry["sha256"] != records.manifest[entry["path"]]
            or sequence["protocol_sha256"] != manifest["protocol_sha256"]
        ):
            raise ValueError("Sequence hash differs from its manifest")
        if (sequence["method"], sequence["warm_start"], sequence["repetition"]) != key or sequence[
            "success"
        ] != entry["success"]:
            raise ValueError("Sequence labels or acceptance differ from the manifest")
        validate_sequence(sequence, protocol)
        sequences.append(sequence)
    if manifest.get("complete") and seen != expected:
        raise ValueError("Completed study has missing declared sequences")
    if manifest["success"] and (seen != expected or not all(s["success"] for s in sequences)):
        raise ValueError("Accepted study has incomplete or failed sequences")
    return protocol, sequences, records.manifest


def summarize(protocol, sequences):
    c = protocol["controls"]
    rows = []
    for warm in c["warm_starts"]:
        for method in c["methods"]:
            selected = [s for s in sequences if s["method"] == method and s["warm_start"] == warm]
            accepted = [s for s in selected if s["success"]]
            failed = [s for s in selected if not s["success"]]

            def values(field):
                return [s[field] for s in accepted if field in s]

            def median(field):
                v = values(field)
                return float(np.median(v)) if v else None

            inner = [i for s in accepted for case in s["cases"] for i in case["inner"]]
            memory = [s["memory"] for s in accepted if s["memory"]["complete"]]
            timings = values("total_seconds")
            rows.append(
                {
                    "problem": c["problem"],
                    "n": c["n"],
                    "slabs": c["slabs"] if c["problem"] == "transient" else 1,
                    "targets": c["targets"],
                    "reference_rank": c["rank"],
                    "recycle_rank": c["recycle_rank"],
                    "window": c["window"],
                    "construction": c["reference_construction"]
                    if c["problem"] == "transient"
                    else "analytical",
                    "method": method,
                    "warm_start": warm,
                    "declared_sequences": c["repeats"],
                    "recorded_sequences": len(selected),
                    "accepted_sequences": len(accepted),
                    "failure_statuses": dict(
                        Counter(
                            s.get("status", s.get("error_type") or "target_failure") for s in failed
                        )
                    ),
                    "failed_target_statuses": dict(
                        Counter(
                            case["status"]
                            for s in failed
                            for case in s.get("cases", [])
                            if case["status"] != "converged"
                        )
                    ),
                    "median_seconds": median("total_seconds"),
                    "minimum_seconds": min(timings) if timings else None,
                    "maximum_seconds": max(timings) if timings else None,
                    "median_preparation_inclusive_seconds": median("preparation_inclusive_seconds"),
                    "median_inner_iterations": median("inner_iterations"),
                    "median_outer_iterations": median("outer_iterations"),
                    "deployed_rank_min": min((i["deployed_rank"] for i in inner), default=None),
                    "deployed_rank_max": max((i["deployed_rank"] for i in inner), default=None),
                    "fallbacks": sum(i["fallback_reason"] is not None for i in inner),
                    "maximum_original_residual": max(
                        (i["original_residual"] for i in inner), default=None
                    ),
                    "verified_memory_sequences": len(memory),
                    "peak_host_rss_bytes": max(
                        (m["peak_host_rss_bytes"] for m in memory), default=None
                    ),
                    "peak_gpu_process_bytes": max(
                        (m["peak_gpu_process_bytes"] for m in memory), default=None
                    ),
                    "maximum_memory_sampling_gap_seconds": max(
                        (m["maximum_sampling_gap_seconds"] for m in memory), default=None
                    ),
                    "median_components_seconds": {
                        k: float(np.median([s["components_seconds"][k] for s in accepted]))
                        for k in accepted[0]["components_seconds"]
                    }
                    if accepted
                    else {},
                }
            )
    return rows


def run(records_roots, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "protocol": "complete-cht-summary-v1",
        "studies": [],
        "rows": [],
        "scope": "medians and ranges of independently repeated accepted complete sequences; failed and missing declared attempts remain separate; component medians need not sum to the median total",
    }
    for root in records_roots:
        protocol, sequences, hashes = load_study(root)
        identifier = Path(root).name
        result["studies"].append(
            {"identifier": identifier, "protocol": protocol, "input_sha256": hashes}
        )
        result["rows"].extend(
            {"study": identifier, **row} for row in summarize(protocol, sequences)
        )
    write_report(output / "summary.json", result)
    columns = (
        [k for k in result["rows"][0] if k != "median_components_seconds"]
        if result["rows"]
        else ["study"]
    )
    with atomic_output(output / "summary.csv") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(result["rows"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-roots", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
