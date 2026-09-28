"""Retain the fixed-quadratic CUDA rank control and its construction costs."""

import argparse
import json
from pathlib import Path

from .coupled_qp_inexact_report import summarize as direction_summary
from .reporting import file_sha256, write_report


def summarize(records, ranks):
    if len(records) != len(ranks) or len(set(ranks)) != len(ranks) or any(r < 0 for r in ranks):
        raise ValueError("Declare one record per distinct nonnegative rank")
    rows, identity = [], None
    for record, rank in zip(records, ranks, strict=True):
        if record is None:
            rows.append({"reference_rank": rank, "status": "missing", "quadratic_verified": False})
            continue
        if record.get("device") != "cuda" or record.get("reference_rank") != rank:
            raise ValueError("Each CUDA record must match its declared reference rank")
        checked = direction_summary([record], [record["linear_tolerance"]])
        key = {k: v for k, v in checked["identity"].items() if k != "reference_rank"}
        key.update(linear_tolerance=record["linear_tolerance"])
        # An interrupted reconstruction may not have reached device initialization.
        gpu = record.get("gpu")
        if record.get("status") == "complete" and gpu is None:
            raise ValueError("A complete CUDA record must identify the GPU")
        if identity is not None and key != identity:
            raise ValueError("Match input, source, accuracy, budget and CPU backend")
        identity = key
        row = checked["rows"][0]
        row["gpu"] = gpu
        components = [
            record.get(k)
            for k in (
                "quadratic_seconds",
                "reference_construction_seconds",
                "preconditioner_construction_seconds",
            )
        ]
        row["quadratic_plus_construction_seconds"] = (
            sum(components) if all(c is not None for c in components) else None
        )
        attempts = [
            a
            for h in record.get("history", [])
            for a in h.get("timing", {}).get("refinement_attempts", [])
        ]
        row["deployed_ranks"] = sorted({a["rank"] for a in attempts})
        row["fallback_attempts"] = sum(a.get("rank") == 0 for a in attempts) if rank else 0
        row["sampled_allocator_reserved_bytes"] = max(
            (
                h["timing"].get("cuda_pool_reserved_bytes", 0)
                for h in record.get("history", [])
                if "timing" in h
            ),
            default=None,
        )
        rows.append(row)
    devices = [r["gpu"] for r in rows if r.get("gpu") is not None]
    if devices and any(gpu != devices[0] for gpu in devices[1:]):
        raise ValueError("Match GPU model, capacity and CUDA runtime")
    return {
        "schema": "coupled-quadratic-cuda-summary-v1",
        "identity": identity,
        "rows": rows,
        "all_quadratics_verified": bool(rows) and all(r.get("quadratic_verified") for r in rows),
        "scope": "Single CUDA attempts on one intermediate quadratic. Construction-inclusive time adds reference and factory construction to the complete QP interval. Reconstruction remains separate. Failed or missing solves provide no speedup; no nonlinear convergence or repeated-timing claim is inferred.",
        "memory_scope": "Largest recorded CuPy allocator reservation, not total process peak memory.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) if path.exists() else None for path in args.records]
    report = summarize(records, args.ranks)
    args.output.mkdir(parents=True, exist_ok=False)
    write_report(
        args.output / "summary.json",
        {**report, "record_sha256": [file_sha256(p) if p.exists() else None for p in args.records]},
    )


if __name__ == "__main__":
    main()
