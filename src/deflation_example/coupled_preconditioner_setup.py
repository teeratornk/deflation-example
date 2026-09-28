"""Matched setup measurements on saved masks, including full-matrix construction.

This compares only two constructions of the same sparse frozen preconditioner.
It does not assemble or replace the coupled Gauss--Newton operator.
"""

import argparse
import gc
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_frozen_preconditioner import frozen_preconditioner_factory
from .coupled_optimize import equation_acceptance, load_problem
from .coupled_retention_replay import rebuild
from .coupled_trace import read_arrays, read_manifest
from .reporting import environment, file_sha256, write_report


def compare(problem, evaluation, masks, *, damping=0.0, repeats=3):
    """Alternate policy order; include the first use and preserve every result."""
    rows, expected = [], {}
    for rep in range(repeats):
        order = ("product", "submatrix") if rep % 2 == 0 else ("submatrix", "product")
        for policy in order:
            gc.collect()
            tick = time.perf_counter()
            factory = frozen_preconditioner_factory(
                problem, evaluation, damping, restriction=policy
            )
            construction = time.perf_counter() - tick
            measurements = []
            for index, indices in enumerate(masks):
                tick = time.perf_counter()
                preconditioner = factory(indices)
                setup = time.perf_counter() - tick
                rhs = np.random.default_rng(20260928 + index).standard_normal(len(indices))
                tick = time.perf_counter()
                z = preconditioner(rhs)
                application = time.perf_counter() - tick
                if index not in expected:
                    expected[index] = z.copy()
                difference = np.linalg.norm(z - expected[index]) / max(
                    np.linalg.norm(expected[index]), 1e-300
                )
                measurements.append(
                    {
                        "mask": index,
                        "inactive": len(indices),
                        "setup_seconds": setup,
                        "application_seconds": application,
                        "relative_action_difference": float(difference),
                        "operator_nonzeros": preconditioner.operator.nnz,
                    }
                )
                del preconditioner
            rows.append(
                {
                    "repetition": rep,
                    "policy": policy,
                    "construction_seconds": construction,
                    "masks": measurements,
                    "setup_including_construction_seconds": construction
                    + sum(r["setup_seconds"] for r in measurements),
                    "verified": all(
                        np.isfinite(r["relative_action_difference"])
                        and r["relative_action_difference"] <= 1e-10
                        for r in measurements
                    ),
                }
            )
            del factory
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_manifest(args.trace)
    if manifest["status"] != "complete":
        raise ValueError("Use a complete trace")
    qindex = len(manifest["quadratics"]) - 1
    qrow = manifest["quadratics"][qindex]
    systems = [r for r in manifest["systems"] if r["quadratic"] == qindex]
    selected = [systems[i] for i in np.unique(np.linspace(0, len(systems) - 1, 5, dtype=int))]
    masks = [read_arrays(args.trace, r["file"], r["sha256"])["indices"] for r in selected]
    arrays = read_arrays(args.trace, qrow["file"], qrow["sha256"])
    cfg = {**manifest["configuration"], "baseline_directory": str(args.baseline)}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-frozen-restriction-setup-v1",
        "status": "rebuilding",
        "trace_sha256": file_sha256(args.trace / "manifest.json"),
        "environment": environment(),
        "quadratic_sha256": qrow["sha256"],
        "masks": [{"sha256": r["sha256"], "pdas_step": r["pdas_step"]} for r in selected],
        "scope": "Setup component diagnostic, three alternating-order repetitions including first use; no complete optimization speedup.",
    }
    write_report(args.output / "record.json", report)
    start = time.perf_counter()
    try:
        with threadpool_limits(cfg["threads"]):
            problem, baseline = load_problem(cfg)
            if baseline["baseline_sha256"] != manifest["baseline_sha256"]:
                raise ValueError("Physical baseline differs")
            evaluation, _, _, _ = rebuild(problem, arrays, qrow)
            if not equation_acceptance(problem.verify(evaluation, local_mass=True), cfg):
                raise ValueError("Reconstructed trajectory fails equation verification")
            report["reconstruction_seconds"] = time.perf_counter() - start
            report["rows"] = compare(problem, evaluation, masks, damping=qrow["damping"])
            report.update(status="complete", verified=all(r["verified"] for r in report["rows"]))
    except Exception as error:
        report.update(status="diagnostic_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        write_report(args.output / "record.json", report)
    if not report["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
