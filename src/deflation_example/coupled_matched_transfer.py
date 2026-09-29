"""Replay full-reference and sequential transfer on identical captured systems.

The initial space comes from the capture run, without a new spectral selection.
All masks are visited chronologically, including initially converged systems.
Independent error solves and coarse diagnostics remain outside solver timers.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_optimize import equation_acceptance, load_problem
from .coupled_reference import SequentialReference
from .coupled_retention_replay import ReplayPreconditioner, rebuild
from .coupled_trace import read_arrays, read_manifest
from .reporting import environment, file_sha256, write_report
from .solvers import CpuCoarseSpace, independent_residual, relative_norm
from .study_solvers import ArrayReference, StudySolver
from .validation import positive_real


def transfer_counts(previous, current, size):
    """The first mask has no preceding transition; later counts are separate."""
    return {
        "inactive": len(current),
        "active": size - len(current),
        "newly_inactive": None
        if previous is None
        else len(np.setdiff1d(current, previous, assume_unique=True)),
        "newly_active": None
        if previous is None
        else len(np.setdiff1d(previous, current, assume_unique=True)),
    }


def coarse_energy(B, residual, error, basis):
    """Evaluate error removal using an independently verified error solution."""
    space = CpuCoarseSpace(B, basis, 1e10)
    details = {
        "deployed_rank": space.rank,
        "coarse_condition": space.condition,
        "fallback": space.fallback,
        "basis_processing": "CPU rank-revealing SVD, relative threshold 1e-12",
    }
    if space.breakdown:
        return {**details, "status": "coarse_factorization_failed"}
    initial_energy = float(error @ (B @ error))
    if not np.isfinite(initial_energy) or initial_energy <= 0:
        return {**details, "status": "zero_or_invalid_initial_energy"}
    correction = space.correct(residual) if space.rank else np.zeros_like(residual)
    remainder = error - correction
    remaining_energy = float(remainder @ (B @ remainder))
    return {
        **details,
        "status": "evaluated",
        "initial_energy": initial_energy,
        "remaining_energy": remaining_energy,
        "energy_fraction_removed": 1 - remaining_energy / initial_energy,
        "scope": "Recorded initial guess and independently solved error equation; not the error of another optimizer path.",
    }


def error_solution(B, rhs, initial, indices, cap, stop):
    residual = rhs - B @ initial
    if not np.any(residual):
        return None, {"status": "initially_converged", "original_residual": 0.0}
    solver = StudySolver(
        "jacobi",
        rank=0,
        rtol=1e-12,
        cg_factor=0.1,
        maxiter=cap,
        refresh=50000,
        residual_policy="refine",
        stop_requested=stop,
    )
    try:
        result, _ = solver.solve(B, residual, indices, initial=np.zeros_like(residual))
        rho = independent_residual(B, result.x, residual)
        verified = result.status == "converged" and np.isfinite(rho) and rho <= 1e-12
        return (result.x if verified else None), {
            "status": "verified" if verified else "error_solve_failed",
            "solver_status": result.status,
            "original_residual": rho,
            "iterations": result.iterations,
        }
    finally:
        solver.close()


def require_capture(record, manifest):
    if (
        record.get("status") != "complete"
        or not record.get("all_problems_verified")
        or manifest.get("status") != "complete"
        or not record["configuration"].get("capture_linear_systems")
        or record["configuration"].get("reference_transfer") != "full"
        or record["configuration"].get("method") != "reference"
        or record.get("baseline_sha256") != manifest.get("baseline_sha256")
        or record["environment"]["source_sha256"] != manifest["environment"]["source_sha256"]
    ):
        raise ValueError(
            "Use a completed, verified full-reference capture with matched source and baseline"
        )
    for key, value in manifest["configuration"].items():
        if record["configuration"].get(key) != value:
            raise ValueError("Capture configuration differs from its optimization record")
    numbers = [s["quadratic"] for s in manifest["systems"]]
    if (
        not numbers
        or numbers != sorted(numbers)
        or any(q < 0 or q >= len(manifest["quadratics"]) for q in numbers)
    ):
        raise ValueError("The complete chronological system history is required")


def replay(optimization, baseline_directory, output, *, budget_seconds=14400):
    budget_seconds = positive_real(budget_seconds, "Replay time budget")
    optimization, output = Path(optimization), Path(output)
    trace = optimization / "inactive-trace-00"
    manifest = read_manifest(trace)
    original = json.loads((optimization / "record.json").read_text())
    require_capture(original, manifest)
    basis = read_arrays(optimization, "reference.npz", original["reference_sha256"])["basis"]
    cfg = {**manifest["configuration"], "baseline_directory": str(baseline_directory)}
    cfg.update(device="cpu", capture_linear_systems=False)
    problem, baseline = load_problem(cfg)
    if baseline["baseline_sha256"] != original["baseline_sha256"] or basis.shape != (
        problem.size,
        cfg["rank"],
    ):
        raise ValueError("The saved reference dimension or physical baseline differs")
    references = {
        "full": ArrayReference(basis.copy(), original["reference_description"]),
        "sequential": SequentialReference(
            ArrayReference(basis.copy(), original["reference_description"]), problem.size
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()

    def stop():
        return time.perf_counter() - start >= budget_seconds

    report = {
        "schema": "coupled-matched-transfer-v1",
        "status": "running",
        "environment": environment(),
        "optimization_sha256": file_sha256(optimization / "record.json"),
        "trace_sha256": file_sha256(trace / "manifest.json"),
        "reference_sha256": original["reference_sha256"],
        "numerical_capture_source": original["environment"]["git_head"],
        "reference_construction_seconds_in_capture": original["components_seconds"][
            "reference_construction_or_restore"
        ],
        "requested_rank": cfg["rank"],
        "expected_systems": len(manifest["systems"]),
        "final_original_residual_target": 1e-10,
        "error_equation_target": 1e-12,
        "rows": [],
        "scope": "CPU matched-system diagnostic with unchanged saved starts, loads and initial reference; excludes reconstruction and independent energy diagnostics from solver timers. No complete-optimization speedup claim.",
    }
    solvers = {
        name: StudySolver(
            "reference",
            reference=ref,
            rank=cfg["rank"],
            rtol=1e-10,
            cg_factor=0.1,
            maxiter=cfg["inner_cap"],
            refresh=cfg["inner_refresh"],
            residual_policy="refine",
            stop_requested=stop,
        )
        for name, ref in references.items()
    }
    preconditioners, previous, number = {}, None, None
    write_report(output / "record.json", report)
    try:
        problem.stop_requested = stop
        for index, system in enumerate(manifest["systems"]):
            if stop():
                report["status"] = "budget_exhausted"
                break
            if system["quadratic"] != number:
                for preconditioner in preconditioners.values():
                    preconditioner.close()
                number = system["quadratic"]
                q = manifest["quadratics"][number]
                arrays = read_arrays(trace, q["file"], q["sha256"])
                tick = time.perf_counter()
                evaluation, H, diagonal, gradient = rebuild(problem, arrays, q)
                if not equation_acceptance(problem.verify(evaluation), cfg):
                    raise ValueError("Reconstructed coupled equations fail verification")
                for name, actual in (("diagonal", diagonal), ("gradient", gradient)):
                    if relative_norm(actual - arrays[name], arrays[name]) > 1e-8:
                        raise ValueError("Reconstructed " + name + " differs")
                report.setdefault("reconstruction_seconds", []).append(time.perf_counter() - tick)
                preconditioners = {
                    name: ReplayPreconditioner(
                        problem, evaluation, q["damping"], "frozen", cfg["frozen_sweeps"]
                    )
                    for name in references
                }
            data = read_arrays(trace, system["file"], system["sha256"])
            I, rhs, initial = data["indices"], data["rhs"], data["initial"]
            B = H.restrict(I)
            B.diagonal = lambda I=I: diagonal[I].copy()
            if system["status"] == "converged":
                saved = read_arrays(trace, system["solution_file"], system["solution_sha256"])
                if independent_residual(B, saved["x"], rhs) > q["linear_tolerance"]:
                    raise ValueError("Reconstructed operator fails the saved solution criterion")
            entry = {
                "system": index,
                "system_sha256": system["sha256"],
                "quadratic": number,
                "recorded_linear_tolerance": q["linear_tolerance"],
                **transfer_counts(previous, I, problem.size),
                "methods": {},
            }
            report["rows"].append(entry)
            # Alternate execution order without changing either transfer history.
            order = ("full", "sequential") if index % 2 == 0 else ("sequential", "full")
            for policy in order:
                tick = time.perf_counter()
                preconditioners[policy].attach(B, I)
                result, timing = solvers[policy].solve(B, rhs, I, initial=initial.copy())
                solve_seconds = time.perf_counter() - tick
                tick = time.perf_counter()
                rho = independent_residual(B, result.x, rhs)
                verify_seconds = time.perf_counter() - tick
                entry["methods"][policy] = {
                    "status": result.status,
                    "verified": result.status == "converged" and rho <= 1e-10,
                    "original_residual": rho,
                    "iterations": result.iterations,
                    "deployed_rank": result.rank,
                    "coarse_condition": result.coarse_condition,
                    "fallback": result.fallback_reason,
                    "solve_seconds": solve_seconds,
                    "independent_verification_seconds": verify_seconds,
                    "total_seconds": solve_seconds + verify_seconds,
                    "timing": timing,
                }
                write_report(output / "record.json", report)
            tick = time.perf_counter()
            # The separate error equation uses the same operator preconditioner.
            error, diagnostic = error_solution(B, rhs, initial, I, cfg["inner_cap"], stop)
            entry["error_equation"] = diagnostic
            if error is not None:
                residual = rhs - B @ initial
                for policy, reference in references.items():
                    entry["methods"][policy]["coarse_error"] = coarse_energy(
                        B, residual, error, reference.restrict(I)
                    )
            full, transferred = (ref.restrict(I) for ref in references.values())
            released = np.ones(len(I), dtype=bool) if previous is None else ~np.isin(I, previous)
            entry["relative_basis_difference"] = float(
                np.linalg.norm(full - transferred) / max(np.linalg.norm(full), np.finfo(float).tiny)
            )
            entry["released_entries_difference_norm"] = (
                None if previous is None else float(np.linalg.norm((full - transferred)[released]))
            )
            entry["diagnostic_seconds"] = time.perf_counter() - tick
            previous = I.copy()
            write_report(output / "record.json", report)
        else:
            report["status"] = "complete"
    except Exception as error:
        report.update(status="replay_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for solver in solvers.values():
            solver.close()
        for preconditioner in preconditioners.values():
            preconditioner.close()
        report["elapsed_seconds"] = time.perf_counter() - start
        report["all_systems_verified"] = (
            report["status"] == "complete"
            and len(report["rows"]) == report["expected_systems"]
            and all(
                len(r["methods"]) == 2 and all(m["verified"] for m in r["methods"].values())
                for r in report["rows"]
            )
        )
        write_report(output / "record.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=14400)
    args = parser.parse_args()
    cfg = json.loads((args.optimization / "record.json").read_text())["configuration"]
    with threadpool_limits(cfg["threads"]):
        report = replay(
            args.optimization, args.baseline, args.output, budget_seconds=args.budget_seconds
        )
    raise SystemExit(0 if report["all_systems_verified"] else 2)


if __name__ == "__main__":
    main()
