"""Bounded regularization screening with freshly assembled constrained quadratics."""

import argparse
import gc
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled_derivatives import GaussNewtonOperator
from .coupled_frozen_preconditioner import frozen_preconditioner_factory
from .coupled_krylov_reference import krylov_reference
from .coupled_optimize import equation_acceptance, load_problem
from .coupled_optimizer import box_kkt, box_quadratic
from .coupled_retention_replay import save_reference
from .coupled_sequence import RestoredEvaluation, prepare_device
from .coupled_targets import desired_temperature
from .coupled_trace import read_arrays, read_manifest
from .reporting import environment, file_sha256, write_arrays, write_report
from .study_solvers import ArrayReference, StudySolver
from .validation import integer, positive_real


ALPHAS = (1e-14, 1e-13, 1e-12, 1e-11)
RANKS = (0, 4, 8, 16)
TARGETS = (7, 15, 14)


def checked_alpha(value):
    alpha = positive_real(value, "Control regularization")
    if alpha not in ALPHAS:
        raise ValueError("Use one of the four predeclared regularization parameters")
    return alpha


def configuration(manifest, alpha):
    """Change regularization explicitly; preserve the physical and accuracy data."""
    cfg = dict(manifest["configuration"])
    required = {
        "slabs": 64,
        "horizon_s": 600.0,
        "target_startup_s": 60.0,
        "lower_K": 337.3,
        "inner_tolerance": 1e-10,
        "inner_cap": 50000,
        "qp_tolerance": 1e-10,
        "qp_cap": 100,
        "nonlinear_tolerance": 1e-8,
        "equation_acceptance_tolerance": 1e-12,
        "conservation_tolerance": 1e-6,
    }
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError("The input trace differs from the fixed study protocol")
    if cfg.get("queries") != [{"target": 7, "upper_K": 357.3}]:
        raise ValueError("The input trace must use the declared nominal target and bound")
    cfg.update(alpha=checked_alpha(alpha), device="cpu", threads=8)
    return cfg


def quadratic_data(problem, evaluation, desired, lower, upper):
    """Recompute the Gauss--Newton model without any previous-alpha secants."""
    _, gradient = problem.objective_gradient(evaluation, desired)
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
    diagonal = problem.preconditioning_diagonal(evaluation, 0.0)
    return H, gradient, diagonal, lower - evaluation.state, upper - evaluation.state


def solve_quadratic(problem, evaluation, desired, lower, upper, solver, cfg):
    """Solve and independently verify one constrained increment model."""
    start = time.perf_counter()
    H, g, d, lo, hi = quadratic_data(problem, evaluation, desired, lower, upper)
    factory = frozen_preconditioner_factory(problem, evaluation, sweeps=3)
    result = box_quadratic(
        H,
        g,
        d,
        lo,
        hi,
        solver,
        tolerance=cfg["qp_tolerance"],
        max_steps=cfg["qp_cap"],
        preconditioner_factory=factory,
    )
    Hx = H @ result.x
    kkt = box_kkt(result.x, Hx + g, lo, hi, max(1.0, np.linalg.norm(g, np.inf)))
    rows = result.history
    linear_verified = all(
        row["linear_status"] in {"converged", "empty"}
        and 0 <= row.get("linear_residual", 0.0) <= cfg["inner_tolerance"]
        for row in rows
    )
    verified = (
        result.status == "converged"
        and linear_verified
        and np.isfinite(list(kkt.values())).all()
        and max(kkt.values()) <= cfg["qp_tolerance"]
    )
    record = {
        "status": result.status,
        "verified": bool(verified),
        "kkt": kkt,
        "quadratic_objective": float(0.5 * result.x @ Hx + g @ result.x),
        "outer_pdas_steps": len(rows),
        "inner_iterations": sum(row["linear_iterations"] for row in rows),
        "deployed_ranks": [row.get("deployed_rank", 0) for row in rows],
        "fallbacks": [row.get("fallback") for row in rows],
        "maximum_original_relative_residual": max(
            (row.get("linear_residual", 0.0) for row in rows), default=0.0
        ),
        "lower_active": int(np.count_nonzero(np.isclose(result.x, lo, rtol=0, atol=1e-12))),
        "upper_active": int(np.count_nonzero(np.isclose(result.x, hi, rtol=0, atol=1e-12))),
        "history": rows,
        "seconds": time.perf_counter() - start,
    }
    return result.x, record


def run_screen(args):
    manifest = read_manifest(args.trace)
    if manifest["status"] != "complete" or not manifest["quadratics"]:
        raise ValueError("Use a complete nominal trace")
    first = manifest["quadratics"][0]
    if (first["iteration"], first["attempt"], first["damping"]) != (0, 0, 0):
        raise ValueError("Use the first undamped initial trajectory")
    cfg = configuration(manifest, args.alpha)
    cfg["baseline_directory"] = str(args.baseline)
    args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "coupled-regularization-screen-v1",
        "environment": environment(),
        "alpha": cfg["alpha"],
        "trace_alpha": manifest["configuration"]["alpha"],
        "trace_sha256": file_sha256(args.trace / "manifest.json"),
        "initial_trajectory_sha256": first["sha256"],
        "ranks": list(RANKS),
        "targets": list(TARGETS),
        "repetitions": 3,
        "status": "running",
        "sequences": [],
        "configuration": {k: v for k, v in cfg.items() if k != "baseline_directory"},
        "scope": "Constrained Gauss-Newton subproblems at a fixed initial trajectory. No complete nonlinear optimization or nonlinear KKT claim.",
    }
    write_report(args.output / "record.json", record)
    sampler = solver = None
    try:
        tick = time.perf_counter()
        problem, baseline = load_problem(cfg)
        if baseline["baseline_sha256"] != manifest["baseline_sha256"]:
            raise ValueError("Physical baseline differs from the declared trace")
        record["baseline_sha256"] = baseline["baseline_sha256"]
        record["assembly_seconds"] = time.perf_counter() - tick
        sampler, _, _, _ = prepare_device("cpu", 0.01)
        sampler.start()
        arrays = read_arrays(args.trace, first["file"], first["sha256"])
        guess = RestoredEvaluation(arrays["state"], arrays["velocity"], arrays["pressure"])
        tick = time.perf_counter()
        evaluation = problem.evaluate(arrays["state"], initial=guess)
        checks = problem.verify(evaluation, local_mass=True)
        if not equation_acceptance(checks, cfg):
            raise ValueError("Nominal coupled equations fail verification")
        record["nominal_equations"] = checks
        record["nominal_reconstruction_seconds"] = time.perf_counter() - tick
        lower = (337.3 - problem.temperature_offset) / problem.temperature_scale
        upper = (357.3 - problem.temperature_offset) / problem.temperature_scale
        if np.any(evaluation.state < lower) or np.any(evaluation.state > upper):
            raise ValueError("The common initial trajectory violates the unchanged bounds")
        del arrays, guess
        H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha)
        diagonal = problem.preconditioning_diagonal(evaluation, 0.0)
        tick = time.perf_counter()
        inverse = frozen_preconditioner_factory(problem, evaluation, sweeps=3)(
            np.arange(problem.size)
        )
        reference = krylov_reference(H, inverse, diagonal, rank=16, steps=48, seed=20260923)
        del inverse
        construction = time.perf_counter() - tick + record["nominal_reconstruction_seconds"]
        record["reference"] = {
            **save_reference(args.output, "reference", reference),
            "construction_seconds": construction,
            "cost_scope": "Nominal coupled reevaluation, full-domain preconditioner and 48-step rank-16 construction, charged in full to every nonzero prefix.",
        }
        targets = [
            desired_temperature(problem, target, cfg["target_count"], cfg["target_startup_s"])
            for target in TARGETS
        ]
        record["state_dofs"] = problem.size
        write_report(args.output / "record.json", record)
        for repetition in range(3):
            order = RANKS[repetition:] + RANKS[:repetition]
            for rank in order:
                tick = time.perf_counter()
                selected = (
                    None
                    if rank == 0
                    else ArrayReference(reference.basis[:, :rank], {"requested_rank": rank})
                )
                solver = StudySolver(
                    "jacobi" if rank == 0 else "reference",
                    reference=selected,
                    rank=rank,
                    window=max(1, rank),
                    rtol=cfg["inner_tolerance"],
                    maxiter=cfg["inner_cap"],
                    refresh=cfg["inner_refresh"],
                    cg_factor=0.1,
                    residual_policy="refine",
                )
                sequence = {"rank": rank, "repetition": repetition, "cases": []}
                states = []
                for target, desired in zip(TARGETS, targets, strict=True):
                    state, row = solve_quadratic(
                        problem, evaluation, desired, lower, upper, solver, cfg
                    )
                    row["target"] = target
                    sequence["cases"].append(row)
                    states.append(state)
                solver.close()
                solver = selected = None
                gc.collect()
                sequence["online_seconds"] = time.perf_counter() - tick
                sequence["construction_seconds_once"] = construction if rank else 0.0
                sequence["setup_inclusive_model_seconds"] = (
                    sequence["online_seconds"] + sequence["construction_seconds_once"]
                )
                sequence["verified"] = all(row["verified"] for row in sequence["cases"])
                filename = f"increments-r{rank}-repeat{repetition}.npz"
                write_arrays(args.output / filename, increments=np.stack(states))
                sequence["arrays"] = {
                    "file": filename,
                    "sha256": file_sha256(args.output / filename),
                }
                record["sequences"].append(sequence)
                write_report(args.output / "record.json", record)
        record["status"] = "complete"
    except Exception as error:
        record.update(
            status="screen_error", error_type=type(error).__name__, error_message=str(error)
        )
        raise
    finally:
        if solver is not None:
            solver.close()
        if sampler is not None:
            record["memory"] = sampler.finish()
            record["memory_scope"] = (
                "Whole-screen sampled process allocation, including the shared reference; not a per-method memory comparison."
            )
        write_report(args.output / "record.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checked_alpha(args.alpha)
    with threadpool_limits(integer(8, "Threads", 1)):
        run_screen(args)


if __name__ == "__main__":
    main()
