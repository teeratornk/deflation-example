"""Matched inactive-system replays; diagnostic work is outside solver timers."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import aslinearoperator, cg
from threadpoolctl import threadpool_limits

from .coupled_derivatives import GaussNewtonOperator
from .coupled_optimize import equation_acceptance, load_problem, solver_options
from .coupled_reference import configured_reference
from .coupled_secant import SecantGaussNewton
from .coupled_selected_reference import CoupledBlockAction, SelectedReference, selected_reference
from .coupled_sequence import RestoredEvaluation, prepare_device
from .coupled_trace import read_arrays, read_manifest
from .reporting import environment, file_sha256, write_arrays, write_report
from .solvers import CpuCoarseSpace, independent_residual
from .spacetime_reference import SpaceTimeReference
from .study_solvers import ArrayReference
from .validation import integer


class ReplayPreconditioner:
    """One quadratic's preconditioner, rebuilt only when its inactive set changes.

    Construct this object inside the resource timer and call ``attach`` inside
    the solve timer. Its reported times are components, never additional totals.
    """

    def __init__(self, problem, evaluation, damping, policy="jacobi", sweeps=3):
        if policy not in {"jacobi", "frozen"}:
            raise ValueError("Choose jacobi or frozen preconditioning")
        if sweeps not in {1, 3, 5}:
            raise ValueError("The new comparison uses one, three or five odd sweeps")
        self.factory = None
        self.closed = False
        if policy == "frozen":
            from .coupled_frozen_preconditioner import frozen_preconditioner_factory

            self.factory = frozen_preconditioner_factory(problem, evaluation, damping, sweeps)
        self.indices = self.preconditioner = None
        self.builds = 0
        self.setup_seconds = 0.0
        self.application_seconds = 0.0
        self.applications = 0

    def attach(self, operator, indices):
        if self.closed:
            raise RuntimeError("A closed replay preconditioner cannot be reused")
        if self.factory is None:
            return
        if self.indices is None or not np.array_equal(self.indices, indices):
            tick = time.perf_counter()
            self.preconditioner = self.factory(indices)
            self.indices = np.asarray(indices).copy()
            self.setup_seconds += time.perf_counter() - tick
            self.builds += 1
        operator.preconditioner = self

    def __call__(self, residual):
        if self.closed or self.preconditioner is None:
            raise RuntimeError("Attach a live replay preconditioner before applying it")
        tick = time.perf_counter()
        result = self.preconditioner(residual)
        self.application_seconds += time.perf_counter() - tick
        self.applications += 1
        return result

    def close(self):
        """Release factors even when the last operator still refers to this wrapper."""
        self.preconditioner = self.factory = self.indices = None
        self.closed = True

    def report(self):
        return {
            "builds": self.builds,
            "setup_seconds": self.setup_seconds,
            "applications": self.applications,
            "application_seconds": self.application_seconds,
            "scope": "Cumulative components within this repetition's solve totals; do not add again.",
        }


def rebuild(problem, arrays, row, *, use_secants=True):
    guess = RestoredEvaluation(arrays["state"], arrays["velocity"], arrays["pressure"])
    evaluation = problem.evaluate(arrays["state"], initial=guess)
    H = GaussNewtonOperator(evaluation.jacobian, problem.weights, problem.alpha, row["damping"])
    pairs = list(zip(arrays["secant_steps"], arrays["secant_gradients"], strict=True))
    if use_secants and pairs:
        H = SecantGaussNewton(H, pairs)
    diagonal = problem.preconditioning_diagonal(evaluation, row["damping"])
    _, gradient = problem.objective_gradient(evaluation, arrays["desired"])
    return evaluation, H, diagonal, gradient


def error_diagnostic(B, rhs, initial, reference, indices, diagonal, *, exact_limit=256):
    """Measure coarse error removal for this initial guess, not a zero-start surrogate.

    The independent error solve is tighter than replay acceptance. Large systems
    receive no condition-number certificate from a few Ritz values.
    """
    r = rhs - B @ initial
    if not np.any(r):
        return {
            "status": "initially_converged",
            "initial_energy": 0.0,
            "energy_fraction_removed": None,
            "spectrum": None,
        }
    preconditioner = aslinearoperator(np.diag(1 / diagonal)) if len(r) <= exact_limit else None
    if len(r) <= exact_limit:
        dense = B @ np.eye(len(r))
        error = linalg.solve(dense, r, assume_a="pos")
    else:
        from scipy.sparse.linalg import LinearOperator

        preconditioner = LinearOperator(B.shape, matvec=lambda x: x / diagonal, dtype=float)
        error, info = cg(B, r, M=preconditioner, rtol=1e-12, atol=0, maxiter=50000)
        if info != 0:
            return {
                "status": "independent_error_solve_failed",
                "cg_info": int(info),
                "original_residual": independent_residual(B, error, r),
            }
    residual = independent_residual(B, error, r)
    if residual > 1e-11:
        return {"status": "independent_error_verification_failed", "original_residual": residual}
    Z = None if reference is None else reference.restrict(indices)
    space = CpuCoarseSpace(B, Z, 1e10)
    if space.breakdown:
        return {"status": "coarse_factorization_failed", "original_residual": residual}
    correction = space.correct(r) if space.rank else np.zeros_like(r)
    energy = float(error @ (B @ error))
    remainder = error - correction
    retained = float(remainder @ (B @ remainder))
    answer = {
        "status": "verified",
        "original_residual": residual,
        "initial_energy": energy,
        "remaining_energy": retained,
        "energy_fraction_removed": 1 - retained / energy if energy > 0 else None,
        "diagnostic_deployed_rank": space.rank,
        "diagnostic_fallback": space.fallback,
        "scope": "Independent CPU-SVD coarse correction; actual recorded initial guess.",
        "spectrum": None,
    }
    if len(r) <= exact_limit:
        inv = 1 / np.sqrt(diagonal)
        scaled = inv[:, None] * dense * inv[None, :]
        if space.rank:
            coupling = inv[:, None] * (dense @ space.Z)
            deflated = scaled - coupling @ linalg.solve(
                space.Z.T @ dense @ space.Z, coupling.T, assume_a="pos"
            )
        else:
            deflated = scaled
        original = linalg.eigvalsh(scaled)
        spectrum = linalg.eigvalsh((deflated + deflated.T) / 2)
        # The nullity follows from the deployed rank; avoid a scale-dependent
        # cutoff that could discard a genuine small nonzero eigenvalue.
        positive = spectrum[space.rank :]
        answer["spectrum"] = {
            "original_condition": float(original[-1] / original[0]),
            "deflated_nonzero_condition": float(positive[-1] / positive[0])
            if len(positive) and positive[0] > 0
            else None,
            "deflated_eigenvalues": spectrum.tolist(),
            "method": "dense symmetric eigensolve",
        }
    return answer


def save_reference(directory, name, reference):
    arrays = (
        {
            "spatial": reference.spatial,
            "temporal": reference.temporal,
            "spatial_columns": reference.spatial_columns,
        }
        if isinstance(reference, SpaceTimeReference)
        else {"basis": reference.basis}
    )
    path = directory / (name + ".npz")
    write_arrays(path, **arrays)
    return {"file": path.name, "sha256": file_sha256(path), "description": reference.description}


def load_reference(directory, row):
    arrays = read_arrays(directory, row["file"], row["sha256"])
    if "basis" in arrays:
        return ArrayReference(arrays["basis"], row["description"])
    return SpaceTimeReference(
        arrays["spatial"], arrays["temporal"], arrays["spatial_columns"], row["description"]
    )


def trace_configuration(args, manifest):
    cfg = dict(manifest["configuration"])
    cfg.update(
        baseline_directory=str(args.baseline),
        device=args.device,
        hybrid_block_max_columns=args.width,
        capture_linear_systems=False,
    )
    return cfg


def construct_bank(args):
    manifest = read_manifest(args.trace)
    if manifest["status"] != "complete" or not manifest["quadratics"]:
        raise ValueError("Reference selection requires a complete diagnostic trace")
    cfg = trace_configuration(args, manifest)
    problem, baseline = load_problem(cfg)
    if baseline["baseline_sha256"] != manifest["baseline_sha256"]:
        raise ValueError("Reference baseline differs from the trace")
    args.output.mkdir(parents=True, exist_ok=False)
    sampler, barrier, _, device = prepare_device(args.device, 0.01)
    sampler.start()
    record = {
        "schema": "coupled-reference-bank-v1",
        "environment": environment(),
        "trace_sha256": file_sha256(args.trace / "manifest.json"),
        "baseline_sha256": baseline["baseline_sha256"],
        "status": "building",
        "references": {},
        "device": device,
    }
    write_report(args.output / "record.json", record)
    action = None
    try:
        for rank in (20, 50, 100, 200, 400):
            tick = time.perf_counter()
            reference = configured_reference(problem, {**cfg, "rank": rank}, baseline)
            seconds = time.perf_counter() - tick
            record["references"][str(rank)] = {
                **save_reference(args.output, f"thermal-{rank}", reference),
                "construction_seconds": seconds,
            }
            write_report(args.output / "record.json", record)
        row = manifest["quadratics"][0]
        arrays = read_arrays(args.trace, row["file"], row["sha256"])
        if row["iteration"] != 0 or row["attempt"] != 0 or row["damping"] != 0:
            raise ValueError("The nominal reference uses the first undamped initial trajectory")
        tick = time.perf_counter()
        evaluation, H, diagonal, _ = rebuild(problem, arrays, row, use_secants=False)
        evaluation_seconds = time.perf_counter() - tick
        if not equation_acceptance(problem.verify(evaluation), cfg):
            raise ValueError("Nominal coupled equations failed independent verification")
        tick = time.perf_counter()
        if args.device != "cpu":
            action = CoupledBlockAction(H)
        upload = time.perf_counter() - tick
        selection_tick = time.perf_counter()
        if getattr(args, "bank_selection", "jacobi") == "frozen":
            from .coupled_frozen_preconditioner import frozen_preconditioner_factory
            from .coupled_preconditioned_reference import preconditioned_reference

            inverse = frozen_preconditioner_factory(problem, evaluation, sweeps=3)(
                np.arange(problem.size)
            )
            selected = preconditioned_reference(
                reference, H, inverse, 200, chunk=args.width, block_action=action
            )
            del inverse
            selection_seconds = time.perf_counter() - selection_tick
            record["preconditioned_selection"] = {
                **save_reference(args.output, "preconditioned-selection", selected),
                "nominal_evaluation_seconds": evaluation_seconds,
                "factor_upload_seconds": upload,
                "construction_seconds": record["references"]["400"]["construction_seconds"]
                + evaluation_seconds
                + upload
                + selection_seconds,
            }
        else:
            selected = selected_reference(
                reference, H, diagonal, 200, chunk=args.width, block_action=action
            )
            write_arrays(args.output / "selection.npz", coefficients=selected.coefficients)
            record["selection"] = {
                "file": "selection.npz",
                "sha256": file_sha256(args.output / "selection.npz"),
                "description": selected.description,
                "nominal_evaluation_seconds": evaluation_seconds,
                "factor_upload_seconds": upload,
                "construction_seconds": record["references"]["400"]["construction_seconds"]
                + evaluation_seconds
                + upload
                + selected.description["construction_seconds"],
            }
        record["status"] = "complete"
    except Exception as error:
        record.update(status="construction_failed", error_type=type(error).__name__)
        raise
    finally:
        tick = time.perf_counter()
        if action is not None:
            action.close()
        barrier()
        record["cleanup_seconds"] = time.perf_counter() - tick
        if "selection" in record:
            record["selection"]["construction_seconds"] += record["cleanup_seconds"]
        if "preconditioned_selection" in record:
            record["preconditioned_selection"]["construction_seconds"] += record["cleanup_seconds"]
        record["memory"] = sampler.finish()
        write_report(args.output / "record.json", record)


def checked_replay_rank(policy, requested):
    allowed = {
        "jacobi": {0},
        "thermal": {20, 50, 100, 200},
        "nominal_coupled": {20, 50, 100, 200},
        "preconditioned_coupled": {20, 50, 100, 200},
        "krylov_coupled": {1, 2, 4, 8},
    }
    requested = integer(requested, "Rank", 0)
    if policy not in allowed or requested not in allowed[policy]:
        raise ValueError("Use a predeclared rank for the selected replay policy")
    return requested


def replay(args):
    requested = checked_replay_rank(args.policy, args.rank)
    manifest = read_manifest(args.trace)
    bank = json.loads((args.bank / "record.json").read_text())
    if (
        manifest["status"] != "complete"
        or bank["status"] != "complete"
        or bank["trace_sha256"] != file_sha256(args.trace / "manifest.json")
    ):
        raise ValueError("Replay requires a completed, checksum-matched trace and reference bank")
    cfg = trace_configuration(args, manifest)
    if args.feedback is not None:
        cfg["feedback_multiplier"] = args.feedback
    problem, baseline = load_problem(cfg)
    if baseline["baseline_sha256"] != manifest["baseline_sha256"]:
        raise ValueError("Replay baseline differs")
    number = integer(args.quadratic, "Quadratic index", 0)
    row = manifest["quadratics"][number]
    arrays = read_arrays(args.trace, row["file"], row["sha256"])
    evaluation, H, diagonal, gradient = rebuild(
        problem, arrays, row, use_secants=args.feedback is None
    )
    if not equation_acceptance(problem.verify(evaluation), cfg):
        raise ValueError("Reconstructed equations fail verification")
    if args.feedback is None:
        for name, actual in (("diagonal", diagonal), ("gradient", gradient)):
            difference = np.linalg.norm(actual - arrays[name]) / max(
                np.linalg.norm(arrays[name]), 1e-30
            )
            if difference > 1e-9:
                raise ValueError("Reconstructed " + name + " differs from the recorded system")
    reference = None
    construction = 0.0
    if args.policy == "thermal":
        reference = load_reference(args.bank, bank["references"][str(requested)])
        construction = bank["references"][str(requested)]["construction_seconds"]
    elif args.policy == "nominal_coupled":
        candidates = load_reference(args.bank, bank["references"]["400"])
        coefficients = read_arrays(args.bank, "selection.npz", bank["selection"]["sha256"])[
            "coefficients"
        ]
        reference = SelectedReference(
            candidates,
            coefficients[:, :requested],
            {
                **bank["selection"]["description"],
                "requested_rank": requested,
                "deployed_rank": min(requested, coefficients.shape[1]),
                "ritz_values": bank["selection"]["description"]["ritz_values"][:requested],
            },
        )
        construction = bank["selection"]["construction_seconds"]
    elif args.policy in {"preconditioned_coupled", "krylov_coupled"}:
        stored = bank[
            "krylov_selection" if args.policy == "krylov_coupled" else "preconditioned_selection"
        ]
        full = load_reference(args.bank, stored)
        reference = ArrayReference(
            full.basis[:, :requested].copy(),
            {
                **full.description,
                "requested_rank": requested,
                "deployed_rank": min(requested, full.rank),
                "ritz_values": full.description["ritz_values"][:requested],
            },
        )
        construction = stored["construction_seconds"]
        del full
    elif requested:
        raise ValueError("Jacobi uses rank zero")
    args.output.mkdir(parents=True, exist_ok=False)
    preparation_tick = time.perf_counter()
    sampler, barrier, solver_class, device = prepare_device(args.device, 0.01)
    preparation_seconds = time.perf_counter() - preparation_tick
    record = {
        "schema": "coupled-retention-replay-v1",
        "environment": environment(),
        "trace_sha256": bank["trace_sha256"],
        "bank_sha256": file_sha256(args.bank / "record.json"),
        "quadratic": number,
        "partition": row["partition"],
        "iteration": row["iteration"],
        "policy": args.policy,
        "rank": requested,
        "width": args.width,
        "feedback": args.feedback,
        "preconditioner": getattr(args, "preconditioner", "jacobi"),
        "frozen_sweeps": getattr(args, "sweeps", 3)
        if getattr(args, "preconditioner", "jacobi") == "frozen"
        else 0,
        "final_residual_tolerance": cfg["inner_tolerance"],
        "device": device,
        "common_device_preparation_seconds": preparation_seconds,
        "construction_seconds_once": construction,
        "rows": [],
        "status": "running",
        "scope": "Matched trace systems with recorded warm starts; diagnostic reconstruction excluded."
        if args.feedback is None
        else "Feedback sensitivity: recomputed flow and Gauss-Newton, no secants; fixed trace masks, loads and initial guesses. Not optimization of the modified model.",
    }
    write_report(args.output / "record.json", record)
    sampler.start()
    solver = preconditioner = None
    try:
        for repetition in range(integer(args.repetitions, "Repetitions", 1)):
            resource_tick = time.perf_counter()
            preconditioner = ReplayPreconditioner(
                problem,
                evaluation,
                row["damping"],
                getattr(args, "preconditioner", "jacobi"),
                getattr(args, "sweeps", 3),
            )
            solver = solver_class(
                "jacobi" if reference is None else "reference",
                reference=reference,
                rank=requested,
                window=max(1, requested),
                rtol=cfg["inner_tolerance"],
                maxiter=cfg["inner_cap"],
                cg_factor=0.1,
                residual_policy="refine",
                **solver_options(cfg, solver_class),
            )
            barrier()
            record.setdefault("resource_creation_seconds", []).append(
                time.perf_counter() - resource_tick
            )
            for index, system in enumerate(manifest["systems"]):
                if system["quadratic"] != number:
                    continue
                data = read_arrays(args.trace, system["file"], system["sha256"])
                I, rhs, initial = data["indices"], data["rhs"], data["initial"]
                B = H.restrict(I)
                B.diagonal = lambda I=I: diagonal[I].copy()
                if args.feedback is None and system["status"] == "converged":
                    saved = read_arrays(
                        args.trace, system["solution_file"], system["solution_sha256"]
                    )
                    if independent_residual(B, saved["x"], rhs) > cfg["inner_tolerance"]:
                        raise ValueError("Reconstructed operator fails the saved solution check")
                barrier()
                tick = time.perf_counter()
                preconditioner.attach(B, I)
                result, timing = solver.solve(B, rhs, I, initial=initial.copy())
                barrier()
                measured = time.perf_counter() - tick
                residual = independent_residual(B, result.x, rhs)
                result_row = {
                    "system": index,
                    "repetition": repetition,
                    "status": result.status,
                    "verified": result.status == "converged" and residual <= cfg["inner_tolerance"],
                    "original_residual": residual,
                    "iterations": result.iterations,
                    "deployed_rank": result.rank,
                    "fallback": result.fallback_reason,
                    "coarse_condition": result.coarse_condition,
                    "solve_seconds": measured,
                    "timing": timing,
                    "preconditioner_cumulative": preconditioner.report(),
                }
                if args.diagnostics and repetition == 0:
                    result_row["error_diagnostic"] = error_diagnostic(
                        B, rhs, initial, reference, I, diagonal[I]
                    )
                record["rows"].append(result_row)
                write_report(args.output / "record.json", record)
            barrier()
            tick = time.perf_counter()
            solver.close()
            preconditioner.close()
            preconditioner = None
            barrier()
            record.setdefault("cleanup_seconds", []).append(time.perf_counter() - tick)
            solver = None
            gc.collect()
        record["status"] = "complete"
    except Exception as error:
        record.update(status="replay_failed", error_type=type(error).__name__)
        raise
    finally:
        if solver is not None:
            solver.close()
        if preconditioner is not None:
            preconditioner.close()
        record["memory"] = sampler.finish()
        record["all_systems_verified"] = bool(record["rows"]) and all(
            r["verified"] for r in record["rows"]
        )
        write_report(args.output / "record.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bank", "replay"))
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "hybrid"), default="hybrid")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--width", type=int, choices=(1, 5, 10, 20), default=5)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--quadratic", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=(
            "jacobi",
            "thermal",
            "nominal_coupled",
            "preconditioned_coupled",
            "krylov_coupled",
        ),
        default="jacobi",
    )
    parser.add_argument("--feedback", type=float, choices=(0.0, 0.5, 1.0))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--preconditioner", choices=("jacobi", "frozen"), default="jacobi")
    parser.add_argument("--sweeps", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--bank-selection", choices=("jacobi", "frozen"), default="jacobi")
    args = parser.parse_args()
    if args.mode == "replay" and args.bank is None:
        parser.error("Replay requires --bank")
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        (construct_bank if args.mode == "bank" else replay)(args)


if __name__ == "__main__":
    main()
