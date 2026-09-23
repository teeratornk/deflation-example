"""Full-domain energy-metric Krylov diagnostics for a fixed coupled operator.

The starting vector is a seeded random vector, independent of target loads and
inactive sets. Ritz values and their residuals describe the sampled subspace;
they do not bound the complete spectrum or certify a runtime improvement.
"""

import argparse
from pathlib import Path
import time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from .coupled_preconditioned_reference import end_indices
from .reporting import environment, file_sha256, write_report
from .study_solvers import ArrayReference
from .validation import integer, real_array


def krylov_reference(operator, inverse, diagonal, rank=8, *, steps=48, seed=20260923):
    """Rayleigh--Ritz extraction of K H in the H inner product.

    Two full orthogonalization passes use cached H-products. Each retained
    direction has a freshly evaluated H norm. Rank loss ends construction;
    no replacement direction is introduced. H and K must be fixed SPD maps.
    """
    start = time.perf_counter()
    n = operator.shape[0]
    steps = min(integer(steps, "Krylov steps", 1), n)
    rank = integer(rank, "Reference rank", 1)
    seed = integer(seed, "Random seed", 0)
    d = real_array(diagonal, "Starting-vector scale")
    if operator.shape != (n, n) or d.shape != (n,) or not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("Use a square operator and a positive finite diagonal")
    Q, HQ, AQ = (np.empty((n, steps)) for _ in range(3))
    counts = {"operator": 0, "inverse": 0}

    def apply(kind, x):
        y = real_array(operator @ x if kind == "operator" else inverse(x), kind)
        counts[kind] += 1
        if y.shape != x.shape or not np.isfinite(y).all():
            raise ValueError("Operator applications must preserve shape and finite values")
        return y

    z = np.random.default_rng(seed).standard_normal(n) / np.sqrt(d)
    size = 0
    termination = "step_limit"
    previous_energy = None
    for j in range(steps):
        for _ in range(2):
            if j:
                z -= Q[:, :j] @ (HQ[:, :j].T @ z)
        hz = apply("operator", z)
        energy = float(z @ hz)
        if not np.isfinite(energy) or energy < 0:
            raise ValueError("The operator failed the positive-energy check")
        if energy == 0 or (previous_energy is not None and energy <= 1e-24 * previous_energy):
            termination = "invariant_subspace"
            break
        Q[:, j], HQ[:, j] = z / np.sqrt(energy), hz / np.sqrt(energy)
        AQ[:, j] = apply("inverse", HQ[:, j])
        if float(HQ[:, j] @ AQ[:, j]) <= 0:
            raise ValueError("The inverse failed the positive-energy check")
        size = j + 1
        z = AQ[:, j].copy()
        # q has unit H norm; the squared Ritz diagonal supplies a local scale.
        previous_energy = float(HQ[:, j] @ AQ[:, j]) ** 2
    Q, HQ, AQ = Q[:, :size], HQ[:, :size], AQ[:, :size]
    metric, projected = Q.T @ HQ, HQ.T @ AQ
    for name, form in (("energy metric", metric), ("preconditioned Ritz form", projected)):
        skew = linalg.norm(form - form.T) / max(linalg.norm(form), 1e-300)
        if not np.isfinite(skew) or skew > 1e-9:
            raise ValueError(name + " fails symmetry verification")
    if not size:
        raise ValueError("The starting vector has zero operator energy")
    values, vectors = linalg.eigh((projected + projected.T) / 2, (metric + metric.T) / 2)
    if values[0] <= 0:
        raise ValueError("The Ritz form is not positive definite")
    selected = end_indices(size)[:rank]
    basis = Q @ vectors[:, selected]
    residuals = []
    for index, column in enumerate(selected):
        residual = AQ @ vectors[:, column] - basis[:, index] * values[column]
        energy = float(residual @ apply("operator", residual))
        if energy < 0:
            raise ValueError("The Ritz residual has negative operator energy")
        residuals.append(float(np.sqrt(energy) / values[column]))
    return ArrayReference(
        basis,
        {
            "selection": "fixed_nominal_energy_krylov_ritz",
            "requested_rank": rank,
            "deployed_rank": basis.shape[1],
            "requested_steps": steps,
            "independent_candidates": size,
            "seed": seed,
            "termination": termination,
            "energy_orthogonality_error": float(linalg.norm(metric - np.eye(size))),
            "ritz_selection": "alternating_low_high",
            "ritz_values": values[selected].tolist(),
            "all_projected_ritz_values": values.tolist(),
            "relative_ritz_residuals_H_norm": residuals,
            "operator_applications": counts["operator"],
            "inverse_applications": counts["inverse"],
            "construction_seconds": time.perf_counter() - start,
            "scope": "Seeded full-domain Krylov space; no target or inactive-set information. Ritz values are sampled spectral information, not global extremal-eigenvalue certificates.",
        },
    )


def main():
    from .coupled_frozen_preconditioner import frozen_preconditioner_factory
    from .coupled_optimize import equation_acceptance, load_problem
    from .coupled_retention_replay import rebuild, save_reference, trace_configuration
    from .coupled_sequence import prepare_device
    from .coupled_trace import read_arrays, read_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.device, args.width = "cpu", 20
    manifest = read_manifest(args.trace)
    if manifest["status"] != "complete" or not manifest["quadratics"]:
        raise ValueError("Use a complete diagnostic trace")
    row = manifest["quadratics"][0]
    if row["iteration"] != 0 or row["attempt"] != 0 or row["damping"] != 0:
        raise ValueError("Use the first undamped initial trajectory")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-reference-bank-v1",
        "environment": environment(),
        "trace_sha256": file_sha256(args.trace / "manifest.json"),
        "status": "building",
        "references": {},
        "scope": "Full-domain construction diagnostic; no complete optimization timing.",
    }
    write_report(args.output / "record.json", report)
    sampler = None
    with threadpool_limits(integer(args.threads, "Threads", 1)):
        try:
            cfg = trace_configuration(args, manifest)
            tick = time.perf_counter()
            problem, baseline = load_problem(cfg)
            report["model_assembly_seconds"] = time.perf_counter() - tick
            if baseline["baseline_sha256"] != manifest["baseline_sha256"]:
                raise ValueError("Baseline differs from the recorded trace")
            report["baseline_sha256"] = baseline["baseline_sha256"]
            sampler, _, _, _ = prepare_device("cpu", 0.01)
            sampler.start()
            arrays = read_arrays(args.trace, row["file"], row["sha256"])
            tick = time.perf_counter()
            evaluation, H, diagonal, _ = rebuild(problem, arrays, row, use_secants=False)
            if not equation_acceptance(problem.verify(evaluation), cfg):
                raise ValueError("Nominal coupled equations fail verification")
            report["nominal_evaluation_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter()
            inverse = frozen_preconditioner_factory(problem, evaluation, sweeps=3)(
                np.arange(problem.size)
            )
            reference = krylov_reference(
                H, inverse, diagonal, args.rank, steps=args.steps, seed=args.seed
            )
            del inverse
            selection_seconds = time.perf_counter() - tick
            report["krylov_selection"] = {
                **save_reference(args.output, "krylov-selection", reference),
                "construction_seconds": report["nominal_evaluation_seconds"] + selection_seconds,
            }
            report["status"] = "complete"
        except Exception as error:
            report.update(status="construction_failed", error_type=type(error).__name__)
            raise
        finally:
            if sampler is not None:
                report["memory"] = sampler.finish()
            write_report(args.output / "record.json", report)


if __name__ == "__main__":
    main()
