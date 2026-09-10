"""Matched full-reference and sequential-transfer replay of accepted CHT traces."""

import argparse
from pathlib import Path
import time

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from .benchmark_cht import build_model, build_reference, desired_and_load, unpack_mask
from .benchmark_extended import digest
from .recycling import transfer_basis
from .reporting import environment, write_report
from .solvers import deflated_cg, independent_residual, orthonormalize
from .tutorials import Records
from .validation import integer


def correction_diagnostics(B, b, basis, deployed_rank, exact, spectral_limit=512):
    """Coarse removal of initial energy error and the remaining scaled spectrum.

    Replay starts from zero. An independently solved system defines the error.
    The rank must match the actual solver. A rank-zero fallback removes zero
    energy. The spectrum is reported only for declared small systems.
    """
    initial_energy = float(exact @ (B @ exact))
    if not np.isfinite(initial_energy) or initial_energy <= 0:
        return {"status": "zero_or_invalid_initial_energy"}
    Z = orthonormalize(basis) if deployed_rank else np.empty((len(b), 0))
    if Z.shape[1] != deployed_rank:
        return {"status": "diagnostic_rank_differs_from_solver", "diagnostic_rank": Z.shape[1]}
    E = Z.T @ (B @ Z)
    if deployed_rank:
        factor = linalg.cho_factor((E + E.T) / 2)
        correction = Z @ linalg.cho_solve(factor, Z.T @ b)
    else:
        correction = np.zeros(len(b))
    remaining = exact - correction
    removed = float(correction @ (B @ correction) / initial_energy)
    remaining_fraction = float(remaining @ (B @ remaining) / initial_energy)
    if not np.isclose(removed + remaining_fraction, 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("Coarse energy decomposition failed its independent check")
    result = {
        "status": "verified",
        "initial_energy": initial_energy,
        "coarse_removed_energy_fraction": removed,
        "remaining_energy_fraction": remaining_fraction,
        "effective_rank": deployed_rank,
    }
    if len(b) <= spectral_limit:
        root = np.sqrt(B.diagonal())
        scaled = B.toarray() / root[:, None] / root[None, :]
        eigenvalues = linalg.eigvalsh(scaled)
        projected = scaled.copy()
        if deployed_rank:
            action = (B @ Z) / root[:, None]
            projected -= action @ linalg.cho_solve(factor, action.T)
        values = linalg.eigvalsh((projected + projected.T) / 2)
        remaining_values = values[deployed_rank:]
        if len(remaining_values) and remaining_values[0] <= 0:
            raise ValueError("Remaining deflated spectrum is not numerically positive")
        result.update(
            original_preconditioned_min=float(eigenvalues[0]),
            original_preconditioned_max=float(eigenvalues[-1]),
            original_preconditioned_condition=float(eigenvalues[-1] / eigenvalues[0]),
            deflated_zero_eigenvalue_max_abs=float(
                np.max(np.abs(values[:deployed_rank]), initial=0)
            ),
            remaining_nonzero_eigenvalues=remaining_values.tolist(),
            effective_condition=None
            if not len(remaining_values)
            else float(remaining_values[-1] / remaining_values[0]),
        )
    return result


def load_trace(root, method, repetition, warm):
    records = Records(root)
    manifest = records.load("results.json")
    protocol = records.load("protocol.json")
    if records.manifest["protocol.json"] != manifest["protocol_sha256"]:
        raise ValueError("Source protocol hash does not match its manifest")
    matches = [
        row
        for row in manifest["sequences"]
        if (row["method"], row["repetition"], row["warm_start"]) == (method, repetition, warm)
    ]
    if len(matches) != 1:
        raise ValueError("Select one recorded source sequence")
    entry = matches[0]
    sequence = records.load(entry["path"])
    if (
        records.manifest[entry["path"]] != entry["sha256"]
        or sequence["protocol_sha256"] != manifest["protocol_sha256"]
    ):
        raise ValueError("Source sequence hash does not match its manifest")
    if not sequence["success"] or not entry["success"]:
        raise ValueError("Primary transfer replay requires an accepted complete source sequence")
    return protocol, sequence, records.manifest


def run(
    records_root,
    output,
    source_method="reference",
    repetition=0,
    warm="outer_inner",
    device="cpu",
    repeats=3,
    spectral_limit=512,
    threads=4,
):
    integer(repeats, "Repetitions", 1)
    integer(repetition, "Source repetition")
    integer(spectral_limit, "Spectral dimension limit")
    integer(threads, "Threads", 1)
    if device not in {"cpu", "cuda"}:
        raise ValueError("Replay device must be cpu or cuda")
    protocol, sequence, inputs = load_trace(records_root, source_method, repetition, warm)
    c = protocol["controls"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    specification = {
        "protocol": "reference-transfer-replay-v1",
        "input_sha256": inputs,
        "source_protocol": protocol,
        "source_method": source_method,
        "source_repetition": repetition,
        "source_warm_start": warm,
        "device": device,
        "repetitions_per_kernel": repeats,
        "spectral_dimension_limit": spectral_limit,
        "threads": threads,
        "initial_guess": "zero for both replay methods",
        "transfer": "full_reference restricts the same original reference at every step; sequential_transfer restricts at the first step, then zero-extends and transfers only its previous restricted values; original reference values are never restored in this control",
        "identity": "one-step full-reference loss R_b Phi - R_b R_a.T R_a Phi equals R_b (I - R_a.T R_a) Phi; sequential transfer can also accumulate loss from earlier masks",
        "rank": "retain actual numerical rank; no replacement directions; coarse fallback recorded",
        "timing": "basis restriction or transfer plus complete kernel cost; matrix/trace reconstruction, independent direct solution and spectral/energy diagnostics are separate common analysis work; both methods follow the same small QR/CG warmup",
        "scope": "verified fixed-inactive-system replay of a complete optimization trace; distinct from new complete optimization timings",
    }
    write_report(output / "protocol.json", specification)
    result = {
        "specification": specification,
        "environment": environment(),
        "rows": [],
        "success": False,
    }
    write_report(output / "results.json", result)
    torch = None
    if device == "cuda":
        from .gpu import require_cuda

        torch = require_cuda()
        torch.set_num_threads(threads)
        torch.ones(1, device="cuda").sum().item()
        result["environment"].update(
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()
        )
    with threadpool_limits(limits=threads):
        tick = time.perf_counter()
        warm_matrix = sparse.diags(
            [-np.ones(31), 3 * np.ones(32), -np.ones(31)], [-1, 0, 1], format="csr"
        )
        warm_rhs = np.linspace(0.1, 1.0, 32)
        warm_basis = np.linalg.eigh(warm_matrix.toarray())[1][:, :4]
        if torch is None:
            warmed = deflated_cg(warm_matrix, warm_rhs, warm_basis)
        else:
            from .gpu import gpu_deflated_cg

            warmed, _ = gpu_deflated_cg(warm_matrix, warm_rhs, warm_basis, basis_backend="gpu_qr")
        if warmed.status != "converged":
            raise RuntimeError("Replay warmup failed original residual acceptance")
        result["common_warmup_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        model = build_model(c)
        reference = build_reference(model, c)
        result["common_construction_seconds"] = time.perf_counter() - tick
        result["reference_storage"] = reference.storage()
        previous = carried = None
        for case in sequence["cases"]:
            desired, load = desired_and_load(model, protocol["targets"][case["query"]])
            if digest(desired) != case["target_sha256"]:
                raise ValueError("Reconstructed target differs from the source trace")
            for step, source in enumerate(case["inner"]):
                if source["status"] != "converged" or source["original_residual"] > c["rtol"]:
                    raise ValueError("Source inner solve failed independent residual acceptance")
                mask = unpack_mask(source["inactive_mask_bits"], model.H.shape[0])
                I, J = np.flatnonzero(mask), np.flatnonzero(~mask)
                B = model.H[I][:, I].tocsr()
                b = load[I] - model.H[I][:, J] @ np.full(len(J), c["bound"])
                if digest(b) != source["rhs_sha256"]:
                    raise ValueError("Reconstructed right-hand side differs from the source trace")
                tick = time.perf_counter()
                exact = spsolve(B, b)
                exact_seconds = time.perf_counter() - tick
                exact_residual = independent_residual(B, exact, b)
                if exact_residual > c["rtol"]:
                    raise ValueError("Independent solution failed original residual acceptance")
                full = reference.restrict(I)
                transferred = (
                    full.copy() if previous is None else transfer_basis(carried, previous, I)
                )
                newly_inactive = (
                    np.empty(0, dtype=int)
                    if previous is None
                    else np.setdiff1d(I, previous, assume_unique=True)
                )
                one_step = (
                    full.copy()
                    if previous is None
                    else transfer_basis(reference.restrict(previous), previous, I)
                )
                released_rows = np.isin(I, newly_inactive)
                identity_rhs = np.zeros_like(full)
                identity_rhs[released_rows] = full[released_rows]
                np.testing.assert_allclose(full - one_step, identity_rhs, atol=1e-13)
                row = {
                    "query": case["query"],
                    "step": step,
                    "n_inactive": len(I),
                    "newly_inactive": len(newly_inactive),
                    "newly_active": 0
                    if previous is None
                    else len(np.setdiff1d(previous, I, assume_unique=True)),
                    "newly_supplied_reference_frobenius": float(np.linalg.norm(identity_rhs)),
                    "accumulated_transfer_difference_frobenius": float(
                        np.linalg.norm(full - transferred)
                    ),
                    "one_step_identity_error": float(
                        np.linalg.norm(full - one_step - identity_rhs)
                    ),
                    "independent_solution_seconds": exact_seconds,
                    "independent_solution_residual": exact_residual,
                    "rhs_sha256": digest(b),
                    "methods": {},
                }
                for repeat in range(repeats):
                    methods = (
                        ("full_reference", "sequential_transfer")
                        if repeat % 2 == 0
                        else ("sequential_transfer", "full_reference")
                    )
                    for method in methods:
                        tick = time.perf_counter()
                        basis = (
                            reference.restrict(I)
                            if method == "full_reference" or previous is None
                            else transfer_basis(carried, previous, I)
                        )
                        transfer_seconds = time.perf_counter() - tick
                        tick = time.perf_counter()
                        if torch is None:
                            solved = deflated_cg(
                                B,
                                b,
                                basis,
                                B.diagonal(),
                                rtol=c["rtol"] * c.get("cg_factor", 1.0),
                                maxiter=c["inner_cap"],
                                refresh=c.get("residual_refresh", 1000),
                                cache_operator_product=c.get("cache_operator_product", False),
                            )
                            kernel_seconds = time.perf_counter() - tick
                            if (
                                solved.status in {"maxiter", "residual_failed"}
                                and solved.residual <= c["rtol"]
                            ):
                                solved.status = "converged"
                        else:
                            from .gpu import gpu_deflated_cg

                            solved, timing = gpu_deflated_cg(
                                B,
                                b,
                                basis,
                                B.diagonal(),
                                rtol=c["rtol"] * c.get("cg_factor", 1.0),
                                acceptance_rtol=c["rtol"],
                                maxiter=c["inner_cap"],
                                basis_backend="gpu_qr",
                                refresh=c.get("residual_refresh", 1000),
                                cache_operator_product=c.get("cache_operator_product", False),
                            )
                            kernel_seconds = timing["total_seconds"]
                        entry = {
                            "repetition": repeat,
                            "transfer_seconds": transfer_seconds,
                            "kernel_seconds": kernel_seconds,
                            "total_seconds": transfer_seconds + kernel_seconds,
                            "iterations": solved.iterations,
                            "requested_rank": c["rank"],
                            "deployed_rank": solved.rank,
                            "coarse_condition": solved.coarse_condition,
                            "fallback_reason": solved.fallback_reason,
                            "status": solved.status,
                            "original_residual": solved.residual,
                        }
                        group = row["methods"].setdefault(method, {"repetitions": []})
                        group["repetitions"].append(entry)
                        if repeat == 0:
                            tick = time.perf_counter()
                            group["diagnostics"] = correction_diagnostics(
                                B, b, basis, solved.rank, exact, spectral_limit
                            )
                            group["diagnostic_seconds"] = time.perf_counter() - tick
                for group in row["methods"].values():
                    accepted = [
                        entry
                        for entry in group["repetitions"]
                        if entry["status"] == "converged"
                        and entry["original_residual"] <= c["rtol"]
                    ]
                    group["accepted_repetitions"] = len(accepted)
                    group["accepted_median_seconds"] = (
                        float(np.median([entry["total_seconds"] for entry in accepted]))
                        if accepted
                        else None
                    )
                    group["all_repetitions_accepted"] = len(accepted) == repeats
                result["rows"].append(row)
                previous, carried = I, transferred
                write_report(output / "results.json", result)
        result["success"] = bool(result["rows"]) and all(
            group["all_repetitions_accepted"]
            for row in result["rows"]
            for group in row["methods"].values()
        )
        write_report(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-method", default="reference")
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--warm", default="outer_inner")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--spectral-limit", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    if not run(**vars(parser.parse_args()))["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
