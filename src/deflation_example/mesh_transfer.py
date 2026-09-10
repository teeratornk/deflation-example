"""Replay full-reference restriction and sequential transfer on mesh PDAS traces."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from .benchmark_cht import unpack_mask
from .benchmark_mesh import build_model
from .benchmark_transfer import correction_diagnostics
from .mesh_reference import build_mesh_reference
from .mesh_showcases import desired_temperature
from .recycling import transfer_basis
from .reporting import environment, write_report
from .solvers import deflated_cg, independent_residual
from .validation import integer


def run(record_path, output, device="cpu", samples=8, repeats=3):
    integer(samples, "Sample count", 1)
    integer(repeats, "Kernel repetitions", 1)
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose CPU or CUDA replay")
    record_path, output = Path(record_path), Path(output)
    source = json.loads(record_path.read_text())
    if not source["success"] or not source.get("cases"):
        raise ValueError("Replay requires an accepted complete optimization sequence")
    c = source["controls"]
    output.mkdir(parents=True, exist_ok=False)
    showcase, model = build_model(c)
    model.H.assembled_restriction = True
    if showcase.preparation.get("input_sha256") != source["mesh"].get("input_sha256"):
        raise ValueError("Replay input bundle differs from the recorded geometry and physics")
    reference = build_mesh_reference(model, showcase.coarse_assembly, showcase.prolongation,
                                     c["rank"], c["construction"], c["spatial_reference"],
                                     c.get("temporal_metric", "euclidean"))
    trace = [(case["query"], j, row) for case in source["cases"]
             for j, row in enumerate(case["inner"])]
    selected = set(np.linspace(0, len(trace)-1, min(samples, len(trace)), dtype=int).tolist())
    if device == "cuda":
        from .gpu import gpu_deflated_cg, require_cuda
        from scipy import sparse
        require_cuda()
        warmed, _ = gpu_deflated_cg(sparse.diags(np.arange(1., 17), format="csr"), np.ones(16),
                                     np.eye(16)[:, :2], basis_backend="gpu_qr")
        if warmed.status != "converged":
            raise RuntimeError("Replay warmup failed residual acceptance")

        def solve(B, b, basis):
            return gpu_deflated_cg(B, b, basis, B.diagonal(), basis_backend="gpu_qr",
                                   rtol=c["rtol"]*c["cg_factor"], acceptance_rtol=c["rtol"],
                                   maxiter=c["inner_cap"], refresh=c["residual_refresh"],
                                   cache_operator_product=True)[0]
    else:
        def solve(B, b, basis):
            return deflated_cg(B, b, basis, B.diagonal(), rtol=c["rtol"],
                               maxiter=c["inner_cap"], refresh=c["residual_refresh"],
                               cache_operator_product=True)
    result = {"protocol": "mesh-reference-transfer-v1", "environment": environment(),
              "source_record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
              "source_controls": c, "device": device,
              "initial_guess": "zero for both specified transfers",
              "scope": "fixed-inactive-system replay; kernel intervals exclude shared reconstruction and independent direct diagnostics",
              "selection": sorted(selected), "rows": [], "success": False,
              "transfer": "Sequential transfer starts with the same restricted reference and then zero-extends only its surviving values. No replacement directions are introduced."}
    previous = previous_full = sequential = None
    for index, (query, outer, row) in enumerate(trace):
        I = np.flatnonzero(unpack_mask(row["inactive_mask_bits"], model.size))
        full = reference.restrict(I)
        if previous is None:
            sequential = full.copy()
            released, activated, identity_error = 0, 0, 0.
        else:
            sequential = transfer_basis(sequential, previous, I)
            released_rows = ~np.isin(I, previous)
            released = int(released_rows.sum())
            activated = int(np.sum(~np.isin(previous, I)))
            one_step = full - transfer_basis(previous_full, previous, I)
            predicted = np.zeros_like(full)
            predicted[released_rows] = full[released_rows]
            identity_error = float(np.linalg.norm(one_step-predicted))
        previous, previous_full = I, full
        entry = {"trace_index": index, "query": query, "outer_step": outer,
                 "inactive_count": len(I), "newly_inactive": released, "newly_active": activated,
                 "transfer_identity_error": identity_error, "methods": {}}
        if index in selected:
            desired = desired_temperature(model, query, c["targets"])
            if hashlib.sha256(np.ascontiguousarray(desired).tobytes()).hexdigest() != source["cases"][query]["target_sha256"]:
                raise ValueError("Replay target differs from the source sequence")
            boundary = np.full(model.size, c["bound"])
            boundary[I] = 0
            b = model.load(desired)[I] - (model.H @ boundary)[I]
            if hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() != row["rhs_sha256"]:
                raise ValueError("Replay right-hand side differs from its source trace")
            B = model.H.restrict(I)
            exact = spsolve(B, b)
            error = independent_residual(B, exact, b)
            if error > c["rtol"]:
                raise ValueError("The independent diagnostic solution failed residual acceptance")
            entry["independent_original_residual"] = error
            for name, basis in (("full_reference", full), ("sequential_transfer", sequential)):
                records = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    solved = solve(B, b, basis)
                    records.append({"seconds": time.perf_counter()-start,
                                    "status": solved.status, "original_residual": solved.residual,
                                    "iterations": solved.iterations, "deployed_rank": solved.rank,
                                    "coarse_condition": solved.coarse_condition,
                                    "fallback_reason": solved.fallback_reason})
                entry["methods"][name] = {"repetitions": records,
                    "correction": correction_diagnostics(B, b, basis, solved.rank, exact)}
        result["rows"].append(entry)
        write_report(output / "transfer.json", result)
    result["success"] = all(
        repetition["status"] == "converged" and repetition["original_residual"] <= c["rtol"]
        for row in result["rows"] for method in row["methods"].values()
        for repetition in method["repetitions"]
    ) and all(row["transfer_identity_error"] <= 1e-12 for row in result["rows"])
    write_report(output / "transfer.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    with threadpool_limits(4):
        report = run(args.record, args.output, args.device, args.samples, args.repeats)
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
