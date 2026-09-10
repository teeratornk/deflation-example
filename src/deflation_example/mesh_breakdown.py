"""Instrument an unchanged mesh protocol and retain failing linear systems.

Diagnostic timings include scalar checks and snapshots. They are separate
from complete-sequence performance measurements. All arrays are numerical;
the snapshot contains the exact CSR matrix, right-hand side, initial guess,
candidate basis and returned iterate of each rejected solve.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .reporting import write_fields, write_report


def run(protocol, output, method="reference"):
    from . import gpu
    from .benchmark_mesh import worker, controls
    from omegaconf import OmegaConf

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    config = controls(OmegaConf.create(json.loads(Path(protocol).read_text())))
    config.pop("output", None)
    write_report(destination / "protocol.json", config)
    original = gpu.gpu_deflated_cg
    calls = []

    def instrumented(A, b, basis=None, diagonal=None, **kwargs):
        result, metrics = original(A, b, basis, diagonal, diagnostics=True, **kwargs)
        number = len(calls)
        row = {
            "call": number,
            "size": len(b),
            "status": result.status,
            "iterations": result.iterations,
            "residual": result.residual,
            "rank": result.rank,
            "coarse_condition": result.coarse_condition,
            "termination_test": metrics["termination_test"],
            "diagnostics": metrics["diagnostics"],
        }
        if result.status != "converged":
            candidate = kwargs.get("device_basis")
            candidate = candidate.cpu().numpy() if candidate is not None else basis
            name = f"system-{number}.npz"
            write_fields(
                destination / name,
                indptr=A.indptr,
                indices=A.indices,
                data=A.data,
                shape=np.array(A.shape),
                rhs=b,
                initial=np.zeros_like(b) if kwargs.get("x0") is None else kwargs["x0"],
                basis=np.empty((len(b), 0)) if candidate is None else candidate,
                diagonal=A.diagonal() if diagonal is None else diagonal,
                solution=result.x,
            )
            row["snapshot"] = name
        calls.append(row)
        write_report(destination / "diagnostics.json", calls)
        return result, metrics

    gpu.gpu_deflated_cg = instrumented
    try:
        return worker(destination / "protocol.json", destination / "sequence", method, 0)
    finally:
        gpu.gpu_deflated_cg = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method", choices=("reference", "jacobi", "recycling"), default="reference"
    )
    args = parser.parse_args()
    raise SystemExit(0 if run(args.protocol, args.output, args.method) else 1)


if __name__ == "__main__":
    main()
