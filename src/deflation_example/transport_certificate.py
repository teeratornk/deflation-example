"""Energy-stability certificates for the thermal operators used in the timed comparisons.

For each geometry and refinement level, the symmetric part of the free-node spatial
operator is factored in symmetric mode without row interchanges. By Sylvester's law
of inertia the pivot signs count its negative, zero and positive eigenvalues. A
positive-definite symmetric part rules out growing backward-Euler modes at every
time step, because Re(lambda) = x*Kx / x*Cx > 0 for every eigenpair K x = lambda C x.

    python -m deflation_example.transport_certificate --output certificates.json \
        --case transformer_2d:0:skew --case engine_3d:3:advective
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .mesh_showcases import build_showcase


def inertia(matrix):
    lu = splu(sparse.csc_matrix(matrix), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
              options={"SymmetricMode": True})
    if not np.array_equal(lu.perm_r, lu.perm_c):
        raise ValueError("Row interchanges occurred; the pivot signs do not give the inertia")
    pivots = lu.U.diagonal()
    return {
        "negative": int((pivots < 0).sum()),
        "zero": int((pivots == 0).sum()),
        "positive": int((pivots > 0).sum()),
        "smallest_pivot_ratio": float(np.abs(pivots).min() / np.abs(pivots).max()),
    }


def certify(geometry, level, form):
    start = time.perf_counter()
    showcase = build_showcase(geometry, level=level, transport_form=form)
    a = showcase.assembly
    free = a.mesh.free
    record = {"geometry": geometry, "level": level, "transport_form": form,
              "free_nodes": int(len(free))}
    for name, matrix in (("operator", a.stiffness), ("transport", a.transport)):
        block = matrix[free][:, free]
        symmetric = 0.5 * (block + block.T)
        if name == "transport":
            # A tiny capacity shift separates a semidefinite transport part from an indefinite one.
            shift = 1e-9 * abs(symmetric).max() / a.capacity[free].max()
            symmetric = symmetric + shift * sparse.diags(a.capacity[free])
            record["transport_shift"] = float(shift)
        record[name] = inertia(symmetric)
    record["energy_stable"] = record["operator"]["negative"] == 0 and record["operator"]["zero"] == 0
    record["seconds"] = time.perf_counter() - start
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True,
                        help="geometry:level:transport_form")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new certificate output")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=Path(__file__).parent).stdout.strip()
    clean = not subprocess.run(["git", "status", "--porcelain", "--", "src"], capture_output=True,
                               text=True, cwd=Path(__file__).parent).stdout.strip()
    records = []
    for case in args.case:
        geometry, level, form = case.split(":")
        records.append(certify(geometry, int(level), form))
        print(json.dumps(records[-1]), flush=True)
    args.output.write_text(json.dumps({"git_head": head, "source_tree_clean": clean,
                                       "certificates": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
