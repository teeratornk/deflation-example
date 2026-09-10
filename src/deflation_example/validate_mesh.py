"""Independent bounded least-squares checks for the mesh-based discretization."""

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits

from .meshes import ThermalMesh, ThermalAssembly
from .mesh_control import build_mesh_control
from .mesh_showcases import build_showcase, desired_temperature
from .reporting import environment, write_report
from .solvers import pdas, kkt_metrics, relative_norm, independent_residual, LinearResult


def transformer_subproblem(assembly):
    """A temperature subproblem on the inlet region, with prescribed cut-boundary values.

    The full mesh's principal spatial operator defines this verification case.
    It retains all lumped nodal weights and the source load at the selected nodes.
    No performance claim uses this auxiliary problem.
    """
    selected = np.flatnonzero(assembly.mesh.nodes[:, 1] < .04)
    cells_mask = np.all(np.isin(assembly.mesh.cells, selected), axis=1)
    old_cells = assembly.mesh.cells[cells_mask]
    selected = np.unique(old_cells)
    mapping = np.full(len(assembly.mesh.nodes), -1, dtype=int)
    mapping[selected] = np.arange(len(selected))
    cells = mapping[old_cells]
    nodes = assembly.mesh.nodes[selected]
    pins = np.flatnonzero(np.any(np.isclose(nodes, nodes.min(axis=0))
                                 | np.isclose(nodes, nodes.max(axis=0)), axis=1))
    mesh = ThermalMesh(nodes, cells, assembly.mesh.materials[cells_mask], pins, True)
    def restrict(A):
        return A[selected][:, selected].tocsr()
    return ThermalAssembly(mesh, assembly.mass[selected], assembly.capacity[selected],
                           restrict(assembly.diffusion), restrict(assembly.transport),
                           restrict(assembly.stabilization), assembly.load[selected])


def validate(settings="physical"):
    if settings not in {"physical", "algebraic"}:
        raise ValueError("Choose physical or algebraic verification settings")
    rows = []
    for geometry, alpha, bound in (("engine_3d", 1e-6, .2), ("transformer_2d", 1e-14, .8)):
        showcase = build_showcase(geometry)
        a = showcase.assembly
        horizon = (.1 if geometry == "engine_3d" else
                   600 / showcase.parameters["physical"]["time_scale_s"])
        if settings == "algebraic":
            horizon = .05
            if geometry == "transformer_2d":
                alpha = 1e-12
        if geometry == "transformer_2d":
            a = transformer_subproblem(a)
        for steps in (None, [.5*horizon, .5*horizon], [.2*horizon, .8*horizon]):
            model = build_mesh_control(a, alpha, time_steps=steps,
                                       initial=np.full(len(a.mesh.free), .03))
            root = np.sqrt(model.weights)
            factor = np.vstack((np.diag(root), np.sqrt(alpha)*root[:, None]*model.A.toarray()))
            for query in range(2):
                desired = desired_temperature(model, query, 2)
                rhs = np.r_[root*desired, np.sqrt(alpha)*root*model.forcing]
                reference = lsq_linear(factor, rhs, bounds=(-np.inf, bound),
                                        method="bvls", tol=1e-13, max_iter=1000)
                load = model.load(desired)
                multiplier = np.where(reference.active_mask == 1, load-model.H@reference.x, 0)
                kkt = kkt_metrics(model.H, load, bound, reference.x, multiplier)

                def solve(B, b, I):
                    x = sparse.linalg.spsolve(B, b)
                    residual = independent_residual(B, x, b)
                    return LinearResult(x, 1, residual,
                                         "converged" if residual <= 1e-10 else "residual_failed")

                result = pdas(model.H, load, bound, tolerance=1e-8, linear_solver=solve,
                               initial_active=np.ones(model.size, dtype=bool))
                relative = relative_norm(result["y"]-reference.x, reference.x)
                rng = np.random.default_rng(83)
                trial = rng.normal(size=model.size)
                forward_error = relative_norm(model.forward(model.recover(trial)[0])-trial, trial)
                adjoint_error = relative_norm(model.apply_transpose(trial)-model.A.T@trial,
                                               model.A.T@trial)
                success = (reference.success and result["status"] == "converged"
                           and max(kkt.values()) <= 1e-8 and relative <= 1e-7
                           and forward_error <= 1e-8 and adjoint_error <= 1e-12)
                rows.append({"geometry": geometry, "time_steps": steps, "query": query,
                             "alpha": alpha, "bound": bound,
                             "dimension": model.size, "success": bool(success),
                             "scope": "full engine mesh" if geometry == "engine_3d"
                             else "transformer inlet-region principal subproblem",
                             "independent_success": bool(reference.success),
                             "independent_kkt": kkt, "pdas_status": result["status"],
                             "pdas_kkt": result["kkt"], "state_relative_difference": relative,
                             "objective_relative_difference": abs(model.objective(result["y"], desired)
                                 -model.objective(reference.x, desired))/max(model.objective(reference.x, desired), 1e-30),
                             "forward_recovery_relative": forward_error,
                             "adjoint_action_relative": adjoint_error})
                print(geometry, steps, query, success, relative, flush=True)
    return {"protocol": "mesh-cht-independent-validation-v2", "settings": settings,
            "environment": environment(),
            "success": all(row["success"] for row in rows), "cases": rows,
            "independent_solver": "SciPy BVLS applied to the weighted tracking/control least-squares factor"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", choices=("physical", "algebraic"), default="physical")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(4):
        result = validate(args.settings)
    write_report(args.output / "validation.json", result)
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
