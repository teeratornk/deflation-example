"""Small complete CHT optimization against independent bounded least squares."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits

from .benchmark_cht import build_reference, desired_and_load, query_parameters
from .benchmark_extended import digest
from .problems import build_problem
from .reporting import environment, write_report
from .solvers import kkt_metrics, pdas, relative_norm
from .study_solvers import StudySolver
from .transient import ThermalTrajectory, build_transient_cht


def validate():
    groups = [
        ("steady", build_problem("cht", 4)),
        ("uniform_trajectory", build_transient_cht(3, 4, initial=np.full(27, 0.0002))),
        (
            "nonuniform_trajectory",
            build_transient_cht(
                3, 4, initial=np.full(27, 0.0002), time_steps=[0.01, 0.02, 0.03, 0.04]
            ),
        ),
    ]
    report = {
        "protocol": "independent-complete-cht-validation-v1",
        "bound": 0.0001,
        "final_inner_rtol": 1e-10,
        "cg_internal_rtol": 1e-11,
        "kkt_tolerance": 1e-8,
        "independent_method": "SciPy bounded-variable least squares (BVLS) on the weighted tracking/control residual, tolerance 1e-13",
        "scope": "nine small distinct problems and three complete PDAS solvers per problem; diagnostic validation, without performance claims",
        "groups": [],
        "success": False,
    }
    for name, model in groups:
        trajectory = isinstance(model, ThermalTrajectory)
        alpha = model.spatial.alpha if trajectory else model.alpha
        n, rank = (3, 12) if trajectory else (4, 10)
        dimension = model.H.shape[0]
        weights = model.weights if trajectory else np.ones(dimension)
        forcing = model.forcing if trajectory else np.zeros(dimension)
        objective_scale = model.objective_scale if trajectory else 1 / (n + 1) ** 3
        root = np.sqrt(weights)
        factor = np.vstack((np.diag(root), np.sqrt(alpha) * root[:, None] * model.A.toarray()))
        controls = {
            "n": n,
            "rank": rank,
            "reference_construction": "mode_dependent",
            "reference_capacity": 1.0,
        }
        reference = build_reference(model, controls)
        adapters = {
            m: StudySolver(m, rank=rank, window=rank, reference=reference, cg_factor=0.1)
            for m in ("jacobi", "reference", "recycling")
        }
        previous = {m: None for m in adapters}
        group = {
            "family": name,
            "n": n,
            "dimension": dimension,
            "rank": rank,
            "time_steps": model.steps.tolist() if trajectory else [],
            "initial_temperature": 0.0002 if trajectory else None,
            "cases": [],
        }
        if trajectory:
            rng = np.random.default_rng(42)
            x, p = rng.normal(size=(2, dimension))
            group["forward_action_relative_error"] = relative_norm(
                model.A @ x - model.apply_linear(x), model.A @ x
            )
            group["adjoint_action_relative_error"] = relative_norm(
                model.A.T @ p - model.apply_transpose(p), model.A.T @ p
            )
            u, _ = model.recover(x)
            group["recovered_forward_solution_relative_error"] = relative_norm(
                model.forward(u) - x, x
            )
        try:
            for query in range(3):
                desired, load = desired_and_load(model, query_parameters(query, 3))
                rhs = np.concatenate((root * desired, np.sqrt(alpha) * root * forcing))
                independent = lsq_linear(
                    factor,
                    rhs,
                    bounds=(-np.inf, report["bound"]),
                    method="bvls",
                    tol=1e-13,
                    max_iter=1000,
                )
                multiplier = np.where(
                    independent.active_mask == 1, load - model.H @ independent.x, 0.0
                )
                independent_kkt = kkt_metrics(
                    model.H, load, report["bound"], independent.x, multiplier
                )
                independent_objective = float(
                    0.5 * objective_scale * np.sum((factor @ independent.x - rhs) ** 2)
                )
                case = {
                    "query": query,
                    "target_sha256": digest(desired),
                    "independent_success": bool(independent.success),
                    "independent_native_status": int(independent.status),
                    "independent_kkt": independent_kkt,
                    "independent_objective": independent_objective,
                    "methods": {},
                }
                for method, adapter in adapters.items():
                    prior = previous[method]
                    state = np.zeros(dimension) if prior is None else prior["y"].copy()
                    inner = []

                    def solve(B, b, I):
                        result, _ = adapter.solve(B, b, I, state[I])
                        inner.append(
                            {
                                "status": result.status,
                                "original_residual": result.residual,
                                "iterations": result.iterations,
                                "deployed_rank": result.rank,
                            }
                        )
                        if result.status == "converged":
                            state[:] = report["bound"]
                            state[I] = result.x
                        return result

                    solved = pdas(
                        model.H,
                        load,
                        report["bound"],
                        initial_active=np.ones(dimension, dtype=bool)
                        if prior is None
                        else prior["active"],
                        tolerance=1e-8,
                        linear_solver=solve,
                    )
                    objective = float(
                        0.5 * objective_scale * np.sum((factor @ solved["y"] - rhs) ** 2)
                    )
                    result = {
                        "status": solved["status"],
                        "kkt": solved["kkt"],
                        "inner": inner,
                        "outer_iterations": solved["iterations"],
                        "inner_iterations": sum(i["iterations"] for i in inner),
                        "state_relative_difference": relative_norm(
                            solved["y"] - independent.x, independent.x
                        ),
                        "relative_objective_difference": abs(objective - independent_objective)
                        / independent_objective,
                        "physical_objective": objective,
                        "constraint_releases": sum(step["left"] for step in solved["history"]),
                        "constraint_activations": sum(
                            step["entered"] for step in solved["history"]
                        ),
                    }
                    if trajectory:
                        result["recovery"] = model.recovery_metrics(
                            solved["y"], desired, solved["multiplier"]
                        )
                    case["methods"][method] = result
                    previous[method] = solved if solved["status"] == "converged" else None
                group["cases"].append(case)
        finally:
            for adapter in adapters.values():
                adapter.close()
        report["groups"].append(group)
    report["success"] = all(
        c["independent_success"]
        and max(c["independent_kkt"].values()) <= 1e-8
        and all(
            m["status"] == "converged"
            and max(m["kkt"].values()) <= 1e-8
            and m["relative_objective_difference"] <= 1e-10
            and m["state_relative_difference"] <= 1e-6
            and all(
                i["status"] == "converged" and i["original_residual"] <= 1e-10 for i in m["inner"]
            )
            and max(m.get("recovery", {"steady": 0}).values()) <= 1e-8
            for m in c["methods"].values()
        )
        for g in report["groups"]
        for c in g["cases"]
    )
    report["success"] = report["success"] and all(
        group.get(key, 0.0) <= 1e-11
        for group in report["groups"]
        for key in (
            "forward_action_relative_error",
            "adjoint_action_relative_error",
            "recovered_forward_solution_relative_error",
        )
    )
    return report


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        report = validate()
        report["environment"] = environment()
        write_report(output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    if not run(**vars(parser.parse_args()))["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
