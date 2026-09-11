"""Optimized bore-in-block temperatures at predeclared upper bounds.

This field illustration uses the existing guarded reference solver and a
sparse-direct PDAS control. It supplies no timing comparison. The same target,
operator and reference are retained across bounds within each problem form.
"""

import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from .benchmark_mesh import build_model, controls
from .mesh_presentation import tetrahedral_slice
from .mesh_reference import build_mesh_reference
from .mesh_showcases import desired_temperature
from .reporting import environment, write_fields, write_report
from .solvers import LinearResult, independent_residual, kkt_metrics, pdas, relative_norm
from .study_solvers import StudySolver
from .validation import integer, positive_real


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def model_settings(c, form):
    return controls(
        OmegaConf.create(
            {
                "geometry": "engine_3d",
                "level": c["level"],
                "transient": form == "transient",
                "slabs": c["slabs"],
                "horizon": c["horizon"],
                "alpha": c["alpha"],
                "rank": c["rank"],
                "spatial_reference": "scaled_schur",
                "temporal_metric": "jacobi",
            }
        )
    )


def settings(config):
    c = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    for key in ("level", "query"):
        integer(c[key], key)
    for key in (
        "query_population",
        "slabs",
        "rank",
        "threads",
        "inner_cap",
        "outer_cap",
        "residual_refresh",
    ):
        integer(c[key], key, 1)
    if c["query"] >= c["query_population"]:
        raise ValueError("The target index must belong to the declared query population")
    for key in (
        "horizon",
        "alpha",
        "rtol",
        "kkt_tolerance",
        "cg_factor",
        "state_comparison_tolerance",
        "forward_tolerance",
    ):
        positive_real(c[key], key)
    if c["cg_factor"] > 1:
        raise ValueError("The internal stopping factor must not exceed one")
    for key, allowed in (("forms", {"steady", "transient"}),):
        if not c[key] or len(set(c[key])) != len(c[key]) or not set(c[key]) <= allowed:
            raise ValueError("Choose distinct steady or transient forms")
    bounds = np.asarray(c["bounds"], dtype=float)
    if (
        bounds.ndim != 1
        or not len(bounds)
        or not np.isfinite(bounds).all()
        or np.any(bounds <= 0)
        or np.any(np.diff(bounds) <= 0)
    ):
        raise ValueError("Bounds must be finite, positive and strictly increasing")
    if c["action"] not in {"solve", "plot"}:
        raise ValueError("Choose solve or plot")
    return c


def solve_case(problem, desired, bound, c, reference=None):
    """Use the same PDAS start, final criteria and original residual checks."""
    adapter = None
    if reference is not None:
        adapter = StudySolver(
            "reference",
            reference=reference,
            rank=c["rank"],
            rtol=c["rtol"],
            cg_factor=c["cg_factor"],
            maxiter=c["inner_cap"],
            refresh=c["residual_refresh"],
            residual_policy="refine",
        )
    state = np.zeros(problem.size)
    inner = []

    def solve(B, b, indices):
        if adapter is None:
            x = spsolve(B, b)
            rho = independent_residual(B, x, b)
            result = LinearResult(x, 1, rho, "converged" if rho <= c["rtol"] else "residual_failed")
        else:
            result, _ = adapter.solve(B, b, indices, state[indices].copy())
        # Evaluate the equation independently of the solver's reported residual.
        rho = independent_residual(B, result.x, b)
        inner.append(
            {
                "status": result.status,
                "original_residual": rho,
                "reported_residual": result.residual,
                "iterations": result.iterations,
                "rank": result.rank,
                "coarse_condition": result.coarse_condition,
                "fallback_reason": result.fallback_reason,
            }
        )
        if not np.isfinite(rho) or rho > c["rtol"]:
            result.status = "residual_failed"
        if result.status == "converged":
            state[:] = bound
            state[indices] = result.x
        return result

    try:
        result = pdas(
            problem.H,
            problem.load(desired),
            bound,
            initial_active=np.ones(problem.size, dtype=bool),
            tolerance=c["kkt_tolerance"],
            maxiter=c["outer_cap"],
            linear_solver=solve,
        )
    finally:
        if adapter is not None:
            adapter.close()
    # These checks use the recovered full state, including active variables.
    kkt = kkt_metrics(problem.H, problem.load(desired), bound, result["y"], result["multiplier"])
    control, adjoint = problem.recover(result["y"])
    forward = relative_norm(problem.forward(control) - result["y"], result["y"])
    rho = max((row["original_residual"] for row in inner), default=0.0)
    success = (
        result["status"] == "converged"
        and np.isfinite(rho)
        and rho <= c["rtol"]
        and all(np.isfinite(v) and v <= c["kkt_tolerance"] for v in kkt.values())
        and np.isfinite(forward)
        and forward <= c["forward_tolerance"]
    )
    w = problem.weights
    metrics = {
        "success": bool(success),
        "status": result["status"],
        "kkt": kkt,
        "maximum_original_residual": rho,
        "forward_relative_error": forward,
        "weighted_tracking_rms": float(np.sqrt(w @ (result["y"] - desired) ** 2 / w.sum())),
        "weighted_control_rms": float(np.sqrt(w @ control**2 / w.sum())),
        "objective": problem.objective(result["y"], desired),
        "state_minimum": float(result["y"].min()),
        "state_maximum": float(result["y"].max()),
        "maximum_bound_violation": float(np.maximum(result["y"] - bound, 0).max()),
        "active_fraction": float(result["active"].mean()),
        "final_active_fraction": float(
            result["active"].reshape(-1, problem.spatial_size)[-1].mean()
        ),
        "outer_iterations": result["iterations"],
        "history": result["history"],
        "inner": inner,
    }
    return metrics, {
        "state": result["y"],
        "active": result["active"],
        "multiplier": result["multiplier"],
        "control": control,
        "adjoint": adjoint,
    }


def run(c, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": "bore-temperature-bounds-v1",
        "settings": c,
        "scope": "Separate CPU field illustrations; no performance comparison",
        "units": "Dimensionless temperatures, control and time of the bore-in-block model",
        "environment": environment(),
        "cases": [],
        "forms": {},
        "success": False,
    }
    write_report(output / "report.json", report)
    with threadpool_limits(c["threads"]):
        for form in c["forms"]:
            showcase, problem = build_model(model_settings(c, form))
            desired = desired_temperature(problem, c["query"], c["query_population"])
            reference = build_mesh_reference(
                problem,
                showcase.coarse_assembly,
                showcase.prolongation,
                c["rank"],
                "mode_dependent",
                "scaled_schur",
                "jacobi",
            )
            mesh = problem.assembly.mesh
            meshfile = output / f"{form}-problem.npz"
            write_fields(
                meshfile,
                nodes=mesh.nodes,
                cells=mesh.cells,
                free=mesh.free,
                materials=mesh.materials,
                desired=desired,
                weights=problem.weights,
                initial=problem.initial,
                steps=problem.steps,
            )
            report["forms"][form] = {
                "dimension": problem.size,
                "spatial_dimension": problem.spatial_size,
                "target_sha256": digest(desired),
                "mesh_inputs": showcase.preparation["input_sha256"],
                "reference": reference.description,
                "reference_storage": reference.storage(),
                "problem_file": meshfile.name,
                "problem_sha256": hashlib.sha256(meshfile.read_bytes()).hexdigest(),
            }
            for index, bound in enumerate(c["bounds"]):
                row = {
                    "form": form,
                    "bound": bound,
                    "target_sha256": digest(desired),
                    "success": False,
                }
                for method in ("reference", "direct"):
                    try:
                        metrics, fields = solve_case(
                            problem, desired, bound, c, reference if method == "reference" else None
                        )
                        filename = output / f"{form}-bound-{index}-{method}.npz"
                        write_fields(filename, **fields)
                        row[method] = {
                            **metrics,
                            "fields": filename.name,
                            "fields_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                        }
                    except (RuntimeError, ValueError, FloatingPointError) as error:
                        row[method] = {
                            "success": False,
                            "status": "exception",
                            "exception_type": type(error).__name__,
                        }
                    print(form, bound, method, row[method]["status"], flush=True)
                if all(row[m]["success"] for m in ("reference", "direct")):
                    with np.load(output / row["reference"]["fields"], allow_pickle=False) as a:
                        with np.load(output / row["direct"]["fields"], allow_pickle=False) as b:
                            difference = relative_norm(a["state"] - b["state"], b["state"])
                    row["direct_state_relative_difference"] = difference
                    row["success"] = bool(
                        np.isfinite(difference) and difference <= c["state_comparison_tolerance"]
                    )
                report["cases"].append(row)
                write_report(output / "report.json", report)
    report["success"] = all(row["success"] for row in report["cases"])
    write_report(output / "report.json", report)
    return report


def load_results(directory):
    """Verify the complete population and file hashes before plotting a result."""
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    c = settings(OmegaConf.create(report["settings"]))
    expected = [(form, bound) for form in c["forms"] for bound in c["bounds"]]
    actual = [(row["form"], row["bound"]) for row in report["cases"]]
    if report["protocol"] != "bore-temperature-bounds-v1" or actual != expected:
        raise ValueError("Incomplete, reordered or duplicated constraint population")
    if not report["success"] or not all(row["success"] for row in report["cases"]):
        raise ValueError("The field figure requires all declared cases to pass verification")
    models = {}
    for form in c["forms"]:
        info = report["forms"][form]
        path = directory / info["problem_file"]
        if (
            path.parent != directory
            or hashlib.sha256(path.read_bytes()).hexdigest() != info["problem_sha256"]
        ):
            raise ValueError("Changed problem fields")
        with np.load(path, allow_pickle=False) as data:
            if digest(data["desired"]) != info["target_sha256"]:
                raise ValueError("Changed desired temperature")
            showcase, problem = build_model(model_settings(c, form))
            if (
                showcase.preparation["input_sha256"] != info["mesh_inputs"]
                or digest(desired_temperature(problem, c["query"], c["query_population"]))
                != info["target_sha256"]
                or not np.array_equal(problem.weights, data["weights"])
            ):
                raise ValueError("The model differs from the saved problem")
            models[form] = (problem, data["desired"].copy())
    for row in report["cases"]:
        if row["target_sha256"] != report["forms"][row["form"]]["target_sha256"]:
            raise ValueError("The desired temperature changed between bounds")
        for method in ("reference", "direct"):
            record = row[method]
            path = directory / record["fields"]
            if (
                path.parent != directory
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["fields_sha256"]
            ):
                raise ValueError("Changed optimized fields")
            residual = np.asarray(record["maximum_original_residual"], dtype=float)
            kkts = np.asarray(list(record["kkt"].values()), dtype=float)
            if (
                not record["success"]
                or record["status"] != "converged"
                or not np.isfinite(residual)
                or residual > c["rtol"]
                or kkts.shape != (5,)
                or not np.isfinite(kkts).all()
                or kkts.max() > c["kkt_tolerance"]
            ):
                raise ValueError("A result fails its original-residual or KKT criterion")
            with np.load(path, allow_pickle=False) as data:
                y = data["state"]
                if (
                    not np.isfinite(y).all()
                    or len(y) != report["forms"][row["form"]]["dimension"]
                    or float(np.maximum(y - row["bound"], 0).max()) > c["kkt_tolerance"]
                ):
                    raise ValueError("The plotted temperature fails the declared bound")
                problem, desired = models[row["form"]]
                kkt = kkt_metrics(
                    problem.H, problem.load(desired), row["bound"], y, data["multiplier"]
                )
                if (
                    not all(np.isfinite(v) and v <= c["kkt_tolerance"] for v in kkt.values())
                    or relative_norm(problem.recover(y)[0] - data["control"], data["control"])
                    > 1e-12
                ):
                    raise ValueError(
                        "The saved fields fail recomputed optimality or control recovery"
                    )
                computed = {
                    "weighted_tracking_rms": float(
                        np.sqrt(problem.weights @ (y - desired) ** 2 / problem.weights.sum())
                    ),
                    "objective": problem.objective(y, desired),
                    "state_minimum": float(y.min()),
                    "state_maximum": float(y.max()),
                    "active_fraction": float(data["active"].mean()),
                }
                if any(
                    not np.isclose(value, record[key], rtol=1e-12, atol=1e-14)
                    for key, value in computed.items()
                ):
                    raise ValueError("The summary differs from the plotted fields")
        with np.load(directory / row["reference"]["fields"], allow_pickle=False) as a:
            with np.load(directory / row["direct"]["fields"], allow_pickle=False) as b:
                error = relative_norm(a["state"] - b["state"], b["state"])
        if (
            not np.isfinite(error)
            or error > c["state_comparison_tolerance"]
            or not np.isclose(
                error, row["direct_state_relative_difference"], rtol=1e-12, atol=1e-15
            )
        ):
            raise ValueError("The direct and reference states differ")
    return report


def plot(directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.lines import Line2D

    directory, output = Path(directory), Path(output)
    report = load_results(directory)
    output.mkdir(parents=True, exist_ok=False)
    c = report["settings"]
    panels = []
    for row in report["cases"]:
        with np.load(
            directory / report["forms"][row["form"]]["problem_file"], allow_pickle=False
        ) as data:
            nodes, cells, free = (data[k] for k in ("nodes", "cells", "free"))
        with np.load(directory / row["reference"]["fields"], allow_pickle=False) as data:
            values = np.zeros((len(nodes), 2))
            values[free, 0] = data["state"].reshape(-1, len(free))[-1]
            values[free, 1] = data["active"].reshape(-1, len(free))[-1]
        xy, triangles, sampled, _ = tetrahedral_slice(nodes, cells, values, axis=2, coordinate=0.5)
        panels.append((row, mtri.Triangulation(xy[:, 0], xy[:, 1], triangles), sampled))
    lower = min(0.0, *(row["reference"]["state_minimum"] for row in report["cases"]))
    upper = max(row["reference"]["state_maximum"] for row in report["cases"])
    fig, axes = plt.subplots(
        len(c["forms"]),
        len(c["bounds"]),
        squeeze=False,
        figsize=(2.8 * len(c["bounds"]), 2.7 * len(c["forms"])),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    for index, (row, tri, values) in enumerate(panels):
        i, j = divmod(index, len(c["bounds"]))
        ax = axes[i, j]
        image = ax.tripcolor(
            tri,
            values[:, 0],
            shading="gouraud",
            cmap="inferno",
            vmin=lower,
            vmax=upper,
            rasterized=True,
        )
        if values[:, 1].min() < 0.5 < values[:, 1].max():
            ax.tricontour(tri, values[:, 1], levels=[0.5], colors="white", linewidths=0.9)
        label = chr(ord("a") + index)
        ax.set_title(f"({label}) $\\psi={row['bound']:g}$", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        if i == len(c["forms"]) - 1:
            ax.set_xlabel("$x_1/L$")
        if j == 0:
            form = "Steady" if row["form"] == "steady" else f"Final time $t={c['horizon']:g}$"
            ax.set_ylabel(form + "\n$x_2/L$")
    fig.colorbar(image, ax=axes, label="Optimized temperature (dimensionless)", shrink=0.86)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="0.4",
                linewidth=1,
                label="White contour: boundary of the interpolated active set",
            )
        ],
        loc="outside lower center",
        frameon=False,
        fontsize=9,
    )
    fig.savefig(
        output / "optimized_temperatures.pdf",
        dpi=200,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return report


@hydra.main(version_base=None, config_path="conf", config_name="temperature_bounds")
def main(config: DictConfig):
    c = settings(config)
    if c["action"] == "plot":
        if c["data"] is None:
            raise ValueError("Plotting requires the saved data directory")
        plot(c["data"], c["output"])
    else:
        report = run(c, c["output"])
        if not report["success"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
