"""Complete CHT sequences with matched warm starts and partitioned costs."""

import hashlib
import json
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits

from .amgx import AmgxSession, amgx_cg, amgx_configuration
from .gpu import gpu_deflated_cg, require_cuda
from .problems import build_problem
from .reporting import environment, summarize_pdas, write_report
from .solvers import calibrate_bound, pdas
from .spectral import analytical_reference
from .timing import PHASES
from .validation import integer, positive_real


METHODS = ("deflated_gpu_qr", "amgx_persistent_resources")
WARM_STARTS = ("cold", "outer", "outer_inner")


def target_parameters(index, count):
    """Unequal Gaussian weights and a vertical translation break target symmetries."""
    integer(count, "Target count", 2)
    integer(index, "Target index")
    if index >= count:
        raise ValueError("Target index is outside the sequence")
    t = index / (count - 1)
    angle = 1.2 * t
    centers = np.array([[0.3, 0.3, 0.3], [0.3, 0.7, 0.3], [0.7, 0.3, 0.7], [0.7, 0.7, 0.7]])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centers[:, :2] = (centers[:, :2] - 0.5) @ rotation.T + 0.5
    centers[:, 2] += 0.035 * (2 * t - 1)
    return {
        "index": index,
        "t": t,
        "angle": angle,
        "sigma": 0.12,
        "centers": centers.tolist(),
        "weights": [1.0, 0.8 + 0.15 * t, 0.65, 0.45],
    }


def target_values(coordinates, parameters):
    return sum(
        weight
        * np.exp(
            -np.sum((coordinates - np.array(center)[:, None]) ** 2, axis=0)
            / (2 * parameters["sigma"] ** 2)
        )
        for weight, center in zip(parameters["weights"], parameters["centers"], strict=True)
    )


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def protocol(config):
    controls = (
        OmegaConf.to_container(config, resolve=True)
        if isinstance(config, DictConfig)
        else dict(config)
    )
    controls.pop("output", None)
    for key in (
        "targets",
        "rank",
        "repeats",
        "threads",
        "calibration_grid",
        "calibration_steps",
        "inner_cap",
        "outer_cap",
    ):
        integer(controls[key], key, 1)
    if controls["targets"] < 2 or controls["repeats"] % 2 != 1:
        raise ValueError("Use distinct targets and an odd number of repetitions")
    for n in controls["grids"]:
        integer(n, "Grid", 4)
        if controls["rank"] >= n**3:
            raise ValueError("Reference rank exceeds the full dimension")
    if len(set(controls["grids"])) != len(controls["grids"]) or not controls["grids"]:
        raise ValueError("Grids must be distinct and nonempty")
    if (
        not controls["warm_starts"]
        or len(set(controls["warm_starts"])) != len(controls["warm_starts"])
        or set(controls["warm_starts"]) - set(WARM_STARTS)
    ):
        raise ValueError("Unknown or duplicate warm-start policy")
    positive_real(controls["rtol"], "Inner tolerance")
    if not 0 < positive_real(controls["inner_stopping_factor"], "Inner stopping factor") <= 1:
        raise ValueError("Inner stopping factor must lie in (0, 1]")
    positive_real(controls["outer_tolerance"], "Outer tolerance")
    if not 0 < controls["calibration_activity"] < 1:
        raise ValueError("Calibration activity must lie strictly between zero and one")
    return {
        "protocol": "extended-cht-pdas-v2",
        "controls": controls,
        "target_parameters": [
            target_parameters(i, controls["targets"]) for i in range(controls["targets"])
        ],
        "alpha": 0.001,
        "conductivity_ratio": 100.0,
        "transport_amplitude": 50.0,
        "reference": "Laplacian tensor sine modes; eigenvalue then lexicographic ordering",
        "rank_policy": "fixed requested rank; GPU thin QR/small-factor SVD cutoff 1e-12; coarse limit 1e10",
        "residual_refresh": 1000,
        "amgx_configuration": amgx_configuration(
            controls["rtol"] * controls["inner_stopping_factor"], controls["inner_cap"], True
        ),
        "amgx_rhs_scaling": "original b and x0; per-solve ABSOLUTE threshold = internal relative tolerance times ||b||_2",
        "resource_policy": "base Config/Resources persist per sequence; new solver Config sets the RHS-dependent threshold; matrix, vectors, solver and hierarchy recreated per inner solve",
        "amendment": "Version 1 native convergence sometimes failed fresh residual acceptance. Version 2 preserves all targets and acceptance limits, leaves the AmgX RHS unscaled, and adds the same internal stopping margin to both solvers. Every Version 1 attempt is retained separately.",
        "warm_start_policy": {
            "cold": "empty active set for every target; zero initial guess for every inner solve",
            "outer": "previous converged target active set; zero initial guess for every inner solve",
            "outer_inner": "previous converged target active set; previous full state restricted to each inner system; update that state after each inner solve",
        },
        "failure_policy": "record a failed target and continue at the next target with empty history; numerical exceptions retained; no parameter adjustment",
        "timing_scope": "independently repeated complete sequences, assembly through final cleanup; calibration and runtime initialization/finalization separately measured",
        "calibration": "one n=calibration_grid bound at t=0.5, fixed as a physical scalar on every grid and target",
    }


def complete_sequence(n, bound, method, warm, specification, torch, api):
    controls = specification["controls"]
    iteration_rtol = controls["rtol"] * controls["inner_stopping_factor"]
    torch.cuda.synchronize()
    start = tick = time.perf_counter()
    problem = build_problem("cht", n)
    parts = {"assembly": time.perf_counter() - tick}
    session = basis = None
    cases, modes, previous = [], [], None
    error = None
    try:
        tick = time.perf_counter()
        if method == METHODS[0]:
            basis, modes = analytical_reference(n, 3, controls["rank"])
            parts["reference_construction"] = time.perf_counter() - tick
        elif method == METHODS[1]:
            session = AmgxSession(api, iteration_rtol, controls["inner_cap"], True).open()
            parts["persistent_resources"] = time.perf_counter() - tick
            modes = []
        else:
            raise ValueError("Unknown sequence method")
        if warm not in WARM_STARTS:
            raise ValueError("Unknown warm-start policy")
        for parameters in specification["target_parameters"]:
            tick = time.perf_counter()
            f = target_values(problem.coordinates, parameters)
            target_seconds = time.perf_counter() - tick
            initial_active = previous["active"] if previous is not None and warm != "cold" else None
            state = (
                previous["y"].copy()
                if previous is not None and warm == "outer_inner"
                else np.zeros(len(f))
            )
            inner = []

            def solve(B, b, I):
                callback_start = time.perf_counter()
                initial = state[I].copy() if warm == "outer_inner" else None
                restricted = basis[I] if basis is not None else None
                restriction_seconds = time.perf_counter() - callback_start
                if basis is not None:
                    result, metrics = gpu_deflated_cg(
                        B,
                        b,
                        restricted,
                        B.diagonal(),
                        x0=initial,
                        rtol=iteration_rtol,
                        acceptance_rtol=controls["rtol"],
                        maxiter=controls["inner_cap"],
                        basis_backend="gpu_qr",
                    )
                else:
                    result, metrics = amgx_cg(
                        B,
                        b,
                        api=api,
                        synchronize=torch.cuda.synchronize,
                        session=session,
                        rtol=iteration_rtol,
                        acceptance_rtol=controls["rtol"],
                        maxiter=controls["inner_cap"],
                        x0=initial,
                        rhs_relative=True,
                    )
                if warm == "outer_inner":
                    state[:] = bound
                    state[I] = result.x
                inner.append(
                    {
                        **metrics,
                        "basis_and_initial_restriction_seconds": restriction_seconds,
                        "n_inactive": len(I),
                        "iterations": result.iterations,
                        "rank": result.rank,
                        "original_residual": result.residual,
                        "status": result.status,
                        "coarse_condition": result.coarse_condition,
                        "fallback_reason": result.fallback_reason,
                        "initial_guess_sha256": digest(
                            np.zeros(len(I)) if initial is None else initial
                        ),
                    }
                )
                return result

            tick = time.perf_counter()
            try:
                outer = pdas(
                    problem.H,
                    f,
                    bound,
                    initial_active=initial_active,
                    tolerance=controls["outer_tolerance"],
                    maxiter=controls["outer_cap"],
                    linear_solver=solve,
                )
                case = {
                    **summarize_pdas(outer, time.perf_counter() - tick),
                    "pdas_timing": outer["timing"],
                    "initial_active_count": 0
                    if initial_active is None
                    else int(initial_active.sum()),
                    "final_active_count": int(outer["active"].sum()),
                    "final_active_sha256": digest(outer["active"]),
                    "solution_sha256": digest(outer["y"]),
                }
                previous = outer if outer["status"] == "converged" else None
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exception:
                case = {
                    "status": "exception",
                    "error_type": type(exception).__name__,
                    "seconds": time.perf_counter() - tick,
                    "outer_iterations": 0,
                    "inner_iterations": sum(i["iterations"] for i in inner),
                }
                previous = None
            case.update(
                index=parameters["index"],
                target_sha256=digest(f),
                target_seconds=target_seconds,
                inner=inner,
                cumulative_seconds=time.perf_counter() - start,
            )
            cases.append(case)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exception:
        error = type(exception).__name__
    finally:
        tick = time.perf_counter()
        if session is not None:
            session.close()
        basis = problem = previous = None
        torch.cuda.synchronize()
        parts["final_cleanup"] = time.perf_counter() - tick
    total = time.perf_counter() - start
    for key in PHASES:
        parts["kernel_" + key] = sum(
            i["components_seconds"][key] for c in cases for i in c["inner"]
        )
    parts["target_construction"] = sum(c["target_seconds"] for c in cases)
    parts["basis_and_initial_restriction"] = sum(
        i["basis_and_initial_restriction_seconds"] for c in cases for i in c["inner"]
    )
    for key in ("preparation", "restriction", "inner_verification", "kkt_and_update"):
        parts["outer_" + key] = sum(
            c.get("pdas_timing", {}).get("components_seconds", {}).get(key, 0.0) for c in cases
        )
    parts["remaining_host_work"] = total - sum(parts.values())
    if parts["remaining_host_work"] < -1e-9:
        raise ValueError("Sequence timing components overlap")
    return {
        "n": n,
        "method": method,
        "warm_start": warm,
        "bound": bound,
        "total_seconds": total,
        "components_seconds": parts,
        "cases": cases,
        "error_type": error,
        "mode_indices": modes,
        "outer_iterations": sum(c["outer_iterations"] for c in cases),
        "inner_iterations": sum(c["inner_iterations"] for c in cases),
        "success": error is None
        and len(cases) == controls["targets"]
        and all(c["status"] == "converged" for c in cases),
    }


def run(config):
    specification = protocol(config)
    controls = specification["controls"]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "sequences").mkdir()
    write_report(output / "protocol.json", specification)
    specification_hash = hashlib.sha256(
        json.dumps(specification, sort_keys=True).encode()
    ).hexdigest()
    report = {
        "protocol_sha256": specification_hash,
        "specification": specification,
        "environment": environment(),
        "sequences": [],
        "success": False,
    }
    write_report(output / "results.json", report)
    torch = require_cuda()
    import pyamgx

    torch.set_num_threads(controls["threads"])
    with threadpool_limits(limits=controls["threads"]):
        tick = time.perf_counter()
        problem = build_problem("cht", controls["calibration_grid"])
        parameters = target_parameters(1, 3)
        bound, fraction = calibrate_bound(
            problem.H,
            target_values(problem.coordinates, parameters),
            controls["calibration_activity"],
            steps=controls["calibration_steps"],
        )
        del problem
        report.update(
            bound=bound,
            calibrated_activity=fraction,
            calibration_seconds=time.perf_counter() - tick,
        )
        tick = time.perf_counter()
        torch.ones(1, device="cuda").sum().item()
        pyamgx.initialize()
        torch.cuda.synchronize()
        report["runtime_initialization_seconds"] = time.perf_counter() - tick
        report["environment"].update(
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()
        )
        try:
            for n in controls["grids"]:
                for repeat in range(controls["repeats"]):
                    order = [
                        (warm, method) for warm in controls["warm_starts"] for method in METHODS
                    ]
                    if repeat % 2:
                        order.reverse()
                    for warm, method in order:
                        sequence = complete_sequence(
                            n, bound, method, warm, specification, torch, pyamgx
                        )
                        sequence.update(repetition=repeat, protocol_sha256=specification_hash)
                        path = f"sequences/n{n}_{warm}_{method}_{repeat}.json"
                        write_report(output / path, sequence)
                        report["sequences"].append(
                            {
                                "path": path,
                                "sha256": hashlib.sha256((output / path).read_bytes()).hexdigest(),
                                **{
                                    k: sequence[k]
                                    for k in (
                                        "n",
                                        "method",
                                        "warm_start",
                                        "repetition",
                                        "total_seconds",
                                        "success",
                                    )
                                },
                            }
                        )
                        write_report(output / "results.json", report)
                        print(
                            f"n={n} {warm} {method} repeat={repeat}: {sequence['total_seconds']:.3f}s success={sequence['success']}",
                            flush=True,
                        )
        finally:
            tick = time.perf_counter()
            pyamgx.finalize()
            torch.cuda.synchronize()
            report["runtime_finalization_seconds"] = time.perf_counter() - tick
            expected = (
                len(controls["grids"])
                * controls["repeats"]
                * len(controls["warm_starts"])
                * len(METHODS)
            )
            report["complete"] = len(report["sequences"]) == expected
            report["success"] = report["complete"] and all(
                s["success"] for s in report["sequences"]
            )
            write_report(output / "results.json", report)
    return report


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_extended")
def main(config):
    if not run(config)["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
