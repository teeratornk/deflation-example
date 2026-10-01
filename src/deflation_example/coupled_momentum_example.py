"""Portable local and full-horizon momentum checks for captured temperatures."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .axisymmetric_flow import FlowResult
from .coupled_momentum_diagnostic import history_path, integrate
from .coupled_optimize import load_problem
from .coupled_projected_example import verification_problem
from .coupled_recovery import identity
from .coupled_trial_capture import read_trial
from .reporting import environment, file_sha256, write_arrays, write_report

NUMERICAL_MODULES = (
    "axisymmetric_flow.py",
    "coupled_flow_solve.py",
    "coupled_control.py",
    "coupled_optimize.py",
    "coupled_pilot.py",
    "meshes.py",
    "oil_properties.py",
    "assess_transformer.py",
)


def load_capture(run, position, trial, baseline=None):
    record = json.loads((run / "record.json").read_text())
    cfg, env = record["configuration"], environment()
    if record["status"] == "running" or not cfg["transient"]:
        raise ValueError("Choose a terminated transient calculation")
    if not 0 <= position < len(cfg["queries"]):
        raise ValueError("Target position is out of range")
    directory = run / "trial-capture" / f"target-{position:02d}"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["identity"] != identity(cfg, record["environment"]["source_sha256"]):
        raise ValueError("Captured configuration or source differs from its record")
    for name in NUMERICAL_MODULES:
        if env["source_sha256"][name] != record["environment"]["source_sha256"][name]:
            raise ValueError("The captured numerical module changed: " + name)
    event, fields = read_trial(directory, trial)
    if baseline is not None:
        cfg = {**cfg, "baseline_directory": str(baseline.resolve())}
    problem, details = load_problem(cfg)
    if details["baseline_sha256"] != record["baseline_sha256"]:
        raise ValueError("The physical baseline differs from the captured problem")
    return (
        problem,
        fields,
        {
            "kind": "capture",
            "record_sha256": file_sha256(run / "record.json"),
            "manifest_sha256": file_sha256(directory / "manifest.json"),
            "trial": event,
            "numerical_source": record["environment"]["git_head"],
            "baseline_sha256": record["baseline_sha256"],
        },
    )


def small_example():
    problem = verification_problem()
    state = np.linspace(0.04, 0.12, problem.size)
    evaluation = problem.evaluate(state)
    fields = {
        "base": {
            "state": state,
            "velocity": np.stack([r.velocity for r in evaluation.flows]),
            "pressure": np.stack([r.pressure for r in evaluation.flows]),
        },
        "candidate": {"state": state + np.linspace(0.001, 0.003, problem.size)},
    }
    return (
        problem,
        fields,
        {
            "kind": "self_contained_verification",
            "scope": "Small dimensionless verification coefficients; no physical performance claim.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--example", action="store_true")
    data.add_argument("--run", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--position", type=int, default=1)
    parser.add_argument("--trial", type=int)
    parser.add_argument("--mode", choices=("local", "path", "trajectory"), default="trajectory")
    parser.add_argument("--trajectory", choices=("retained", "rejected"), default="rejected")
    parser.add_argument("--slab", type=int, default=1)
    parser.add_argument("--subdivision", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--budget-seconds", type=float, default=3600)
    parser.add_argument("--prefix-budget-seconds", type=float, default=600)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run is not None and args.trial is None:
        parser.error("Captured inputs require --trial")
    if args.threads < 1 or any(
        not np.isfinite(v) or v <= 0 for v in (args.budget_seconds, args.prefix_budget_seconds)
    ):
        parser.error("Use positive threads and a finite positive time budget")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "coupled-momentum-diagnostic-v1",
        "status": "initializing",
        "environment": environment(),
        "mode": args.mode,
        "trajectory": args.trajectory,
        "subdivision": args.subdivision,
        "slab_zero_based": args.slab,
        "budget_seconds": args.budget_seconds,
        "prefix_budget_seconds": args.prefix_budget_seconds,
        "threads": args.threads,
    }
    write_report(args.output / "record.json", report)
    try:
        with threadpool_limits(args.threads):
            problem, fields, source = (
                small_example()
                if args.example
                else load_capture(args.run, args.position, args.trial, args.baseline)
            )
            report.update(
                source=source,
                original_slabs=problem.slabs,
                horizon_s=float(problem.physical_steps.sum()),
            )
            state = (
                fields["base"]["state"]
                if args.trajectory == "retained"
                else fields["candidate"]["state"]
            )
            previous = None
            if args.mode in {"local", "path"}:
                if not 0 < args.slab < problem.slabs:
                    raise ValueError("Local modes require an interior time slab")
                if args.mode == "path" and args.trajectory != "rejected":
                    raise ValueError("A history path is defined toward the rejected trajectory")
                if args.trajectory == "retained":
                    previous = FlowResult(
                        fields["base"]["velocity"][args.slab - 1],
                        fields["base"]["pressure"][args.slab - 1],
                        "saved",
                        [],
                    )
                else:
                    prefix, arrays = integrate(
                        problem,
                        state,
                        fields["base"],
                        stop_slab=args.slab,
                        budget_seconds=args.prefix_budget_seconds,
                    )
                    write_arrays(args.output / "prefix.npz", **arrays)
                    report.update(
                        prefix=prefix, prefix_sha256=file_sha256(args.output / "prefix.npz")
                    )
                    if not prefix["verified"]:
                        report.update(status="prefix_failed", verified=False)
                        write_report(args.output / "record.json", report)
                        raise SystemExit(2)
                    previous = FlowResult(
                        arrays["velocity"][-1], arrays["pressure"][-1], "converged", []
                    )

            def callback(rows):
                write_report(args.output / "progress.json", {"steps": rows})

            if args.mode == "path":
                result, arrays = history_path(
                    problem,
                    fields["base"],
                    state,
                    args.slab,
                    previous.velocity,
                    [0, 0.5, 0.75, 0.8, 0.82, 0.824, 0.825, 0.83, 0.85, 1],
                    budget_seconds=args.budget_seconds,
                    callback=callback,
                )
            else:
                options = (
                    {"start_slab": args.slab, "stop_slab": args.slab + 1, "previous": previous}
                    if args.mode == "local"
                    else {}
                )
                result, arrays = integrate(
                    problem,
                    state,
                    fields["base"],
                    args.subdivision,
                    budget_seconds=args.budget_seconds,
                    callback=callback,
                    **options,
                )
            write_arrays(args.output / "fields.npz", **arrays)
            report.update(result, fields_sha256=file_sha256(args.output / "fields.npz"))
            write_report(args.output / "record.json", report)
    except Exception as error:
        report.update(status="diagnostic_error", error_type=type(error).__name__, error=str(error))
        write_report(args.output / "record.json", report)
        raise
    if args.mode != "path" and not report["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
