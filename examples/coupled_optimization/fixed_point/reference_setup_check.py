"""Reproduce the same-mesh launcher error and verify the corrected construction.

This construction-only diagnostic runs on a compute node. It changes no saved
control, optimization setting, or original result, and supplies no solve timing.
"""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from hydra import compose, initialize_config_module
import numpy as np
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from deflation_example.coupled_optimize import load_problem
from deflation_example.coupled_reference import configured_reference
from deflation_example.reporting import environment, file_sha256, write_report
import workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = workflow.read(workflow.HERE / "protocol.json")
    command, _ = workflow.optimization_command(
        SimpleNamespace(**vars(args), phase="optimize"),
        protocol,
        "newton",
        method="reference",
    )
    with initialize_config_module(config_module="deflation_example.conf", version_base=None):
        configuration = OmegaConf.to_container(
            compose(
                config_name="coupled_sequence",
                overrides=command[3:],
            ),
            resolve=True,
        )
    report = {
        "schema": "fixed-point-reference-setup-diagnostic-v1",
        "environment": environment(),
        "configuration": configuration,
        "protocol_sha256": file_sha256(workflow.HERE / "protocol.json"),
        "status": "running",
        "scope": "Construction-only diagnostic; not an optimization or speedup comparison.",
    }
    write_report(args.output / "record.json", report)
    with threadpool_limits(args.threads):
        problem, baseline = load_problem(configuration)
        old = {**configuration, "reference_baseline_directory": configuration["baseline_directory"]}
        try:
            configured_reference(problem, old, baseline)
        except ValueError as exc:
            report["original_exception"] = {"type": type(exc).__name__, "message": str(exc)}
            if str(exc) != "The reference mesh must be the immediate nested parent":
                raise
        else:
            raise AssertionError("The original setup did not reproduce the diagnosed error")
        start = time.perf_counter()
        basis = configured_reference(problem, configuration, baseline)
        report["diagnostic_construction_seconds"] = time.perf_counter() - start
        # Sample rows only; avoid deploying a full inactive basis in this check.
        indices = np.unique(np.linspace(0, problem.size - 1, min(4096, problem.size), dtype=int))
        values = basis.restrict(indices)
        if values.shape != (len(indices), configuration["rank"]) or not np.isfinite(values).all():
            raise AssertionError("The corrected reference has invalid dimensions or entries")
        report.update(
            status="verified",
            state_dofs=problem.size,
            requested_rank=configuration["rank"],
            reference_description=basis.description,
            storage=basis.storage(),
            sampled_rows=len(indices),
            sampled_numerical_rank=int(np.linalg.matrix_rank(values)),
        )
    write_report(args.output / "record.json", report)
    print(
        json.dumps({key: report[key] for key in ("status", "original_exception", "requested_rank")})
    )


if __name__ == "__main__":
    main()
