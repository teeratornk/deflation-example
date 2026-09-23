"""Checksum-bound inactive systems from a diagnostic optimization trajectory.

Capture adds I/O and is never a performance measurement. Operators are rebuilt
from the saved temperatures, flow guesses, damping, and secants. Every attempted
inner solve is retained, including failures and fixed-mask correction equations.
"""

import json
from pathlib import Path

import numpy as np

from .reporting import environment, file_sha256, write_arrays, write_report


class CoupledTrace:
    def __init__(self, directory, configuration, baseline_sha256):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        # Machine-local input locations belong in the private invocation, not the archive.
        config = {
            k: v
            for k, v in configuration.items()
            if k
            not in {
                "output",
                "baseline_directory",
                "reference_baseline_directory",
                "initial_state_snapshot",
                "initial_state_assessment",
                "initial_trajectory_directory",
                "initial_control_directory",
            }
        }
        self.record = {
            "schema": "coupled-inactive-trace-v1",
            "status": "capturing",
            "environment": environment(),
            "configuration": config,
            "baseline_sha256": baseline_sha256,
            "quadratics": [],
            "systems": [],
            "scope": "Diagnostic capture with I/O; not a complete optimization timing.",
        }
        self.current = None
        self.pending = None
        self.save()

    def save(self):
        write_report(self.directory / "manifest.json", self.record)

    def begin_quadratic(
        self,
        iteration,
        attempt,
        evaluation,
        desired,
        gradient,
        diagonal,
        damping,
        secants,
        lower,
        upper,
    ):
        if self.pending is not None:
            raise RuntimeError("An unfinished inner solve must retain its pending status")
        name = f"quadratic-{len(self.record['quadratics']):04d}.npz"
        n = len(evaluation.state)
        write_arrays(
            self.directory / name,
            state=evaluation.state,
            desired=desired,
            gradient=gradient,
            diagonal=diagonal,
            lower=lower,
            upper=upper,
            velocity=np.stack([f.velocity for f in evaluation.flows]),
            pressure=np.stack([f.pressure for f in evaluation.flows]),
            secant_steps=np.array([s for s, _ in secants]).reshape(-1, n),
            secant_gradients=np.array([g for _, g in secants]).reshape(-1, n),
        )
        self.current = len(self.record["quadratics"])
        self.record["quadratics"].append(
            {
                "file": name,
                "sha256": file_sha256(self.directory / name),
                "iteration": int(iteration),
                "attempt": int(attempt),
                "damping": float(damping),
                "partition": "selection" if iteration <= 2 else "held_out",
            }
        )
        self.save()

    def before_solve(self, step, indices, rhs, initial, correction):
        if self.current is None or self.pending is not None:
            raise RuntimeError("Begin a quadratic and complete the preceding solve first")
        name = f"system-{len(self.record['systems']):04d}.npz"
        write_arrays(self.directory / name, indices=indices, rhs=rhs, initial=initial)
        self.pending = len(self.record["systems"])
        self.record["systems"].append(
            {
                "file": name,
                "sha256": file_sha256(self.directory / name),
                "quadratic": self.current,
                "pdas_step": int(step),
                "equation": "correction" if correction else "state",
                "status": "pending",
            }
        )
        self.save()

    def after_solve(self, result, timing):
        if self.pending is None:
            raise RuntimeError("No captured solve is pending")
        row = self.record["systems"][self.pending]
        name = row["file"].replace(".npz", "-solution.npz")
        write_arrays(self.directory / name, x=result.x)
        row.update(
            status=result.status,
            iterations=result.iterations,
            original_residual=result.residual,
            deployed_rank=result.rank,
            solution_file=name,
            solution_sha256=file_sha256(self.directory / name),
            timing=timing,
        )
        self.pending = None
        self.save()

    def finish(self):
        if self.pending is not None:
            raise RuntimeError("An incomplete solve cannot produce a complete trace")
        self.record["status"] = "complete"
        self.save()


def read_arrays(directory, filename, digest):
    root = Path(directory).resolve()
    path = (root / filename).resolve()
    if path.parent != root or file_sha256(path) != digest:
        raise ValueError("Trace file location or checksum differs")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Trace arrays must be finite")
    return arrays


def read_manifest(directory):
    record = json.loads((Path(directory) / "manifest.json").read_text())
    if record.get("schema") != "coupled-inactive-trace-v1":
        raise ValueError("Unknown inactive-system trace")
    return record
