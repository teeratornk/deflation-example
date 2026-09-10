"""Portable provenance and output serialization; no solver or CLI dependencies."""

import hashlib
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import numpy as np
import scipy
from threadpoolctl import threadpool_info


def environment():
    """Record the installed source, even outside Git or after wheel installation."""
    from . import __version__
    source = Path(__file__).resolve().parent
    head, source_tree_clean = None, None
    candidate = source.parent.parent
    if (candidate / ".git").exists():
        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
            source_tree_clean = not bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--", "src"],
                    cwd=candidate,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                ).strip()
            )
        except (OSError, subprocess.SubprocessError):
            pass
    fields = {
        "user_api",
        "internal_api",
        "num_threads",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
    }
    blas = [{k: v for k, v in item.items() if k in fields} for item in threadpool_info()]
    try:
        package_version = importlib.metadata.version("deflation-example")
    except importlib.metadata.PackageNotFoundError:
        package_version = __version__
    cpu_model = None
    if platform.system() == "Linux":
        try:
            cpu_model = next(
                (
                    line.split(":", 1)[1].strip()[:160]
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name") and ":" in line
                ),
                None,
            )
        except OSError:
            pass
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "package": package_version,
        "source_package": __version__,
        "git_head": head,
        "source_tree_clean": source_tree_clean,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))
        },
        "blas": blas,
    }


def json_safe(value):
    """Keep failure records valid JSON without disguising nonfinite quantities."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


@contextmanager
def atomic_output(path, binary=False):
    """Replace a file owned by this run only after its writer succeeds."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb" if binary else "w",
            encoding=None if binary else "utf-8",
            newline=None if binary else "",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_report(path, report):
    """Serialization or write failures leave the previous JSON record intact."""
    content = json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n"
    with atomic_output(path) as stream:
        stream.write(content)


def write_fields(path, **fields):
    with atomic_output(path, binary=True) as stream:
        np.savez_compressed(stream, **fields)


def summarize_pdas(result, seconds):
    return {
        "status": result["status"],
        "kkt": result["kkt"],
        "objective_reduced": result["objective"],
        "outer_iterations": result["iterations"],
        "inner_iterations": sum(row["linear_iterations"] for row in result["history"]),
        "seconds": seconds,
        "history": result["history"],
    }


def write_csv(output, report):
    import csv

    with atomic_output(output / "kernels.csv") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "theta",
                "method",
                "status",
                "iterations",
                "residual",
                "rank",
                "total_seconds",
                "fallback_reason",
            ],
        )
        writer.writeheader()
        for row in report["cases"]:
            for method, values in row["kernels"].items():
                writer.writerow(
                    {
                        "theta": row["theta"],
                        "method": method,
                        **{k: values[k] for k in writer.fieldnames if k not in {"theta", "method"}},
                    }
                )
