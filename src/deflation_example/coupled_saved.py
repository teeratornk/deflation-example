"""Select one verified coupled source without changing its values."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .validation import integer


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_saved_solution(directory, method, position=None):
    directory = Path(directory)
    record = json.loads((directory / "record.json").read_text())
    cfg = dict(record["configuration"])
    if position is None:
        rows = [row for row in record.get("results", []) if row["method"] == method]
        if len(rows) != 1 or rows[0]["status"] != "converged":
            raise ValueError("Select a verified optimization result")
        field = directory / (method + "-fields.npz")
    else:
        position = integer(position, "Target position", 0)
        if cfg.get("method") != method:
            raise ValueError("The selected method differs from the saved sequence")
        rows = [row for row in record.get("cases", []) if row["position"] == position]
        if len(rows) != 1 or not rows[0].get("verified", False):
            raise ValueError("Select a verified target from the saved sequence")
        cfg.update(query=rows[0]["target"], upper_K=rows[0]["upper_K"])
        field = directory / f"target-{position:02d}.npz"
    with np.load(field, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in ("state", "control", "desired")}
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Saved temperature, target and source must be finite")
    if not arrays["state"].size or any(
        array.shape != arrays["state"].shape for array in arrays.values()
    ):
        raise ValueError("Saved temperature, target and source must have identical nonempty shapes")
    return record, cfg, arrays, file_digest(field)


def require_matching_baseline(optimization_record, baseline_record):
    if optimization_record["baseline_sha256"] != baseline_record["baseline_sha256"]:
        raise ValueError("The supplied baseline differs from the saved optimization")
