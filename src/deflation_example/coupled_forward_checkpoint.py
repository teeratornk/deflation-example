"""Atomic, hash-bound snapshots for independently verified forward trajectories.

The JSON pointer is committed last. A crash between field and JSON writes leaves
an unreferenced field archive rather than an apparently complete checkpoint.
Each output directory belongs to one run; resumed runs use a different directory.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .reporting import file_sha256, write_arrays, write_report
from .validation import integer


def protocol_digest(protocol):
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def save_snapshot(directory, name, result, details, protocol):
    """Save the state and its own verification; never replace an earlier attempt."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Snapshot name must be a plain filename stem")
    archive, record = directory / f"{name}.npz", directory / f"{name}.json"
    if archive.exists() or record.exists():
        raise FileExistsError(record)
    arrays = {
        "state": result.state,
        "velocity": result.flow.velocity,
        "pressure": result.flow.pressure,
    }
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Cannot checkpoint nonfinite forward fields")
    write_arrays(archive, **arrays)
    document = {
        "schema": "coupled-forward-snapshot-v1",
        "protocol_sha256": protocol_digest(protocol),
        "fields": archive.name,
        "fields_sha256": file_sha256(archive),
        "status": result.status,
        "seconds": result.seconds,
        "details": details,
    }
    write_report(record, document)
    return {"record": record.name, "record_sha256": file_sha256(record)}


def load_snapshot(directory, entry, protocol):
    directory = Path(directory)
    for name in (entry["record"],):
        if Path(name).name != name:
            raise ValueError("Snapshot records must be local filenames")
    record = directory / entry["record"]
    if file_sha256(record) != entry["record_sha256"]:
        raise ValueError("Forward checkpoint record checksum differs")
    document = json.loads(record.read_text())
    if document.get("schema") != "coupled-forward-snapshot-v1":
        raise ValueError("Unknown forward checkpoint format")
    if document["protocol_sha256"] != protocol_digest(protocol):
        raise ValueError("Forward checkpoint protocol differs")
    name = document["fields"]
    if Path(name).name != name:
        raise ValueError("Snapshot fields must be a local filename")
    archive = directory / name
    if file_sha256(archive) != document["fields_sha256"]:
        raise ValueError("Forward checkpoint field checksum differs")
    with np.load(archive, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in ("state", "velocity", "pressure")}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Forward checkpoint contains nonfinite fields")
    return document, arrays


def save_step(directory, slab, result, details, protocol, entries):
    """Publish one completed attempt after both its fields and details are durable."""
    slab = integer(slab, "Checkpoint step", 0)
    if slab != len(entries):
        raise ValueError("Forward steps must be persisted in order")
    entry = save_snapshot(directory, f"step-{slab:05d}", result, details, protocol)
    updated = [*entries, entry]
    write_report(
        Path(directory) / "index.json",
        {"schema": "coupled-forward-index-v1", "protocol": protocol, "steps": updated},
    )
    return updated


def load_steps(directory, protocol):
    """Read every committed step; a failed last step is returned for inspection."""
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if index.get("schema") != "coupled-forward-index-v1" or index["protocol"] != protocol:
        raise ValueError("Forward restart requires the identical source and numerical protocol")
    steps = []
    for slab, entry in enumerate(index["steps"]):
        document, arrays = load_snapshot(directory, entry, protocol)
        if document["details"]["slab_zero_based"] != slab:
            raise ValueError("Forward checkpoint time order differs")
        if slab and steps[-1][0]["status"] != "converged":
            raise ValueError("Forward checkpoint continues after an unverified step")
        steps.append((document, arrays))
    return steps
