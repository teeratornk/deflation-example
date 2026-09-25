"""Atomic, source-bound recovery of complete optimizer and solver state."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .reporting import file_sha256, json_safe, write_arrays, write_report


def identity(configuration, source):
    encoded = json.dumps({"configuration": configuration, "source": source}, sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


class RecoveryStore:
    """Two alternating archives with a manifest committed after the new archive.

    An interrupted write leaves the previously referenced slot untouched. The
    manifest binds every scalar and array to its configuration and source.
    Checkpoints contain data only; deserialization never uses pickle.
    """

    def __init__(self, directory, fingerprint):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.manifest = self.directory / "latest.json"

    def save(self, payload):
        old = json.loads(self.manifest.read_text()) if self.manifest.exists() else None
        if old is not None and old["identity"] != self.fingerprint:
            raise ValueError("Checkpoint configuration or source differs")
        slot = 1 - old["slot"] if old is not None else 0
        arrays = {}

        def pack(value):
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject or not np.isfinite(value).all():
                    raise ValueError("Checkpoint arrays must be finite numeric data")
                key = f"array_{len(arrays)}"
                arrays[key] = value
                return {"__array__": key}
            if isinstance(value, dict):
                if "__array__" in value:
                    raise ValueError("Reserved checkpoint key")
                return {k: pack(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [pack(v) for v in value]
            return json_safe(value)

        metadata = pack(payload)
        arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
        path = self.directory / f"state-{slot}.npz"
        write_arrays(path, **arrays)
        write_report(
            self.manifest,
            {
                "schema": "coupled-recovery-v1",
                "identity": self.fingerprint,
                "slot": slot,
                "sha256": file_sha256(path),
            },
        )

    def load(self):
        if not self.manifest.exists():
            return None
        meta = json.loads(self.manifest.read_text())
        if (
            meta.get("schema") != "coupled-recovery-v1"
            or meta.get("identity") != self.fingerprint
            or meta.get("slot") not in (0, 1)
        ):
            raise ValueError("Checkpoint configuration, source or format differs")
        path = self.directory / f"state-{meta['slot']}.npz"
        if file_sha256(path) != meta["sha256"]:
            raise ValueError("Checkpoint checksum differs")
        with np.load(path, allow_pickle=False) as archive:

            def unpack(value):
                if isinstance(value, dict):
                    if set(value) == {"__array__"}:
                        a = archive[value["__array__"]].copy()
                        if a.dtype.hasobject or not np.isfinite(a).all():
                            raise ValueError("Invalid checkpoint array")
                        return a
                    return {k: unpack(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [unpack(v) for v in value]
                return value

            return unpack(json.loads(str(archive["metadata"])))
