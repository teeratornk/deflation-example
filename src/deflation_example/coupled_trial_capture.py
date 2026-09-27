"""Checksummed trial inputs and outcomes for coupled-flow failure reproduction."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .reporting import file_sha256, write_arrays, write_report


class TrialCapture:
    """Store one starting trajectory per outer step and each actual trial state.

    A started entry survives interrupted flow evaluation. No optimizer state is
    restored from these files; source-bound recovery remains a separate system.
    """

    def __init__(self, directory, fingerprint):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {"schema": "coupled-trials-v1", "identity": fingerprint, "trials": []}
        self.bases = {}
        self.pending = {}
        self.candidate_signatures = {}
        self.seconds = 0.0
        self._write()

    def _write(self):
        write_report(self.directory / "manifest.json", self.manifest)

    def _arrays(self, name, **arrays):
        path = self.directory / name
        if path.exists():
            raise FileExistsError("Preserve the existing trial archive")
        if any(np.asarray(a).dtype.kind not in "biufc" for a in arrays.values()):
            raise ValueError("Trial archives contain numeric arrays only")
        write_arrays(path, **arrays)
        return {"file": name, "sha256": file_sha256(path)}

    def __call__(self, event, evaluation, candidate, failed_flow):
        began = time.perf_counter()
        try:
            key = (event["iteration"], event["attempt"], event["trial"])
            if any(isinstance(k, bool) or not isinstance(k, int) or k < 0 for k in key):
                raise ValueError("Trial indices must be nonnegative integers")
            name = "step-{:04d}-attempt-{:03d}-trial-{:02d}".format(*key)
            array = np.ascontiguousarray(candidate)
            signature = (array.shape, str(array.dtype), hashlib.sha256(array).hexdigest())
            if event["phase"] == "started":
                if key in self.pending:
                    raise ValueError("Duplicate trial start")
                if key[0] not in self.bases:
                    self.bases[key[0]] = self._arrays(
                        f"step-{key[0]:04d}-base.npz",
                        state=evaluation.state,
                        velocity=np.stack([f.velocity for f in evaluation.flows]),
                        pressure=np.stack([f.pressure for f in evaluation.flows]),
                    )
                row = {
                    "start": dict(event),
                    "base": self.bases[key[0]],
                    "candidate": self._arrays(name + ".npz", state=candidate),
                    "outcome": None,
                }
                self.pending[key] = row
                self.candidate_signatures[key] = signature
                self.manifest["trials"].append(row)
            elif event["phase"] == "finished":
                if key not in self.pending or self.pending[key]["outcome"] is not None:
                    raise ValueError("Trial completion requires one unfinished start")
                row = self.pending[key]
                if (
                    signature != self.candidate_signatures[key]
                    or event["step"] != row["start"]["step"]
                ):
                    raise ValueError("The completed trial must match its recorded candidate")
                if failed_flow is not None:
                    row["failed_flow"] = self._arrays(
                        name + "-failed-flow.npz",
                        velocity=failed_flow.velocity,
                        pressure=failed_flow.pressure,
                    )
                row["outcome"] = dict(event)
            else:
                raise ValueError("Unknown trial-capture phase")
            self._write()
        finally:
            self.seconds += time.perf_counter() - began


def read_trial(directory, index):
    """Read exactly the recorded input; permit unfinished trials, never pickle."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "coupled-trials-v1":
        raise ValueError("Unexpected trial manifest")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(manifest["trials"])
    ):
        raise ValueError("Trial index is out of range")
    row = manifest["trials"][index]
    arrays = {}
    for kind in ("base", "candidate", "failed_flow"):
        if kind not in row:
            continue
        item = row[kind]
        path = (directory / item["file"]).resolve()
        if not path.is_relative_to(directory) or Path(item["file"]).is_absolute():
            raise ValueError("Trial archive path leaves its directory")
        if file_sha256(path) != item["sha256"]:
            raise ValueError("Trial archive checksum differs")
        with np.load(path, allow_pickle=False) as data:
            arrays[kind] = {name: data[name].copy() for name in data.files}
    return row, arrays
