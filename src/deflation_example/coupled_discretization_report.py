"""Accuracy-based selection for the bounded six-case discretization study."""

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled_discretization import PROTOCOL
from .coupled_optimize import adjoint_acceptance, equation_acceptance
from .coupled_trust_gate import inner_accuracy
from .reporting import environment, file_sha256, write_report


def verified_optimization(record):
    cfg = record["configuration"]
    cases = record.get("cases", [])
    return bool(
        record.get("status") == "complete"
        and record.get("all_problems_verified") is True
        and len(cases) == 3
        and [r["target"] for r in cases] == PROTOCOL["targets"]
        and all(
            r["status"] == "converged"
            and r["verified"]
            and len(r["kkt"]) == 5
            and np.isfinite(list(r["kkt"].values())).all()
            and min(r["kkt"].values()) >= 0
            and max(r["kkt"].values()) <= 1e-8
            and equation_acceptance(r["equations"], cfg)
            and adjoint_acceptance(r["adjoint"])
            and inner_accuracy(r, cfg)
            for r in cases
        )
    )


def select(rows):
    expected = {(form, n) for form in PROTOCOL["transport_forms"] for n in PROTOCOL["time_slabs"]}
    keys = [(r["transport_form"], r["slabs"]) for r in rows]
    if len(keys) != 6 or set(keys) != expected:
        raise ValueError("Retain exactly the six declared transport/time-grid configurations")
    eligible = [r for r in rows if r["passed"]]
    if any(
        r["status"] in {"missing", "running", "verification_pending", "replay_pending"}
        for r in rows
    ):
        action, chosen = "wait_for_declared_outcomes", None
    elif eligible:
        chosen = min(eligible, key=lambda r: (r["slabs"], r["transport_form"] != "advective"))
        action = "gpu_feasibility"
    else:
        action, chosen = "stop_bounded_study", None
    return {
        "action": action,
        "selected": None if chosen is None else {k: chosen[k] for k in ("transport_form", "slabs")},
        "selection_rule": PROTOCOL["selection"],
        "scope": "Accuracy and discrete temperature sensitivity select the configuration. No runtime or speedup enters this decision.",
    }


def sequence_row(directory, form, slabs, replay_root):
    row = {"transport_form": form, "slabs": slabs, "status": "missing", "passed": False}
    path = directory / "record.json"
    if not path.exists():
        return row
    record = json.loads(path.read_text())
    cfg = record["configuration"]
    if (
        cfg["transport_form"] != form
        or cfg["slabs"] != slabs
        or cfg.get("discretization_protocol") != PROTOCOL["identifier"]
        or cfg["method"] != "jacobi"
        or cfg["rank"] != 0
        or cfg["device"] != "cpu"
    ):
        raise ValueError("Sequence ordering and protocol must match the six declared cases")
    row.update(
        status=record["status"],
        record_sha256=file_sha256(path),
        numerical_source=record["environment"]["git_head"],
        cases=[
            {k: c.get(k) for k in ("target", "status", "kkt", "inner_iterations")}
            for c in record.get("cases", [])
        ],
        seconds=record.get("attempt_seconds"),
        error=record.get("error"),
    )
    row["optimization_verified"] = verified_optimization(record)
    if not row["optimization_verified"]:
        return row
    check_path = directory / "independent-verification.json"
    if not check_path.exists():
        row["status"] = "verification_pending"
        return row
    checks = json.loads(check_path.read_text())
    if checks.get("status") != "complete":
        row["status"] = "verification_pending"
        return row
    if checks["optimization_record_sha256"] != row["record_sha256"]:
        raise ValueError("Independent derivatives belong to a different optimization record")
    if not checks["all_verified"]:
        row.update(
            status="independent_verification_failed",
            independent_verified=False,
            independent_checks=checks["cases"],
        )
        return row
    for position, c in enumerate(checks["cases"]):
        if c.get("field_sha256") != file_sha256(directory / f"target-{position:02d}.npz"):
            raise ValueError("Independent derivatives must identify the saved returned fields")
    row["independent_verified"] = bool(
        checks["all_verified"]
        and len(checks["cases"]) == 3
        and [c["target"] for c in checks["cases"]] == PROTOCOL["targets"]
        and all(c["verified"] and c["derivatives_passed"] for c in checks["cases"])
    )
    if not row["independent_verified"]:
        row["status"] = "independent_verification_failed"
        return row
    replays = []
    row["replays"] = replays
    for position in range(3):
        for subdivision in PROTOCOL["replay_subdivisions"]:
            folder = replay_root / f"{form}-{slabs}-target-{position}-x{subdivision}"
            item = {"position": position, "subdivision": subdivision, "passed": False}
            replays.append(item)
            p = folder / "verification.json"
            if not p.exists():
                item["status"] = "pending"
                continue
            audit = json.loads(p.read_text())
            if (
                audit["optimization_record_sha256"] != row["record_sha256"]
                or audit["optimization_field_sha256"]
                != file_sha256(directory / f"target-{position:02d}.npz")
                or audit["forward_record_sha256"] != file_sha256(folder / "record.json")
                or audit["forward_fields_sha256"] != file_sha256(folder / "states.npz")
                or audit["position"] != position
                or audit["subdivision"] != subdivision
            ):
                raise ValueError(
                    "Replay verification must identify the unchanged source and returned fields"
                )
            temperature = audit["temperature"]
            passed = bool(
                audit["passed"]
                and audit["complete_trajectory_verified"]
                and temperature is not None
                and np.isfinite(temperature["maximum_trajectory_difference_K"])
                and 0 <= temperature["maximum_trajectory_difference_K"] <= 0.05
            )
            item.update(status=audit["forward_status"], passed=passed, temperature=temperature)
    row["passed"] = all(r["passed"] for r in replays)
    row["status"] = (
        "replay_pending"
        if any(r["status"] == "pending" for r in replays)
        else ("all_checks_met" if row["passed"] else "replay_gate_failed")
    )
    return row


def local_rows(directories):
    if len(directories) != 27:
        raise ValueError("Retain all 27 declared local cases")
    rows = []
    for index, directory in enumerate(directories):
        row = {
            "trajectory": ("nominal", "retained", "alternate")[index // 9],
            "subdivision": (1, 2, 4)[index % 9 // 3],
            "perturbation_K": (-1e-6, 0, 1e-6)[index % 3],
            "status": "missing",
            "verified": False,
        }
        path = directory / "record.json"
        if path.exists():
            record = json.loads(path.read_text())
            for key in ("subdivision", "perturbation_K"):
                if key in record and record[key] != row[key]:
                    raise ValueError("Local records must use the declared case ordering")
            if "result_sha256" in record and record["result_sha256"] != file_sha256(
                directory / "result.npz"
            ):
                raise ValueError("Local returned fields changed")
            steps = record.get("steps", [])
            row.update(
                status=record["status"],
                verified=record.get("verified", False),
                steps=steps,
                record_sha256=file_sha256(path),
                error=record.get("error"),
            )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, nargs=6, required=True)
    parser.add_argument("--intervals", type=Path, nargs=27, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous study summary")
    design = [(f, n) for f in PROTOCOL["transport_forms"] for n in PROTOCOL["time_slabs"]]
    rows = [
        sequence_row(p, f, n, args.replay_root)
        for p, (f, n) in zip(args.sequences, design, strict=True)
    ]
    write_report(
        args.output,
        {
            "schema": "coupled-discretization-summary-v1",
            "environment": environment(),
            "sequences": rows,
            "local": local_rows(args.intervals),
            "decision": select(rows),
        },
    )


if __name__ == "__main__":
    main()
