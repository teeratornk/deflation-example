"""Matched GPU comparisons after the complete discretization accuracy gate."""

import argparse
import copy
import json
from pathlib import Path

from .coupled_discretization import PROTOCOL, verify_returned
from .coupled_discretization_report import select, sequence_row, verified_optimization
from .coupled_trust_run import run
from .reporting import environment, file_sha256, write_report


ARMS = {
    "rank0": ("jacobi", 0),
    "reference8": ("reference", 8),
    "reference16": ("reference", 16),
    "recycling8": ("recycling", 8),
    "recycling16": ("recycling", 16),
}


def configuration(cpu, arm, repetition):
    if (
        arm not in ARMS
        or type(repetition) is not int
        or not 0 <= repetition <= 5
        or cpu.get("discretization_protocol") != PROTOCOL["identifier"]
        or cpu["device"] != "cpu"
        or cpu["method"] != "jacobi"
        or cpu["rank"] != 0
        or cpu.get("initial_state_policy") != "physical_initial"
    ):
        raise ValueError("Use a declared arm, repetition 0--5, and the verified CPU protocol")
    cfg = copy.deepcopy(cpu)
    # Source-study labels remain in its immutable input record. They are not
    # inherited as validation or timing claims for this new comparison.
    for key in tuple(cfg):
        if key.startswith("study_") or key == "verification_gate_sha256":
            cfg.pop(key)
    method, rank = ARMS[arm]
    cfg.update(
        method=method,
        rank=rank,
        device="hybrid",
        recycle_window=48,
        repetition=repetition,
        discretization_gpu_arm=arm,
        discretization_gpu_phase="feasibility" if repetition == 0 else "repeated_sequence",
    )
    return cfg


def require_feasibility(root, cpu, source_hashes):
    """No missing or failed comparator can be dropped before repeated timing."""
    for arm in ARMS:
        directory = root / arm
        record = json.loads((directory / "record.json").read_text())
        checks = json.loads((directory / "independent-verification.json").read_text())
        if (
            record["configuration"] != configuration(cpu, arm, 0)
            or record["environment"]["source_sha256"] != source_hashes
            or not verified_optimization(record)
            or checks.get("status") != "complete"
            or not checks.get("all_verified")
            or checks["optimization_record_sha256"] != file_sha256(directory / "record.json")
            or len(checks["cases"]) != 3
        ):
            raise ValueError("All five matched arms must pass feasibility before repeated timing")
        for position, check in enumerate(checks["cases"]):
            if (
                not check["verified"]
                or not check["derivatives_passed"]
                or check["field_sha256"] != file_sha256(directory / f"target-{position:02d}.npz")
            ):
                raise ValueError("Feasibility verification must match every returned target")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cpu-sequence", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--feasibility-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    decision = select(summary["sequences"])
    if decision["action"] != "gpu_feasibility" or decision != summary["decision"]:
        raise ValueError("The complete accuracy gate has not selected a GPU configuration")
    chosen = decision["selected"]
    row = sequence_row(
        args.cpu_sequence, chosen["transport_form"], chosen["slabs"], args.replay_root
    )
    frozen = next(r for r in summary["sequences"] if all(r[k] == v for k, v in chosen.items()))
    if not row["passed"] or row["record_sha256"] != frozen["record_sha256"]:
        raise ValueError("The selected CPU sequence and replays must match the frozen gate")
    source = json.loads((args.cpu_sequence / "record.json").read_text())
    cfg = configuration(source["configuration"], args.arm, args.repetition)
    if args.repetition:
        if args.feasibility_root is None:
            raise ValueError("Repeated timing requires the five feasibility records")
        require_feasibility(
            args.feasibility_root, source["configuration"], environment()["source_sha256"]
        )
    record = run(cfg, args.output, budget_seconds=PROTOCOL["sequence_budget_seconds"])
    checks = verify_returned(record, args.output)
    checks["optimization_record_sha256"] = file_sha256(args.output / "record.json")
    checks["discretization_summary_sha256"] = file_sha256(args.summary)
    write_report(args.output / "independent-verification.json", checks)
    if not checks["all_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
