"""A launcher correction keeps failed attempts and cannot replace numerical failures."""

import copy
import pytest

from deflation_example.reporting import file_sha256, write_report
from test_fixed_point_reporting import module


def fixture(tmp_path):
    audit = module()
    root, repaired = tmp_path / "original", tmp_path / "repaired"
    root.mkdir()
    protocol = tmp_path / "protocol.json"
    write_report(protocol, {"version": 3})
    env = {"git_head": "frozen", "source_sha256": {"solver.py": "same"}}
    selection = {"environment": env, "families": {"momentum": {"selected": "anderson5"}}}
    write_report(root / "selection.json", selection)
    entries = []
    for policy in ("newton", "anderson5"):
        gate = root / f"derivatives/{policy}/derivatives/record.json"
        gate.parent.mkdir(parents=True)
        write_report(gate, {"status": "verified", "environment": env})
        relative = f"optimize/{policy}/reference/rep-0/record.json"
        original = root / relative
        original.parent.mkdir(parents=True)
        write_report(
            original,
            {
                "status": "sequence_error",
                "error_type": "ValueError",
                "cases": [],
                "environment": env,
                "baseline_sha256": "baseline",
                "configuration": {"repetition": 0, "queries": [{"target": 7}], "rank": 200},
                "sequence_seconds": 3.0,
                "components_seconds": {"setup": 3.0},
            },
        )
        entries.append({"record": relative, "sha256": file_sha256(original)})
    declaration = {
        "schema": "fixed-point-reference-setup-amendment-v1",
        "protocol_sha256": file_sha256(protocol),
        "numerical_source": "frozen",
        "changed_setting": "reference_baseline_directory",
        "new_value": None,
        "replacement_tasks": [2, 3, 8, 9, 14, 15],
        "withdrawn_before_execution_tasks": [8, 9, 14, 15],
        "original_setup_errors": entries,
    }
    path = tmp_path / "amendment.json"
    write_report(path, declaration)
    return audit, root, repaired, protocol, selection, path, declaration


def test_repair_preserves_setup_failures_and_missing_results(tmp_path):
    audit, root, repaired, protocol, selection, path, _ = fixture(tmp_path)
    repair = audit.reference_repair(path, protocol, root, repaired, selection)
    assert len(repair["original_setup_errors"]) == 2
    assert all(
        r["status"] == "sequence_error" and r["seconds"] == 3
        for r in repair["original_setup_errors"]
    )
    for rep in range(3):
        spec = {"phase": "optimize", "group": "reference", "policy": "newton", "repetition": rep}
        row, record = audit.repaired_reference_outcome(
            repaired, f"optimize/newton/reference/rep-{rep}/record.json", spec, repair
        )
        assert record is None and row["status"] == "missing"
        assert row["seconds"] is None and row["all_attempt_sequence_seconds"] is None
        assert row["original_setup_error_seconds"] == (3 if rep == 0 else 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("protocol_sha256", "wrong"),
        ("new_value", "different"),
        ("changed_setting", "rank"),
        ("replacement_tasks", [2]),
        ("numerical_source", "other"),
        ("original_setup_errors", []),
    ],
)
def test_repair_rejects_different_protocol_or_population(tmp_path, field, value):
    audit, root, repaired, protocol, selection, path, declaration = fixture(tmp_path)
    declaration[field] = value
    write_report(path, declaration)
    with pytest.raises(ValueError):
        audit.reference_repair(path, protocol, root, repaired, selection)


@pytest.mark.parametrize(
    "status,cases", [("complete", []), ("sequence_error", [{"status": "iteration_cap"}])]
)
def test_repair_cannot_hide_numerical_outcome(tmp_path, status, cases):
    audit, root, repaired, protocol, selection, path, declaration = fixture(tmp_path)
    entry = declaration["original_setup_errors"][0]
    original = root / entry["record"]
    record = audit.read(original)
    record.update(status=status, cases=cases)
    write_report(original, record)
    entry["sha256"] = file_sha256(original)
    write_report(path, declaration)
    with pytest.raises(ValueError, match="numerical solve"):
        audit.reference_repair(path, protocol, root, repaired, selection)


def test_repair_rejects_executed_unstarted_task(tmp_path):
    audit, root, repaired, protocol, selection, path, _ = fixture(tmp_path)
    record = root / "optimize/newton/reference/rep-1/record.json"
    record.parent.mkdir(parents=True)
    write_report(record, {"status": "running"})
    with pytest.raises(ValueError, match="unstarted"):
        audit.reference_repair(path, protocol, root, repaired, selection)


@pytest.mark.parametrize("change", [None, "rank", "baseline", "gate", "live"])
def test_corrected_attempt_checks_settings_and_retains_cost(tmp_path, change):
    audit, root, repaired, protocol, selection, path, _ = fixture(tmp_path)
    repair = audit.reference_repair(path, protocol, root, repaired, selection)
    relative = "optimize/newton/reference/rep-0/record.json"
    record = copy.deepcopy(repair["templates"]["newton"])
    record.update(sequence_seconds=5.0, components_seconds={"setup": 5.0})
    if change == "rank":
        record["configuration"]["rank"] = 100
    elif change == "baseline":
        record["baseline_sha256"] = "different"
    elif change == "live":
        record.pop("baseline_sha256")
        record.update(status="running", sequence_seconds=None)
    target = repaired / relative
    target.parent.mkdir(parents=True)
    write_report(target, record)
    attempt = {
        "reference_setup_policy": "same-mesh-reference-no-separate-coarse-baseline-v1",
        "protocol_sha256": file_sha256(protocol),
        "launcher_sha256": "launcher",
        "gate_record_sha256": {} if change == "gate" else repair["gate_hashes"],
    }
    write_report(target.parent.parent / "rep-0-attempt.json", attempt)
    spec = {"phase": "optimize", "group": "reference", "policy": "newton", "repetition": 0}
    if change not in (None, "live"):
        with pytest.raises(ValueError):
            audit.repaired_reference_outcome(repaired, relative, spec, repair)
    else:
        row, _ = audit.repaired_reference_outcome(repaired, relative, spec, repair)
        if change == "live":
            assert row["seconds"] is None and row["all_attempt_sequence_seconds"] is None
            assert row["input_identity_status"] == "awaiting_final_record"
        else:
            assert row["seconds"] == 5 and row["all_attempt_sequence_seconds"] == 8
        assert not row["verified"]
        assert audit.groups([row])[0]["median_seconds"] is None
