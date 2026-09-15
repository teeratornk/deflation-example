"""Reported timings and roots retain their complete-population definitions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deflation_example.reporting import file_sha256, write_report


def module():
    path = Path(__file__).parents[1] / "examples/coupled_optimization/fixed_point/audit.py"
    spec = importlib.util.spec_from_file_location("fixed_point_audit", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_empty_study_keeps_every_missing_screen(tmp_path):
    audit = module()
    summary = audit.audit(tmp_path / "inputs", tmp_path / "summary", plots=False)
    assert len(summary["rows"]) == 60
    assert all(r["status"] == "missing" and r["seconds"] is None for r in summary["rows"])
    assert not summary["selection_complete"]
    assert all(g["median_seconds"] is None for g in summary["groups"])


def test_declared_complete_population_has_correct_counts():
    audit = module()
    protocol = audit.read(audit.HERE / "protocol.json")
    selection = {"families": {key: {"selected": "anderson3"} for key in ("forward", "momentum")}}
    rows = list(audit.declared_records(protocol, selection))
    assert len(rows) == 60 + 12 + 2 + 18
    assert len({p for p, _ in rows}) == len(rows)


def test_failed_or_unfinished_repetition_has_no_median():
    audit = module()
    rows = [
        {
            "phase": "optimize",
            "group": "jacobi",
            "policy": "newton",
            "status": s,
            "verified": ok,
            "seconds": t,
        }
        for s, ok, t in (
            ("complete", True, 10),
            ("iteration_cap", False, 1),
            ("missing", False, None),
        )
    ]
    group = audit.groups(rows)[0]
    assert group["verified_seconds"] == [10]
    assert group["median_seconds"] is None
    assert group["statuses"] == ["complete", "iteration_cap", "missing"]


def optimizer_record():
    return {
        "status": "complete",
        "all_problems_verified": True,
        "configuration": {"queries": [{"target": 7}]},
        "cases": [{"status": "converged", "verified": True}],
        "sequence_seconds": 12,
        "components_seconds": {"setup": 3, "solve": 9},
    }


def test_optimizer_checks_component_sum_and_resumed_cost(tmp_path):
    audit = module()
    path = tmp_path / "record.json"
    record = optimizer_record()
    write_report(path, record)
    row, _ = audit.outcome(path, {"phase": "optimize"})
    assert row["verified"] and row["seconds"] == 12
    record["stage"] = {"resume": {"prior_seconds": 4}}
    write_report(path, record)
    row, _ = audit.outcome(path, {"phase": "optimize"})
    assert row["verified"] and row["seconds"] is None
    assert row["timing_status"] == "requires_complete_attempt_accounting"
    record["components_seconds"]["setup"] = 4
    write_report(path, record)
    with pytest.raises(ValueError, match="component timers"):
        audit.outcome(path, {"phase": "optimize"})


def test_trajectory_cannot_be_verified_from_partial_steps(tmp_path):
    audit = module()
    path = tmp_path / "record.json"
    write_report(
        path,
        {
            "status": "converged",
            "slabs": 2,
            "steps": [{"verified": True, "seconds": 1}],
            "sum_step_seconds": 1,
            "all_attempt_process_seconds": None,
        },
    )
    row, _ = audit.outcome(path, {"phase": "trajectory"})
    assert not row["verified"] and row["seconds"] is None


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_duration_is_not_a_speedup(value):
    with pytest.raises(ValueError, match="duration"):
        module().duration(value)


def test_root_agreement_checks_both_archives_and_matching_inputs(tmp_path):
    audit = module()
    case = tmp_path / "screen/forward/ordinary"
    identity = {
        "environment": {"source_sha256": {"module": "abc"}},
        "optimization_field_sha256": "control",
        "baseline_sha256": "base",
        "fine_baseline_sha256": "fine",
        "replay_fields_sha256": "replay",
        "time_s": 1,
        "initial": "previous",
        "time_scheme": "backward_euler",
        "temperature_scale_K": 10,
        "temperature_offset_K": 300,
        "row": {"verified": True},
    }
    for policy, value in (("newton", 0.5), ("anderson3", 0.6)):
        folder = case / policy
        folder.mkdir(parents=True)
        np.savez(folder / "fields.npz", state=[value], velocity=[[1, 2]], pressure=[0])
        write_report(
            folder / "record.json",
            {**identity, "policy": policy, "field_sha256": file_sha256(folder / "fields.npz")},
        )
    assert audit.root_agreement(tmp_path)[0]["maximum_temperature_difference_K"] == pytest.approx(1)
    path = case / "anderson3/record.json"
    record = audit.read(path)
    write_report(path, {**record, "time_s": 2})
    with pytest.raises(ValueError, match="matched inputs"):
        audit.root_agreement(tmp_path)
    write_report(path, record)
    np.savez(case / "newton/fields.npz", state=[99], velocity=[[1, 2]], pressure=[0])
    with pytest.raises(ValueError, match="checksum"):
        audit.root_agreement(tmp_path)


def test_all_complete_repetitions_have_median_and_range():
    rows = [
        {
            "phase": "trajectory",
            "group": "fine",
            "policy": "newton",
            "status": "converged",
            "verified": True,
            "seconds": t,
        }
        for t in (12, 10, 11)
    ]
    result = module().groups(rows)[0]
    assert result["median_seconds"] == 11
    assert result["minimum_seconds"] == 10 and result["maximum_seconds"] == 12


def test_complete_time_figure_retains_failures(tmp_path):
    pytest.importorskip("matplotlib")
    audit = module()
    rows = [
        {
            "phase": "optimize",
            "group": "reference",
            "policy": "anderson3",
            "status": "iteration_cap",
            "verified": False,
            "seconds": 2,
        }
    ]
    audit.plot_comparisons({"groups": audit.groups(rows)}, tmp_path)
    assert (tmp_path / "optimize-complete-times.pdf").stat().st_size > 100
