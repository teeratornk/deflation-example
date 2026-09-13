"""The ablation design and summaries retain matched settings and all outcomes."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from deflation_example import coupled_ablations as ablations
from test_coupled_protocol import row as equation_row
from test_coupled_report import record


def protocol():
    return json.loads((Path(__file__).parents[1] / ablations.DEFAULT_PROTOCOL).read_text())


def test_design_has_31_distinct_cases_with_fixed_physical_horizon_and_accuracy():
    design = ablations.cases(protocol())
    assert len(design) == 31
    gpu = ablations.cases(protocol(), "cuda")
    assert len(gpu) == 7
    assert {case["group"] for case in gpu} == {"baseline", "rank20", "rank200"}
    for case in design:
        cfg = case["settings"]
        assert cfg["horizon_s"] == 600
        assert cfg["target_startup_s"] == 60
        assert cfg["inner_tolerance"] == 1e-10
        assert cfg["nonlinear_tolerance"] == 1e-8
        assert len(cfg["queries"]) == 1
        if cfg["method"] == "jacobi":
            assert cfg["rank"] == 0
        else:
            assert cfg["rank"] == cfg["recycle_window"]
    changed = protocol()
    changed["groups"].append(deepcopy(changed["groups"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        ablations.cases(changed)


def test_config_composition_preserves_hydra_solver_and_declared_controls(tmp_path):
    case = ablations.cases(protocol())[-1]
    cfg = ablations.configuration(case, tmp_path / "baseline", tmp_path / "output")
    assert cfg.reference_transfer == "sequential"
    assert cfg.reference_construction == "mode_dependent"
    assert cfg.reference_temporal_metric == "jacobi"
    assert cfg.mode == "optimize"
    assert cfg.method == "reference"
    assert cfg.queries[0].target == 7


def test_derivative_gate_checks_grid_baseline_and_numerical_results():
    cfg = ablations.cases(protocol())[0]["settings"]
    gate = {
        "status": "verified",
        "baseline_sha256": "baseline",
        "derivative_identity": ablations.derivative_identity(cfg),
        "derivatives": {
            "taylor_orders": [2, 2, 2],
            "relative_dot_product_error": 1e-12,
            "equations": [equation_row()],
        },
    }
    ablations.check_gate(gate, cfg, "baseline")
    with pytest.raises(ValueError, match="matching"):
        ablations.check_gate(gate, {**cfg, "slabs": 32}, "baseline")
    with pytest.raises(ValueError, match="matching"):
        ablations.check_gate(gate, cfg, "different")
    gate["derivatives"]["taylor_orders"] = [2, 2, float("nan")]
    with pytest.raises(ValueError, match="numerical checks"):
        ablations.check_gate(gate, cfg, "baseline")


def save_case(root, design, index, data):
    case = ablations.cases(design)[index]
    directory = root / case["id"] / "rep-0"
    (directory / "sequence").mkdir(parents=True)
    data["configuration"].update(case["settings"], device="cpu")
    (directory / "case.json").write_text(
        json.dumps(
            {
                "case": case,
                "protocol_sha256": ablations.protocol_digest(design),
                "status": "complete",
                "repetition": 0,
                "environment": data["environment"],
            }
        )
    )
    path = directory / "sequence" / "record.json"
    path.write_text(json.dumps(data))
    return path


def test_summary_keeps_missing_failed_and_verified_cases(tmp_path):
    design = protocol()
    good = record("jacobi", 0, 3)
    good["cases"][0]["inner_iterations"] = 12
    save_case(tmp_path, design, 0, good)
    failed = record("reference", 0, 4)
    failed["cases"][0].update(status="nonlinear_iteration_cap", verified=False)
    failed.update(verified_problems=0, all_problems_verified=False)
    save_case(tmp_path, design, 1, failed)
    report = ablations.summarize(design, tmp_path)
    assert len(report["rows"]) == 31
    assert report["rows"][0]["verified"]
    assert report["rows"][0]["inner_iterations"] == 12
    assert not report["rows"][1]["verified"]
    assert report["rows"][1]["target_statuses"] == ["nonlinear_iteration_cap"]
    assert report["rows"][1]["sequence_seconds"] == 4
    assert report["rows"][2]["status"] == "missing"
    assert not any("speedup" in key for row in report["rows"] for key in row)
    out = tmp_path / "summary"
    ablations.write_summary(report, out)
    assert (out / "summary.csv").is_file()
    pytest.importorskip("matplotlib")
    assert ablations.plot_summary(report, out)
    assert (out / "ablations.pdf").is_file()


def test_summary_rejects_changed_settings_and_mixed_sources(tmp_path):
    design = protocol()
    first = record("jacobi", 0, 3)
    path = save_case(tmp_path, design, 0, first)
    altered = deepcopy(first)
    altered["configuration"]["inner_tolerance"] = 1e-6
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="settings"):
        ablations.summarize(design, tmp_path)
    path.write_text(json.dumps(first))
    second = record("reference", 0, 2)
    second["environment"]["source_sha256"] = {"solver.py": "another"}
    save_case(tmp_path, design, 1, second)
    with pytest.raises(ValueError, match="populations separate"):
        ablations.summarize(design, tmp_path)


def test_early_runtime_failure_is_not_relabelled_as_a_completed_solve(tmp_path):
    data = record("jacobi", 0, 3)
    data.update(
        status="sequence_error",
        cases=[],
        verified_problems=0,
        all_problems_verified=False,
        error_type="MemoryError",
    )
    save_case(tmp_path, protocol(), 0, data)
    row = ablations.summarize(protocol(), tmp_path)["rows"][0]
    assert not row["verified"]
    assert row["error_type"] == "MemoryError"
    assert row["status"] == "sequence_error"


def test_launcher_error_overrides_an_unfinished_sequence_record(tmp_path):
    data = record("jacobi", 0, 3)
    data["status"] = "running"
    del data["sequence_seconds"]
    path = save_case(tmp_path, protocol(), 0, data)
    manifest_path = path.parent.parent / "case.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="run_error", error_type="RuntimeError")
    manifest_path.write_text(json.dumps(manifest))
    row = ablations.summarize(protocol(), tmp_path)["rows"][0]
    assert row["status"] == "run_error"
    assert row["record_status"] == "running"
    assert row["error_type"] == "RuntimeError"
    assert not row["verified"]


def test_nullable_transition_counts_preserve_coverage_in_the_complete_report(tmp_path):
    data = record("jacobi", 0, 3)
    data["cases"][0]["history"] = [
        {
            "attempts": [
                {
                    "qp_history": [
                        {"timing": {"newly_active": None, "newly_inactive": None}},
                        {"timing": {"newly_active": 0, "newly_inactive": 5}},
                    ]
                }
            ]
        }
    ]
    save_case(tmp_path, protocol(), 0, data)
    row = ablations.summarize(protocol(), tmp_path)["rows"][0]
    assert row["verified"]
    assert row["newly_active"] is None
    assert row["newly_inactive"] is None
    assert row["newly_active_recorded_total"] == 0
    assert row["newly_inactive_recorded_total"] == 5
    assert row["newly_inactive_recorded_steps"] == 1
    assert row["newly_inactive_unrecorded_steps"] == 1


@pytest.mark.parametrize("steps", [[], [{}], [{"timing": None}], [{"timing": {}}]])
def test_absent_transition_counts_remain_unavailable(steps):
    result = ablations.transition_summary(steps)
    for name in ("newly_active", "newly_inactive"):
        assert result[name] is None
        assert result[name + "_recorded_total"] is None
        assert result[name + "_recorded_steps"] == 0
        assert result[name + "_unrecorded_steps"] == len(steps)


def test_complete_transition_counts_include_true_zeros():
    result = ablations.transition_summary(
        [
            {"timing": {"newly_active": 0, "newly_inactive": 2}},
            {"timing": {"newly_active": 0, "newly_inactive": 3}},
        ]
    )
    assert result["newly_active"] == 0
    assert result["newly_inactive"] == 5
    assert result["newly_inactive_unrecorded_steps"] == 0


def execution_record(status):
    return {
        "schema": "coupled-ablation-execution-status-v1",
        "protocol_sha256": ablations.protocol_digest(protocol()),
        "device": "cpu",
        "rows": [{"case": "baseline-jacobi", "repetition": 0, "status": status}],
    }


@pytest.mark.parametrize("status", ["TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"])
def test_interrupted_jobs_do_not_remain_running(tmp_path, status):
    data = record("jacobi", 0, 3)
    data["status"] = "running"
    del data["sequence_seconds"]
    save_case(tmp_path, protocol(), 0, data)
    report = ablations.summarize(protocol(), tmp_path)
    ablations.apply_execution_status(report, execution_record(status))
    row = report["rows"][0]
    assert row["status"] == "execution_" + status.lower()
    assert row["record_status"] == "running"
    assert not row["verified"]
    assert "sequence_seconds" not in row


def test_execution_failure_does_not_erase_a_verified_numerical_record(tmp_path):
    save_case(tmp_path, protocol(), 0, record("jacobi", 0, 3))
    report = ablations.summarize(protocol(), tmp_path)
    ablations.apply_execution_status(report, execution_record("FAILED"))
    row = report["rows"][0]
    assert row["numerical_record_verified"]
    assert not row["verified"]
    assert row["sequence_seconds"] == 3


def test_completed_launcher_requires_complete_numerical_output(tmp_path):
    report = ablations.summarize(protocol(), tmp_path)
    ablations.apply_execution_status(report, execution_record("COMPLETED"))
    assert report["rows"][0]["status"] == "incomplete_output"
    assert not report["rows"][0]["verified"]


def test_execution_status_rejects_wrong_device_and_duplicate_cases(tmp_path):
    report = ablations.summarize(protocol(), tmp_path)
    wrong = execution_record("COMPLETED")
    wrong["device"] = "cuda"
    with pytest.raises(ValueError, match="protocol and device"):
        ablations.apply_execution_status(report, wrong)
    duplicate = execution_record("COMPLETED")
    duplicate["rows"] *= 2
    with pytest.raises(ValueError, match="distinct declared cases"):
        ablations.apply_execution_status(report, duplicate)
