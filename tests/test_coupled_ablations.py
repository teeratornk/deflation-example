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
