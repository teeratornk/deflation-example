"""Cancelled jobs and directory labels cannot manufacture supporting evidence."""

from copy import deepcopy

from deflation_example.coupled_completion_audit import forward_archive, retention_audit
from deflation_example.coupled_retention_report import summarize
from deflation_example.reporting import file_sha256, write_report
from test_coupled_retention_report import records


def forward_record():
    return {
        "status": "converged",
        "complete_trajectory_verified": True,
        "seconds": 20.0,
        "forward_slabs": 1,
        "steps": [{"seconds": 19.0}],
        "independent_audit": {
            "complete_trajectory_verified": True,
            "maximum_recomputed_equations": {
                "momentum_relative_residual": 1e-13,
                "continuity_relative_residual": 1e-16,
                "thermal_relative_residual": 1e-13,
                "mass_relative_imbalance": 1e-14,
                "energy_relative_defect": 1e-13,
            },
        },
    }


def test_forward_archive_keeps_cancelled_and_absent_attempts(tmp_path):
    worker = tmp_path / "worker"
    worker.mkdir()
    write_report(worker / "record.json", forward_record())
    closure = {
        "jobs": [
            {"job": "1", "folder": "worker", "status": "completed_verified"},
            {"job": "2", "folder": "worker", "status": "cancelled_incomplete"},
            {"job": "3", "status": "cancelled_before_start"},
            {"job": "4", "folder": "missing", "status": "completed_verified"},
        ]
    }
    result = forward_archive(tmp_path, closure)
    assert [r["complete_verified"] for r in result["outcomes"]] == [True, False, False, False]
    assert result["outcomes"][1]["complete_seconds"] is None
    assert result["outcomes"][1]["saved_step_seconds"] == 19
    assert result["complete_pimple_speedup"] is None


def test_forward_archive_requires_independent_nonnegative_residuals(tmp_path):
    record = forward_record()
    record["independent_audit"]["maximum_recomputed_equations"]["thermal_relative_residual"] = -1
    write_report(tmp_path / "record.json", record)
    result = forward_archive(tmp_path, {"jobs": [{"folder": ".", "status": "completed_verified"}]})
    assert not result["outcomes"][0]["complete_verified"]


def test_dispatch_stage_name_does_not_define_the_evidence_partition(tmp_path):
    manifest, selection = records()
    manifest["schema"] = "coupled-inactive-trace-v1"
    trace = tmp_path / "coupled-retention-capture-v8-10/linear-systems"
    trace.mkdir(parents=True)
    write_report(trace / "manifest.json", manifest)
    digest = file_sha256(trace / "manifest.json")
    for item in selection:
        item["trace_sha256"] = digest
    heldout = deepcopy(selection)
    for item in heldout:
        item.update(partition="held_out", quadratic=1)
        for row in item["rows"]:
            row["system"] = 1
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    for stage, reports in (("selection", selection), ("widths", []), ("heldout", heldout)):
        jobs = []
        for i, report in enumerate(reports):
            job = f"{stage}-{i}"
            folder = tmp_path / f"coupled-retention-replay-v8-{job}"
            folder.mkdir()
            write_report(folder / "record.json", report)
            jobs.append({"role": "replay", "job": job})
        write_report(campaign / (stage + "-jobs.json"), {"capture": "10", "jobs": jobs})
    for folder, reports, partition in (
        ("widths-summary", selection, "selection"),
        ("heldout-summary", selection, "selection"),
        ("assessment-summary", heldout, "held_out"),
    ):
        (campaign / folder).mkdir()
        write_report(campaign / folder / "summary.json", summarize(manifest, reports, partition))
    result = retention_audit(tmp_path, campaign)
    assert result["all_summaries_reproduced"]
    assert [r["partition"] for r in result["summaries"]] == ["selection", "selection", "held_out"]
