"""Presentation preserves timing boundaries, failure status and memory criteria."""

from copy import deepcopy
import hashlib

from omegaconf import OmegaConf
import pytest

from deflation_example import benchmark_cht, benchmark_cht_presentation as presentation
from deflation_example.memory import ProcessMemory
from deflation_example.reporting import write_report


def test_cost_groups_are_disjoint_and_use_an_actual_median_sequence():
    rows = [
        {"success": True, "total_seconds": t, "repetition": r}
        for r, t in enumerate((5, 1, 3, 2, 4))
    ]
    assert presentation.median_sequence(rows)["repetition"] == 2
    assert presentation.median_sequence(rows[:2]) is None
    sequence = {
        "total_seconds": 10,
        "components_seconds": {
            "assembly": 1,
            "kernel_iteration": 5,
            "kernel_upload": 1,
            "kernel_verification": 2,
            "remaining_host_work": 1,
        },
    }
    groups = presentation.timing_groups(sequence)
    assert list(groups.values()) == [1, 5, 1, 3]
    sequence["total_seconds"] += 1
    with pytest.raises(ValueError, match="sum"):
        presentation.timing_groups(sequence)


def memory_row(rank, seconds, memory_gib):
    return {
        "problem": "transient",
        "n": 24,
        "slabs": 8,
        "method": "recycling",
        "warm_start": "outer_inner",
        "reference_rank": 100,
        "recycle_rank": rank,
        "accepted_sequences": 5,
        "declared_sequences": 5,
        "verified_memory_sequences": 5,
        "peak_host_rss_bytes": 2 * 2**30,
        "peak_gpu_process_bytes": memory_gib * 2**30,
        "median_seconds": seconds,
        "deployed_rank_min": 0,
        "deployed_rank_max": rank,
        "fallbacks": 0,
    }


def test_memory_envelopes_keep_failed_or_over_budget_choices_separate(tmp_path):
    rows = [memory_row(20, 10, 2), memory_row(100, 8, 10), memory_row(200, 6, 32)]
    failed = memory_row(300, 1, 1)
    failed["accepted_sequences"] = 4
    results = presentation.memory_envelopes(rows + [failed])
    assert [row["selected_rank"] for row in results] == [20, 100, 200]
    assert all(row["measured_requested_ranks"] == [20, 100, 200, 300] for row in results)
    assert results[0]["selected_deployed_rank_range"] == [0, 20]
    assert results[0]["feasible_choices"] == 1
    presentation.memory_budget_table(results, tmp_path)
    table = (tmp_path / "memory_budget_rows.tex").read_text()
    assert "Recycling & 20 & 10.000" in table
    assert "Recycling & 200 & 6.000" in table


def test_complete_record_to_figures_and_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark_cht,
        "ProcessMemory",
        lambda *args: ProcessMemory(interval=60, query=lambda: (100, 0)),
    )
    c = OmegaConf.structured(benchmark_cht.StudyConfig)
    c.device, c.methods, c.problem = "cpu", ["jacobi", "reference", "recycling"], "transient"
    c.n, c.slabs, c.targets, c.rank, c.window, c.bound = 3, 3, 3, 5, 5, 0.0001
    protocol = benchmark_cht.specification(c)
    source = tmp_path / "tiny"
    source.mkdir()
    write_report(source / "protocol.json", protocol)
    checksum = hashlib.sha256((source / "protocol.json").read_bytes()).hexdigest()
    manifest = {"protocol_sha256": checksum, "success": True, "complete": True, "sequences": []}
    for method in c.methods:
        sequence = benchmark_cht.complete_sequence(protocol, method, "outer_inner")
        sequence.update(repetition=0, protocol_sha256=checksum)
        assert sequence["success"]
        path = f"{method}.json"
        write_report(source / path, sequence)
        manifest["sequences"].append(
            {
                "method": method,
                "warm_start": "outer_inner",
                "repetition": 0,
                "success": True,
                "path": path,
                "sha256": hashlib.sha256((source / path).read_bytes()).hexdigest(),
            }
        )
    write_report(source / "results.json", manifest)
    output = tmp_path / "presentation"
    report = presentation.run([source], output, ["tiny"], ["tiny"])
    assert len(report["rows"]) == 3
    assert len(report["physical_slices"]) == 3
    assert report["physical_slices"][0]["x2_coordinate"] == 0.5
    assert len(report["component_sequences"]) == 3
    for row in report["component_sequences"]:
        assert sum(row["groups_seconds"].values()) == pytest.approx(row["complete_seconds"])
    assert (output / "complete_sequences.pdf").is_file()
    assert "3^3" in (output / "complete_rows.tex").read_text()
    damaged = deepcopy(manifest)
    damaged["sequences"][0]["success"] = False
    write_report(source / "results.json", damaged)
    with pytest.raises(ValueError, match="acceptance"):
        presentation.run([source], tmp_path / "bad", ["tiny"], [])
