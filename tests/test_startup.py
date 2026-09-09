"""Fresh-process totals include child launch and preserve failed repetitions."""

from types import SimpleNamespace

import pytest

from deflation_example import benchmark_startup


@pytest.mark.parametrize("succeeded", [True, False])
def test_process_partition_and_failures(tmp_path, monkeypatch, succeeded):
    response = SimpleNamespace(
        returncode=0 if succeeded else 1,
        stdout='STARTUP_RECORD {"components_seconds": {"initialize": 0.001}}\n',
    )
    monkeypatch.setattr(benchmark_startup.subprocess, "run", lambda *a, **kw: response)
    ticks = iter([1.0, 1.01, 2.0, 2.02])
    monkeypatch.setattr(benchmark_startup.time, "perf_counter", lambda: next(ticks))
    report = benchmark_startup.run(tmp_path / "records", repeats=2)
    assert report["complete"] and report["success"] == succeeded
    assert len(report["rows"]) == 2
    if succeeded:
        for row in report["rows"]:
            assert sum(row["components_seconds"].values()) == pytest.approx(row["total_seconds"])
    else:
        assert all(row["returncode"] == 1 for row in report["rows"])
