"""The construction CLI retains measurements and unsuccessful attempts."""

import json
import sys

import numpy as np
import pytest

import deflation_example.coupled_reference_screen as module
from test_axisymmetric_flow import annular_rectangle
from test_coupled_reference import problem


@pytest.mark.parametrize("solver", ["dense", "tridiagonal"])
@pytest.mark.parametrize("failure", [None, MemoryError])
def test_reference_screen_runs_both_procedures_and_retains_failures(
    monkeypatch, tmp_path, solver, failure
):
    p = problem(annular_rectangle(3), [300, 300])
    p.alpha = 1e-14
    p.size = 2 * p.spatial_size
    monkeypatch.setattr(module, "load_problem", lambda cfg: (p, {"baseline_sha256": "baseline"}))
    calls = []

    class Sampler:
        def __init__(self, *args):
            pass

        def start(self):
            calls.append("start")

        def finish(self):
            calls.append("finish")
            return {"complete": True, "peak_host_rss_bytes": 1234}

    monkeypatch.setattr(module, "ProcessMemory", Sampler)
    if failure:

        def fail(*args):
            raise failure()

        monkeypatch.setattr(module, "configured_reference", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen",
            "--baseline",
            str(tmp_path),
            "--slabs",
            "2",
            "--rank",
            "2",
            "--temporal-solver",
            solver,
            "--output",
            str(tmp_path / "output"),
        ],
    )
    if failure:
        with pytest.raises(RuntimeError, match="MemoryError"):
            module.main()
    else:
        module.main()
    assert calls == ["start", "finish"]
    report = json.loads((tmp_path / "output/record.json").read_text())
    assert report["status"] == ("memory_limited" if failure else "complete")
    assert report["sampled_process_memory"]["peak_host_rss_bytes"] == 1234
    if not failure:
        assert report["deployed_reference_rank"] == 2
        assert report["trajectory_state_degrees_of_freedom"] == p.size
        with np.load(tmp_path / "output/reference.npz", allow_pickle=False) as fields:
            assert fields["temporal"].shape == (2, 2)
            assert np.isfinite(fields["temporal"]).all()
