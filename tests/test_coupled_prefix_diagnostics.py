"""Incomplete trajectories support explicitly delimited diagnostics only."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deflation_example.coupled_prefix_diagnostics import (
    aligned_counts,
    pair_diagnostics,
    read_forward,
    verified_prefix,
)
from deflation_example.reporting import write_fields, write_report


def run(slabs, failed=False):
    times = np.arange(1, slabs + 1) / slabs
    checks = dict.fromkeys(
        (
            "momentum_relative_residual",
            "continuity_relative_residual",
            "thermal_relative_residual",
            "mass_relative_imbalance",
            "energy_relative_defect",
        ),
        1e-14,
    )
    checks["procedure"] = "returned_state_verification"
    rows = [{"status": "converged", "time_s": t, "history": [deepcopy(checks)]} for t in times]
    if failed:
        rows[-1]["status"] = "hybrid_no_convergence"
    record = {
        "steps": rows,
        "forward_slabs": slabs,
        "status": rows[-1]["status"],
        "configuration": {"horizon_s": 1, "query": 0, "target_count": 1},
    }
    fields = {
        "state": np.column_stack((times, 2 * times)),
        "times_s": times,
        "velocity": np.zeros((slabs, 3, 2)),
        "pressure": np.zeros((slabs, 2)),
    }
    if failed:
        fields["state"][-1] = 1e9
    return record, fields, slabs - int(failed), Path(f"grid-{slabs}")


def test_verified_prefix_keeps_failure_and_does_not_include_failed_fields(tmp_path):
    record, fields, count, _ = run(4, failed=True)
    assert verified_prefix(record) == count == 3
    write_report(tmp_path / "record.json", record)
    write_fields(tmp_path / "states.npz", **fields)
    assert read_forward(tmp_path)[2] == 3
    record["status"] = "converged"
    write_report(tmp_path / "record.json", record)
    with pytest.raises(ValueError, match="every step"):
        read_forward(tmp_path)


@pytest.mark.parametrize("value", [1e-4, -1e-15, np.nan])
def test_independent_verification_overrides_a_converged_label(value):
    record = run(2)[0]
    record["steps"][0]["history"][-1]["momentum_relative_residual"] = value
    with pytest.raises(ValueError, match="original equations"):
        verified_prefix(record)


def test_unverified_gap_cannot_be_skipped():
    record = run(4)[0]
    record["steps"][1]["status"] = "failed"
    with pytest.raises(ValueError, match="jump"):
        verified_prefix(record)


def test_prefix_alignment_truncates_both_grids_to_a_shared_verified_time():
    assert aligned_counts(np.arange(1, 5), np.arange(1, 9) / 2, 4, 7, 2) == (3, 6)
    with pytest.raises(ValueError, match="shared verified"):
        aligned_counts(np.arange(1, 5), np.arange(1, 9) / 2, 4, 1, 2)


def test_interpolation_uses_actual_initial_field_and_excludes_failure(monkeypatch):
    import deflation_example.coupled_prefix_diagnostics as module

    problem = SimpleNamespace(
        temperature_scale=20,
        temperature_offset=341.3,
        assembly=SimpleNamespace(mass=np.array([1.0, 3.0])),
        free=np.array([0, 1]),
        initial=np.array([0.0, 0.0]),
        mesh=SimpleNamespace(
            cells=np.array([[0, 1, 2]]), nodes=np.zeros((3, 2)), materials=np.array([1])
        ),
        time_scale=1.0,
        spatial_size=2,
    )
    monkeypatch.setattr(module, "desired_temperature", lambda p, *args: np.zeros(p.size))
    coarse, fine = run(2), run(4, failed=True)
    pair = pair_diagnostics(problem, coarse, fine)
    assert pair["comparison_end_s"] == 0.5
    assert pair["maximum_temperature_difference_K"] == 0
    assert not pair["complete_horizon"]
    # Change a non-shared verified time. Endpoint-only tests would miss this.
    fine[1]["state"][0, 0] += 0.1
    pair = pair_diagnostics(problem, coarse, fine)
    assert pair["maximum_temperature_difference_K"] == pytest.approx(2)
    assert pair["maximum_weighted_temperature_rms_K"] == pytest.approx(1)
    assert pair["peak"]["time_s"] == 0.25
    assert len(pair["times_s"]) == 2
