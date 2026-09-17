import json

import numpy as np
import pytest

from deflation_example.axisymmetric_flow import FlowResult
from deflation_example.coupled_forward import CoupledResult
from deflation_example.coupled_forward_checkpoint import load_steps, save_snapshot, save_step


def fields(status="converged"):
    return CoupledResult(
        np.arange(3.0), FlowResult(np.ones((4, 2)), np.zeros(2), status, []), status, [], 1.5
    )


def test_round_trip_and_protocol_identity(tmp_path):
    protocol = {"source": "version", "control": "checksum", "tolerance": 1e-12}
    entries = save_step(tmp_path, 0, fields(), {"slab_zero_based": 0}, protocol, [])
    save_step(tmp_path, 1, fields("cap"), {"slab_zero_based": 1}, protocol, entries)
    loaded = load_steps(tmp_path, protocol)
    assert [d["status"] for d, _ in loaded] == ["converged", "cap"]
    np.testing.assert_array_equal(loaded[1][1]["state"], fields().state)
    with pytest.raises(ValueError, match="identical"):
        load_steps(tmp_path, {**protocol, "tolerance": 1e-8})
    with pytest.raises(FileExistsError):
        save_snapshot(tmp_path, "step-00000", fields(), {}, protocol)


def test_tampered_fields_cannot_be_used_for_restart(tmp_path):
    save_step(tmp_path, 0, fields(), {"slab_zero_based": 0}, {}, [])
    archive = tmp_path / "step-00000.npz"
    archive.write_bytes(archive.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="field checksum"):
        load_steps(tmp_path, {})


def test_tampered_verification_cannot_be_used_for_restart(tmp_path):
    save_step(tmp_path, 0, fields("cap"), {"slab_zero_based": 0}, {}, [])
    record = tmp_path / "step-00000.json"
    data = json.loads(record.read_text())
    data["status"] = "converged"
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="record checksum"):
        load_steps(tmp_path, {})


def test_interrupted_snapshot_never_advances_the_index(tmp_path, monkeypatch):
    import deflation_example.coupled_forward_checkpoint as module

    entries = save_step(tmp_path, 0, fields(), {"slab_zero_based": 0}, {}, [])
    writer = module.write_report

    def fail_record(path, record):
        if path.name == "step-00001.json":
            raise OSError("simulated interrupted write")
        return writer(path, record)

    monkeypatch.setattr(module, "write_report", fail_record)
    with pytest.raises(OSError):
        save_step(tmp_path, 1, fields(), {"slab_zero_based": 1}, {}, entries)
    assert len(load_steps(tmp_path, {})) == 1


def test_checkpoint_order_and_finite_values_are_required(tmp_path):
    with pytest.raises(ValueError, match="in order"):
        save_step(tmp_path, 1, fields(), {"slab_zero_based": 1}, {}, [])
    invalid = fields()
    invalid.state[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        save_snapshot(tmp_path, "invalid", invalid, {}, {})
