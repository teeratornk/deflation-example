"""A staged run may start from a coarser saved optimum on a declared finer grid."""

import json

import numpy as np
from omegaconf import OmegaConf
import pytest

from deflation_example import coupled_campaign as campaign
from deflation_example import coupled_sequence as sequence
from test_coupled_campaign import base_config, prepared
from test_coupled_campaign_support import plan, protocol
from test_coupled_derivatives import small_coupled_problem


def test_stage_overrides_declare_an_off_grid_resolution_run(tmp_path):
    proto = protocol()
    frozen = {
        "ranks": {
            "jacobi": {"rank": 0, "recycle_window": 1},
            "reference": {"rank": 400, "recycle_window": 400},
            "recycling": {"rank": 200, "recycle_window": 200},
        },
        "inner_refresh": 1000,
    }
    overrides = campaign.stage_overrides(
        proto,
        plan(frozen),
        "reference",
        0,
        [1],
        tmp_path / "out",
        slabs=128,
        rank=200,
        window=200,
        refresh=1000,
    )
    assert "slabs=128" in overrides and "rank=200" in overrides
    with pytest.raises(ValueError):
        campaign.stage_overrides(
            proto, plan(frozen), "reference", 0, [1], tmp_path / "x", slabs=128
        )
    with pytest.raises(ValueError):
        campaign.stage_overrides(
            proto, plan(frozen), "reference", 0, [1], tmp_path / "x", slabs=64, rank=200
        )


def test_staged_runner_starts_from_a_prolonged_coarser_optimum(monkeypatch, tmp_path):
    coarse = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    prepared(monkeypatch, coarse)
    config = base_config(coarse, "jacobi", tmp_path / "coarse")
    config["slabs"] = 2
    config["stage"] = {"positions": [0], "restore": None, "resume": None}
    first = sequence.run(OmegaConf.create(config))
    assert first["all_problems_verified"]

    fine = small_coupled_problem([0.1, 0.1, 0.175, 0.175], uniform_capacity=True)
    prepared(monkeypatch, fine)
    config = base_config(fine, "jacobi", tmp_path / "fine")
    config.update(
        slabs=4,
        initial_control_directory=str(tmp_path / "coarse"),
        initial_control_method="jacobi",
        initial_control_position=0,
        initial_control_repeat=2,
    )
    config["stage"] = {"positions": [0], "restore": None, "resume": None}
    second = sequence.run(OmegaConf.create(config))
    assert second["status"] == "complete", second.get("error_type")
    assert second["all_problems_verified"], second["cases"][0]["status"]
    note = second["initial_control"]
    assert note["temporal_repetition"] == 2 and note["saved_slabs"] == 2
    with np.load(tmp_path / "coarse/target-00.npz") as data:
        saved = data["state"].reshape(2, -1)
    history = second["cases"][0]["history"]
    assert history and history[0]["iteration"] == 0
    stored = json.loads((tmp_path / "fine/record.json").read_text())
    assert stored["initial_control"]["field_sha256"] == note["field_sha256"]
    assert saved.shape[1] * 4 == fine.size
    with pytest.raises(ValueError):
        config["stage"] = {"positions": [0, 1], "restore": None, "resume": None}
        config["output"] = str(tmp_path / "bad")
        sequence.run(OmegaConf.create(config))
