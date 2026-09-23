"""The final study predeclares its population without inventing resolved choices."""

import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_final_design_matches_the_public_runner_and_remains_unfrozen():
    root = Path(__file__).parents[1]
    protocol = json.loads(
        (root / "examples/coupled_optimization/final_study/protocol.json").read_text()
    )
    with initialize_config_dir(
        version_base="1.3", config_dir=str(root / "src/deflation_example/conf")
    ):
        default = compose(config_name="coupled_sequence")
    # Unknown setting names must not silently leave a runner default in effect.
    assert set(protocol["common"]) <= set(default)
    config = OmegaConf.merge(default, protocol["common"])
    assert protocol["frozen_selection"] is None
    assert "slabs" not in protocol["common"]
    assert protocol["physical_policy"]["safety_margin_K"] == 0
    assert config.horizon_s == 600 and config.target_startup_s == 60
    assert config.lower_K == 337.3
    assert len(config.queries) == 5
    assert len({q.target for q in config.queries}) == 3
    assert len({(q.target, q.upper_K) for q in config.queries}) == 5
    assert config.inner_tolerance == 1e-10
    assert config.nonlinear_tolerance == 1e-8
    assert config.reference_transfer == "full"
    assert config.reference_construction == "mode_dependent"
    assert config.reference_temporal_metric == "jacobi"
    assert config.warm_start
    assert not config.evaluation_progress and not config.linear_progress
    assert len(protocol["method_order_by_repetition"]) == protocol["repetitions"] == 5
    for order in protocol["method_order_by_repetition"]:
        assert sorted(order) == sorted(protocol["methods"])
    assert protocol["rank_selection"]["candidate_ranks"] == [200, 300, 400]
    prior = protocol["rank_selection"]["prior_screen"]
    assert prior["candidate_ranks"] == [20, 100, 200] and prior["device"] == "cpu"
    assert {row["rank"] for row in prior["outcomes"] if row["method"] == "reference"} == {
        20,
        100,
        200,
    }


def test_v2_amendment_keeps_v1_resolution_and_declares_headline_population():
    root = Path(__file__).parents[1]
    protocol = json.loads(
        (root / "examples/coupled_optimization/final_study/protocol.json").read_text()
    )
    assert protocol["schema"] == "coupled-final-study-design-v2"
    resolution = protocol["resolution"]
    assert resolution["temperature_change_K"] == 0.05
    assert resolution["tracking_relative_change"] == 0.01
    assert resolution["forward_procedure"] == "monolithic_newton"
    assert resolution["forward_tolerance"] == 1e-12 and resolution["forward_newton_cap"] == 30
    assert protocol["methods"] == ["jacobi", "reference", "recycling"]
    assert protocol["headline_population"] == ["jacobi", "reference"]
    assert set(protocol["headline_population"]) < set(protocol["methods"])
    assert protocol["screen_population"]["recycling"]["complete_sequences"] == 2
    assert protocol["common"]["device"] == "hybrid"
    assert protocol["common"]["hybrid_block_min_columns"] == 20
    assert protocol["device_policy"]["headline_device"] == "hybrid"
    v2 = protocol["resolution_v2"]["criteria"]
    assert v2["tracking_relative_change"]["threshold"] == 0.01
    assert v2["mass_weighted_space_time_rms_K"]["threshold"] == 0.05
    assert v2["pointwise_maximum_K"]["threshold"] == 0.05
    assert "not met" in protocol["resolution_v2"]["status_of_v1"]
    assert protocol["bdf2_interpretation_rule"]["declared_before_reading"] is True
    assert protocol["feasibility_v2"]["margin"].startswith("No safety margin")
    assert protocol["amendment"]["date"] == "2026-09-15"
    assert "inner_refresh" not in protocol["common"]
