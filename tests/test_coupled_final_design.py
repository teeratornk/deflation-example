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
    assert protocol["rank_selection"]["candidate_ranks"] == [20, 100, 200]
