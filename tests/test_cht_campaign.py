from omegaconf import OmegaConf

from deflation_example.benchmark_cht import specification
from deflation_example.benchmark_cht_campaign import BASE, campaign_specification, studies


def test_frozen_campaign_has_matched_primary_populations_and_rank_controls():
    declared = studies()
    assert len(declared) == 17
    for name, settings in declared.items():
        protocol = specification(OmegaConf.create(BASE | settings))
        controls = protocol["controls"]
        assert controls["targets"] == 16
        assert controls["repeats"] == 5
        assert controls["phase"] == "final"
        assert controls["rank"] == controls["recycle_rank"] == controls["window"]
        assert controls["cg_factor"] == controls["amgx_factor"] == 0.1
        assert len(protocol["targets"]) == 16
        if "rank" not in name:
            assert controls["methods"] == ["jacobi", "reference", "recycling", "amgx"]
    assert campaign_specification()["memory_comparison"]["gpu_budgets_bytes"] == [
        8 * 2**30,
        16 * 2**30,
        64 * 2**30,
    ]
