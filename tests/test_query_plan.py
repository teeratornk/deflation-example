"""Declared query plans pair targets with bounds for rating-style sequences."""

import pytest
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from deflation_example import benchmark_mesh as study
from deflation_example.benchmark_cht import unpack_mask


@pytest.fixture(autouse=True)
def bounded_blas_threads():
    with threadpool_limits(2):
        yield


@pytest.mark.parametrize(
    "overrides",
    [
        {"targets": 2, "query_targets": [0, 1]},
        {"targets": 2, "query_targets": [0, 1], "query_bounds": [0.2]},
        {"targets": 2, "query_targets": [0, 2], "query_bounds": [0.2, 0.2]},
        {"targets": 2, "query_targets": [0, 0], "query_bounds": [0.2, 0.2]},
        {"targets": 2, "query_targets": [0, 1], "query_bounds": [0.2, -0.1]},
    ],
)
def test_inconsistent_query_plans_are_refused(overrides):
    with pytest.raises(ValueError):
        study.controls(OmegaConf.create(overrides))


def test_default_plan_is_one_query_per_target_under_the_fixed_bound():
    c = study.controls(OmegaConf.create({"targets": 3, "bound": 0.25}))
    assert study.query_plan(c) == [(0, 0.25), (1, 0.25), (2, 0.25)]


def test_a_looser_bound_releases_constraints_for_the_same_target():
    c = study.controls(
        OmegaConf.create(
            {
                "targets": 2,
                "query_targets": [0, 1, 0, 1],
                "query_bounds": [0.2, 0.2, 0.3, 0.3],
                "methods": ["jacobi"],
                "save_fields": False,
            }
        )
    )
    record = study.sequence(c, "jacobi")[0]
    assert record["success"], record.get("failure")
    cases = record["cases"]
    assert [case["bound"] for case in cases] == [0.2, 0.2, 0.3, 0.3]
    assert [case["target_index"] for case in cases] == [0, 1, 0, 1]
    size = int(record["problem_size"])

    def active(case):
        return int(unpack_mask(case["active_mask_bits"], size).sum())

    for tight, loose in ((cases[0], cases[2]), (cases[1], cases[3])):
        assert tight["target_sha256"] == loose["target_sha256"]
        assert active(loose) <= active(tight)
