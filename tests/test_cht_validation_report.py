"""The independent reference validates optimality and temporal coupling."""

from threadpoolctl import threadpool_limits

from deflation_example.benchmark_cht_validation import validate


def test_nine_distinct_small_problems_have_independent_optimality_checks():
    with threadpool_limits(limits=1):
        report = validate()
    assert report["success"]
    assert len(report["groups"]) == 3
    assert sum(len(g["cases"]) for g in report["groups"]) == 9
    for group in report["groups"]:
        assert len({c["target_sha256"] for c in group["cases"]}) == 3
        for case in group["cases"]:
            assert max(case["independent_kkt"].values()) <= 1e-8
            for result in case["methods"].values():
                assert result["state_relative_difference"] < 1e-6
                assert result["relative_objective_difference"] < 1e-10
        assert all(
            group["cases"][0]["methods"][m]["constraint_releases"] > 0 for m in case["methods"]
        )
