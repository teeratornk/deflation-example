import pytest

from deflation_example.coupled_qp_globalization_report import summarize


def records():
    base = dict(
        schema="coupled-quadratic-globalization-v1",
        trace_sha256="trace",
        quadratic_sha256="qp",
        quadratic=2,
        linear_tolerance=1e-8,
        quadratic_tolerance=0.01,
        initial="zero",
        restriction="product",
        budget_seconds=7200,
        status="complete",
        environment={"git_head": "same-source", "cpu_model": "same-cpu"},
        quadratic_status="converged",
        quadratic_verified=True,
        kkt=dict(zip(("primal", "stationarity", "dual", "lower", "upper"), (0, 1e-4, 0, 0, 0))),
    )
    return {method: {**base, "method": method} for method in ("pdas", "projected")}


def test_all_outcomes_retained_without_turning_a_failure_into_a_speedup():
    rows = records()
    rows["pdas"].update(quadratic_verified=False, quadratic_status="active_set_cap")
    result = summarize(rows)
    assert result["verified_procedures"] == ["projected"]
    assert result["next_step"] == "verify_nonlinear_integration"
    assert result["rows"][0]["quadratic_status"] == "active_set_cap"
    assert "speedup" not in result


def test_missing_and_unsuccessful_projected_runs_remain_explicit():
    rows = records()
    rows["projected"].update(status="diagnostic_error", quadratic_verified=False, error="test")
    del rows["pdas"]
    result = summarize(rows)
    assert result["next_step"] == "inspect_quadratic_histories"
    assert result["rows"][0]["status"] == "missing"
    assert result["rows"][1]["error"] == "test"


@pytest.mark.parametrize(
    "key,value", [("linear_tolerance", 1e-6), ("quadratic_sha256", "different")]
)
def test_unmatched_protocol_is_rejected(key, value):
    rows = records()
    rows["projected"][key] = value
    with pytest.raises(ValueError, match="matched"):
        summarize(rows)


def test_false_quadratic_success_is_rejected():
    rows = records()
    rows["projected"]["kkt"] = dict.fromkeys(range(5), 0.5)
    with pytest.raises(ValueError, match="KKT"):
        summarize(rows)
