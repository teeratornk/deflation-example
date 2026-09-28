from copy import deepcopy

import pytest

from deflation_example.coupled_qp_inexact_report import summarize


def record(target=1e-2):
    return dict(
        schema="coupled-quadratic-globalization-v1",
        method="projected",
        trace_sha256="trace",
        quadratic_sha256="quadratic",
        quadratic=2,
        linear_tolerance=target,
        quadratic_tolerance=0.01,
        initial="zero temperature step",
        restriction="product",
        budget_seconds=7200,
        status="complete",
        environment={"git_head": "source", "cpu_model": "cpu"},
        quadratic_status="linear_budget_exhausted",
        quadratic_verified=False,
        kkt=dict(
            primal_absolute=0,
            stationarity=2,
            dual_feasibility=0,
            lower_complementarity=0,
            upper_complementarity=0,
        ),
        inner_iterations=9,
        quadratic_change=-3,
        history=[
            dict(
                linear_status="converged",
                linear_residual=target / 10,
                linear_iterations=5,
                projected_steps=[dict(quadratic_change=-1)],
                reduced_search=dict(quadratic_change=-2),
            ),
            dict(linear_status="budget_exhausted", linear_residual=1, linear_iterations=4),
        ],
    )


def test_missing_and_failed_arms_remain_visible_without_speedup():
    result = summarize([record(), None, record(1e-8)], [1e-2, 1e-4, 1e-8])
    assert len(result["rows"]) == 3
    assert result["rows"][1]["status"] == "missing"
    assert result["rows"][0]["completed_linear_solves"] == 1
    assert result["rows"][0]["reduced_updates"] == 1
    assert result["verified_direction_tolerances"] == []
    assert result["next_step"] == "inspect_quadratic_histories"
    assert "speedup" not in result


def test_verified_result_still_requires_separate_nonlinear_tests():
    value = record()
    value.update(quadratic_status="converged", quadratic_verified=True)
    value["kkt"]["stationarity"] = 0.005
    result = summarize([value], [1e-2])
    assert result["verified_direction_tolerances"] == [1e-2]
    assert result["next_step"] == "verify_nonlinear_integration"


@pytest.mark.parametrize(
    "key,value",
    [
        ("quadratic_sha256", "other"),
        ("quadratic_tolerance", 0.1),
        ("initial", "warm"),
        ("budget_seconds", 8000),
        ("restriction", "submatrix"),
    ],
)
def test_unmatched_conditions_are_rejected(key, value):
    other = record(1e-4)
    other[key] = value
    with pytest.raises(ValueError, match="Match"):
        summarize([record(), other], [1e-2, 1e-4])


@pytest.mark.parametrize("key", ["git_head", "source_sha256", "cpu_model", "scipy", "blas"])
def test_source_and_backend_must_match(key):
    other = record(1e-4)
    other["environment"][key] = "different"
    with pytest.raises(ValueError, match="Match"):
        summarize([record(), other], [1e-2, 1e-4])


@pytest.mark.parametrize("change", [0, 1, float("nan")])
def test_invalid_search_decrease_rejected(change):
    value = record()
    value["history"][0]["reduced_search"]["quadratic_change"] = change
    with pytest.raises(ValueError, match="decrease"):
        summarize([value], [1e-2])


def test_residual_count_and_objective_checks():
    base = record()
    for key, invalid in [("inner_iterations", 10), ("quadratic_change", -4)]:
        value = deepcopy(base)
        value[key] = invalid
        with pytest.raises(ValueError):
            summarize([value], [1e-2])
    base["history"][0]["linear_residual"] = 0.02
    with pytest.raises(ValueError, match="fresh residual"):
        summarize([base], [1e-2])


def test_false_kkt_success_rejected():
    value = record()
    value.update(quadratic_status="converged", quadratic_verified=True)
    with pytest.raises(ValueError, match="KKT"):
        summarize([value], [1e-2])


@pytest.mark.parametrize("targets", [[], [0], [1], [float("nan")], [1e-2, 1e-2]])
def test_invalid_declarations_rejected(targets):
    with pytest.raises(ValueError, match="Declare"):
        summarize([None] * len(targets), targets)


def test_unexpected_tolerance_and_method_rejected():
    with pytest.raises(ValueError, match="declared"):
        summarize([record()], [1e-4])
    value = record()
    value["method"] = "pdas"
    with pytest.raises(ValueError, match="declared"):
        summarize([value], [1e-2])


def test_partial_error_record_preserves_failure():
    value = record()
    value.update(status="diagnostic_error", error="reconstruction failed")
    del value["history"]
    result = summarize([value], [1e-2])
    assert result["rows"][0]["error"] == "reconstruction failed"
    assert not result["verified_direction_tolerances"]


def test_cli_records_digests_and_missing_declared_arm(tmp_path, monkeypatch):
    import json
    from deflation_example.coupled_qp_inexact_report import main
    from deflation_example.reporting import file_sha256

    existing = tmp_path / "record.json"
    existing.write_text(json.dumps(record()))
    output = tmp_path / "summary"
    monkeypatch.setattr(
        "sys.argv",
        [
            "report",
            "--records",
            str(existing),
            str(tmp_path / "missing.json"),
            "--tolerances",
            "1e-2",
            "1e-4",
            "--output",
            str(output),
        ],
    )
    main()
    result = json.loads((output / "summary.json").read_text())
    assert result["record_sha256"] == [file_sha256(existing), None]
    assert result["rows"][1]["status"] == "missing"
    with pytest.raises(FileExistsError):
        main()
