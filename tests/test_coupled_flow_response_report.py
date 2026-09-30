"""Summary populations and failed sensitivity measurements remain explicit."""

import copy

import pytest

from deflation_example.coupled_flow_response_report import summarize


def record():
    settings = [(name, 0.0, False) for name in ("retained", "preceding_time", "preceding_target")]
    settings += [("retained", sign * h, False) for h in (1e-4, 1e-6, 1e-8) for sign in (-1, 1)]
    settings += [("retained", sign * 1e-6, True) for sign in (-1, 1)]
    return {
        "schema": "coupled-local-flow-response-v1",
        "status": "complete",
        "record_sha256": "record",
        "fields_sha256": ["a", "b"],
        "position": 1,
        "target": 8,
        "environment": {"git_head": "source"},
        "budget_seconds_per_case": 180,
        "slab_zero_based": 8,
        "tangent": {"velocity_derivative_norm_m_s_per_K": 3},
        "cases": [
            {
                "initial_guess": seed,
                "step_K": step,
                "continuation": cont,
                "status": "converged",
                "verified": True,
                "velocity_change_norm_m_s": 3 * abs(step),
                "history": [],
            }
            for seed, step, cont in settings
        ],
    }


def test_failed_response_is_not_a_sensitivity_measurement():
    source = record()
    source["cases"][4].update(status="line_search_failed", verified=False)
    result = summarize([source])
    assert result["case_count"] == 11
    assert result["verified_count"] == 10
    rows = result["slabs"][0]["cases"]
    assert rows[4]["measured_velocity_response_m_s_per_K"] is None
    assert rows[3]["measured_velocity_response_m_s_per_K"] == 3
    assert rows[0]["measured_velocity_response_m_s_per_K"] is None


@pytest.mark.parametrize(
    "mutation", ["duplicate", "source", "population", "running", "verified_failure"]
)
def test_summary_rejects_mixed_or_mislabeled_evidence(mutation):
    first, other = record(), record()
    other["slab_zero_based"] = 7
    if mutation == "duplicate":
        other = copy.deepcopy(first)
    elif mutation == "source":
        other["environment"]["git_head"] = "different"
    elif mutation == "population":
        other["cases"].pop()
    elif mutation == "running":
        other["status"] = "running"
    else:
        other["cases"][0]["status"] = "iteration_cap"
    with pytest.raises(ValueError):
        summarize([first, other])


def test_plot_uses_comparable_axis_limits(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    from deflation_example.coupled_flow_response_report import plot

    first, other = record(), record()
    other["slab_zero_based"] = 9
    other["tangent"]["velocity_derivative_norm_m_s_per_K"] = 300
    summary = summarize([first, other])
    checked = []

    def inspect(figure, *args, **kwargs):
        assert figure.axes[0].get_ylim() == figure.axes[1].get_ylim()
        assert figure.axes[0].get_ylim()[1] >= 300
        checked.append(True)

    monkeypatch.setattr(Figure, "savefig", inspect)
    plot(summary, tmp_path)
    assert len(checked) == 2
