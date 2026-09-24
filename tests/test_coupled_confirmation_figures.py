"""Synthetic plot fixtures are not measurements or publication evidence."""

import numpy as np
import pytest

from deflation_example.coupled_confirmation_figures import figure_data, measured_intervals, plot
from test_coupled_confirmation_report import population


def measured_population():
    settings, records, fields = population()
    for row in records:
        seconds = row["sequence_seconds"]
        row["components_seconds"] = {
            "model_assembly_and_baseline_verification": 0.1,
            "reference_construction": 0.2,
            "solver_resources": 0.1,
            "target_optimizations_and_verification": seconds - 0.5,
            "cleanup": 0.05,
            "bookkeeping": 0.05,
        }
        for case in row["cases"]:
            case["seconds"] = (seconds - 0.6) / 3
    return settings, records, fields


def test_every_endpoint_equals_its_measured_sequence_total():
    settings, records, fields = measured_population()
    data = figure_data(records, settings, fields)
    assert data["summary"]["publication_gate_passed"]
    assert len(data["methods"]) == 4
    for method in data["methods"]:
        assert len(method["outcomes"]) == 5
        for row in method["outcomes"]:
            assert row["cumulative_seconds"][-1] == row["sequence_seconds"]
            assert np.all(np.diff(row["cumulative_seconds"]) >= 0)
            assert sum(row["cost_components_seconds"].values()) == pytest.approx(
                row["sequence_seconds"]
            )


def test_failed_repetitions_stay_visible_without_component_comparison(tmp_path):
    settings, records, fields = measured_population()
    records[-1]["cases"][0].update(verified=False, status="linear_breakdown")
    records[-1].update(all_problems_verified=False, verified_problems=2)
    data = figure_data(records, settings, fields)
    assert not data["summary"]["publication_gate_passed"]
    reference = data["methods"][-1]
    assert len(reference["outcomes"]) == 5
    assert reference["representative_record_index"] is None
    assert reference["outcomes"][-1]["statuses"][0] == "linear_breakdown"
    plot(data, tmp_path)
    assert (tmp_path / "complete_sequences.pdf").stat().st_size > 1000
    assert (tmp_path / "complete_sequences.png").stat().st_size > 1000


def test_inconsistent_query_subintervals_are_rejected():
    _, records, _ = measured_population()
    records[0]["cases"][0]["seconds"] = records[0]["sequence_seconds"] + 1
    with pytest.raises(ValueError, match="Query intervals"):
        measured_intervals(records[0])


def test_component_display_uses_an_actual_sequence():
    settings, records, fields = measured_population()
    data = figure_data(records, settings, fields)
    for method in data["methods"]:
        chosen = method["representative_record_index"]
        assert records[chosen]["configuration"]["repetition"] == 2


def test_fixed_flow_figures_keep_the_physics_label(tmp_path):
    settings, records, fields = measured_population()
    for record in records:
        record["configuration"]["physics"] = "prescribed_flow"
        for case in record["cases"]:
            case["pdas_history"] = case.pop("history")[0]["attempts"][0]["qp_history"]
            case["pdas_steps"] = 1
    data = figure_data(records, settings, fields)
    assert data["physics"] == "prescribed_flow"
    plot(data, tmp_path)
    assert (tmp_path / "complete_sequences.png").stat().st_size > 1000
