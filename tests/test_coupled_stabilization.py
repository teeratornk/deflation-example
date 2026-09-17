"""The stabilisation report must read its records and refuse a vacuous comparison."""

import json

import numpy as np
import pytest

from deflation_example.coupled_stabilization import (
    build,
    conditioning,
    convergence,
    macros,
    replay_cost,
)


def convergence_record(tmp_path, unstabilised=(5e-14, 1.6e-14, 6e-15)):
    directory = tmp_path / "rates"
    directory.mkdir()

    def case(name, errors, streamline, consistent, bound=None):
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        return {
            "case": name,
            "streamline": streamline,
            "consistent": consistent,
            "divisions": [4, 8, 16],
            "maximum_errors_K": list(errors),
            "error_ratios": ratios,
            "observed_order": [float(np.log2(r)) for r in ratios],
            "row_bound": bound,
        }

    (directory / "record.json").write_text(
        json.dumps(
            {
                "schema": "coupled-stabilisation-convergence-v1",
                "speed_m_s": 40.0,
                "conductivity_W_m_K": 0.01,
                "cases": [
                    case("unstabilised", unstabilised, False, False),
                    case("declared_symmetric_term", (2.3e-1, 1.5e-1, 8.5e-2), True, False),
                    case(
                        "residual_weighted",
                        (2.0e-2, 4.8e-3, 1.2e-3),
                        True,
                        True,
                        {
                            "cells": 512,
                            "cells_bounded": 512,
                            "smallest_factor": 0.33,
                            "median_factor": 0.33,
                        },
                    ),
                ],
            }
        )
    )
    return directory


def conditioning_record(tmp_path, name, bad, pivot, bound=None):
    directory = tmp_path / name
    directory.mkdir()

    def checks(matrix, minimum, diagonal):
        return {
            "matrix": matrix,
            "rows": 10830,
            "row_sum_minimum": minimum,
            "row_sum_maximum": 1.0,
            "rows_with_nonpositive_sum": bad,
            "diagonal_minimum": diagonal,
            "rows_with_nonpositive_diagonal": 0,
            "dominance_slack_minimum": -1.0,
            "rows_not_diagonally_dominant": 7921,
        }

    entry = {
        "streamline_length": "edge",
        "checks": [
            checks("source_action", -0.003 if bad else 4e-6, 1e-5),
            checks("storage_over_step", -4252.0 if bad else 5.7, -654.0 if bad else 6.6),
            checks("step_operator", -4252.0 if bad else 5.7, 33.8),
        ],
        "step_operator_factorisation": {"pivot_magnitude_ratio": pivot},
    }
    if bound is not None:
        entry["row_limit"] = bound
    (directory / "record.json").write_text(
        json.dumps(
            {
                "schema": "coupled-consistent-storage-conditioning-v1",
                "lengths": [entry, {**entry, "streamline_length": "flow"}],
            }
        )
    )
    return directory


def replay_record(tmp_path, name, consistent, statuses, iterations):
    directory = tmp_path / name
    directory.mkdir()
    steps = [
        {
            "slab_zero_based": n,
            "status": status,
            "seconds": 20.0,
            "history": [{}] * (count + 1),
        }
        for n, (status, count) in enumerate(zip(statuses, iterations, strict=True))
    ]
    (directory / "record.json").write_text(
        json.dumps(
            {
                "schema": "coupled-fixed-source-newton-replay-v1",
                "status": "converged" if all(s == "converged" for s in statuses) else statuses[-1],
                "forward_slabs": len(steps),
                "configuration": {"consistent_stabilization": consistent},
                "steps": steps,
            }
        )
    )
    return directory


def test_the_report_reads_the_correction_from_its_records(tmp_path):
    report = build(
        convergence_record(tmp_path),
        conditioning_record(tmp_path, "before", 35, 1.5e-4),
        conditioning_record(
            tmp_path,
            "after",
            0,
            1.06e-3,
            {
                "fluid_cells": 14848,
                "cells_limited": 14828,
                "fraction_limited": 0.9987,
                "smallest_factor": 0.009,
                "median_factor": 0.457,
            },
        ),
        replays=[replay_record(tmp_path, "shipped", False, ["converged"] * 4, [4, 5, 4, 3])],
    )
    values = macros(report)
    # An operator that is exact on this field has no order to report.
    assert "coupledStabUnstabilisedOrder" not in values
    assert values["coupledStabUnstabilisedFinestError"] == "6.00e-15"
    assert float(values["coupledStabShippedOrder"]) < 1
    assert float(values["coupledStabWeightedOrder"]) >= 2
    assert report["convergence"]["order_gain"] > 1
    assert values["coupledStabUnboundedBadRows"] == "35"
    assert values["coupledStabBoundedBadRows"] == "0"
    assert values["coupledStabTotalRows"] == "10830"
    assert float(values["coupledStabPivotImprovement"]) > 6
    assert values["coupledStabShippedNewtonMedian"] == "4"
    assert report["conditioning"]["lengths_agree_once_bounded"] == {"edge": 0, "flow": 0}


def test_a_comparison_that_would_prove_nothing_is_refused(tmp_path):
    """If the unstabilised operator is not exact, the error is not all artificial."""
    with pytest.raises(ValueError, match="not exact on this field"):
        convergence(convergence_record(tmp_path, unstabilised=(1e-3, 5e-4, 2e-4)))


def test_a_stalled_replay_reports_where_it_stopped(tmp_path):
    row = replay_cost(
        replay_record(
            tmp_path,
            "stalled",
            True,
            ["converged", "converged", "newton_line_search_stagnation"],
            [55, 34, 41],
        )
    )
    assert row["consistent_stabilization"] is True
    assert row["reached_the_horizon"] is False
    assert row["terminating_status"] == "newton_line_search_stagnation"
    # Only the converged slabs contribute a cost; the stalled one is not a cost.
    assert row["converged_slabs"] == 2
    assert row["median_newton_iterations"] == pytest.approx(44.5)


def test_the_streamline_length_must_be_in_both_records(tmp_path):
    with pytest.raises(ValueError, match="streamline length"):
        conditioning(
            conditioning_record(tmp_path, "before", 35, 1.5e-4),
            conditioning_record(tmp_path, "after", 0, 1.06e-3),
            length="spectral",
        )


def test_the_stabilisation_macros_are_written_and_gated(tmp_path):
    """The subsection is included only when its own macro file exists."""
    from deflation_example.coupled_presentation import write_stabilization

    output = tmp_path / "artifacts"
    output.mkdir()
    write_stabilization({"stabilization": None}, output)
    assert not (output / "stabilization.tex").exists()

    report = build(
        convergence_record(tmp_path),
        conditioning_record(tmp_path, "before", 35, 1.5e-4),
        conditioning_record(
            tmp_path,
            "after",
            0,
            1.06e-3,
            {
                "fluid_cells": 14848,
                "cells_limited": 14828,
                "fraction_limited": 0.9987,
                "smallest_factor": 0.009,
                "median_factor": 0.457,
            },
        ),
    )
    write_stabilization({"stabilization": report}, output)
    written = (output / "stabilization.tex").read_text()
    assert written.startswith("\\newcommand{\\coupledStab")
    assert written.count("\\newcommand") == len(macros(report))
    assert "coupledStabUnstabilisedOrder" not in written
