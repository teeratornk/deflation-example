"""Bundling, twin substitution and main-table rows of the prescribed-speedup-v21 evidence."""

import json

import pytest

from deflation_example import v21_evidence as v21

INNER_KEYS = (
    "basis_processing",
    "coarse_or_hierarchy_setup",
    "initialization",
    "handle_creation",
    "resource_creation",
    "iteration",
    "conversion",
    "upload",
    "download",
    "verification",
)


def timed_record(seconds, inner=100, success=True, statuses=("converged", "converged")):
    iteration = 0.5 * seconds / len(statuses)
    cases = []
    for k, status in enumerate(statuses):
        parts = dict.fromkeys(INNER_KEYS, 0.0)
        parts["iteration"] = iteration
        cases.append(
            {
                "status": status,
                "inner_iterations": inner // len(statuses),
                "outer_iterations": 2,
                "cumulative_seconds": seconds * (k + 1) / len(statuses),
                "inner": [
                    {
                        "iterations": inner // len(statuses),
                        "original_residual": 1e-11,
                        "components_seconds": parts,
                        "total_seconds": iteration,
                        "restriction_and_transfer_seconds": 0.0,
                    }
                ],
                "outer_history": [{"stationarity": 1e-9, "entered": 0, "left": 0}],
                "outer_timing": {
                    "components_seconds": {
                        "restriction": 0.0,
                        "inner_verification": 0.0,
                        "kkt_and_update": 0.0,
                    }
                },
            }
        )
    return {
        "success": success,
        "seconds": seconds,
        "components_seconds": {
            "assembly": 0.1 * seconds,
            "reference_construction": 0.0,
            "solver_resources": 0.0,
        },
        "cases": cases,
        "memory": {"peak_gpu_process_bytes": 2**30},
        "problem_size": 500,
        "environment": {"git_head": "abc", "source_tree_clean": True},
    }


def write_population(path, times, methods=("jacobi", "reference", "recycling", "amgx"), fail=()):
    path.mkdir(parents=True)
    repeats = len(next(iter(times.values())))
    (path / "protocol.json").write_text(
        json.dumps({"methods": list(methods), "repeats": repeats, "phase": "final"})
    )
    for method in methods:
        for repetition, seconds in enumerate(times[method]):
            ok = (method, repetition) not in fail
            record = timed_record(
                seconds,
                success=ok,
                statuses=("converged", "converged") if ok else ("converged", "maxiter"),
            )
            (path / f"{method}-{repetition}").mkdir()
            (path / f"{method}-{repetition}/record.json").write_text(json.dumps(record))
    (path / "attempts.json").write_text(
        json.dumps(
            [
                {"method": m, "repetition": r, "status": "completed"}
                for m in methods
                for r in range(repeats)
            ]
        )
    )


@pytest.fixture
def evidence(tmp_path):
    v20_runs, v21_runs, campaign = tmp_path / "v20", tmp_path / "v21", tmp_path / "campaign"
    write_population(
        v20_runs / "wave3/C-S4b-transformer-x4",
        {"jacobi": [30, 31], "reference": [20, 21], "recycling": [40, 41], "amgx": [35, 36]},
    )
    write_population(
        v21_runs / "wave2/C-S4b-transformer-x4-skew",
        {"jacobi": [30, 32], "reference": [10, 12], "recycling": [25, 27], "amgx": [50, 51]},
        fail={("amgx", 1)},
    )
    write_population(
        v21_runs / "wave8/G-engine-L4-subset",
        {"jacobi": [9, 9], "reference": [3, 3]},
        methods=("jacobi", "reference"),
    )
    for name in v21.CAMPAIGN_FILES:
        (campaign / name).parent.mkdir(parents=True, exist_ok=True)
        (campaign / name).write_text("{}")
    out = tmp_path / "evidence"
    v21.bundle([(v20_runs, ""), (v21_runs, v21.PREFIX)], campaign, out)
    return out


def test_v21_populations_are_prefixed_and_carry_cost_components(evidence):
    summary = v21.summarize(evidence)
    assert "wave3/C-S4b-transformer-x4" in summary
    assert "v21-wave2/C-S4b-transformer-x4-skew" in summary
    run = json.loads(
        (
            evidence / "populations/v21-wave2/C-S4b-transformer-x4-skew/runs/reference-0.json"
        ).read_text()
    )
    assert run["components"]["Iteration"] == pytest.approx(5.0)
    assert sum(run["components"].values()) == pytest.approx(run["seconds"])


def test_main_rows_substitute_twins_and_report_range_and_counts(evidence):
    summary = v21.summarize(evidence)
    spec = v21.twin([("Transformer", "4 slabs", "wave3/C-S4b-transformer-x4", "reference")])
    text, values = v21.main_rows(summary, spec)
    cells = [c.strip() for c in text.strip().rstrip("\\").split(" & ")]
    # Recycling 26 s against reference 11 s; range over repetitions 25/12 to 27/10.
    assert cells[4] == "recycling 26.0"
    assert cells[5] == "2.36 [2.08, 2.70]"
    assert cells[7] == "AmgX 1/2"
    assert values[0]["population"] == "v21-wave2/C-S4b-transformer-x4-skew"


def test_reduced_comparator_coverage_is_marked(evidence):
    summary = v21.summarize(evidence)
    text, _ = v21.main_rows(
        summary, [("Bore 4", "steady", "v21-wave8/G-engine-L4-subset", "reference")]
    )
    assert text.startswith("Bore 4$^\\dagger$")
    full, _ = v21.main_rows(summary, [("T", "x4", "wave3/C-S4b-transformer-x4", "reference")])
    assert "dagger" not in full


def test_a_changed_digest_is_refused(evidence):
    path = evidence / "populations/v21-wave2/C-S4b-transformer-x4-skew/runs/reference-0.json"
    path.write_text(path.read_text().replace('"seconds": 10', '"seconds": 1'))
    with pytest.raises(ValueError):
        v21.verify(evidence)


def test_every_advective_transformer_population_of_the_manuscript_has_a_skew_twin():
    specs = v21.SCALE_ROWS + v21.TRANSIENT_ROWS + v21.OPERATION_ROWS + v21.SI_ROWS + v21.SINGLE_ROWS
    for _, _, key, *_ in specs:
        if "transformer" in key and "-skew" not in key and "O1s" not in key and "O4s" not in key:
            raise AssertionError(f"{key} has no skew twin")
    assert all(
        key in v21.TWINS
        for _, _, key, _ in v21.ORIGINAL_ROWS
        if key.startswith(("wave3", "wave9/O5", "wave9/O3"))
    )


def test_rows_without_a_converged_reference_or_any_converged_method(tmp_path):
    runs, campaign = tmp_path / "v21", tmp_path / "campaign"
    write_population(
        runs / "wave2/failed-reference",
        {"jacobi": [8, 9], "reference": [3, 3], "recycling": [7, 7], "amgx": [6, 6]},
        fail={("reference", 0)},
    )
    write_population(
        runs / "wave2/all-failed",
        {"jacobi": [8, 9], "reference": [3, 3], "recycling": [7, 7], "amgx": [6, 6]},
        fail={(m, r) for m in ("jacobi", "reference", "recycling", "amgx") for r in (0, 1)},
    )
    for name in v21.CAMPAIGN_FILES:
        (campaign / name).parent.mkdir(parents=True, exist_ok=True)
        (campaign / name).write_text("{}")
    out = tmp_path / "evidence"
    v21.bundle([(runs, v21.PREFIX)], campaign, out)
    summary = v21.summarize(out)
    text, values = v21.main_rows(
        summary,
        [
            ("A", "ref fails", "v21-wave2/failed-reference", "reference"),
            ("B", "all fail", "v21-wave2/all-failed", None),
        ],
    )
    first, second = text.strip().split("\n")
    cells = [c.strip() for c in first.rstrip("\\").split(" & ")]
    assert cells[3] == "--" and cells[4] == "AmgX 6.0" and cells[5] == "--"
    assert cells[7] == "reference 1/2"
    assert "no method converges at every query" in second
    assert [v["converged"] for v in values] == [False, False]
