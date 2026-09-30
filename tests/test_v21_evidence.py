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
                "kkt": {"stationarity": 1e-9, "primal": 0.0, "dual": 2e-10},
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


def test_transfer_rows_follow_the_supporting_table_format(tmp_path):
    def trace(energy, iterations):
        repetition = {
            "status": "converged",
            "original_residual": 1e-11,
            "iterations": iterations,
            "seconds": 0.5,
            "deployed_rank": 100,
        }
        method = {
            "repetitions": [repetition] * 3,
            "correction": {"coarse_removed_energy_fraction": energy},
        }
        return {
            "success": True,
            "source_controls": {"rtol": 1e-10},
            "rows": [
                {
                    "methods": {"full_reference": method, "sequential_transfer": method},
                    "newly_active": 2,
                    "newly_inactive": 5,
                }
            ]
            * 2,
        }

    for path in v21.TRANSFERS.values():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(json.dumps(trace(0.25, 40)))
    text, records = v21.transfer_rows(tmp_path)
    first = text.splitlines()[0]
    assert first == r"Bore 2, steady & 2 & 4/10 & Full reference & 2 & 100 & 0.250 & 80 & 1.000 \\"
    assert text.splitlines()[1].startswith(" &  &  & Sequential transfer")
    assert len(records) == 6 and records[0]["newly_inactive"] == 10


def test_complete_rows_and_accuracy_of_the_primary_cases(evidence, monkeypatch):
    monkeypatch.setattr(
        v21, "PRIMARY_ROWS", [("Transformer", "4 slabs", "v21-wave2/C-S4b-transformer-x4-skew")]
    )
    summary = v21.summarize(evidence)
    text, _ = v21.primary_complete_rows(summary)
    cells = [c.strip() for c in text.strip().removesuffix(r"\\[4pt]").split(" & ")]
    assert cells[:3] == ["Transformer", "4 slabs", "100"]
    assert cells[4] == r"\shortstack{11.000\\{[10.000, 12.000]}\\1.000}"
    assert cells[6] == r"\shortstack{50.000 (1/2)\\{[50.000, 50.000]}\\1.000}"
    sentence, values = v21.primary_accuracy(evidence)
    assert values["kkt"] == pytest.approx(1e-9) and values["residual"] == pytest.approx(1e-11)
    assert r"$1.00\times10^{-9}$" in sentence


def test_certificate_rows_report_operator_inertia_and_stability(tmp_path):
    def certificate(geometry, level, form, negative):
        return {
            "geometry": geometry,
            "level": level,
            "transport_form": form,
            "free_nodes": 10830,
            "operator": {"negative": negative, "zero": 0, "positive": 10830 - negative},
            "transport": {"negative": 4753, "zero": 0, "positive": 6077},
            "energy_stable": negative == 0,
        }

    path = tmp_path / v21.CERTIFICATES
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "source_tree_clean": True,
                "certificates": [
                    certificate("transformer_2d", 0, "advective", 11),
                    certificate("engine_3d", 3, "advective", 0),
                ],
            }
        )
    )
    text, _ = v21.certificate_rows(tmp_path)
    first, second = text.strip().splitlines()
    assert first == r"Transformer & advective & 10{,}830 & 11/0/10819 & 4{,}753 & no \\"
    assert second.startswith("Bore 3 & advective") and second.endswith(r"& yes \\")


def test_records_without_inner_components_keep_their_kkt(tmp_path):
    record = timed_record(5.0)
    for case in record["cases"]:
        for row in case["inner"]:
            del row["components_seconds"]
    (tmp_path / "direct-0").mkdir()
    (tmp_path / "direct-0/record.json").write_text(json.dumps(record))
    digest = v21._with_components(tmp_path, {"accepted": True, "source": "direct-0/record.json"})
    assert digest["components"] is None and digest["max_kkt"] == pytest.approx(1e-9)


def test_diverged_replays_are_printed_as_powers_of_ten():
    assert v21._kelvin([0.4, 2.871]) == "2.87"
    assert v21._kelvin([9.27e145]) == r"$9.3\times10^{145}$"
    assert v21._kelvin([]) == "--"
