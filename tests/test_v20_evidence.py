"""Digests, readings and manuscript rows of the prescribed-speedup-v20 evidence."""

import json

import pytest

from deflation_example import v20_evidence as evidence


def mesh_record(seconds, inner, success=True, statuses=("converged", "converged"), level=0):
    cases = [
        {
            "status": status,
            "inner_iterations": inner // len(statuses),
            "outer_iterations": 3,
            "cumulative_seconds": seconds * (k + 1) / len(statuses),
            "inner": [{"iterations": inner // len(statuses), "original_residual": 1e-11}],
            "outer_history": [
                {"stationarity": 1e-9, "entered": 2, "left": 1},
                {"stationarity": 3e-9, "entered": 0, "left": 0},
            ],
        }
        for k, status in enumerate(statuses)
    ]
    return {
        "success": success,
        "seconds": seconds,
        "cases": cases,
        "memory": {"peak_gpu_process_bytes": 2**30},
        "problem_size": 1000,
        "mesh": {"reference_level": level},
        "environment": {"git_head": "abc", "source_tree_clean": True},
    }


def write_mesh(path, runs, repeats, methods, reference_level=0, attempts=None):
    path.mkdir(parents=True)
    controls = {
        "methods": methods,
        "repeats": repeats,
        "phase": "final",
        "reference_level": reference_level,
    }
    (path / "protocol.json").write_text(json.dumps(controls))
    for (method, repetition), record in runs.items():
        (path / f"{method}-{repetition}").mkdir()
        (path / f"{method}-{repetition}/record.json").write_text(json.dumps(record))
    rows = attempts or [{"method": m, "repetition": r, "status": "completed"} for (m, r) in runs]
    (path / "attempts.json").write_text(json.dumps(rows))


def cht_record(seconds, inner, success=True, status="complete"):
    return {
        "success": success,
        "status": status if not success else "complete",
        "total_seconds": seconds,
        "inner_iterations": inner,
        "outer_iterations": 10,
        "memory": {"peak_gpu_process_bytes": 2**31},
        "cases": [{"status": "converged", "cumulative_seconds": seconds, "inner": []}],
    }


def write_cht(path, runs, repeats, methods, n=4, problem="steady"):
    path.mkdir(parents=True)
    controls = {
        "methods": methods,
        "repeats": repeats,
        "phase": "final",
        "n": n,
        "slabs": 8,
        "problem": problem,
    }
    (path / "declared_protocol.json").write_text(json.dumps({"controls": controls}))
    for (method, repetition), record in runs.items():
        name = f"sequence-{repetition}-outer_inner-{method}.json"
        (path / name).write_text(json.dumps(record))


def campaign(tmp_path):
    directory = tmp_path / "campaign"
    directory.mkdir()
    for name in evidence.CAMPAIGN_FILES:
        (directory / name).write_text("{}\n")
    return directory


@pytest.fixture
def tree(tmp_path):
    runs = tmp_path / "runs"
    # Body-fitted: reference 10/12/14 s, Jacobi 20/24/22 s, recycling 30 s once and
    # one failed repetition, AmgX never recorded for repetition 2.
    write_mesh(
        runs / "wave1/M",
        {
            ("reference", 0): mesh_record(10.0, 100),
            ("reference", 1): mesh_record(12.0, 120),
            ("reference", 2): mesh_record(14.0, 140),
            ("jacobi", 0): mesh_record(20.0, 400),
            ("jacobi", 1): mesh_record(24.0, 480),
            ("jacobi", 2): mesh_record(22.0, 440),
            ("recycling", 0): mesh_record(30.0, 300),
            ("recycling", 1): mesh_record(
                9.0, 90, success=False, statuses=("converged", "maxiter")
            ),
            ("amgx", 0): mesh_record(40.0, 10),
            ("amgx", 1): mesh_record(41.0, 10),
        },
        3,
        ["jacobi", "reference", "recycling", "amgx"],
        attempts=[
            {"method": "amgx", "repetition": 2, "status": "timeout"},
        ],
    )
    # Cartesian: reference 5 s, Jacobi 8 s, AmgX cycling (not accepted).
    write_cht(
        runs / "wave1/C",
        {
            ("reference", 0): cht_record(5.0, 50),
            ("jacobi", 0): cht_record(8.0, 200),
            ("amgx", 0): cht_record(3.0, 5, success=False, status="sequence_failed"),
        },
        1,
        ["jacobi", "reference", "amgx"],
    )
    # A split population: one method and one repetition per directory.
    for arm, method, level, seconds in (
        ("jacobi", "jacobi", 0, 50.0),
        ("ref2", "reference", 2, 25.0),
        ("ref3", "reference", 3, 80.0),
    ):
        for repetition in range(2):
            write_mesh(
                runs / f"wave8/G-engine-L4-steady-o400-{arm}-rep{repetition}",
                {(method, 0): mesh_record(seconds + repetition, 1000, level=level)},
                1,
                [method],
                reference_level=level,
            )
    # A tightened AmgX population augmenting wave8/G-steady96.
    write_cht(
        runs / "wave8/G-steady96",
        {("reference", 0): cht_record(10.0, 100), ("jacobi", 0): cht_record(18.0, 400)},
        1,
        ["jacobi", "reference"],
    )
    write_cht(runs / "wave10/H-amgx001-steady96", {("amgx", 0): cht_record(14.0, 20)}, 1, ["amgx"])
    out = tmp_path / "evidence"
    evidence.bundle(runs, campaign(tmp_path), out)
    return out


def test_body_fitted_readings_use_accepted_repetitions_only(tree):
    summary = evidence.summarize(tree)["wave1/M"]
    rows, reading = summary["methods"], summary["readings"]["reference"]
    assert rows["reference"]["median_seconds"] == 12.0
    assert rows["jacobi"]["median_seconds"] == 22.0
    assert rows["recycling"] == {**rows["recycling"], "accepted": 1, "declared": 3}
    assert rows["recycling"]["median_seconds"] == 30.0
    assert rows["amgx"]["accepted"] == 2 and "timeout" in rows["amgx"]["failures"]
    assert reading["fastest_alternative"] == "jacobi"
    assert reading["ratio"] == pytest.approx(22.0 / 12.0)
    assert reading["jacobi_iteration_ratio"] == pytest.approx(440 / 120)


def test_a_failed_alternative_is_not_the_fastest(tree):
    reading = evidence.summarize(tree)["wave1/C"]["readings"]["reference"]
    assert reading["fastest_alternative"] == "jacobi"
    assert reading["ratio"] == pytest.approx(8.0 / 5.0)


def test_split_populations_are_read_together_by_reference_level(tree):
    summary = evidence.summarize(tree)["wave8/G-engine-L4-steady-o400"]
    assert set(summary["methods"]) == {"jacobi", "reference-ref2", "reference-ref3"}
    assert summary["methods"]["reference-ref2"]["declared"] == 2
    assert summary["methods"]["reference-ref2"]["median_seconds"] == 25.5
    assert summary["readings"]["reference-ref2"]["ratio"] == pytest.approx(50.5 / 25.5)
    assert summary["readings"]["reference-ref3"]["ratio"] == pytest.approx(50.5 / 80.5)


def test_a_tightened_amgx_is_an_added_alternative(tree):
    summary = evidence.summarize(tree)["wave8/G-steady96"]
    assert summary["methods"]["amgx001"]["median_seconds"] == 14.0
    reading = summary["readings"]["reference"]
    assert reading["fastest_alternative"] == "amgx001"
    assert reading["ratio"] == pytest.approx(1.4)


def test_digests_carry_the_source_hash(tree, tmp_path):
    digest = json.loads((tree / "populations/wave1/M/runs/reference-1.json").read_text())
    source = (tmp_path / "runs/wave1/M/reference-1/record.json").read_bytes()
    assert digest["source_sha256"] == evidence._sha(source)
    missing = json.loads((tree / "populations/wave1/M/runs/amgx-2.json").read_text())
    assert missing["accepted"] is False and missing["source"] is None


def test_a_changed_evidence_file_is_refused(tree):
    path = tree / "populations/wave1/M/runs/reference-0.json"
    path.write_text(path.read_text().replace('"seconds": 10.0', '"seconds": 1.0'))
    with pytest.raises(ValueError, match="hash mismatch"):
        evidence.summarize(tree)


def test_an_unlisted_evidence_file_is_refused(tree):
    (tree / "populations/extra.json").write_text("{}")
    with pytest.raises(ValueError, match="different files"):
        evidence.verify(tree)


def test_the_bundle_destination_must_be_new(tree, tmp_path):
    with pytest.raises(ValueError, match="must be new"):
        evidence.bundle(tmp_path / "runs", tmp_path / "campaign", tree)


def test_table_rows_state_the_reading(tree):
    summary = evidence.summarize(tree)
    text, values = evidence.table_rows(
        summary, [("Bore", "level 4", "wave8/G-engine-L4-steady-o400", "reference-ref2")]
    )
    assert [v["ratio"] for v in values] == [pytest.approx(50.5 / 25.5)]
    assert text == r"Bore & level 4 & 1{,}000 & 25.5 & Jacobi-CG 50.5 & 1.98 & 1.00 & -- \\" + "\n"
    with pytest.raises(ValueError, match="No population"):
        evidence.table_rows(summary, [("x", "y", "wave1/missing", "reference")])
    with pytest.raises(ValueError, match="No reading"):
        evidence.table_rows(summary, [("x", "y", "wave1/C", "block_reference")])


def test_a_population_without_a_completed_method_is_stated(tmp_path):
    runs = tmp_path / "runs"
    write_cht(
        runs / "wave6/F",
        {
            ("reference", 0): cht_record(1.0, 1, success=False, status="sequence_failed"),
            ("jacobi", 0): cht_record(1.0, 1, success=False, status="sequence_failed"),
        },
        1,
        ["jacobi", "reference"],
    )
    evidence.bundle(runs, campaign(tmp_path), tmp_path / "evidence")
    summary = evidence.summarize(tmp_path / "evidence")
    text, values = evidence.table_rows(summary, [("T", "a", "wave6/F", None)])
    assert values == [] and "no method completes" in text
    with pytest.raises(ValueError, match="A method completes"):
        evidence.table_rows(
            {"wave6/F": {**summary["wave6/F"], "methods": {"jacobi": {"accepted": 1}}}},
            [("T", "a", "wave6/F", None)],
        )


def test_incomplete_methods_are_named_and_the_reference_must_complete(tree):
    summary = evidence.summarize(tree)
    text, _ = evidence.table_rows(summary, [("M", "d", "wave1/M", "reference")])
    assert text.rstrip().endswith(r"& AmgX, recycling \\")
    broken = {
        "wave1/M": {
            **summary["wave1/M"],
            "methods": {
                **summary["wave1/M"]["methods"],
                "reference": {**summary["wave1/M"]["methods"]["reference"], "accepted": 2},
            },
        }
    }
    with pytest.raises(ValueError, match="unaccepted repetition"):
        evidence.table_rows(broken, [("M", "d", "wave1/M", "reference")])


def test_campaign_counts_do_not_repeat_augmented_runs(tree):
    runs = {k: p["methods"] for k, p in evidence.populations(tree).items()}
    populations, sequences, solves = evidence.campaign_counts(runs, ["wave8/G-steady96"])
    assert (populations, sequences, solves) == (1, 3, 3)


def test_digests_carry_the_final_outer_state_and_source(tree):
    digest = json.loads((tree / "populations/wave1/M/runs/reference-0.json").read_text())
    case = digest["cases"][0]
    assert case["settled"] is True
    assert case["final_stationarity"] == 3e-9
    assert case["max_residual"] == 1e-11
    assert (digest["git_head"], digest["source_tree_clean"]) == ("abc", True)


def test_scientific_values_are_latex_math():
    assert evidence._sci(1.04e-8) == r"1.0\times10^{-8}"
    assert evidence._sci(9.4e-8) == r"9.4\times10^{-8}"


def test_temporal_rows_read_verified_cases_and_moduli(tmp_path):
    def summary(verified, change):
        rows = [
            {
                "slabs": n,
                "verified": verified,
                "temperature_max_difference_K": 0.1 * (64 // n) if n < 64 else 0.0,
                "replay": {"temperature_change_max_K": change},
            }
            for n in (4, 8, 16, 32, 64)
        ]
        return {"rows": rows}

    def stability(moduli):
        return {
            "rows": [
                {
                    "transport_form": form,
                    "largest_computed_amplification_modulus": m,
                    "status": "eigenpair_verified",
                }
                for form, values in moduli.items()
                for m in values
            ]
        }

    for _, form, folder in evidence.TEMPORAL:
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "summary.json").write_text(
            json.dumps(summary(form == "skew", 0.4 if form == "skew" else 2e150))
        )
    for folder in evidence.STABILITY.values():
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "stability.json").write_text(
            json.dumps(stability({"skew": [0.5, 0.8], "advective": [0.6, 15.2]}))
        )
    text, values = evidence.temporal_rows(tmp_path)
    lines = text.splitlines()
    assert lines[1] == r"600 s & skew & 5/5 & 0.80 & 0.40 & 0.80 \\"
    assert lines[0] == r"600 s & advective & 0/5 & -- & -- & 15.20 \\"
    assert values[("1 h", "skew")]["change"] == 0.4


def test_a_missing_difference_to_an_unverified_finest_grid_prints_a_dash(tmp_path):
    rows = [
        {"slabs": n, "verified": n < 32, "replay": {"temperature_change_max_K": 1e152}}
        for n in (4, 8, 16, 32, 64)
    ]
    stability = {
        "rows": [
            {
                "transport_form": form,
                "largest_computed_amplification_modulus": 0.5,
                "status": "eigenpair_verified",
            }
            for form in ("advective", "skew")
        ]
    }
    for _, _, folder in evidence.TEMPORAL:
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "summary.json").write_text(json.dumps({"rows": rows}))
    for folder in evidence.STABILITY.values():
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "stability.json").write_text(json.dumps(stability))
    text, values = evidence.temporal_rows(tmp_path)
    assert text.splitlines()[0] == r"600 s & advective & 3/5 & -- & $1.0\times10^{152}$ & 0.50 \\"
    assert values[("600 s", "advective")]["difference"] is None


def test_a_per_slab_run_reads_against_its_global_twin(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    write_cht(
        runs / "wave2/G",
        {("reference", 0): cht_record(48.8, 100), ("jacobi", 0): cht_record(37.6, 300)},
        1,
        ["jacobi", "reference"],
        problem="transient",
    )
    write_cht(runs / "wave2/P", {("reference", 0): cht_record(48.2, 90)}, 1, ["reference"])
    evidence.bundle(runs, campaign(tmp_path), tmp_path / "evidence")
    monkeypatch.setattr(
        evidence, "PAIRED", {"wave2/G": [("wave2/P", "reference", "reference-perslab6")]}
    )
    readings = evidence.summarize(tmp_path / "evidence")["wave2/G"]["readings"]
    assert readings["reference"]["ratio"] == pytest.approx(37.6 / 48.8)
    assert readings["reference-perslab6"]["fastest_alternative"] == "jacobi"
    assert readings["reference-perslab6"]["ratio"] == pytest.approx(37.6 / 48.2)
