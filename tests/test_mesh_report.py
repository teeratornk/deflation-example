import json

import numpy as np
from omegaconf import OmegaConf
import pytest
from threadpoolctl import threadpool_limits

from deflation_example.benchmark_mesh import controls, sequence
from deflation_example.mesh_presentation import summarize
from deflation_example.mesh_report import components, generate, plot_sequences


@pytest.fixture
def records(tmp_path):
    c = controls(OmegaConf.create({"methods": ["jacobi"], "targets": 1, "save_fields": False}))
    c.pop("output")
    with threadpool_limits(2):
        record, _, _ = sequence(c, "jacobi")
    assert record["success"]
    record.update(
        controls=c, repetition=0, startup_after_import_seconds=0.1, finalization_seconds=0.01
    )
    source = tmp_path / "records"
    (source / "jacobi-0").mkdir(parents=True)
    (source / "protocol.json").write_text(json.dumps(c))
    (source / "attempts.json").write_text(
        json.dumps(
            [
                {
                    "method": "jacobi",
                    "repetition": 0,
                    "status": "completed",
                    "process_wall_seconds": record["seconds"] + 0.2,
                    "record": "jacobi-0/record.json",
                }
            ]
        )
    )
    (source / "jacobi-0/record.json").write_text(json.dumps(record))
    return source


def test_report_reconstructs_complete_cost_and_acceptance(records, tmp_path):
    result = generate([records], tmp_path / "report", plots=False)[0]
    m = result["methods"][0]
    assert m["accepted"] == m["requested"] == 1
    assert m["peak_host_bytes"] == []
    assert m["complete_memory_measurements"] == 0
    np.testing.assert_allclose(sum(m["components_seconds"][0].values()), m["complete_seconds"][0])
    assert m["max_kkt"] <= 1e-8 and m["max_original_residual"] <= 1e-10
    assert (tmp_path / "report/complete_rows.tex").exists()


@pytest.mark.parametrize("change", ["duplicate", "label", "residual", "time", "nan_kkt"])
def test_report_rejects_inconsistent_records(records, change):
    path = records / "jacobi-0/record.json"
    r = json.loads(path.read_text())
    if change == "duplicate":
        attempts = records / "attempts.json"
        rows = json.loads(attempts.read_text())
        attempts.write_text(json.dumps(rows + rows))
    elif change == "label":
        r["repetition"] = 1
    elif change == "residual":
        r["cases"][0]["inner"][0]["original_residual"] = 1e-5
    elif change == "time":
        r["cases"][0]["cumulative_seconds"] = r["seconds"] + 10
    else:
        r["cases"][0]["kkt"]["primal"] = None
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        summarize(records)


def test_process_failure_retains_an_accepted_numerical_record_as_failed_attempt(records, tmp_path):
    path = records / "attempts.json"
    attempts = json.loads(path.read_text())
    attempts[0]["status"] = "timeout"
    path.write_text(json.dumps(attempts))
    report = generate([records], tmp_path / "report", plots=False)[0]
    assert report["methods"][0]["accepted"] == 0
    assert report["methods"][0]["failures"] == ["timeout"]


def test_component_check_rejects_nested_timer_disagreement(records):
    r = json.loads((records / "jacobi-0/record.json").read_text())
    r["cases"][0]["inner"][0]["components_seconds"]["iteration"] += 10
    with pytest.raises(ValueError, match="Inner timing"):
        components(r)


@pytest.mark.parametrize("targets", [1, 4, 16, 17])
def test_sequence_plot_counts_completed_targets_and_preserves_repetitions(
    targets, tmp_path, monkeypatch
):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved.append(figure))
    report = {
        "controls": {
            "geometry": "engine_3d",
            "level": 1,
            "transient": False,
            "rank": 100,
            "targets": targets,
            "repeats": 5,
        }
    }
    records = {
        method: [
            {
                "initial_setup_seconds": 0.25,
                "cases": [
                    {"cumulative_seconds": (i + 1) * slope + rep * 0.01} for i in range(targets)
                ],
                "seconds": targets * slope + rep * 0.01 + 0.2,
            }
            for rep in range(5)
        ]
        for method, slope in (("jacobi", 2), ("reference", 1), ("recycling", 3), ("amgx", 4))
    }
    plot_sequences([(report, records)], tmp_path / "plot.pdf")
    axis = saved[0].axes[0]
    ticks = axis.get_xticks()
    assert axis.get_xlabel() == "Completed targets"
    assert ticks[0] == 0 and ticks[-1] == targets
    assert all(float(tick).is_integer() for tick in ticks)
    if targets == 16:
        np.testing.assert_array_equal(ticks, [0, 4, 8, 12, 16])
    assert len(axis.lines) == 4 * 6  # Five repetitions and their pointwise median.
    for method_index, method in enumerate(records):
        for rep in range(5):
            line = axis.lines[method_index * 6 + rep]
            assert line.get_xdata()[-1] == targets
            assert line.get_ydata()[-1] == records[method][rep]["seconds"]
    assert "Fastest alternative: Jacobi-CG" in axis.texts[0].get_text()
