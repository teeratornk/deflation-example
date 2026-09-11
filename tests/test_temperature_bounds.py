"""Keep bound illustrations separate, complete and tied to verified states."""

from pathlib import Path
import json
import shutil

import numpy as np
from omegaconf import OmegaConf
import pytest

from deflation_example.temperature_bounds import load_results, plot, run, settings


def configuration(**overrides):
    path = Path(__file__).parents[1] / "src/deflation_example/conf/temperature_bounds.yaml"
    c = OmegaConf.load(path)
    del c["hydra"]
    return settings(OmegaConf.merge(c, overrides))


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("bound-fields") / "result"
    c = configuration(level=0, rank=20, bounds=[0.1, 0.3], slabs=2, threads=2)
    result = run(c, root)
    assert result["success"]
    return root


@pytest.mark.parametrize(
    "change",
    [
        {"bounds": []},
        {"bounds": [0.2, 0.2]},
        {"bounds": [0.3, 0.1]},
        {"bounds": [-0.1]},
        {"bounds": [float("nan")]},
        {"query": 16},
        {"forms": ["steady", "steady"]},
        {"forms": ["unknown"]},
        {"cg_factor": 2},
        {"slabs": 0},
    ],
)
def test_invalid_population_is_rejected(change):
    with pytest.raises((ValueError, TypeError)):
        configuration(**change)


def test_all_bounds_use_the_same_target_operator_and_reference(results):
    report = load_results(results)
    assert len(report["cases"]) == 4
    for form in ("steady", "transient"):
        rows = [r for r in report["cases"] if r["form"] == form]
        assert len({r["target_sha256"] for r in rows}) == 1
        assert rows[0]["reference"]["objective"] >= rows[1]["reference"]["objective"]
        for row in rows:
            assert row["direct_state_relative_difference"] <= 1e-7
            assert row["reference"]["maximum_bound_violation"] <= 1e-8
            assert row["reference"]["forward_relative_error"] <= 1e-7
            assert all(r["original_residual"] <= 1e-10 for r in row["reference"]["inner"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "failed", "nan", "target", "summary"])
def test_changed_or_failed_records_cannot_be_plotted(results, tmp_path, change):
    root = tmp_path / "changed"
    shutil.copytree(results, root)
    path = root / "report.json"
    report = json.loads(path.read_text())
    if change == "missing":
        report["cases"].pop()
    elif change == "duplicate":
        report["cases"].append(report["cases"][0])
    elif change == "failed":
        report["cases"][0]["reference"]["status"] = "maxiter"
    elif change == "nan":
        report["cases"][0]["reference"]["kkt"]["stationarity"] = None
    elif change == "target":
        report["cases"][0]["target_sha256"] = "changed"
    else:
        report["cases"][0]["reference"]["state_maximum"] += 0.1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_results(root)


def test_changed_temperature_file_is_rejected(results, tmp_path):
    root = tmp_path / "changed"
    shutil.copytree(results, root)
    report = json.loads((root / "report.json").read_text())
    path = root / report["cases"][0]["reference"]["fields"]
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    arrays["state"] = np.minimum(arrays["state"], 0)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="Changed optimized fields"):
        load_results(root)


def test_plot_uses_all_optimized_states_on_one_color_scale(results, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *a, **kw: saved.append(figure))
    plot(results, tmp_path / "plot")
    axes = [a for a in saved[0].axes if a.get_title()]
    assert len(axes) == 4
    assert len({ax.collections[0].get_clim() for ax in axes}) == 1
    assert [ax.get_title() for ax in axes] == [
        "(a) $\\psi=0.1$",
        "(b) $\\psi=0.3$",
        "(c) $\\psi=0.1$",
        "(d) $\\psi=0.3$",
    ]


def test_existing_outputs_are_preserved(results):
    before = (results / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(configuration(), results)
    assert (results / "report.json").read_bytes() == before
