"""Check time-grid transfers, fixed-source replay and complete case accounting."""

import json
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf
import pytest

from deflation_example.temporal_resolution import (
    forward_replay,
    model,
    prepare,
    prolong_control,
    prolong_state,
    run_case,
    settings,
    summarize,
)


def config(**overrides):
    path = Path(__file__).parents[1] / "src/deflation_example/conf/temporal_resolution.yaml"
    c = OmegaConf.load(path)
    del c["hydra"]
    del c["defaults"]
    return settings(OmegaConf.merge(c, overrides))


@pytest.mark.parametrize(
    "override",
    [
        {"slabs": [2, 4]},
        {"slabs": [2, 4, 7]},
        {"slabs": [1, 2, 2]},
        {"slabs": [0, 1, 2]},
        {"nominal_query": 16},
        {"query_population": 1},
        {"replay_factors": [1, 2]},
        {"cg_factor": 2},
        {"initial_temperature": 2},
    ],
)
def test_invalid_design(override):
    with pytest.raises((ValueError, TypeError)):
        config(**override)


def test_state_interpolation_preserves_initial_value_and_all_original_nodes():
    original = np.array([[3.0, 5.0], [5.0, 9.0]])
    fine = prolong_state(original, np.array([1.0, 1.0]), 4)
    np.testing.assert_allclose(fine, [[2, 3], [3, 5], [4, 7], [5, 9]])
    np.testing.assert_array_equal(fine[1::2], original)
    np.testing.assert_array_equal(prolong_control(original, 4), np.repeat(original, 2, axis=0))
    for action in (
        lambda: prolong_control(original, 3),
        lambda: prolong_state(original, np.ones(2), 3),
    ):
        with pytest.raises(ValueError):
            action()


def test_same_grid_forward_replay_preserves_source_initial_and_forcing():
    c = config(
        geometry="engine_3d",
        alpha=1e-6,
        horizon=0.1,
        initial_temperature=0.03,
        slabs=[1, 2, 4],
        rank=5,
    )
    _, p = model(c, 4)
    u = np.random.default_rng(907).normal(size=(4, p.spatial_size))
    y, residual = forward_replay(p, u, 4)
    np.testing.assert_allclose(y.ravel(), p.forward(u.ravel()), rtol=1e-12, atol=1e-12)
    assert residual <= 1e-10


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    directory = tmp_path_factory.mktemp("temporal")
    c = config(
        root=str(directory / "runs"),
        geometry="engine_3d",
        alpha=1e-6,
        bound=0.3,
        horizon=0.1,
        slabs=[1, 2, 4],
        query_population=3,
        nominal_query=1,
        rank=5,
        threads=2,
        direct_check=True,
    )
    design = prepare(c)
    assert len(set(design["queries"])) == 2
    scores = design["desired_exceedance_scores"]
    eligible = [q for q in range(3) if q != 1]
    assert design["queries"][1] == max(eligible, key=lambda q: scores[q])
    for index in range(6):
        result = run_case({**c, "case": index})
        assert result["verified"], result
        assert result["replay"]["maximum_original_residual"] <= 1e-10
    return c


def test_summary_preserves_every_case_and_defines_finest_reference(completed, tmp_path):
    report = summarize({**completed, "output": str(tmp_path / "summary")})
    assert report["all_verified"] and len(report["rows"]) == 6
    for row in report["rows"]:
        assert row["replay"]["slabs"] == [8, 16]
        assert row["replay"]["maximum_bound_violation_K"] >= 0
        if row["slabs"] == 4:
            assert row["temperature_rms_difference_K"] == 0
            assert row["control_relative_rms_difference"] == 0
    with pytest.raises(FileExistsError):
        run_case(completed)


def test_missing_cases_remain_explicit(completed, tmp_path):
    root = tmp_path / "incomplete"
    root.mkdir()
    shutil.copy(Path(completed["root"]) / "design.json", root)
    report = summarize({**completed, "root": str(root), "output": str(tmp_path / "summary")})
    assert not report["all_verified"] and len(report["rows"]) == 6
    assert all(r["status"] == "missing" for r in report["rows"])


def test_changed_case_is_rejected(completed, tmp_path):
    root = tmp_path / "changed"
    shutil.copytree(completed["root"], root)
    path = root / "case-00/record.json"
    record = json.loads(path.read_text())
    record["query"] = 90
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="selected case"):
        summarize({**completed, "root": str(root), "output": str(tmp_path / "summary")})


def test_changed_protocol_cannot_run(completed):
    with pytest.raises(ValueError, match="predeclared"):
        run_case({**completed, "bound": 0.35})
