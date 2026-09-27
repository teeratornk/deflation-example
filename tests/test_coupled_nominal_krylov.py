"""The complete reference uses a fresh, fixed, verified initial trajectory."""

import numpy as np
from omegaconf import OmegaConf
import pytest

from deflation_example import coupled_sequence as sequence
from deflation_example.coupled_nominal_krylov import configured_krylov_reference
from test_coupled_derivatives import small_coupled_problem
from test_coupled_sequence import configuration


def setup():
    problem = small_coupled_problem(
        [0.2, 0.35], uniform_capacity=True, consistent=True, streamline_rule="smooth_p8"
    )
    cfg = configuration(problem)
    cfg.update(
        device="cpu",
        method="reference",
        rank=2,
        reference_krylov_steps=6,
        reference_krylov_seed=20260923,
        reference_selection="krylov_coupled",
        inner_preconditioner="frozen",
        frozen_sweeps=3,
    )
    cfg["queries"][1]["upper_K"] = cfg["queries"][0]["upper_K"]
    return problem, cfg


def test_reference_is_deterministic_and_unchanged_by_restriction():
    problem, cfg = setup()
    state = np.full(problem.size, 0.06)
    a = configured_krylov_reference(problem, cfg, initial_state=state)
    b = configured_krylov_reference(problem, cfg, initial_state=state)
    full = np.arange(problem.size)
    np.testing.assert_array_equal(a.restrict(full), b.restrict(full))
    expected = a.restrict(full).copy()
    a.restrict(full[::2])[:] = 0
    np.testing.assert_array_equal(a.restrict(full), expected)
    np.testing.assert_array_equal(state, np.full(problem.size, 0.06))
    assert a.rank == 2
    assert a.description["nominal_alpha"] == problem.alpha
    assert a.description["total_reference_construction_seconds"] >= (
        a.description["construction_seconds"]
        + a.description["nominal_evaluation_seconds"]
        + a.description["nominal_preconditioner_seconds"]
    )


def test_sequential_ablation_starts_from_identical_modes_and_loses_released_entries():
    problem, cfg = setup()
    cfg["reference_krylov_selection"] = "lowest"
    state = np.full(problem.size, 0.06)
    full = configured_krylov_reference(problem, cfg, initial_state=state)
    seq = configured_krylov_reference(
        problem, {**cfg, "reference_transfer": "sequential"}, initial_state=state
    )
    first, second = np.arange(problem.size)[::2], np.arange(problem.size)
    seq.begin_system(first)
    np.testing.assert_array_equal(seq.restrict(first), full.restrict(first))
    seq.begin_system(second)
    np.testing.assert_array_equal(seq.restrict(second)[::2], full.restrict(first))
    np.testing.assert_array_equal(seq.restrict(second)[1::2], 0)
    assert seq.reference is None
    assert seq.description["ritz_selection"] == "lowest"


@pytest.mark.parametrize("defect", ["nan", "bounds", "shape", "equations"])
def test_inconsistent_initial_temperature_or_equations_fail(monkeypatch, defect):
    problem, cfg = setup()
    state = np.full(problem.size, 0.06)
    if defect == "nan":
        state[0] = np.nan
    elif defect == "bounds":
        state[0] = 999
    elif defect == "shape":
        state = state[:-1]
    else:
        import deflation_example.coupled_nominal_krylov as construction

        monkeypatch.setattr(construction, "equation_acceptance", lambda *args: False)
    with pytest.raises(ValueError):
        configured_krylov_reference(problem, cfg, initial_state=state)


@pytest.mark.parametrize(
    "key,value",
    [
        ("device", "hybrid"),
        ("reference_transfer", "unknown"),
        ("inner_preconditioner", "jacobi"),
        ("frozen_sweeps", 5),
        ("rank", 7),
    ],
)
def test_changed_construction_policy_is_rejected(key, value):
    problem, cfg = setup()
    with pytest.raises(ValueError):
        configured_krylov_reference(problem, {**cfg, key: value})


def test_complete_sequence_constructs_only_once_and_matches_rank_zero(monkeypatch, tmp_path):
    problem, cfg = setup()
    cfg.update(
        memory_interval=0.01,
        repetition=0,
        recycle_window=2,
        inner_tolerance=1e-11,
        inner_cap=1000,
        threads=1,
        output=str(tmp_path / "reference"),
    )
    monkeypatch.setattr(
        sequence,
        "load_problem",
        lambda cfg: (
            problem,
            {
                "baseline_sha256": "test",
                "configuration": {},
                "input_sha256": {},
                "seconds": 0.0,
            },
        ),
    )
    import deflation_example.coupled_nominal_krylov as construction

    calls = []
    original = construction.configured_krylov_reference

    def build(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(construction, "configured_krylov_reference", build)
    ref = sequence.run(OmegaConf.create(cfg))
    assert ref["all_problems_verified"], ref
    assert calls == [1]
    plain = sequence.run(
        OmegaConf.create(
            {
                **cfg,
                "method": "jacobi",
                "rank": 0,
                "output": str(tmp_path / "plain"),
            }
        )
    )
    assert plain["all_problems_verified"], plain
    assert calls == [1]
    for position in range(2):
        with np.load(tmp_path / "reference" / f"target-{position:02d}.npz") as a:
            with np.load(tmp_path / "plain" / f"target-{position:02d}.npz") as b:
                np.testing.assert_allclose(a["state"], b["state"], atol=1e-6, rtol=0)
    assert ref["components_seconds"]["reference_construction"] > 0
    assert sum(ref["components_seconds"].values()) == pytest.approx(ref["sequence_seconds"])
