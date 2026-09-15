"""The pluggable coarse space leaves the host CG kernel arithmetically unchanged."""

import numpy as np
import pytest

from deflation_example.solvers import CpuCoarseSpace, deflated_cg


def spd_system(seed=11, size=24):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(size, size))
    H = B.T @ B + np.eye(size)
    f = rng.normal(size=size)
    U = np.linalg.eigh(H)[1]
    return H, f, U


def test_explicit_host_factory_is_bitwise_identical_to_default():
    H, f, U = spd_system()
    for basis, cache in ((U[:, :3], True), (U[:, :3], False), (None, True)):
        default = deflated_cg(H, f, basis, np.diag(H), refresh=7, cache_operator_product=cache)
        explicit = deflated_cg(
            H,
            f,
            basis,
            np.diag(H),
            refresh=7,
            cache_operator_product=cache,
            coarse_factory=CpuCoarseSpace,
        )
        assert default.status == explicit.status == "converged"
        assert default.iterations == explicit.iterations
        assert default.rank == explicit.rank
        assert default.coarse_condition == explicit.coarse_condition
        assert np.array_equal(default.x, explicit.x)


def test_factory_receives_operator_basis_and_limit_once_per_solve():
    H, f, U = spd_system(seed=3)
    calls = []

    def spy(A, basis, condition_limit):
        space = CpuCoarseSpace(A, basis, condition_limit)
        calls.append((A.shape, None if basis is None else basis.shape, condition_limit, space))
        return space

    result = deflated_cg(H, f, U[:, :4], np.diag(H), condition_limit=1e8, coarse_factory=spy)
    assert result.status == "converged" and result.rank == 4
    assert len(calls) == 1
    shape, basis_shape, limit, space = calls[0]
    assert shape == H.shape and basis_shape == (H.shape[0], 4) and limit == 1e8
    assert space.applications >= 1
    assert space.storage()["resident_operator_product_bytes"] == H.shape[0] * 4 * 8
    result = deflated_cg(H, f, None, np.diag(H), coarse_factory=spy)
    assert result.status == "converged" and result.rank == 0
    assert len(calls) == 2 and calls[1][3].applications == 0


def test_factory_breakdown_and_fallback_follow_host_semantics():
    H = np.diag([1.0, 100.0])
    limited = deflated_cg(H, np.ones(2), np.eye(2), np.diag(H), condition_limit=10)
    explicit = deflated_cg(
        H, np.ones(2), np.eye(2), np.diag(H), condition_limit=10, coarse_factory=CpuCoarseSpace
    )
    assert limited.rank == explicit.rank == 0
    assert limited.fallback_reason == explicit.fallback_reason == "coarse_condition_limit"

    class Broken(CpuCoarseSpace):
        def __init__(self, A, basis, condition_limit):
            super().__init__(A, basis, condition_limit)
            self.breakdown = True

    result = deflated_cg(H, np.ones(2), np.eye(2), np.diag(H), coarse_factory=Broken)
    assert result.status == "breakdown" and result.rank == 2

    with pytest.raises(ValueError):
        deflated_cg(H, np.ones(2), coarse_factory="not callable")


def test_sequence_runner_forwards_shared_inner_solver_options(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    from deflation_example import coupled_sequence as sequence
    from deflation_example.study_solvers import ArrayReference, StudySolver
    from test_coupled_derivatives import small_coupled_problem
    from test_coupled_sequence import configuration

    captured = {}

    class Recording(StudySolver):
        def __init__(self, method, **kwargs):
            captured.update(kwargs)
            super().__init__(method, **kwargs)

    problem = small_coupled_problem([0.2, 0.35], uniform_capacity=True)
    monkeypatch.setattr(sequence, "StudySolver", Recording)
    monkeypatch.setattr(
        sequence,
        "load_problem",
        lambda cfg: (
            problem,
            {"baseline_sha256": "test", "configuration": {}, "input_sha256": {}, "seconds": 0.25},
        ),
    )
    monkeypatch.setattr(
        sequence,
        "configured_reference",
        lambda *args: ArrayReference(np.eye(problem.size)[:, :3], {"construction": "test"}),
    )
    config = configuration(problem)
    config.update(
        method="reference",
        device="cpu",
        memory_interval=0.01,
        repetition=0,
        rank=3,
        recycle_window=6,
        inner_tolerance=1e-11,
        inner_cap=1000,
        inner_refresh=37,
        hybrid_block_min_columns=9,
        hybrid_coarse_device="cuda",
        threads=1,
        output=str(tmp_path / "sequence"),
    )
    report = sequence.run(OmegaConf.create(config))
    assert report["status"] == "complete", report.get("error_type")
    assert captured["refresh"] == 37
    # Hybrid-only settings must not reach the CPU control.
    assert "block_min_columns" not in captured and "coarse_device" not in captured
    assert report["configuration"]["inner_refresh"] == 37
