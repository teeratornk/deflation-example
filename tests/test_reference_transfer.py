"""Zero-extension transfer of a full-domain reference: the retention ablation."""

import numpy as np
from omegaconf import OmegaConf
import pytest
from threadpoolctl import threadpool_limits

from deflation_example import benchmark_mesh as study
from deflation_example.recycling import transfer_basis
from deflation_example.reference_transfer import ZeroExtensionReference
from deflation_example.study_solvers import ArrayReference


@pytest.fixture(autouse=True)
def bounded_blas_threads():
    with threadpool_limits(2):
        yield


def inactive_sequence(rng, size, steps):
    sets = []
    for _ in range(steps):
        mask = rng.random(size) < 0.6
        mask[rng.integers(size)] = True
        sets.append(np.flatnonzero(mask))
    return sets


def expected_bases(full, sets):
    basis = full[sets[0]]
    yield basis
    for previous, current in zip(sets, sets[1:]):
        basis = transfer_basis(basis, previous, current)
        yield basis


def test_first_restriction_is_direct_and_later_ones_are_zero_extensions():
    rng = np.random.default_rng(3)
    full = rng.standard_normal((40, 5))
    reference = ZeroExtensionReference(ArrayReference(full, {"name": "test"}))
    sets = inactive_sequence(rng, 40, 6)
    for indices, expected in zip(sets, expected_bases(full, sets)):
        reference.begin_system(indices)
        np.testing.assert_array_equal(reference.restrict(indices), expected)
    assert reference.description["transfer_policy"] == "zero_extension_without_learning"


def test_released_entries_stay_zero_and_unchanged_sets_are_kept():
    full = np.arange(1.0, 13.0).reshape(6, 2)
    reference = ZeroExtensionReference(ArrayReference(full, {}))
    reference.begin_system(np.array([0, 1, 2]))
    reference.begin_system(np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(reference.restrict(np.array([1, 2, 3, 4]))[2:], 0)
    reference.begin_system(np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(reference.restrict(np.array([0, 1, 2, 3, 4]))[[0, 3, 4]], 0)
    transfers = reference.transfers
    reference.begin_system(np.array([0, 1, 2, 3, 4]))
    assert reference.transfers == transfers
    with pytest.raises(ValueError):
        reference.restrict(np.array([0, 1]))


def test_tensor_path_matches_the_host_transfer():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    full = rng.standard_normal((60, 4))

    class TensorReference(ArrayReference):
        def restrict_device(self, indices):
            return torch.as_tensor(self.basis[indices], dtype=torch.float64)

    reference = ZeroExtensionReference(TensorReference(full, {}), torch, device="cpu")
    sets = inactive_sequence(rng, 60, 8)
    for indices, expected in zip(sets, expected_bases(full, sets)):
        reference.begin_system(indices)
        np.testing.assert_array_equal(reference.restrict_device(indices).numpy(), expected)


def test_complete_sequence_runs_the_ablation_with_the_same_reference():
    c = study.controls(
        OmegaConf.create(
            {"targets": 2, "methods": ["reference", "reference_zero"], "save_fields": False}
        )
    )
    record, _, _ = study.sequence(c, "reference_zero")
    assert record["success"]
    description = record["storage"]["reference_description"]
    assert description["transfer_policy"] == "zero_extension_without_learning"
    inner = [i for case in record["cases"] for i in case["inner"]]
    assert all("reference_transfer_seconds" in i for i in inner)
    direct, _, _ = study.sequence(c, "reference")
    assert direct["success"]
    assert "transfer_policy" not in direct["storage"]["reference_description"]
