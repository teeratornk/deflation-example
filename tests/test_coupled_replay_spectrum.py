import numpy as np
import pytest

from deflation_example.coupled_replay_spectrum import pencils, verified_slab
from deflation_example.coupled_step_spectrum import step_linearization
from test_coupled_derivatives import small_coupled_problem


def test_only_existing_verified_states_can_supply_a_local_map():
    assert verified_slab(np.array([1.0, 2.0, 3.0]), 2, 2.0) == 1
    for time in (0.5, 3.0, float("nan")):
        with pytest.raises(ValueError, match="verified physical time"):
            verified_slab(np.array([1.0, 2.0, 3.0]), 2, time)


def test_pencil_blocks_retain_consistent_storage_and_signed_source():
    p = small_coupled_problem([0.2, 0.35], uniform_capacity=True, consistent=True)
    previous = p.full_temperature(p.initial)
    source = np.linspace(-0.1, 0.1, len(p.mesh.nodes))
    full = step_linearization(
        p, p.initial, p.initial_flow.velocity, 1, control=source, previous=previous
    )
    actual = pencils(p, source, previous, p.initial, p.initial_flow.velocity, 1)
    n = len(p.flow_free)
    expected = ((full[0], full[1]), (full[0][:n, :n], full[1][:n, :n]), (full[2], full[3]))
    for (_, H, C), (a, b) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(H.toarray(), a.toarray())
        np.testing.assert_array_equal(C.toarray(), b.toarray())
