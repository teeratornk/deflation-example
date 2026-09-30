"""A complete temporal basis per spatial mode gives the same restricted coarse space.

With S spatial modes and all K temporal factors of each, the reference spans
{e_t (x) phi_j}. Restriction selects rows, so R (Q (x) Phi) = R (I (x) Phi) (Q (x) I)
with Q invertible, and the two restricted bases have the same column space. Every
coarse projector built from that space is therefore identical in exact arithmetic.
"""

import numpy as np
import pytest

from deflation_example.mesh_control import build_mesh_control
from deflation_example.mesh_reference import build_mesh_reference
from deflation_example.mesh_showcases import build_showcase


@pytest.fixture(scope="module")
def transformer():
    data = build_showcase("transformer_2d", transport_form="skew")
    model = build_mesh_control(data.assembly, alpha=1e-14, time_steps=[1.2e-5] * 4)
    return data, model


def orthonormal(basis):
    q, r = np.linalg.qr(basis)
    keep = np.abs(np.diag(r)) > 1e-12 * np.abs(np.diag(r)).max()
    return q[:, keep]


def test_complete_temporal_coverage_and_the_tensor_basis_restrict_to_one_space(transformer):
    data, model = transformer
    modes, slabs = 3, 4
    reference = build_mesh_reference(model, data.coarse_assembly, data.prolongation,
                                     modes * slabs, "mode_dependent", "scaled_schur", "jacobi",
                                     spatial_modes=modes)
    n = model.spatial_size
    tensor = np.zeros((model.size, modes * slabs))
    for j in range(modes):
        for t in range(slabs):
            tensor[t * n:(t + 1) * n, j * slabs + t] = reference.spatial[:, j]
    rng = np.random.default_rng(7)
    for fraction in (0.95, 0.8, 0.5):
        inactive = np.sort(rng.choice(model.size, int(fraction * model.size), replace=False))
        per_mode = orthonormal(reference.restrict(inactive))
        explicit = orthonormal(tensor[inactive])
        assert per_mode.shape[1] == explicit.shape[1] == modes * slabs
        # Orthogonal projectors agree, hence so does the span.
        overlap = np.linalg.svd(per_mode.T @ explicit, compute_uv=False)
        assert overlap.min() > 1 - 1e-10
        # A coarse correction with any SPD operator on the inactive set agrees as well.
        # The operator a a^T + m I is applied implicitly.
        a = rng.standard_normal((len(inactive), 40))
        rhs = rng.standard_normal(len(inactive))

        def correction(z):
            za = z.T @ a
            coarse = za @ za.T + len(inactive) * (z.T @ z)
            return z @ np.linalg.solve(coarse, z.T @ rhs)

        first = correction(reference.restrict(inactive))
        second = correction(tensor[inactive])
        assert np.linalg.norm(first - second) <= 1e-9 * np.linalg.norm(first)


def test_truncated_temporal_selection_spans_a_proper_subspace(transformer):
    data, model = transformer
    truncated = build_mesh_reference(model, data.coarse_assembly, data.prolongation, 6,
                                     "mode_dependent", "scaled_schur", "jacobi",
                                     spatial_modes=3, temporal_modes=2)
    assert truncated.description["selection_policy"] == "3 spatial modes x 2 temporal factors"
    n = model.spatial_size
    basis = orthonormal(truncated.restrict(np.arange(model.size)))
    missing = 0
    for t in range(4):
        indicator = np.zeros(model.size)
        indicator[t * n:(t + 1) * n] = truncated.spatial[:, 0]
        residual = indicator - basis @ (basis.T @ indicator)
        missing += np.linalg.norm(residual) > 1e-8 * np.linalg.norm(indicator)
    assert missing > 0


def test_gpu_restriction_matches_the_host_restriction(transformer):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No GPU")
    from deflation_example.mesh_reference import DeviceMeshReference

    data, model = transformer
    reference = build_mesh_reference(model, data.coarse_assembly, data.prolongation, 12,
                                     "mode_dependent", "scaled_schur", "jacobi", spatial_modes=3)
    device = DeviceMeshReference(reference, torch)
    inactive = np.sort(np.random.default_rng(3).choice(model.size, model.size // 2, replace=False))
    np.testing.assert_allclose(device.restrict_device(inactive).cpu().numpy(),
                               reference.restrict(inactive), rtol=0, atol=1e-14)
