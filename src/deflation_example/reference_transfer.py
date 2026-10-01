"""Zero-extension transfer of a full-domain reference: the retention ablation.

Reference deflation restricts the stored full-domain reference directly to every
inactive set. This ablation restricts it only for the first inactive system and
then carries the previous restricted basis to each later inactive set by zero
extension, R_b R_a^T Z_a. Entries at newly released degrees of freedom are zero,
and no directions are learned from the solves. The construction, rank and solver
are otherwise those of reference deflation, so a complete sequence measures how
much of its cost depends on direct restriction.
"""

import numpy as np

from .recycling import inactive_indices, transfer_basis


class ZeroExtensionReference:
    """Wrap a full-domain reference so that each later restriction is a zero extension.

    With ``torch`` the restricted basis stays on the GPU: the first restriction uses
    ``reference.restrict_device`` and every transfer gathers the retained rows on the
    device. Without ``torch`` the host path uses ``recycling.transfer_basis``.
    """

    def __init__(self, reference, torch=None, device="cuda"):
        self.reference = reference
        self.torch = torch
        self.device = device
        self.rank = reference.rank
        self.description = {
            **reference.description,
            "transfer_policy": "zero_extension_without_learning",
        }
        self.indices = None
        self.basis = None
        self.transfers = 0

    def begin_system(self, indices):
        indices = inactive_indices(indices)
        if self.indices is not None and np.array_equal(indices, self.indices):
            return
        if self.indices is None:
            first = self.torch is not None and callable(
                getattr(self.reference, "restrict_device", None)
            )
            self.basis = (
                self.reference.restrict_device(indices)
                if first
                else self._to_device(self.reference.restrict(indices))
            )
        else:
            self.basis = self._transfer(self.basis, self.indices, indices)
            self.transfers += 1
        self.indices = indices

    def _to_device(self, basis):
        if self.torch is None:
            return np.array(basis, dtype=float, copy=True)
        return self.torch.as_tensor(
            np.asarray(basis, dtype=float), dtype=self.torch.float64, device=self.device
        )

    def _transfer(self, basis, previous, current):
        if self.torch is None:
            return transfer_basis(basis, previous, current)
        torch = self.torch
        out = torch.zeros((len(current), basis.shape[1]), dtype=basis.dtype, device=basis.device)
        if not len(previous) or not len(current):
            return out
        order = np.argsort(previous, kind="stable")
        old = torch.as_tensor(previous[order], device=basis.device)
        new = torch.as_tensor(current, device=basis.device)
        position = torch.searchsorted(old, new).clamp(max=len(previous) - 1)
        hit = old[position] == new
        rows = torch.as_tensor(order, device=basis.device)[position[hit]]
        out[hit] = basis[rows]
        return out

    def _current(self, indices):
        indices = inactive_indices(indices)
        if self.indices is None or not np.array_equal(indices, self.indices):
            raise ValueError("Begin the inactive system before restricting the transferred basis")

    def restrict_device(self, indices):
        self._current(indices)
        return self.basis.clone()

    def restrict(self, indices):
        self._current(indices)
        if self.torch is None:
            return self.basis.copy()
        return self.basis.cpu().numpy().copy()

    def storage(self):
        return self.reference.storage()
