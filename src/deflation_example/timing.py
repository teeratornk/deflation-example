"""Nonoverlapping wall-clock accounting for numerical benchmarks."""

import math
import time


PHASES = (
    "conversion", "resource_creation", "handle_creation", "upload",
    "basis_processing", "coarse_or_hierarchy_setup", "initialization",
    "iteration", "download", "verification", "cleanup", "synchronization",
    "host_bookkeeping",
)


class PhaseTimer:
    """Partition one elapsed interval; no component is nested in another.

    Explicit device barriers have their own component. Implicit waits inside
    library calls remain charged to that call's phase. These are wall-clock
    intervals, not isolated device execution times.
    """

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.start = self.last = clock()
        self.components = dict.fromkeys(PHASES, 0.0)
        self.closed = False

    def mark(self, phase):
        if self.closed or phase not in self.components:
            raise ValueError("Invalid phase or already completed timer")
        now = self.clock()
        elapsed = now - self.last
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Clock must be finite and monotone")
        self.components[phase] += elapsed
        self.last = now

    def synchronize(self, barrier):
        self.mark("host_bookkeeping")
        barrier()
        self.mark("synchronization")

    def finish(self):
        self.mark("host_bookkeeping")
        self.closed = True
        total = self.last - self.start
        if not math.isclose(sum(self.components.values()), total, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Timing components do not sum to elapsed time")
        return {"components_seconds": self.components.copy(), "total_seconds": total}
