"""Rate-limited progress for a logical solve containing CPU CG kernels."""

import time

from .reporting import write_report
from .validation import positive_real


class LinearProgressWriter:
    """Keep local error-equation residuals separate from final acceptance.

    Initial and final kernel events are always emitted. Other events are written
    at most once per interval; no matrix application is added for a heartbeat.
    The enclosing solver writes the final original-system result separately.
    """

    def __init__(self, destination, metadata, interval=30.0, clock=None):
        self.destination, self.metadata = destination, dict(metadata)
        self.interval = positive_real(interval, "Heartbeat interval")
        self.clock = time.perf_counter if clock is None else clock
        self.start = self.clock()
        self.last = self.start

    def __call__(self, event):
        now = self.clock()
        if event["stage"] in {"initial", "final"} or now - self.last >= self.interval:
            write_report(
                self.destination,
                {
                    **self.metadata,
                    "status": "running",
                    "elapsed_seconds": now - self.start,
                    "kernel": dict(event),
                },
            )
            self.last = now
