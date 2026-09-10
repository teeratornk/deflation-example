"""Comparable sampled process memory for complete optimization benchmarks."""

import os
import threading
import time

from .validation import positive_real


class ProcessMemory:
    """Sample RSS and this process's NVML allocation on its CUDA device.

    NVML includes allocations from PyTorch and AmgX. Host RSS uses the same
    process boundary for every solver. Sampling is a lower bound on a true
    instantaneous peak; its period and observed maximum gap are reported.
    Use a fresh process per timed sequence to isolate allocator histories.
    Neither process identifiers nor GPU identifiers enter the saved report.
    """

    def __init__(self, device="cpu", interval=0.01, *, query=None):
        self.interval = positive_real(interval, "Memory sampling interval")
        self._shutdown = lambda: None
        if query is None:
            try:
                import psutil
            except ImportError as error:
                raise RuntimeError("Memory measurements require the [study] extra") from error
            process = psutil.Process()

            def gpu_bytes():
                return 0

            if device == "cuda":
                try:
                    import pynvml
                except ImportError as error:
                    raise RuntimeError(
                        "GPU memory measurements require the [study] extra"
                    ) from error
                from .gpu import require_cuda

                torch = require_cuda()
                pynvml.nvmlInit()
                self._shutdown = pynvml.nvmlShutdown
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(
                        str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
                    )
                except Exception:
                    self._shutdown()
                    raise

                def gpu_bytes():
                    rows = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    own = [r.usedGpuMemory for r in rows if r.pid == os.getpid()]
                    if not own or any(value is None for value in own):
                        raise RuntimeError("NVML does not provide this process's GPU memory")
                    return sum(own)

            elif device != "cpu":
                raise ValueError("Memory device must be cpu or cuda")

            def query():
                return process.memory_info().rss, gpu_bytes()

        self.query = query
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.count = 0
        self.initial = None
        self.peak_host = self.peak_gpu = 0
        self.last_sample = None
        self.maximum_gap = 0.0
        self.error = None
        self.thread = None
        self.closed = False

    def sample(self):
        with self.lock:
            try:
                rss, gpu = self.query()
                if min(rss, gpu) < 0:
                    raise ValueError("Memory measurements must be nonnegative")
                now = time.perf_counter()
                if self.initial is None:
                    self.initial = (int(rss), int(gpu))
                if self.last_sample is not None:
                    self.maximum_gap = max(self.maximum_gap, now - self.last_sample)
                self.last_sample = now
                self.peak_host = max(self.peak_host, int(rss))
                self.peak_gpu = max(self.peak_gpu, int(gpu))
                self.count += 1
            except Exception as error:
                self.error = type(error).__name__
                self.stop_event.set()

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Memory sampler is already running")
        self.sample()
        if self.error:
            self._shutdown()
            raise RuntimeError("Initial process memory measurement failed")

        def collect():
            while not self.stop_event.wait(self.interval):
                self.sample()

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        if self.thread is None or self.closed:
            raise RuntimeError("Memory sampler has not been started")
        self.stop_event.set()
        self.thread.join()
        self.sample()
        self._shutdown()
        self.closed = True
        return {
            "measurement": "sampled process RSS and NVML compute-process GPU allocation",
            "sampling_interval_seconds": self.interval,
            "maximum_sampling_gap_seconds": self.maximum_gap,
            "samples": self.count,
            "initial_host_rss_bytes": self.initial[0],
            "peak_host_rss_bytes": self.peak_host,
            "initial_gpu_process_bytes": self.initial[1],
            "peak_gpu_process_bytes": self.peak_gpu,
            "complete": self.error is None,
            "error_type": self.error,
        }
