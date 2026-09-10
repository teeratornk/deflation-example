"""Comparable sampled process memory for complete optimization benchmarks."""

import os
import multiprocessing
import threading
import time

from .validation import positive_real


def _memory_process(connection, observed_pid, device_uuid, interval):
    """Sample another process without sharing its Python interpreter lock."""

    def shutdown():
        pass

    state = {
        "count": 0,
        "initial": None,
        "peak_host": 0,
        "peak_gpu": 0,
        "last": None,
        "gap": 0.0,
        "error": None,
    }
    try:
        import psutil

        process = psutil.Process(observed_pid)
        handle = None
        if device_uuid is not None:
            import pynvml

            pynvml.nvmlInit()
            shutdown = pynvml.nvmlShutdown
            handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid)

        def sample():
            host, gpu = process.memory_info().rss, 0
            if handle is not None:
                rows = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                own = [r.usedGpuMemory for r in rows if r.pid == observed_pid]
                if not own or any(v is None or v < 0 or v >= 2**63 for v in own):
                    raise RuntimeError("NVML does not provide process GPU memory")
                gpu = sum(own)
            now = time.perf_counter()
            if state["initial"] is None:
                state["initial"] = (host, gpu)
            if state["last"] is not None:
                state["gap"] = max(state["gap"], now - state["last"])
            state.update(
                last=now,
                count=state["count"] + 1,
                peak_host=max(state["peak_host"], host),
                peak_gpu=max(state["peak_gpu"], gpu),
            )

        sample()
        connection.send(dict(state))
        while not connection.poll(interval):
            sample()
        connection.recv()
        sample()
    except Exception as error:
        state["error"] = type(error).__name__
    finally:
        try:
            shutdown()
        except Exception as error:
            state["error"] = state["error"] or type(error).__name__
        try:
            connection.send(state)
        except (BrokenPipeError, EOFError):
            pass
        connection.close()


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
        self.external = query is None
        self.process = self.connection = None
        self.device_uuid = None
        if self.external:
            if device == "cuda":
                from .gpu import require_cuda

                torch = require_cuda()
                self.device_uuid = str(
                    torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
                )
            elif device != "cpu":
                raise ValueError("Memory device must be cpu or cuda")
            self.closed = False
            return
        self._shutdown = lambda: None
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
        if self.external:
            if self.process is not None:
                raise RuntimeError("Memory sampler is already running")
            context = multiprocessing.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(
                target=_memory_process,
                args=(child, os.getpid(), self.device_uuid, self.interval),
                daemon=True,
            )
            self.process.start()
            child.close()
            if not self.connection.poll(60):
                self.process.terminate()
                self.process.join(5)
                self.connection.close()
                raise RuntimeError("Memory sampler did not initialize")
            self.initial_state = self.connection.recv()
            if self.initial_state["error"]:
                self.process.join(5)
                self.connection.close()
                raise RuntimeError("Initial process memory measurement failed")
            return self
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
        if self.external:
            if self.process is None or self.closed:
                raise RuntimeError("Memory sampler has not been started")
            state = dict(self.initial_state)
            try:
                if not self.connection.poll():
                    self.connection.send("stop")
                    if not self.connection.poll(10):
                        raise RuntimeError("Memory sampler did not finish")
                state = self.connection.recv()
            except (RuntimeError, EOFError, BrokenPipeError) as error:
                state["error"] = type(error).__name__
            finally:
                self.process.join(5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(5)
                self.connection.close()
                self.closed = True
            return {
                "measurement": "sampled process RSS and NVML compute-process GPU allocation",
                "sampler": "separate process; unaffected by the solver interpreter lock",
                "sampling_interval_seconds": self.interval,
                "maximum_sampling_gap_seconds": state["gap"],
                "samples": state["count"],
                "initial_host_rss_bytes": state["initial"][0],
                "peak_host_rss_bytes": state["peak_host"],
                "initial_gpu_process_bytes": state["initial"][1],
                "peak_gpu_process_bytes": state["peak_gpu"],
                "complete": state["error"] is None,
                "error_type": state["error"],
            }
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
