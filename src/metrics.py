from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import Any, Callable

import psutil


@dataclass
class StepMetrics:
    elapsed_sec: float
    memory_peak_mb: float
    status: str
    error: str = ""


def _process_tree_rss_mb(process: psutil.Process) -> float:
    """
    Returns RSS memory for the current process plus child processes.

    This is important for PySpark: the Spark JVM usually runs as a child
    process, so measuring only the Python process would underreport memory.
    """
    total = 0.0
    processes = [process]

    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass

    for item in processes:
        try:
            total += item.memory_info().rss / 1024 / 1024
        except psutil.Error:
            continue

    return total


def measure_step(
    func: Callable[..., Any],
    *args: Any,
    sample_interval_sec: float = 0.05,
    **kwargs: Any,
) -> tuple[Any, StepMetrics]:
    """
    Executes func and measures wall-clock time plus peak RSS memory.

    Memory is measured for the current Python process and its child processes.
    This keeps the metric more representative for Pandas, Dask and local Spark.
    """
    process = psutil.Process(os.getpid())
    stop_event = threading.Event()
    peak_mb = _process_tree_rss_mb(process)

    def sample_memory() -> None:
        nonlocal peak_mb
        while not stop_event.is_set():
            current_mb = _process_tree_rss_mb(process)
            if current_mb > peak_mb:
                peak_mb = current_mb
            time.sleep(sample_interval_sec)

    sampler = threading.Thread(target=sample_memory, daemon=True)

    start = time.perf_counter()
    sampler.start()

    result: Any = None
    status = "success"
    error = ""

    try:
        result = func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = repr(exc)
    finally:
        elapsed_sec = time.perf_counter() - start
        stop_event.set()
        sampler.join(timeout=1)

    return result, StepMetrics(
        elapsed_sec=round(elapsed_sec, 6),
        memory_peak_mb=round(peak_mb, 3),
        status=status,
        error=error,
    )
