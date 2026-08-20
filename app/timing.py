"""
Reusable timing utilities for measuring pipeline stage latencies.

Replaces the repeated `t0 = time.time(); ...; latency = (time.time()-t0)*1000`
pattern with clean context managers.

Usage (async):
    async with timed_stage("stt") as t:
        result = await transcribe_audio(...)
    print(t.latency_ms)  # e.g. 142.5

Usage (sync):
    with timed_stage_sync("retrieval") as t:
        passages = retrieve_passages(...)
    print(t.latency_ms)  # e.g. 8.3
"""

import time
from contextlib import contextmanager, asynccontextmanager


class StageTimer:
    """Holds the measured latency for a single pipeline stage."""

    __slots__ = ("name", "latency_ms", "_start")

    def __init__(self, name: str) -> None:
        self.name = name
        self.latency_ms: float = 0.0
        self._start: float = 0.0

    def _begin(self) -> None:
        self._start = time.perf_counter()

    def _end(self) -> None:
        self.latency_ms = (time.perf_counter() - self._start) * 1000.0


@asynccontextmanager
async def timed_stage(name: str):
    """
    Async context manager that measures wall-clock time in milliseconds.

    Example::

        async with timed_stage("stt") as t:
            transcript = await transcribe_audio(audio)
        latency = t.latency_ms
    """
    timer = StageTimer(name)
    timer._begin()
    try:
        yield timer
    finally:
        timer._end()


@contextmanager
def timed_stage_sync(name: str):
    """
    Sync context manager that measures wall-clock time in milliseconds.

    Example::

        with timed_stage_sync("retrieval") as t:
            passages = retrieve_passages(query)
        latency = t.latency_ms
    """
    timer = StageTimer(name)
    timer._begin()
    try:
        yield timer
    finally:
        timer._end()
