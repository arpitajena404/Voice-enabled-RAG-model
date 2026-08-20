"""
Tests for the timing utility (app/timing.py).

Covers:
  - Async timed_stage produces latency_ms >= 0 for a known sleep
  - Sync timed_stage_sync produces latency_ms >= 0 for a known sleep
  - Timer name is preserved
"""

import asyncio
import time
import pytest
from app.timing import timed_stage, timed_stage_sync, StageTimer


class TestTimedStageAsync:
    """Tests for the async timed_stage context manager."""

    @pytest.mark.asyncio
    async def test_latency_is_positive(self):
        async with timed_stage("test_stage") as t:
            await asyncio.sleep(0.05)  # 50ms

        assert t.latency_ms >= 40.0, f"Expected >= 40ms, got {t.latency_ms}"
        assert t.latency_ms < 200.0, f"Expected < 200ms, got {t.latency_ms}"

    @pytest.mark.asyncio
    async def test_name_is_preserved(self):
        async with timed_stage("stt") as t:
            pass
        assert t.name == "stt"

    @pytest.mark.asyncio
    async def test_zero_work_produces_small_latency(self):
        async with timed_stage("noop") as t:
            pass
        # Should be essentially zero, but definitely less than 10ms
        assert t.latency_ms < 10.0


class TestTimedStageSyncManager:
    """Tests for the sync timed_stage_sync context manager."""

    def test_latency_is_positive(self):
        with timed_stage_sync("retrieval") as t:
            time.sleep(0.05)  # 50ms

        assert t.latency_ms >= 40.0, f"Expected >= 40ms, got {t.latency_ms}"
        assert t.latency_ms < 200.0, f"Expected < 200ms, got {t.latency_ms}"

    def test_name_is_preserved(self):
        with timed_stage_sync("guardrails") as t:
            pass
        assert t.name == "guardrails"

    def test_zero_work_produces_small_latency(self):
        with timed_stage_sync("noop") as t:
            pass
        assert t.latency_ms < 10.0


class TestStageTimer:
    """Direct unit tests for StageTimer."""

    def test_begin_end_records_time(self):
        timer = StageTimer("manual")
        timer._begin()
        time.sleep(0.02)
        timer._end()
        assert timer.latency_ms >= 15.0
        assert timer.latency_ms < 100.0

    def test_initial_latency_is_zero(self):
        timer = StageTimer("fresh")
        assert timer.latency_ms == 0.0
