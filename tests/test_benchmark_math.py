"""
Tests for benchmark percentile math.

Verifies that P50, P70, and P100 calculations using numpy.percentile
produce the expected values for known input arrays.
"""

import numpy as np
import pytest


def compute_percentiles(times: list[float]) -> dict:
    """
    Mirror the percentile computation used in benchmark_pipeline.py.
    Returns dict with p50, p70, p100, mean.
    """
    return {
        "p50": float(np.percentile(times, 50)),
        "p70": float(np.percentile(times, 70)),
        "p100": float(np.percentile(times, 100)),
        "mean": float(np.mean(times)),
    }


class TestPercentileMath:
    """Validate percentile computations against known inputs."""

    def test_simple_sorted_array(self):
        times = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = compute_percentiles(times)

        assert result["p50"] == 30.0
        assert result["p100"] == 50.0
        assert result["mean"] == 30.0
        # P70 of [10,20,30,40,50] via linear interpolation: 38.0
        assert abs(result["p70"] - 38.0) < 0.01

    def test_single_element(self):
        times = [42.0]
        result = compute_percentiles(times)

        assert result["p50"] == 42.0
        assert result["p70"] == 42.0
        assert result["p100"] == 42.0
        assert result["mean"] == 42.0

    def test_two_elements(self):
        times = [100.0, 200.0]
        result = compute_percentiles(times)

        assert result["p50"] == 150.0  # median of 2 elements
        assert result["p100"] == 200.0
        assert result["mean"] == 150.0

    def test_identical_values(self):
        times = [5.0, 5.0, 5.0, 5.0]
        result = compute_percentiles(times)

        assert result["p50"] == 5.0
        assert result["p70"] == 5.0
        assert result["p100"] == 5.0
        assert result["mean"] == 5.0

    def test_large_spread(self):
        times = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0, 300.0, 400.0, 500.0]
        result = compute_percentiles(times)

        assert result["p100"] == 500.0
        # Mean should be (1+2+3+4+5+100+200+300+400+500)/10 = 151.5
        assert abs(result["mean"] - 151.5) < 0.01

    def test_p100_is_max(self):
        """P100 must always equal the maximum value."""
        times = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        result = compute_percentiles(times)
        assert result["p100"] == max(times)

    def test_p50_is_median(self):
        """P50 must equal the median for an odd-length sorted array."""
        times = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        result = compute_percentiles(times)
        assert result["p50"] == 40.0  # median of 7 elements

    def test_unsorted_input_works(self):
        """Percentile computation should work on unsorted input."""
        times = [50.0, 10.0, 40.0, 20.0, 30.0]
        result = compute_percentiles(times)

        assert result["p50"] == 30.0
        assert result["p100"] == 50.0
