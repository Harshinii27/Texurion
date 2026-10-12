"""Histogram statistics and distance metrics."""

import numpy as np
import pytest

from core.histogram import get_histogram, histogram_distance, histogram_statistics


def test_get_histogram_shape_and_sum():
    lbp = np.array([[0, 1, 2, 3], [4, 5, 6, 7]] * 10, dtype=np.int64)
    hist = get_histogram(lbp, bins=8)
    assert hist.shape == (8,)
    assert pytest.approx(float(hist.sum()), abs=1e-12) == 1.0


def test_statistics_known_distribution():
    hist = np.zeros(4)
    hist[1] = 1.0
    stats = histogram_statistics(hist)
    assert stats["mean"] == pytest.approx(1.0)
    assert stats["entropy"] == pytest.approx(0.0)  # deterministic single bin
    assert stats["uniformity"] == pytest.approx(1.0)


def test_chi_square_self_is_zero():
    hist = np.array([0.25, 0.25, 0.5, 0.0])
    assert histogram_distance(hist, hist, "chi_square") == pytest.approx(0.0)


def test_cosine_self_is_zero_and_orthogonal():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    assert histogram_distance(a, a, "cosine") == pytest.approx(0.0)
    assert histogram_distance(a, b, "cosine") == pytest.approx(1.0)


def test_different_histograms_positive_distance():
    a = np.array([0.9, 0.1])
    b = np.array([0.1, 0.9])
    assert histogram_distance(a, b, "chi_square") > 0.0


def test_unknown_metric_rejected():
    a = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        histogram_distance(a, a, "nope")