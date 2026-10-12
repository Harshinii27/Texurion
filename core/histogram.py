"""Histogram utilities for LBP feature maps and fused feature vectors."""

import numpy as np


def get_histogram(lbp, bins=26):
    """Return a normalised histogram of the given LBP map.

    ``bins`` is the fixed feature-vector size used consistently between
    training and prediction (default 26 matches a ``uniform`` LBP at
    radius=3 / points=24).
    """

    if lbp is None or lbp.size == 0:
        raise ValueError("Cannot build a histogram from an empty array.")

    bins = max(1, int(bins))

    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype("float64")

    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


def histogram_statistics(hist):
    """Descriptive statistics for a normalised histogram."""

    hist = np.asarray(hist, dtype="float64")
    hist = np.clip(hist, 0, None)

    total = hist.sum()
    probs = hist / total if total > 0 else hist

    values = np.arange(probs.size)
    mean = float(np.sum(values * probs))
    variance = float(np.sum(((values - mean) ** 2) * probs))
    std = float(np.sqrt(variance))
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    entropy = max(0.0, entropy)  # guard against floating-point negatives
    uniformity = float(np.sum(probs ** 2))

    return {
        "mean": mean,
        "std": std,
        "entropy": entropy,
        "uniformity": uniformity,
    }


def histogram_distance(a, b, metric="chi_square"):
    """Distance between two normalised (LBP) histograms.

    Supported metrics: ``chi_square`` and ``cosine``.
    """

    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.shape != b.shape:
        raise ValueError("Histograms must have the same length.")

    if metric == "chi_square":
        both = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (a - b) ** 2 / np.where(both == 0, 1.0, both)
        return float(np.nansum(term))

    if metric == "cosine":
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 1.0
        return float(1.0 - np.dot(a, b) / denom)

    raise ValueError(f"Unknown distance metric '{metric}'.")