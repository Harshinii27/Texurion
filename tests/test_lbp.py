"""LBP: map dimensions, histogram normalization, edge cases, engine parity."""

import numpy as np
import pytest

from core.histogram import get_histogram
from core.lbp import (
    LBP_METHODS,
    generate_lbp,
    lbp_circular_numba,
    lbp_circular_python,
    lbp_circular_vectorized,
    number_of_patterns,
    validate_lbp_params,
)


def _random_gray(seed, h=48, w=64):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def test_lbp_map_same_shape():
    gray = _random_gray(1)
    _, lbp = generate_lbp(gray=gray, radius=2, points=16, method="uniform")
    assert lbp.shape == gray.shape


def test_uniform_lbp_values_are_labels():
    gray = _random_gray(2)
    _, lbp = generate_lbp(gray=gray, radius=3, points=16, method="uniform")
    unique = set(np.unique(lbp).tolist())
    # uniform labels are 0..P plus the "non-uniform" bucket P+1
    assert max(unique) <= 17


def test_histogram_normalized_to_one():
    gray = _random_gray(3)
    _, lbp = generate_lbp(gray=gray, radius=3, points=24, method="uniform")
    hist = get_histogram(lbp, bins=number_of_patterns("uniform", 24))
    assert hist.ndim == 1
    assert pytest.approx(float(hist.sum()), abs=1e-6) == 1.0


def test_small_image_rejected():
    gray = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_lbp_params(gray, 3, 24, "uniform")


def test_unknown_method_rejected():
    gray = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_lbp_params(gray, 3, 24, "nope")


def test_engine_equivalence_uniform():
    gray = _random_gray(4, 40, 40)
    a = lbp_circular_vectorized(gray, 3, 24, method="uniform")
    b = lbp_circular_numba(gray, 3, 24, method="uniform")
    c = lbp_circular_python(gray, 3, 24, method="uniform")
    assert np.array_equal(a, b)
    assert np.array_equal(b, c)


def test_engine_equivalence_default():
    gray = _random_gray(5, 40, 40)
    a = lbp_circular_vectorized(gray, 2, 16, method="default")
    b = lbp_circular_numba(gray, 2, 16, method="default")
    c = lbp_circular_python(gray, 2, 16, method="default")
    assert np.array_equal(a, b)
    assert np.array_equal(b, c)


def test_custom_engine_rejects_unimplemented_method():
    gray = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError):
        lbp_circular_vectorized(gray, 2, 16, method="ror")


def test_number_of_patterns_rules():
    assert number_of_patterns("uniform", 8) == 10
    assert number_of_patterns("uniform", 24) == 26
    # nri_uniform densely labels 0..P*(P-1)+2 -> P*(P-1)+3 bins.
    assert number_of_patterns("nri_uniform", 8) == 59
    assert number_of_patterns("nri_uniform", 16) == 243
    assert number_of_patterns("default", 8) == 256
    assert number_of_patterns("default", 3) == 8
    assert number_of_patterns("ror", 8) == 256
    assert number_of_patterns("var", 16) == 256


@pytest.mark.parametrize("method", ["uniform", "nri_uniform", "ror", "default", "var"])
def test_multiscale_histogram_is_non_degenerate(method):
    # Every supported method must yield a normalised, non-empty histogram
    # (the code/response must fall inside the chosen histogram range).
    from core.lbp import lbp_multiscale_histograms

    gray = _random_gray(7, 48, 48)
    hist = lbp_multiscale_histograms(gray, [(3, 16)], method=method)
    assert hist.ndim == 1 and hist.size == number_of_patterns(method, 16)
    assert pytest.approx(float(hist.sum()), abs=1e-6) == 1.0


def test_methods_exist():
    assert set(LBP_METHODS) == {"uniform", "nri_uniform", "ror", "var", "default"}