"""Gabor filter bank: sizes, finite values and magnitude behaviour."""

import numpy as np

from core.gabor import (
    DEFAULT_FREQUENCIES,
    DEFAULT_ORIENTATIONS,
    gabor_descriptor_size,
    gabor_feature_vector,
    gabor_magnitude_map,
    make_gabor_kernels,
)


def test_kernel_count_mathches_bank():
    kernels = make_gabor_kernels()
    assert len(kernels) == len(DEFAULT_FREQUENCIES) * len(DEFAULT_ORIENTATIONS)


def test_descriptor_length_matches_vector():
    gray = np.random.default_rng(42).integers(0, 256, (64, 64)).astype("uint8")
    vector = gabor_feature_vector(gray)
    assert vector.shape == (gabor_descriptor_size(),)
    assert np.all(np.isfinite(vector))
    assert np.any(vector > 0)


def test_magnitude_map_preserves_display_dims():
    gray = np.zeros((32, 32), dtype="uint8")
    gray[8:24, 8:24] = 255
    kernels = make_gabor_kernels()
    magnitude = gabor_magnitude_map(gray, kernels)
    assert magnitude.shape == (32, 32, 3)  # RGB display view
    assert magnitude.dtype == np.uint8
    assert np.all(np.isfinite(magnitude))


def test_custom_bank_size():
    kernels = make_gabor_kernels(frequencies=(0.1, 0.2), orientations=(0, 45))
    vector = gabor_feature_vector(np.zeros((16, 16), dtype="uint8"), kernels=kernels)
    assert vector.shape == (8,)  # 2 freqs * 2 orientations * 2 stats