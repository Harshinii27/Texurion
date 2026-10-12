"""Image IO: decoding, validation, unsupported formats and missing files."""

import os

import cv2
import numpy as np
import pytest

from core.io import (
    SUPPORTED_EXTENSIONS,
    ImageValidationError,
    decode_image,
    ensure_max_size,
    is_supported_ext,
    read_image,
    to_grayscale,
)


def test_read_valid_image(brick_bgr, jpeg_bytes):
    image = decode_image(jpeg_bytes, name="sample.jpg")
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8


def test_supported_extensions():
    assert ".jpg" in SUPPORTED_EXTENSIONS
    assert ".png" in SUPPORTED_EXTENSIONS
    assert is_supported_ext("a/b.png")
    assert not is_supported_ext("a/b.gif")


def test_decode_invalid_bytes_raises():
    with pytest.raises(ImageValidationError):
        decode_image(b"this is not an image at all", name="bad.jpg")


def test_decode_empty_file_raises():
    with pytest.raises(ImageValidationError):
        decode_image(b"", name="empty.jpg")


def test_decode_unsupported_extension_raises(jpeg_bytes):
    with pytest.raises(ImageValidationError):
        decode_image(jpeg_bytes, name="photo.gif")


def test_missing_file_raises():
    with pytest.raises(ImageValidationError):
        read_image(os.path.join("images", "does_not_exist.png"))


def test_unsupported_extension_path_raises(jpeg_bytes, tmp_path):
    path = tmp_path / "photo.png"
    path.write_bytes(jpeg_bytes)

    # Same contents, wrong extension -> rejected by extension policy.
    wrong = tmp_path / "photo.txt"
    wrong.write_bytes(jpeg_bytes)
    with pytest.raises(ImageValidationError):
        read_image(str(wrong))


def test_grayscale_conversion(brick_bgr):
    gray = to_grayscale(brick_bgr)
    assert gray.ndim == 2
    assert gray.shape == brick_bgr.shape[:2]


def test_downscales_large_images(brick_bgr):
    max_side = 320
    small = ensure_max_size(brick_bgr, max_side)
    assert max(small.shape[:2]) <= max_side
    aspect_before = brick_bgr.shape[1] / brick_bgr.shape[0]
    aspect_after = small.shape[1] / small.shape[0]
    assert abs(aspect_before - aspect_after) < 1e-2


def test_keeps_small_images_unchanged(tiny_checker):
    assert ensure_max_size(tiny_checker, 1000) is tiny_checker


def test_decode_roundtrip_from_disk(brick_bytes):
    image = decode_image(brick_bytes, name="brick.jpeg")
    assert image.shape[0] > 0 and image.shape[1] > 0