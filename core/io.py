"""Robust image loading and validation.

All decoding is performed with OpenCV against raw bytes so that uploaded
files (Streamlit) and files on disk follow the same code path. Corrupt or
unsupported content raises :class:`ImageValidationError` instead of
returning ``None`` silently.
"""

import os

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

MAX_PIXELS = 200_000_000  # 200 MP: generous safety ceiling


class ImageValidationError(ValueError):
    """Raised when an image cannot be decoded or is rejected by policy."""


def _extension_of(name):
    name = (name or "").lower()
    return os.path.splitext(name)[1]


def is_supported_ext(name):
    return _extension_of(name) in SUPPORTED_EXTENSIONS


def decode_image(data, name=None):
    """Decode ``bytes`` into a BGR (3-channel) ``numpy`` array.

    Parameters
    ----------
    data : bytes
        Raw encoded image bytes.
    name : str, optional
        Filename, used only to improve error messages.

    Raises
    ------
    ImageValidationError
        If the payload is empty, cannot be decoded, or is degenerate.
    """

    if not data:
        raise ImageValidationError(
            f"File is empty: {name or 'unknown image'}")

    if name and is_supported_ext(name) is False and not _is_valid_header(data, name):
        raise ImageValidationError(
            f"Unsupported image format: {name}. Supported: "
            + ", ".join(SUPPORTED_EXTENSIONS))

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageValidationError(
            f"Corrupt or undecodable image: {name or 'unknown image'}")

    h, w = image.shape[:2]
    if h < 1 or w < 1:
        raise ImageValidationError(f"Image has invalid dimensions: {name}")
    if h * w > MAX_PIXELS:
        raise ImageValidationError(
            f"Image too large ({h * w:,} px > {MAX_PIXELS:,} px): {name}")

    return image


def _is_valid_header(data, name):
    return False


def read_image(path):
    """Read an image file from disk, returning a BGR ``numpy`` array.

    Raises
    ------
    ImageValidationError
        If the path does not exist, the extension is unsupported, or the
        contents cannot be decoded.
    """

    if not path or not os.path.isfile(path):
        raise ImageValidationError(f"File not found: {path}")

    if not is_supported_ext(path):
        raise ImageValidationError(
            f"Unsupported extension: {path}. Supported: "
            + ", ".join(SUPPORTED_EXTENSIONS))

    with open(path, "rb") as handle:
        data = handle.read()

    return decode_image(data, name=os.path.basename(path))


def to_grayscale(image):
    """Return a single-channel version of the given BGR/RGB/gray array."""

    if image is None:
        raise ImageValidationError("image is None")
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ensure_max_size(image, max_side, interpolation=cv2.INTER_AREA):
    """Downscale an image so its larger side is at most ``max_side``."""

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, new_size, interpolation=interpolation)