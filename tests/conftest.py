"""Shared fixtures for the Texture Analysis test suite."""

import io
import os
import zipfile

import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_IMAGE = os.path.join(PROJECT_ROOT, "images", "brick.jpeg")


@pytest.fixture(scope="session")
def brick_bgr():
    image = cv2.imread(SAMPLE_IMAGE)
    assert image is not None, f"sample image missing: {SAMPLE_IMAGE}"
    return image


@pytest.fixture(scope="session")
def brick_bytes():
    with open(SAMPLE_IMAGE, "rb") as handle:
        return handle.read()


@pytest.fixture
def jpeg_bytes(brick_bgr):
    ok, buf = cv2.imencode(".jpg", brick_bgr)
    assert ok
    return buf.tobytes()


@pytest.fixture
def tiny_checker():
    yy, xx = np.mgrid[0:32, 0:32]
    pattern = ((xx // 4 + yy // 4) % 2).astype("uint8") * 255
    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def tmpzip():
    """Build an in-memory ZIP archive from {member_name: bytes}."""

    def _build(members):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        buffer.seek(0)
        return buffer

    return _build


@pytest.fixture(autouse=True)
def _chdir_to_project(tmp_path, monkeypatch):
    """Ensure 'core' is importable regardless of where pytest was launched."""

    monkeypatch.chdir(PROJECT_ROOT)
    return PROJECT_ROOT