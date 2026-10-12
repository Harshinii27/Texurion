"""Batch: ZIP safety, path-traversal rejection, concurrency, CSV export."""

import csv
import os

import numpy as np
import pytest

from core.batch import (
    BatchError,
    collect_batch_inputs,
    process_batch,
    read_zip_images,
    safe_zip_members,
    write_results_csv,
)
from core.pipeline import FeatureConfig, extract_features


def test_safe_zip_rejects_parent_traversal(tmpzip, jpeg_bytes):
    archive = tmpzip({"../evil.jpg": jpeg_bytes})
    with pytest.raises(BatchError):
        list(safe_zip_members(archive))


def test_safe_zip_rejects_absolute_path(tmpzip, jpeg_bytes):
    archive = tmpzip({"/etc/evil.jpg": jpeg_bytes})
    with pytest.raises(BatchError):
        list(safe_zip_members(archive))


def test_safe_zip_rejects_drive_letter(tmpzip, jpeg_bytes):
    archive = tmpzip({"C:/windows/evil.jpg": jpeg_bytes})
    with pytest.raises(BatchError):
        list(safe_zip_members(archive))


def test_safe_zip_skips_non_images(tmpzip, jpeg_bytes):
    archive = tmpzip({"ok/a.jpg": jpeg_bytes,
                      "ok/readme.txt": b"not an image",
                      "ok/plain.bmp": jpeg_bytes})
    members = list(safe_zip_members(archive))
    # .jpg supported; .bmp supported; .txt ignored
    assert members == ["ok/a.jpg", "ok/plain.bmp"]


def test_read_zip_images_decodes_members(tmpzip, jpeg_bytes):
    archive = tmpzip({"a/one.jpg": jpeg_bytes, "b/two.png": jpeg_bytes})
    items = list(read_zip_images(archive))
    assert len(items) == 2
    for name, data, err in items:
        assert err is None
        assert data is not None


def test_collect_batch_inputs_from_uploads(tmpzip, jpeg_bytes):
    archive = tmpzip({"x/a.jpg": jpeg_bytes})
    items, errors = collect_batch_inputs([("one.jpg", jpeg_bytes),
                                          ("list.zip", archive.getvalue())])
    assert len(errors) == 0
    names = {n for n, _d in items}
    assert "one.jpg" in names
    assert "x/a.jpg" in names


def test_process_batch_reports_failure_without_stopping():
    def worker(name, data):
        if name == "bad.jpg":
            raise RuntimeError("deliberate failure")
        return {"len": len(data)}

    items = [("good1.jpg", b"abc"), ("bad.jpg", b"xyz"), ("good2.jpg", b"q")]
    results = process_batch(items, worker, workers=2)
    by_name = {r["name"]: r for r in results}
    assert by_name["bad.jpg"]["status"] == "failed"
    assert "deliberate failure" in by_name["bad.jpg"]["error"]
    assert by_name["good1.jpg"]["status"] == "ok"
    assert by_name["good2.jpg"]["status"] == "ok"
    assert by_name["good1.jpg"]["duration_ms"] > 0
    assert by_name["good1.jpg"]["result"]["len"] == 3


def test_process_batch_entropy_is_finite(tiny_checker, tmp_path):
    import cv2

    ok, buf = cv2.imencode(".png", tiny_checker)
    assert ok

    def worker(name, data):
        from core.io import decode_image
        cfg = FeatureConfig(max_side=128)
        vec = extract_features(decode_image(data, name=name), cfg)["vector"]
        return {"len": int(vec.size)}

    results = process_batch([("t.png", buf.tobytes())], worker)
    assert results[0]["status"] == "ok"
    assert results[0]["result"]["len"] > 0


def test_write_results_csv_rows_and_header(tmp_path):
    results = [
        {"name": "a.jpg", "status": "ok", "duration_ms": 12.5,
         "error": None,
         "result": {"label": "brick", "confidence": "0.8", "fused_entropy": "1.2"}},
        {"name": "b.jpg", "status": "failed", "duration_ms": 1.0,
         "error": "boom", "result": None},
    ]
    path = write_results_csv(results, str(tmp_path / "batch.csv"))
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["filename", "status", "duration_ms", "error",
                       "predicted_class", "confidence", "entropy"]
    assert rows[1][0] == "a.jpg" and rows[1][1] == "ok"
    assert rows[2][0] == "b.jpg" and rows[2][1] == "failed"
    assert rows[2][3] == "boom"