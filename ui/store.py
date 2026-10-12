"""Local-first persistence for saved Texurion analyses.

Records are stored as individual JSON files plus PNG previews under
``saved_analyses/`` in the project directory. This is server-local storage:
records do NOT synchronise across devices or across separately deployed
instances. The location is configurable through the ``TEXURION_DATA_DIR``
environment variable.
"""

import csv
import io
import json
import os
import shutil
from datetime import datetime
from uuid import uuid4

from . import runtime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "saved_analyses")


def data_dir():
    root = os.environ.get("TEXURION_DATA_DIR", DEFAULT_DIR)
    os.makedirs(root, exist_ok=True)
    return root


def _new_id(name):
    stem = "".join(c if c.isalnum() else "_" for c in os.path.splitext(name)[0])[:24]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{stem}_{uuid4().hex[:4]}"


def _record_path(record_id):
    return os.path.join(data_dir(), f"{record_id}.json")


def _asset_dir(record_id):
    path = os.path.join(data_dir(), f"{record_id}_assets")
    os.makedirs(path, exist_ok=True)
    return path


def _write_png(array, path):
    with open(path, "wb") as handle:
        handle.write(runtime.encode_snapshot_png(array))


def save_analysis(name, config, roi, run, views, model_prediction=None,
                  workflow=None):
    """Persist one analysis. Returns the new record id."""

    record_id = _new_id(name)
    assets = _asset_dir(record_id)
    files = {}

    preview_map = {
        "original": views.get("original"),
        "grayscale": views.get("grayscale"),
        "lbp": views.get("lbp"),
        "gabor": views.get("gabor"),
    }
    for key, image in preview_map.items():
        if image is None:
            continue
        filename = f"{key}.png"
        _write_png(image, os.path.join(assets, filename))
        files[key] = filename

    try:
        from report import render_histogram_png

        hist_path = os.path.join(assets, "histogram.png")
        if run.hist is not None:
            render_histogram_png(
                run.hist, hist_path,
                title="Fused multiscale LBP histogram")
            files["histogram"] = "histogram.png"
    except Exception:  # noqa: BLE001
        pass

    record = {
        "id": record_id,
        "name": os.path.basename(name),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workflow": workflow,
        "config": config.to_dict(),
        "roi": list(map(int, roi)),
        "roi_pixels": int(run.total_pixels),
        "metrics": {k: float(v) for k, v in run.metrics.items()},
        "descriptor_label": run.label,
        "roughness": float(run.roughness),
        "vector_length": int(run.vector.size) if run.vector is not None else 0,
        "summary": runtime.metric_summary(run),
        "model_prediction": model_prediction,
        "files": files,
    }

    with open(_record_path(record_id), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    return record_id


def list_analyses():
    """Return saved records (metadata + paths), newest first."""

    root = data_dir()
    records = []
    for filename in os.listdir(root):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(root, filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
            record["_path"] = path
            record["_assets"] = os.path.join(root, f"{record['id']}_assets")
            records.append(record)
        except (OSError, json.JSONDecodeError):
            continue
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return records


def load_analysis(record_id):
    path = _record_path(record_id)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    record["_assets"] = os.path.join(data_dir(), f"{record_id}_assets")
    return record


def asset_path(record, key):
    filename = (record.get("files") or {}).get(key)
    if not filename:
        return None
    path = os.path.join(record.get("_assets", ""), filename)
    return path if os.path.isfile(path) else None


def delete_analysis(record_id):
    removed = False
    path = _record_path(record_id)
    if os.path.isfile(path):
        os.remove(path)
        removed = True
    assets = os.path.join(data_dir(), f"{record_id}_assets")
    if os.path.isdir(assets):
        shutil.rmtree(assets, ignore_errors=True)
    return removed


def record_json_bytes(record):
    clean = {k: v for k, v in record.items() if not k.startswith("_")}
    return json.dumps(clean, indent=2).encode("utf-8")


def record_csv_bytes(record):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["field", "value"])
    for key in ("id", "name", "created_at", "workflow", "descriptor_label",
                "roughness", "roi_pixels", "vector_length"):
        writer.writerow([key, record.get(key, "")])
    if record.get("model_prediction"):
        writer.writerow(["predicted_class", record["model_prediction"].get("label", "")])
        writer.writerow(["predicted_confidence",
                         record["model_prediction"].get("confidence", "")])
    writer.writerow(["roi", record.get("roi", "")])
    writer.writerow([])
    writer.writerow(["metric", "value"])
    for metric, value in (record.get("metrics") or {}).items():
        writer.writerow([metric, value])
    return buffer.getvalue().encode("utf-8")
