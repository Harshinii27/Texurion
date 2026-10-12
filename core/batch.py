"""Safe batch processing: ZIP validation, concurrency and CSV export.

ZIP members are validated against path traversal (absolute paths, drive
letters, ``..``) *before* any content is read, and files are decoded from
memory - the archive is never extracted to disk during processing.
"""

import csv
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from .io import MAX_PIXELS, decode_image, is_supported_ext

DEFAULT_WORKERS = 4


class BatchError(ValueError):
    """Raised when a batch cannot be set up (e.g. unsafe archive)."""


def safe_zip_members(zip_path_or_file):
    """Yield safe image member names from a ZIP archive.

    Rejects members whose name could escape the archive (absolute paths,
    drive specs, ``..`` segments). Raises :class:`BatchError` if the archive
    is not a ZIP file. Directory entries and non-image members are skipped.
    """

    if isinstance(zip_path_or_file, str):
        try:
            archive = zipfile.ZipFile(zip_path_or_file)
        except (zipfile.BadZipFile, OSError):
            raise BatchError(f"Not a valid ZIP archive: {zip_path_or_file}")
    else:
        try:
            archive = zipfile.ZipFile(zip_path_or_file)
        except (zipfile.BadZipFile, OSError):
            raise BatchError("Uploaded file is not a valid ZIP archive.")

    seen = set()
    with archive:
        for info in archive.infolist():
            raw = info.filename.replace("\\", "/")
            # Reject absolute paths, drive-absolute paths and parent
            # traversal *before* normalising, so a leading "/" is not lost.
            if raw.startswith("/") or (len(raw) >= 2 and raw[1] == ":"):
                raise BatchError(f"Unsafe path in archive: {info.filename}")
            name = raw.lstrip("/")
            segments = name.split("/")
            if any(segment == ".." for segment in segments) or "//" in name:
                raise BatchError(f"Unsafe path in archive: {info.filename}")
            if info.is_dir():
                continue
            if not is_supported_ext(name):
                continue
            if name in seen:
                continue
            seen.add(name)
            yield name


def read_zip_images(zip_path_or_file):
    """Yield ``(safe_name, bytes)`` for every safe, supported image member.

    Members that fail to decode are reported with their error instead of
    aborting the whole archive.
    """

    if isinstance(zip_path_or_file, str):
        archive = zipfile.ZipFile(zip_path_or_file)
    else:
        archive = zipfile.ZipFile(zip_path_or_file)

    with archive:
        for name in safe_zip_members(zip_path_or_file):
            try:
                info = archive.getinfo(name)
                if info.file_size > MAX_PIXELS * 4:
                    yield name, None, f"member too large ({info.file_size} bytes)"
                    continue
                data = archive.read(name)
                yield name, data, None
            except Exception as exc:  # noqa: BLE001
                yield name, None, f"read error: {exc}"


def collect_batch_inputs(files_uploaded, zip_paths=None):
    """Collect ``(safe_name, data_or_path)`` from uploaded files and zips.

    ``files_uploaded`` is an iterable of ``(name, bytes)`` pairs. Errors are
    returned through the ``errors`` array.
    """

    items = []
    errors = []

    for name, data in files_uploaded or []:
        base = os.path.basename(str(name).replace("\\", "/"))
        if not base:
            continue
        if is_supported_ext(base):
            items.append((base, data))
        elif str(base).lower().endswith(".zip"):
            # Explode in-memory ZIP uploads just like disk archives.
            try:
                stream = io.BytesIO(data)
                for member, member_data, err in read_zip_images(stream):
                    if err:
                        errors.append((member, err))
                    elif member_data:
                        items.append((member, member_data))
            except BatchError as exc:
                errors.append((base, str(exc)))

    for zip_path in zip_paths or []:
        try:
            for name, data, err in read_zip_images(zip_path):
                if err:
                    errors.append((name, err))
                elif data:
                    items.append((name, data))
        except BatchError as exc:
            errors.append((os.path.basename(zip_path), str(exc)))

    return items, errors


def process_batch(files, fn, workers=DEFAULT_WORKERS):
    """Process image bytes concurrently.

    Parameters
    ----------
    files : iterable of (name, bytes)
        Already validated image payloads.
    fn : callable (name: str, data: bytes) -> dict or raises
        Pure worker function; exceptions are captured as failures.
    workers : int
        Thread-pool size. Bounded by default; safe for NumPy/OpenCV work
        that releases the GIL intermittently and for I/O-bound decoding.

    Returns
    -------
    list of dict with keys ``name, status, duration_ms, error, result``.
    """

    workers = max(1, min(int(workers), 8))
    results = []
    total = len(files)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for name, data in files:
            future = executor.submit(_run_one, fn, name, data)
            future_map[future] = name

        for idx, future in enumerate(as_completed(future_map), start=1):
            name = future_map[future]
            try:
                entry = future.result()
            except Exception as exc:  # noqa: BLE001
                entry = {
                    "name": name,
                    "status": "failed",
                    "duration_ms": None,
                    "error": str(exc),
                    "result": None,
                }
            entry["index"] = idx
            entry["total"] = total
            results.append(entry)

    results.sort(key=lambda r: r.get("index", 0))
    return results


def _run_one(fn, name, data):
    started = time.perf_counter()
    try:
        result = fn(name, data)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": name,
            "status": "failed",
            "duration_ms": (time.perf_counter() - started) * 1000,
            "error": str(exc),
            "result": None,
        }

    return {
        "name": name,
        "status": "ok",
        "duration_ms": (time.perf_counter() - started) * 1000,
        "error": None,
        "result": result,
    }


def write_results_csv(results, path, format_row=None):
    """Write batch results to CSV.

    ``format_row`` maps a result dict to a list of row values; the default
    emits filename, status, duration, error and (when present) the result's
    ``label`` / ``confidence`` / ``entropy`` fields.
    """

    def _default_row(e):
        result = e.get("result") or {}
        return [
            e.get("name", ""),
            e.get("status", ""),
            _fmt_ms(e.get("duration_ms")),
            e.get("error") or "",
            result.get("label", ""),
            result.get("confidence", ""),
            result.get("entropy", ""),
        ]

    formatter = format_row or _default_row
    header = ["filename", "status", "duration_ms", "error",
              "predicted_class", "confidence", "entropy"]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for entry in results:
            writer.writerow(formatter(entry))

    return path


def _fmt_ms(value):
    if value is None:
        return ""
    return f"{float(value):.2f}"



def verify_image_bytes(name, data):
    """Decode to ensure a payload is a valid supported image.

    Returns nothing on success; raises otherwise. Small wrapper so batch
    workers can pre-validate before sending blank/invalid uploads to the
    pipeline.
    """

    decode_image(data, name=name)
    return True