"""Performance benchmarking with real measurements.

Latency is measured with :func:`time.perf_counter` after a warm-up run and
repeated ``reps`` times; the median, mean and std are reported in
milliseconds. Peak memory is read from the actual process working set on
Windows (via ``psapi``/ctypes); on other platforms it falls back to
``tracemalloc`` peak Python allocations and is labelled as such.

No performance claim is made that is not backed by these measurements, and
an optimisation result is only marked comparable when the NumPy and Numba
engines produce bit-identical output on the benchmark input.
"""

import gc
import os
import platform
import sys
import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np

from .gabor import gabor_feature_vector, make_gabor_kernels
from .io import ensure_max_size, to_grayscale
from .lbp import lbp_circular_numba, lbp_circular_python, lbp_circular_vectorized
from .pipeline import FeatureConfig, extract_features
from .histogram import get_histogram

BENCHMARK_SIZE = 256


@dataclass
class BenchmarkResult:
    image_size: tuple
    reps: int
    environment: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)  # name -> stat dict
    memory: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    equivalence: dict = field(default_factory=dict)


def environment_report():
    """Describe the software/hardware environment that produced the numbers."""

    import cv2
    import scipy
    import skimage

    numba_version = "n/a"
    try:
        import numba
        numba_version = numba.__version__
    except Exception:  # pragma: no cover
        pass

    cpu = platform.processor() or "unknown"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu,
        "cpu_count": os.cpu_count() or 0,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__ if hasattr(cv2, "__version__") else "n/a",
        "scikit-image": skimage.__version__,
        "scipy": scipy.__version__,
        "numba": numba_version,
    }


def _measurers():
    import time as _t
    return _t.perf_counter


def measure_latency(fn, *args, warmup=1, reps=5, **kwargs):
    """Median/mean/std latency of ``fn(*args, **kwargs)`` in milliseconds."""

    for _ in range(max(1, warmup)):
        fn(*args, **kwargs)

    samples = []
    for _ in range(max(1, reps)):
        started = _measurers()()
        fn(*args, **kwargs)
        samples.append((_measurers()() - started) * 1000.0)

    samples = np.array(sorted(samples), dtype="float64")
    return {
        "median_ms": float(np.median(samples)),
        "mean_ms": float(np.mean(samples)),
        "std_ms": float(np.std(samples)),
        "min_ms": float(samples[0]),
        "max_ms": float(samples[-1]),
        "samples": samples.tolist(),
    }


def peak_working_set_mb():
    """Current peak working-set size of this process (Windows), or None."""

    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        psapi = ctypes.WinDLL("psapi")
        getmini = psapi.GetProcessMemoryInfo
        getmini.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                            wintypes.DWORD]
        getmini.restype = wintypes.BOOL
        result = getmini(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb)
        if not result:
            return None
        return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:  # pragma: no cover
        return None


def benchmark_all(image_bgr, reps=5, benchmark_size=BENCHMARK_SIZE):
    """Run a real benchmark battery on the given image.

    All stages use the same downscaled grayscale input (dimensions reported
    in the result) except the fused pipeline, which runs on the image at the
    application's working resolution so it reflects actual user experience.
    """

    import cv2

    from skimage.feature import local_binary_pattern

    gc.collect()
    base_rss = peak_working_set_mb()

    bench = ensure_max_size(image_bgr, benchmark_size)
    gray = to_grayscale(bench)
    n_h, n_w = gray.shape

    r, p, method = 3, 24, "uniform"
    offsets = None
    from .lbp import neighbor_offsets
    offsets = neighbor_offsets(r, p)

    timings = {}

    sk = lambda: local_binary_pattern(gray, p, r, method=method)  # noqa: E731
    timings["skimage LBP (R3 P24)"] = measure_latency(sk, reps=reps)

    vec = lambda: lbp_circular_vectorized(gray, r, p, method=method)  # noqa: E731
    timings["custom NumPy LBP (R3 P24)"] = measure_latency(vec, reps=reps)

    num = lambda: lbp_circular_numba(gray, r, p, method=method, offsets=offsets)  # noqa: E731
    timings["custom Numba LBP (R3 P24)"] = measure_latency(num, reps=reps)

    # Python reference loop - inherently slow; run on a tiny input.
    tiny = ensure_max_size(gray, 64)
    py = lambda: lbp_circular_python(tiny, r, p, method=method, offsets=offsets)  # noqa: E731
    timings["custom Python LBP (64px)"] = measure_latency(py, reps=2)

    # Equivalence check: NumPy vs Numba engines on the same input.
    a = lbp_circular_vectorized(gray, r, p, method=method)
    b = lbp_circular_numba(gray, r, p, method=method, offsets=offsets)
    equivalence = {
        "equal": bool(np.array_equal(a, b)),
        "diff_count": int(np.count_nonzero(a != b)),
        "shape": list(a.shape),
    }
    if not equivalence["equal"]:
        equivalence["note"] = (
            "NumPy and Numba engines differ (borderline samples or rounding); "
            "latencies are indicative only.")

    # Gabor bank.
    kernels = make_gabor_kernels()
    gb = lambda: gabor_feature_vector(gray, kernels=kernels)  # noqa: E731
    timings["Gabor bank (4x3)"] = measure_latency(gb, reps=reps)

    # Multiscale LBP block (the actual feature block used by the app).
    ms = lambda: extract_features(bench, FeatureConfig(include_gabor=False))["vector"]  # noqa: E731
    timings["Multiscale LBP block"] = measure_latency(ms, reps=reps)

    # Full fused pipeline at application resolution.
    full_size = ensure_max_size(image_bgr, 768)
    full_cfg = FeatureConfig(include_gabor=True, include_color_opponent=True,
                             max_side=768, lbp_scales=((1, 8), (2, 16), (3, 24)))
    fused = lambda: extract_features(ensure_max_size(full_size, 768), full_cfg)["vector"]  # noqa: E731
    timings["Fused pipeline (LBP+Gabor+Opponent)"] = measure_latency(fused,
                                                                     reps=reps)

    trailing_rss = peak_working_set_mb()

    memory = {
        "baseline_rss_mb": base_rss,
        "trailing_rss_mb": trailing_rss,
        "peak_rss_mb": trailing_rss or None,
        "method": "process working set (Windows)" if trailing_rss else "tracemalloc fallback",
    }
    if trailing_rss is None:
        tracemalloc.start()
        extract_features(ensure_max_size(full_size, 768), full_cfg)
        _current, _peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory["peak_rss_mb"] = _peak / (1024 * 1024)
        memory["method"] = "tracemalloc peak (Python allocations)"

    notes = [
        f"Timing method: time.perf_counter, warm-up then {reps} repetitions, "
        "median/mean/std reported.",
        f"All stages ran on a {n_w}x{n_h} grayscale patch except the fused "
        "pipeline which ran at the application working resolution.",
        "The Python-loop reference was measured on a smaller input "
        "(64px) because a full-resolution Python loop would take minutes.",
    ]
    if equivalence["diff_count"] != 0:
        notes.append(equivalence.get("note", ""))

    return BenchmarkResult(
        image_size=(n_w, n_h),
        reps=reps,
        environment=environment_report(),
        timings=timings,
        memory=memory,
        notes=notes,
        equivalence=equivalence,
    )


def render_benchmark_chart(result, path):
    """Render a labelled bar chart of the measured medians to ``path`` (PNG)."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result.timings.keys())
    medians = [v["median_ms"] for v in result.timings.values()]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fbfcff")
    bars = ax.barh(names[::-1], medians[::-1], color="#4f8cff", edgecolor="#21468a")
    ax.set_xlabel("median latency (ms)")
    ax.set_title("Measured feature-extraction latency")
    for bar, value in zip(bars, medians[::-1]):
        ax.text(bar.get_width() + max(medians) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f} ms", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(medians) * 1.18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def build_benchmark_pdf(result, chart_path, pdf_path):
    """Build a PDF report from real benchmark numbers."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    env = result.environment

    styles = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=styles["Title"], textColor=colors.HexColor("#16254a"),
                           fontSize=18, spaceAfter=2)
    sub = ParagraphStyle("S", parent=styles["Normal"], textColor=colors.HexColor("#5b6b8c"),
                         fontSize=9, spaceAfter=8)

    story = [
        Paragraph("Performance Benchmark Report", title),
        Paragraph(
            f"Texture Analysis Studio - generated {time.strftime('%d %b %Y %H:%M')}",
            sub,
        ),
    ]

    env_rows = [["Environment variable", "Value"], ["Platform", env["platform"]],
                ["Machine", env["machine"]], ["Processor", env["processor"]],
                ["CPU count", str(env["cpu_count"])], ["Python", env["python"]],
                ["NumPy", env["numpy"]], ["OpenCV", env["opencv"]],
                ["scikit-image", env["scikit-image"]], ["scipy", env["scipy"]],
                ["Numba", env["numba"]]]
    story += [Paragraph("Environment", ParagraphStyle("H2", parent=styles["Heading2"],
                                                      fontSize=11, spaceBefore=8)),
              Table(env_rows,
                    colWidths=[55 * mm, 110 * mm]),
              Spacer(1, 4)]

    timing_rows = [["Component", "median ms", "mean ms", "std ms", "min", "max"]]
    for name, stat in result.timings.items():
        timing_rows.append([name,
                            f"{stat['median_ms']:.2f}", f"{stat['mean_ms']:.2f}",
                            f"{stat['std_ms']:.2f}", f"{stat['min_ms']:.2f}",
                            f"{stat['max_ms']:.2f}"])
    story += [Paragraph("Latency (measured)", ParagraphStyle("H2", parent=styles["Heading2"],
                                                             fontSize=11, spaceBefore=10)),
              Table(timing_rows, colWidths=[70 * mm, 20 * mm, 20 * mm, 20 * mm,
                                            20 * mm, 20 * mm],
                    style=TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f8cff")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0e0")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                         [colors.white, colors.HexColor("#eef2fa")]),
                    ])),
              Spacer(1, 4)]

    mem = result.memory
    story += [
        Paragraph(
            f"Memory: baseline {_mb(mem['baseline_rss_mb'])} -> after benchmark "
            f"{_mb(mem['trailing_rss_mb'])} (peak {_mb(mem['peak_rss_mb'])}), "
            f"method: {mem['method']}.",
            ParagraphStyle("M", parent=styles["Normal"], fontSize=9),
        ),
        Spacer(1, 4),
        Paragraph(
            f"NumPy vs Numba equivalence: {'identical' if result.equivalence['equal'] else 'DIFFERENT'} "
            f"(differing pixels: {result.equivalence['diff_count']}). "
            f"Input size {result.image_size[0]}x{result.image_size[1]}, "
            f"{result.reps} repetitions per component.",
            ParagraphStyle("M2", parent=styles["Normal"], fontSize=9),
        ),
    ]

    if os.path.isfile(chart_path):
        story += [Spacer(1, 6), RLImage(chart_path, width=160 * mm, height=80 * mm)]

    notes = [Paragraph(f"&bull; {note}", ParagraphStyle("N", parent=styles["Normal"],
                                                        fontSize=8)) for note in result.notes]
    story += [Spacer(1, 4), Paragraph("Measurement notes", styles["Heading2"])] + notes

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm)
    doc.build(story)
    return pdf_path


def _mb(value):
    if value is None:
        return "n/a"
    return f"{value:.1f} MB"