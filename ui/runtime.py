"""Shared analysis runtime: run the real pipeline and explain its output.

Nothing here fabricates a result. Every number comes from the existing
``core`` algorithms (LBP + optional Gabor / opponent-colour, GLCM metrics and
histogram statistics).
"""

import io
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from classifier import _roughness_score
from core.io import decode_image, ensure_max_size, to_grayscale
from core.pipeline import (
    PipelineError,
    crop_roi,
    extract_features,
    texture_metrics,
)

MAX_SAVED_IMAGE_SIDE = 900


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------

@dataclass
class AnalysisRun:
    image: np.ndarray
    gray: np.ndarray
    hist: np.ndarray
    metrics: dict = field(default_factory=dict)
    config: object = None
    vector: np.ndarray = None
    sizes: dict = field(default_factory=dict)
    box: tuple = (0, 0, 0, 0)
    total_pixels: int = 0
    status: str = "ok"
    error: str = None
    label: str = "—"
    roughness: float = 0.0

    @property
    def descriptor_provider(self):
        return "rule-based descriptor (not a trained classifier)"


def descriptor_label(roughness: float) -> str:
    if roughness < 0.24:
        return "Smooth"
    if roughness < 0.48:
        return "Fine"
    if roughness < 0.72:
        return "Rough"
    return "Coarse"


def run_analysis(image, config, box):
    """Fused extraction on an ROI box (working-resolution coordinates)."""

    try:
        work = ensure_max_size(image, config.max_side)
        cropped, used_box = crop_roi(work, box)
        gray = to_grayscale(cropped)
        extracted = extract_features(cropped, config)

        lbp_block = np.asarray(extracted["blocks"]["lbp"], dtype="float64")
        total = float(lbp_block.sum())
        fused_hist = lbp_block / total if total > 0 else lbp_block

        metrics = texture_metrics(gray, fused_hist)
        rough = _roughness_score(metrics, len(fused_hist))

        return AnalysisRun(
            image=work, gray=gray, hist=fused_hist, metrics=metrics,
            config=config, vector=extracted["vector"], sizes=extracted["sizes"],
            box=used_box, total_pixels=int(gray.size), status="ok",
            label=descriptor_label(rough), roughness=rough,
        )
    except (ValueError, PipelineError, Exception) as exc:  # noqa: BLE001
        try:
            fallback = ensure_max_size(image, config.max_side)
        except Exception:  # noqa: BLE001
            fallback = image
        return AnalysisRun(
            image=fallback, gray=None, hist=None, config=config, box=box,
            status="error", error=str(exc),
        )


def decode_upload(uploaded):
    return decode_image(uploaded.getvalue(), name=uploaded_name_of(uploaded))


def upload_name(uploaded):
    return uploaded_name_of(uploaded)


def uploaded_name_of(uploaded):
    return os.path.basename(
        str(getattr(uploaded, "name", "image")).replace("\\", "/"))


def decode_bytes(data, name):
    return decode_image(data, name=name)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_overlay(work, box):
    """RGB preview of the working image with the ROI rectangle drawn on it."""

    display = ensure_max_size(work, 640)
    sy = display.shape[0] / work.shape[0]
    sx = display.shape[1] / work.shape[1]
    x0, y0, x1, y1 = box
    out = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    cv2.rectangle(
        out, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)),
        (255, 215, 0), 2, lineType=cv2.LINE_AA)
    return out


def histogram_figure(hist, title, accent="#C2185B"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=140)
    hist = np.asarray(hist, dtype="float64")
    x = np.arange(hist.size)
    ax.bar(x, hist, width=0.9, color=accent, edgecolor="none")
    ax.plot(x, hist, color="#FFD166", linewidth=1.1)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def difference_image(gray_a, gray_b):
    """Absolute per-pixel difference of two aligned grayscale images (uint8)."""

    a = np.asarray(gray_a, dtype="float64")
    b = np.asarray(gray_b, dtype="float64")
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    diff = np.abs(a[:h, :w] - b[:h, :w])
    diff = diff / (diff.max() or 1.0)
    return (diff * 255).astype("uint8")


def rgb_to_bytes(rgb, fmt="PNG", quality=95):
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype("uint8")
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ext = ".jpg" if fmt.upper() in ("JPG", "JPEG") else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext == ".jpg" else []
    ok, buffer = cv2.imencode(ext, arr, params)
    if not ok:
        raise ValueError(f"Could not encode image as {ext}.")
    return buffer.tobytes()


def apply_smoothing(rgb, kind, kernel):
    """Optional export-only smoothing (never applied to LBP/Gabor features)."""

    if kind == "None" or kernel <= 1:
        return np.asarray(rgb)
    arr = np.asarray(rgb)
    if kind == "Gaussian blur":
        return cv2.GaussianBlur(arr, (int(kernel), int(kernel)), 0)
    size = int(kernel)
    if arr.ndim == 3:
        size = min(size, 5)
    if size % 2 == 0:
        size += 1
    return cv2.medianBlur(arr, size)


def png_bytes(rgb):
    """Encode any RGB/gray array to PNG bytes (used by the saved-analysis store)."""

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype("uint8")
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("Could not encode PNG.")
    return buffer.tobytes()


# ---------------------------------------------------------------------------
# Metric explanations (used by "Explain this result" and Orio)
# ---------------------------------------------------------------------------

METRIC_HELP = {
    "Contrast": {
        "plain": "How strong the intensity differences between neighbouring "
                 "pixels are. Higher = more pronounced local variation.",
        "reading": "A high value suggests a visually 'busier' pattern; a low "
                   "value suggests a flatter surface.",
        "method": "Mean GLCM contrast across 4 directions at distance 1 on a "
                  "32-level quantised grayscale image.",
        "limit": "It is a pixel-level indicator, not a calibrated physical "
                 "roughness measurement (e.g. Ra).",
    },
    "Dissimilarity": {
        "plain": "Similar to contrast but grows linearly with intensity "
                 "differences instead of quadratically.",
        "reading": "Useful alongside contrast to see whether differences are "
                   "spread out or dominated by a few large jumps.",
        "method": "Mean GLCM dissimilarity across 4 directions.",
        "limit": "Depends on the chosen quantisation (32 levels here).",
    },
    "Homogeneity": {
        "plain": "How similar neighbouring pixels are. Values near 1 mean a "
                 "very uniform surface.",
        "reading": "High = smooth/uniform; low = locally varied.",
        "method": "Mean GLCM homogeneity across 4 directions.",
        "limit": "Influenced by image sharpness and lighting, not only the "
                 "material.",
    },
    "Energy": {
        "plain": "Also called Angular Second Moment: high when the intensity "
                 "pattern is ordered and repetitive.",
        "reading": "High energy = few dominant patterns; low = complex texture.",
        "method": "Mean GLCM energy across 4 directions.",
        "limit": "Scale-dependent; compare images captured at similar settings.",
    },
    "ASM": {
        "plain": "Angular Second Moment, the sum of squared co-occurrence "
                 "probabilities (Energy is its square root).",
        "reading": "Same interpretation as Energy.",
        "method": "Mean GLCM ASM across 4 directions.",
        "limit": "Scale-dependent.",
    },
    "Correlation": {
        "plain": "How predictable a pixel is from its neighbour; high values "
                 "indicate strong linear structure or directionality.",
        "reading": "High = organised, directional texture; low = fine noise.",
        "method": "Mean GLCM correlation across 4 directions.",
        "limit": "Sensitive to repeated patterns and to image resizing.",
    },
    "Entropy": {
        "plain": "Shannon entropy of the normalised LBP histogram (base 2). "
                 "It measures how many different local patterns appear.",
        "reading": "High entropy = many diverse patterns; low = a few "
                   "dominant patterns.",
        "method": "Entropy computed from the fused, normalised LBP histogram.",
        "limit": "Bounded by the number of histogram bins; not comparable "
                 "across different feature configurations.",
    },
    "Uniformity": {
        "plain": "Sum of squared histogram probabilities. High when a few LBP "
                 "patterns dominate the image.",
        "reading": "High = consistent/regular local texture.",
        "method": "Computed from the fused LBP histogram.",
        "limit": "Bounded by bin count.",
    },
    "Mean": {
        "plain": "The average histogram bin index, i.e. where the mass of the "
                 "LBP distribution sits.",
        "reading": "Shifts with the dominant local pattern type.",
        "method": "Expectation of the histogram.",
        "limit": "Bin-index units are not physical.",
    },
    "Std Dev": {
        "plain": "Spread of the LBP histogram around its mean index.",
        "reading": "Higher = more varied local patterns.",
        "method": "Standard deviation of the histogram.",
        "limit": "Bin-index units are not physical.",
    },
    "Roughness": {
        "plain": "A single 0-1 heuristic that blends contrast, inverse "
                 "homogeneity and entropy into a 'how rough does this look' "
                 "indicator.",
        "reading": "Higher = rougher-looking texture.",
        "method": "Rule-based fusion: 0.5*contrast + 0.3*(1-homogeneity) + "
                  "0.2*entropy (each normalised to 0-1).",
        "limit": "This is a descriptor, NOT a trained classifier or a physical "
                 "roughness measurement.",
    },
}


def explain_metric(name: str) -> dict:
    return METRIC_HELP.get(name, {
        "plain": "A texture measurement computed from the analysed region.",
        "reading": "Compare it against other images analysed with the same "
                   "settings.",
        "method": "Computed by the Texurion feature pipeline.",
        "limit": "Interpret together with the visualisations and methodology.",
    })


def metric_summary(run: AnalysisRun) -> str:
    """Plain-language summary grounded only in computed values."""

    if run.hist is None:
        return "No result available."
    entropy = run.metrics.get("Entropy", 0.0)
    homogeneity = run.metrics.get("Homogeneity", 0.0)
    contrast = run.metrics.get("Contrast", 0.0)

    if homogeneity >= 0.65 and contrast < 25:
        observation = "The analysed region looks relatively uniform at the pixel level."
    elif contrast >= 60:
        observation = "The analysed region shows strong local intensity variation."
    else:
        observation = "The analysed region shows a moderate mix of local patterns."

    detail = (
        f"Descriptor label: {run.label} (roughness {run.roughness:.3f}). "
        f"Entropy {entropy:.3f} bits, homogeneity {homogeneity:.3f}, "
        f"contrast {contrast:.2f}."
    )
    return f"{observation} {detail}"


def supported_metric_table(run: AnalysisRun):
    import pandas as pd

    rows = [{"metric": k, "value": round(float(v), 6),
             "type": "GLCM" if k in ("Contrast", "Dissimilarity", "Homogeneity",
                                     "Energy", "Correlation", "ASM")
             else "Histogram statistic"}
            for k, v in run.metrics.items()]
    return pd.DataFrame(rows)


def encode_snapshot_png(rgb, max_side=MAX_SAVED_IMAGE_SIDE):
    """Encode and downscale a preview for storage/export."""

    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    small = ensure_max_size(arr, max_side)
    return png_bytes(small)


def streamlit_image_to_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()
