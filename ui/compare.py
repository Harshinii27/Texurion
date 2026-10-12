"""Two-image comparison engine.

Pixel-level similarity, LBP-histogram distances at the texture level, and
full feature-vector similarity are computed separately. They are related but
NOT interchangeable, and the report states which is which.
"""

import numpy as np

from core.histogram import histogram_distance
from core.io import ensure_max_size, to_grayscale
from core.pipeline import extract_features, texture_metrics

RADAR_METRICS = ("Contrast", "Homogeneity", "Energy", "Correlation", "Entropy",
                 "Uniformity")


def _fused_lbp_hist(image, config):
    extracted = extract_features(image, config)
    lbp = np.asarray(extracted["blocks"]["lbp"], dtype="float64")
    total = float(lbp.sum())
    return (lbp / total if total > 0 else lbp), extracted["vector"]


def compare_images(image_a, image_b, config):
    """Return a dict of real, computed comparison measurements."""

    from skimage.metrics import structural_similarity as ssim

    gray_a = to_grayscale(ensure_max_size(image_a, config.max_side))
    gray_b = to_grayscale(ensure_max_size(image_b, config.max_side))

    # Pixel metrics require identical geometry: align B onto A's grid.
    import cv2

    b_aligned = cv2.resize(
        gray_b, (gray_a.shape[1], gray_a.shape[0]),
        interpolation=cv2.INTER_AREA)

    mae = float(np.mean(np.abs(gray_a.astype("float64") - b_aligned.astype("float64"))))
    ssim_value = float(ssim(gray_a, b_aligned, data_range=255))
    diff = np.abs(gray_a.astype("float64") - b_aligned.astype("float64"))
    diff_norm = diff / (diff.max() or 1.0)
    diff_image = (diff_norm * 255).astype("uint8")

    hist_a, vector_a = _fused_lbp_hist(image_a, config)
    hist_b, vector_b = _fused_lbp_hist(image_b, config)

    chi2 = histogram_distance(hist_a, hist_b, metric="chi_square")
    cosine_hist = histogram_distance(hist_a, hist_b, metric="cosine")

    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    feature_cosine = 1.0 - float(np.dot(vector_a, vector_b) / denom) if denom else 1.0
    feature_similarity = max(0.0, 1.0 - feature_cosine)

    metrics_a = texture_metrics(gray_a, hist_a)
    metrics_b = texture_metrics(gray_b, hist_b)
    deltas = {k: float(metrics_b[k] - metrics_a[k])
              for k in metrics_a if k in metrics_b}

    return {
        "ssim": ssim_value,
        "mae": mae,
        "lbp_chi_square": float(chi2),
        "lbp_cosine": float(cosine_hist),
        "feature_cosine": float(feature_cosine),
        "feature_similarity": feature_similarity,
        "diff_image": diff_image,
        "gray_a": gray_a,
        "gray_b": gray_b,
        "hist_a": hist_a,
        "hist_b": hist_b,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "deltas": deltas,
        "config": config.to_dict(),
        "shapes": {
            "a": (int(image_a.shape[1]), int(image_a.shape[0])),
            "b": (int(image_b.shape[1]), int(image_b.shape[0])),
        },
    }


def radar_figure(result):
    """Radar chart of normalised metrics for the two images."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [m for m in RADAR_METRICS if m in result["metrics_a"]]
    if len(labels) < 3:
        return None

    values_a = np.array([result["metrics_a"][m] for m in labels], dtype="float64")
    values_b = np.array([result["metrics_b"][m] for m in labels], dtype="float64")
    peak = np.maximum(np.maximum(np.abs(values_a), np.abs(values_b)), 1e-9)
    norm_a = values_a / peak
    norm_b = values_b / peak

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    norm_a = np.concatenate([norm_a, norm_a[:1]])
    norm_b = np.concatenate([norm_b, norm_b[:1]])

    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=140, subplot_kw={"polar": True})
    ax.plot(angles, norm_a, color="#C2185B", linewidth=1.8, label="Image A")
    ax.fill(angles, norm_a, color="#C2185B", alpha=0.18)
    ax.plot(angles, norm_b, color="#1B7F5A", linewidth=1.8, label="Image B")
    ax.fill(angles, norm_b, color="#1B7F5A", alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=8)
    ax.set_title("Normalised texture profile", fontsize=10)
    fig.tight_layout()
    return fig


def summary_text(result):
    if result["feature_similarity"] >= 0.9:
        relation = "very similar texture feature profiles"
    elif result["feature_similarity"] >= 0.7:
        relation = "moderately similar texture feature profiles"
    else:
        relation = "clearly different texture feature profiles"

    return (
        f"The two images show {relation}. Pixel similarity (SSIM) is "
        f"{result['ssim']:.3f} and mean absolute pixel difference is "
        f"{result['mae']:.2f}/255. LBP histogram chi-square distance is "
        f"{result['lbp_chi_square']:.4f}. Pixel, texture-histogram and "
        "feature-level similarity are distinct measures; treat them together."
    )
