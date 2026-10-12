"""Fused feature-extraction pipeline.

The pipeline combines optional blocks into a single documented feature
vector:

* **Multiscale LBP** histograms (scikit-image or custom engines).
* **Gabor filter-bank** mean/std statistics.
* **Opponent-colour LBP** histograms for RGB input.

A :class:`FeatureConfig` holds every extraction choice so the exact same
preprocessing and dimensions are used at training and prediction time.
All processing is CPU-only and returns plain NumPy arrays so it can be used
by Streamlit, the desktop GUI, CLIs and tests alike.
"""

import numpy as np

from .gabor import (
    DEFAULT_FREQUENCIES,
    DEFAULT_ORIENTATIONS,
    gabor_descriptor_size,
    gabor_feature_vector,
    make_gabor_kernels,
)
from .histogram import get_histogram, histogram_distance, histogram_statistics
from .io import ensure_max_size, to_grayscale
from .lbp import (
    lbp_multiscale_histograms,
    number_of_patterns,
    opponent_color_lbp_histograms,
)

DEFAULT_MAX_SIDE = 1024


class PipelineError(ValueError):
    """Raised for invalid pipeline configuration or inputs."""


class FeatureConfig:
    """Immutable description of the feature pipeline.

    Attributes
    ----------
    method : str
        LBP method name (scikit-image names).
    engine : str
        ``'skimage'``, ``'numba'`` or ``'numpy'``.
    lbp_scales : tuple of (radius, points)
        Scales for the multiscale LBP block.
    include_gabor : bool
        Whether the Gabor block is included.
    gabor_frequencies / gabor_orientations
        Gabor bank configuration (must match between train and predict).
    include_color_opponent : bool
        Whether opponent-colour LBP histograms are included.
    max_side : int
        Images are downscaled so their largest side equals this before
        feature extraction (consistent pre-processing).
    """

    __slots__ = (
        "method",
        "engine",
        "lbp_scales",
        "include_gabor",
        "gabor_frequencies",
        "gabor_orientations",
        "include_color_opponent",
        "max_side",
    )

    def __init__(
        self,
        method="uniform",
        engine="skimage",
        lbp_scales=((1, 8), (2, 16), (3, 24)),
        include_gabor=False,
        gabor_frequencies=DEFAULT_FREQUENCIES,
        gabor_orientations=DEFAULT_ORIENTATIONS,
        include_color_opponent=False,
        max_side=DEFAULT_MAX_SIDE,
    ):
        lbp_scales = tuple(tuple(map(int, s)) for s in lbp_scales or ())
        for radius, points in lbp_scales:
            if radius < 1 or points < 1:
                raise PipelineError(f"Invalid LBP scale (radius, points)={(radius, points)}")

        self.method = method
        self.engine = engine
        self.lbp_scales = lbp_scales
        self.include_gabor = bool(include_gabor)
        self.gabor_frequencies = tuple(float(f) for f in gabor_frequencies)
        self.gabor_orientations = tuple(float(o) for o in gabor_orientations)
        self.include_color_opponent = bool(include_color_opponent)
        self.max_side = int(max_side)

    def descriptor_length(self, color_channels=1):
        """Total length of the fused feature vector produced by this config."""

        # One grayscale histogram plus three opponent-colour histograms
        # (RG, RB, GB) when the opponent block is enabled.
        n_histograms = 4 if self.include_color_opponent else 1
        lbp_len = (
            sum(number_of_patterns(self.method, p) for _r, p in self.lbp_scales)
            * n_histograms
        )
        gabor_len = self.gabor_descriptor_length() if self.include_gabor else 0
        return int(lbp_len + gabor_len)

    def gabor_descriptor_length(self):
        return gabor_descriptor_size(
            self.gabor_frequencies, self.gabor_orientations)

    def to_dict(self):
        return {
            "method": self.method,
            "engine": self.engine,
            "lbp_scales": [list(map(int, s)) for s in self.lbp_scales],
            "include_gabor": bool(self.include_gabor),
            "gabor_frequencies": list(self.gabor_frequencies),
            "gabor_orientations": list(self.gabor_orientations),
            "include_color_opponent": bool(self.include_color_opponent),
            "max_side": int(self.max_side),
            "descriptor_length": int(self.descriptor_length()),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            method=data["method"],
            engine=data["engine"],
            lbp_scales=[tuple(pair) for pair in data["lbp_scales"]],
            include_gabor=data["include_gabor"],
            gabor_frequencies=data["gabor_frequencies"],
            gabor_orientations=data["gabor_orientations"],
            include_color_opponent=data["include_color_opponent"],
            max_side=data["max_side"],
        )


def preprocess_image(image, config):
    """Downscale an image to the configuration's working resolution."""

    return ensure_max_size(image, config.max_side)


def crop_roi(image, box):
    """Crop a rectangular region ``(x0, y0, x1, y1)`` from an image.

    Coordinates are validated and clamped to the image bounds. Returns a
    ``(cropped, box)`` tuple where ``box`` is the final (possibly clamped)
    rectangle.
    """

    h, w = image.shape[:2]
    try:
        x0, y0, x1, y1 = (int(v) for v in box)
    except (TypeError, ValueError):
        raise PipelineError(
            "ROI must be a 4-tuple of integers (x0, y0, x1, y1).")

    if x1 <= x0 or y1 <= y0:
        raise PipelineError("ROI must have positive width and height.")

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    if x1 <= x0 or y1 <= y0:
        raise PipelineError("ROI lies outside the image bounds.")

    return image[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def extract_lbp_block(gray, config):
    """Multiscale LBP block for a grayscale image."""

    return lbp_multiscale_histograms(
        gray,
        config.lbp_scales,
        method=config.method,
        engine=config.engine,
    )


def extract_gabor_block(gray, config, kernels=None):
    """Gabor mean/std block."""

    if kernels is None:
        kernels = make_gabor_kernels(
            frequencies=config.gabor_frequencies,
            orientations=config.gabor_orientations,
        )
    return gabor_feature_vector(gray, kernels=kernels)


def extract_opponent_block(image, config):
    """Opponent-colour LBP histograms block (requires a 3-channel image)."""

    return opponent_color_lbp_histograms(
        image,
        config.lbp_scales,
        method=config.method,
        engine=config.engine,
    )


def extract_features(image, config):
    """Extract the fused, documented feature vector for an image.

    Returns a dict with ``vector`` (1-D float64), ``sizes`` (block -> length)
    and ``blocks`` (block -> np.ndarray). ``sizes`` documents how the vector
    is laid out so callers can interpret results.
    """

    image = preprocess_image(image, config)
    h, w = image.shape[:2]
    if h < 3 or w < 3:
        raise PipelineError(
            f"Image too small for analysis ({w}x{h}); need at least 3x3.")

    gray = to_grayscale(image)

    blocks = {}
    sizes = {}

    lbp_vec = extract_lbp_block(gray, config)
    blocks["lbp"] = lbp_vec
    sizes["lbp"] = int(lbp_vec.size)

    if config.include_gabor:
        gabor_vec = extract_gabor_block(gray, config)
        blocks["gabor"] = gabor_vec
        sizes["gabor"] = int(gabor_vec.size)

    if config.include_color_opponent:
        if image.ndim != 3:
            raise PipelineError(
                "Opponent-colour LBP requires an RGB image upload.")
        opponent_vec = extract_opponent_block(image, config)
        blocks["color_opponent"] = opponent_vec
        sizes["color_opponent"] = int(opponent_vec.size)

    vector = np.concatenate([blocks[k] for k in blocks])
    vector = vector.astype("float64")

    return {"vector": vector, "sizes": sizes, "blocks": blocks}


def texture_metrics(gray, hist=None):
    """Real texture descriptors (GLCM + histogram statistics).

    Returns plain computed numbers; these are descriptive, not a trained
    classification. Contrast/homogeneity/energy/ASM/correlation come from a
    grey-level co-occurrence matrix; entropy and uniformity come from the
    normalised histogram.
    """

    from skimage.feature import graycomatrix, graycoprops

    gray = np.asarray(gray)
    if gray.max() <= gray.min():
        quantised = np.zeros_like(gray, dtype="uint8")
    else:
        quantised = ((gray.astype("float32") - gray.min())
                     / (gray.max() - gray.min()) * 31).astype("uint8")

    glcm = graycomatrix(
        quantised,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )

    metrics = {
        "Contrast": float(graycoprops(glcm, "contrast").mean()),
        "Dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
        "Homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "Energy": float(graycoprops(glcm, "energy").mean()),
        "Correlation": float(graycoprops(glcm, "correlation").mean()),
        "ASM": float(graycoprops(glcm, "ASM").mean()),
    }

    if hist is not None:
        stats = histogram_statistics(hist)
        metrics["Entropy"] = stats["entropy"]
        metrics["Uniformity"] = stats["uniformity"]
        metrics["Mean"] = stats["mean"]
        metrics["Std Dev"] = stats["std"]

    return metrics


def display_views(image, config, lbp_viz_radius=3, lbp_viz_points=None):
    """Return RGB previews for the analyzer: ``(original, gray, lbp, gabor)``.

    ``gabor`` is ``None`` unless Gabor is enabled in ``config``.
    These views downscale for display and do not affect feature extraction.
    """

    import cv2

    from .lbp import lbp_visualize, validate_lbp_params

    image = preprocess_image(image, config)
    gray = to_grayscale(image)

    original_rgb = cv2.cvtColor(
        ensure_max_size(image, 640), cv2.COLOR_BGR2RGB)
    gray_rgb = cv2.cvtColor(
        ensure_max_size(gray, 640), cv2.COLOR_GRAY2RGB)

    radius, points, _method = validate_lbp_params(
        gray, lbp_viz_radius, lbp_viz_points or 8 * lbp_viz_radius, "uniform")
    from skimage.feature import local_binary_pattern

    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    lbp_rgb = lbp_visualize(lbp, max_side=640)

    gabor_rgb = None
    if config.include_gabor:
        kernels = make_gabor_kernels(
            frequencies=config.gabor_frequencies,
            orientations=config.gabor_orientations,
        )
        from .gabor import gabor_magnitude_map

        summed = None
        for kernel, _t, _n in kernels:
            resp = np.abs(
                cv2.filter2D(gray.astype("float32"), cv2.CV_32F, kernel))
            summed = resp if summed is None else summed + resp
        mn, mx = float(summed.min()), float(summed.max())
        if mx - mn < 1e-8:
            gabor8 = np.zeros_like(gray, dtype="uint8")
        else:
            gabor8 = ((summed - mn) / (mx - mn) * 255).astype("uint8")
        gabor_rgb = cv2.cvtColor(
            ensure_max_size(gabor8, 640), cv2.COLOR_GRAY2RGB)

    return original_rgb, gray_rgb, lbp_rgb, gabor_rgb


def similarity_ranks(query_vector, gallery, metric="chi_square", top_k=5):
    """Rank a gallery of real feature vectors by distance to a query.

    ``gallery`` is an iterable of ``(label, vector)``. Returns a list of
    ``(label, distance)`` sorted ascending. Distances are computed, never
    fabricated.
    """

    results = []
    for label, vector in gallery:
        distance = histogram_distance(query_vector, vector, metric=metric)
        results.append((label, distance))

    results.sort(key=lambda item: item[1])
    return results[:top_k]