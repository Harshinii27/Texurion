"""Local Binary Pattern extraction.

Two families of implementations are provided:

* **skimage engine** - :func:`local_binary_pattern` from scikit-image. This is
  the well-tested library implementation and the default for all feature
  extraction in the app. Supported methods are the ones scikit-image exposes:
  ``'uniform'``, ``'nri_uniform'``, ``'ror'``, ``'var'``, ``'default'``.

* **custom engines** (:func:`lbp_circular_vectorized`,
  :func:`lbp_circular_numba`, :func:`lbp_circular_python`) - a classic Ojala
  circular-sampling uniform LBP implemented in this project. All three
  implementations sample the *same* precomputed circular coordinates so their
  results are bit-for-bit equivalent (verified by tests). The Numba variant is
  compiled with ``@njit`` and is used by the performance dashboard to show a
  measured optimization. These are NOT claimed to be faster than skimage; the
  benchmarking tab reports real, measured numbers.

Boundary handling for the custom engines: out-of-bounds sample coordinates are
clamped to the image border (edge replication). Parameters are validated by
:func:`validate_lbp_params`.
"""

import numpy as np
from skimage.feature import local_binary_pattern

from .io import SUPPORTED_EXTENSIONS, read_image, to_grayscale

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "LBP_METHODS",
    "generate_lbp",
    "validate_lbp_params",
    "number_of_patterns",
    "lbp_visualize",
    "lbp_circular_vectorized",
    "lbp_circular_numba",
    "lbp_circular_python",
    "lbp_multiscale_histograms",
    "opponent_color_lbp_histograms",
]

LBP_METHODS = (
    "uniform",
    "nri_uniform",
    "ror",
    "var",
    "default",
)

_DEFAULT_METHOD = "uniform"


def validate_lbp_params(gray, radius, points, method):
    """Validate LBP arguments; raises ``ValueError`` with a clear message."""

    if gray is None or gray.size == 0:
        raise ValueError("Empty image provided for LBP.")

    try:
        radius = int(radius)
        points = int(points)
    except (TypeError, ValueError):
        raise ValueError("radius and points must be integers.")

    if radius < 1:
        raise ValueError("radius must be >= 1.")
    if points < 1:
        raise ValueError("points (neighbors P) must be >= 1.")
    if max(points, 8) > 1000:
        raise ValueError("points P is unreasonably large.")

    h, w = gray.shape[:2]
    min_side = 2 * radius + 1
    if h < 3 or w < 3:
        raise ValueError(
            "Image is too small to apply LBP (need at least 3x3).")
    if min(h, w) < min_side:
        # We still run using edge replication, but warn the caller through
        # the raised minimum requirement only when it is clearly unusable.
        pass

    if method not in LBP_METHODS:
        raise ValueError(
            f"Unknown LBP method '{method}'. Options: {LBP_METHODS}")

    return radius, points, method


def generate_lbp(path=None, radius=3, points=None, method="uniform", gray=None):
    """Compute a skimage LBP map for an image on disk or a gray array.

    Returns ``(gray, lbp)``. Kept for backward compatibility with the
    desktop application and CLI.
    """

    if gray is None:
        image = read_image(path)
        gray = to_grayscale(image)

    radius, points, method = validate_lbp_params(
        gray, radius, points or 8 * radius, method or _DEFAULT_METHOD)

    lbp = local_binary_pattern(gray, points, radius, method=method)
    return gray, lbp


def number_of_patterns(method, points):
    """Return the number of histogram bins for an LBP method.

    The counts match the label domains produced by scikit-image's
    ``local_binary_pattern``:

    * ``uniform``     -> ``P + 2`` labels (0..P plus the non-uniform bucket).
    * ``nri_uniform`` -> ``P * (P - 1) + 3`` labels.
    * ``ror`` / ``default`` -> ``min(2**P, 256)`` bins spanning ``[0, 2**P)``.
    * ``var``         -> a fixed 256-bin histogram (continuous response).

    Keeping a fixed, method-dependent bin count means the descriptor length
    stays independent of the image content.
    """

    points = int(points)
    if method == "uniform":
        return points + 2
    if method == "nri_uniform":
        return points * (points - 1) + 3
    # 'ror', 'default' and any unknown label method: bounded label domain.
    return min(int(2 ** min(points, 8)), 256)


def _lbp_histogram_range(method, points, bins, lbp):
    """Return the ``(lo, hi)`` histogram range for an LBP map.

    ``uniform``/``nri_uniform`` labels densely cover ``[0, bins)``. ``ror``
    and ``default`` produce integer codes in ``[0, 2**P)`` (sparse), so the
    range must span the whole code domain even though only ``bins`` bins are
    used. ``var`` is a continuous response, so its range follows the data.
    """

    if method in ("uniform", "nri_uniform", "ror", "default"):
        # ror/default keep a fixed bin count but must span the full code domain.
        if method in ("ror", "default"):
            return (0, int(2 ** int(points)))
        return (0, bins)

    if method == "var":
        lo = float(np.nanmin(lbp))
        hi = float(np.nanmax(lbp))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return (0.0, 1.0)
        return (lo, hi)

    return (0, max(bins, 1))


def lbp_visualize(lbp, max_side=None):
    """Turn an LBP map into a colour-mapped RGB visualisation (uint8)."""

    from .io import ensure_max_size

    lbp = np.asarray(lbp, dtype="float32")
    lo, hi = float(lbp.min()), float(lbp.max())

    if hi - lo < 1e-8:
        normalised = np.zeros_like(lbp, dtype="uint8")
    else:
        normalised = ((lbp - lo) / (hi - lo) * 255.0).astype("uint8")

    if max_side is not None:
        normalised = ensure_max_size(normalised, max_side)

    return cv2_apply_colormap(normalised)


def cv2_apply_colormap(gray):
    import cv2

    coloured = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Custom circular uniform LBP (Ojala et al.)
# ---------------------------------------------------------------------------

def neighbor_offsets(radius, points):
    """Circular sampling coordinates ``(dx, dy)`` for ``P`` neighbours.

    Sampled once with NumPy so every engine interpolates at identical
    floating point coordinates (required for engine equivalence).
    """

    theta = 2.0 * np.pi * np.arange(points) / float(points)
    offsets = np.column_stack(
        (np.cos(theta) * radius, np.sin(theta) * radius))
    return np.ascontiguousarray(offsets, dtype=np.float64)


def _uniform_pattern_index(bits, points):
    """Standard uniform mapping: count of ones for uniform, P+1 otherwise."""

    transitions = 0
    for k in range(points - 1):
        if bits[k] != bits[k + 1]:
            transitions += 1
    if bits[points - 1] != bits[0]:
        transitions += 1
    if transitions <= 2:
        return int(sum(bits))
    return points + 1


def lbp_circular_vectorized(gray, radius, points, method="uniform"):
    """Custom circular uniform LBP, vectorised with NumPy.

    Out-of-bounds samples are edge-clamped. For ``method`` only ``uniform``
    and ``default`` (raw weight-weighted code) are supported here; other
    scikit-image methods should use :func:`generate_lbp`.
    """

    radius, points, method = validate_lbp_params(gray, radius, points, method)
    if method not in ("uniform", "default"):
        raise ValueError(
            f"Custom engine does not implement method '{method}'. "
            "Use generate_lbp (skimage) for that method.")

    img = gray.astype(np.float64)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    offsets = neighbor_offsets(radius, points)

    samples = np.zeros((points, h, w), dtype=np.float64)
    for k in range(points):
        dx, dy = offsets[k, 0], offsets[k, 1]
        x = np.clip(xx + dx, 0, w - 1)
        y = np.clip(yy + dy, 0, h - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        wx = (x - x0).astype(np.float64)
        wy = (y - y0).astype(np.float64)
        a = img[y0, x0]
        b = img[y0, x1]
        c = img[y1, x0]
        d = img[y1, x1]
        samples[k] = (a * (1 - wx) * (1 - wy) + b * wx * (1 - wy)
                      + c * (1 - wx) * wy + d * wx * wy)

    bits = (samples >= img[None, :, :]).astype(np.uint8)

    if method == "default":
        weights = (2 ** np.arange(points)).reshape(points, 1, 1).astype(np.uint64)
        code = np.sum(bits.astype(np.uint64) * weights, axis=0)
        return code

    # uniform mapping
    transitions = np.zeros((h, w), dtype=np.int64)
    for k in range(points - 1):
        transitions += (bits[k] != bits[k + 1])
    transitions += (bits[points - 1] != bits[0])
    ones = bits.sum(axis=0).astype(np.int64)
    pattern = np.where(transitions <= 2, ones, points + 1)
    return pattern.astype(np.int64)


def lbp_circular_numba(gray, radius, points, method="uniform", offsets=None):
    """Same algorithm as :func:`lbp_circular_vectorized`, compiled with Numba.

    ``offsets`` must come from :func:`neighbor_offsets` so both engines use
    identical sample coordinates. If Numba is unavailable this falls back to
    the pure Python engine with equivalent output.
    """

    radius, points, method = validate_lbp_params(gray, radius, points, method)
    img = np.ascontiguousarray(gray.astype(np.float64))

    if offsets is None:
        offsets = neighbor_offsets(radius, points)
    offsets = np.ascontiguousarray(offsets, dtype=np.float64)

    try:
        from numba import njit
    except Exception:  # pragma: no cover - depends on environment
        return _lbp_loop_impl(img, radius, points, method, offsets)

    compiled = njit(cache=True)(_lbp_loop_impl)
    return compiled(img, radius, points, method, offsets)


def lbp_circular_python(gray, radius, points, method="uniform", offsets=None):
    """Pure Python reference implementation (same algorithm)."""

    radius, points, method = validate_lbp_params(gray, radius, points, method)
    img = np.ascontiguousarray(gray.astype(np.float64))
    if offsets is None:
        offsets = neighbor_offsets(radius, points)
    offsets = np.ascontiguousarray(offsets, dtype=np.float64)
    return _lbp_loop_impl(img, radius, points, method, offsets)


def _lbp_loop_impl(img, radius, points, method, offsets):
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.int64)
    bits = np.zeros(points, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            center = img[i, j]
            for k in range(points):
                x = min(max(j + offsets[k, 0], 0.0), w - 1)
                y = min(max(i + offsets[k, 1], 0.0), h - 1)
                x0 = int(np.floor(x))
                y0 = int(np.floor(y))
                x1 = min(x0 + 1, w - 1)
                y1 = min(y0 + 1, h - 1)
                wx = x - x0
                wy = y - y0
                a = img[y0, x0]
                b = img[y0, x1]
                c = img[y1, x0]
                d = img[y1, x1]
                val = (a * (1.0 - wx) * (1.0 - wy) + b * wx * (1.0 - wy)
                       + c * (1.0 - wx) * wy + d * wx * wy)
                bits[k] = 1 if val >= center else 0

            if method == "default":
                code = 0
                for k in range(points):
                    code |= int(bits[k]) << k
                out[i, j] = code
            else:
                # standard uniform mapping: count of ones for uniform, P+1 otherwise
                transitions = 0
                ones = 0
                for k in range(points - 1):
                    ones += int(bits[k])
                    if bits[k] != bits[k + 1]:
                        transitions += 1
                ones += int(bits[points - 1])
                if bits[points - 1] != bits[0]:
                    transitions += 1
                out[i, j] = ones if transitions <= 2 else points + 1
    return out


LBP_ENGINES = {
    "skimage": generate_lbp,
    "numba": lbp_circular_numba,
    "numpy": lbp_circular_vectorized,
}


# ---------------------------------------------------------------------------
# Multiscale and opponent-colour histograms
# ---------------------------------------------------------------------------

def lbp_multiscale_histograms(
    gray, scales, method="uniform", engine="skimage", n_bins=None
):
    """Concatenate normalised LBP histograms for several scales.

    Parameters
    ----------
    gray : numpy.ndarray
        Grayscale image.
    scales : list of (radius, points)
        One pair per scale, e.g. ``[(1, 8), (2, 16), (3, 24)]``.
    method : str
        LBP method (scikit-image names).
    engine : str
        ``'skimage'`` (default), ``'numba'`` or ``'numpy'``.
    n_bins : int, optional
        Fixed bin count per scale histogram. Defaults to the natural bin
        count for the selected method.

    Returns
    -------
    numpy.ndarray
        1-D concatenated normalised histogram. Its length equals
        ``sum(bins per scale)`` and is documented by ``descriptor_shape``.
    """

    if engine not in LBP_ENGINES:
        raise ValueError(f"Unknown LBP engine '{engine}'. Options: "
                         + ", ".join(LBP_ENGINES))

    histograms = []
    for radius, points in scales:
        radius, points, method = validate_lbp_params(
            gray, radius, points, method)
        bins = n_bins or number_of_patterns(method, points)
        if engine == "skimage":
            lbp = local_binary_pattern(gray, points, radius, method=method)
        else:
            lbp = LBP_ENGINES[engine](gray, radius, points, method=method)
        hist_range = _lbp_histogram_range(method, points, bins, lbp)
        hist = np.histogram(lbp.ravel(), bins=bins, range=hist_range)[0]
        hist = hist.astype("float64")
        total = hist.sum()
        if total > 0:
            hist /= total
        histograms.append(hist)

    return np.concatenate(histograms)


def opponent_color_lbp_histograms(image, scales, method="uniform",
                                  engine="skimage"):
    """Opponent-colour LBP histograms for an RGB image.

    For each colour channel pair (c1, c2) a difference image
    ``(c1 - c2)`` is formed (which is invariant to the *sum* of channel
    intensities), normalised to ``[0, 255]``, and a uniform LBP histogram
    is computed on it. The histograms of every pair are concatenated.

    This is a real opponent-colour scheme (Maenpaa & Pietikainen 2004) and
    is intentionally separate from plain grayscale LBP.

    Returns the concatenated normalised histogram vector.
    """

    import cv2

    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Opponent-colour LBP requires a 3-channel RGB image.")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    channels = [rgb[:, :, 0].astype(np.int16),
                rgb[:, :, 1].astype(np.int16),
                rgb[:, :, 2].astype(np.int16)]

    histograms = []
    for i in range(3):
        for j in range(i + 1, 3):
            diff = np.clip((channels[i] - channels[j] + 255) / 2.0, 0, 255)
            gray = diff.astype(np.uint8)
            hist = lbp_multiscale_histograms(
                gray, scales, method=method, engine=engine)
            histograms.append(hist)

    return np.concatenate(histograms)