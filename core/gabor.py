"""Gabor filter-bank features.

A configurable bank of 2-D Gabor kernels (multiple orientations and
frequencies) is convolved with the grayscale image. For every kernel the
mean and standard deviation of the response magnitude are retained, giving a
fixed-size feature vector of ``2 * frequencies * orientations`` values.
"""

import numpy as np

DEFAULT_ORIENTATIONS = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
DEFAULT_FREQUENCIES = (0.1, 0.25, 0.4)
DEFAULT_KERNEL_SIZE = 17


def make_gabor_kernels(
    ksizes=DEFAULT_KERNEL_SIZE,
    frequencies=DEFAULT_FREQUENCIES,
    orientations=DEFAULT_ORIENTATIONS,
    sigma=None,
    gamma=0.5,
):
    """Build the kernel bank.

    Returns a list of ``(kernel, theta, nu)`` tuples.
    """

    import cv2

    if sigma is None:
        sigma = max(1.0, ksizes / 4.0)

    kernels = []
    for nu in frequencies:
        for theta in orientations:
            kernel = cv2.getGaborKernel(
                (ksizes, ksizes), sigma, theta, nu, gamma)
            kernels.append((kernel / (kernel.sum() + 1e-12), theta, nu))
    return kernels


def gabor_feature_vector(gray, kernels=None, **kwargs):
    """Compute the Gabor magnitude mean/std feature vector.

    Returns a 1-D ``numpy`` array of length
    ``2 * len(frequencies) * len(orientations)``. Dimensionality is fixed by
    the kernel bank configuration, which makes the features safe to reuse
    between training and prediction.
    """

    if gray is None or gray.size == 0:
        raise ValueError("Cannot compute Gabor features on an empty image.")

    import cv2

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    gray = gray.astype("float32") / 255.0

    if kernels is None:
        kernels = make_gabor_kernels(**kwargs)

    features = []
    for kernel, _theta, _nu in kernels:
        response = cv2.filter2D(gray, cv2.CV_32F, kernel)
        magnitude = np.abs(response)
        features.append(float(magnitude.mean()))
        features.append(float(magnitude.std()))

    return np.asarray(features, dtype="float64")


def gabor_descriptor_size(frequencies=DEFAULT_FREQUENCIES,
                          orientations=DEFAULT_ORIENTATIONS):
    """Fixed number of features produced by the Gabor bank."""

    return 2 * len(frequencies) * len(orientations)


def gabor_magnitude_map(gray, kernel, max_side=None):
    """Magnitude response of one (or more) kernels, rescaled to RGB display.

    ``kernel`` may be a single kernel or a list/tuple of kernel banks (the
    responses are summed before normalisation). Returns a uint8 RGB view of
    the grayscale image dimensions.
    """

    import cv2

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if isinstance(kernel, (list, tuple)):
        summed = None
        for entry in kernel:
            response = cv2.filter2D(
                gray.astype("float32"), cv2.CV_32F, entry[0])
            magnitude = np.abs(response)
            summed = magnitude if summed is None else summed + magnitude
    else:
        response = cv2.filter2D(gray.astype("float32"), cv2.CV_32F, kernel)
        summed = np.abs(response)

    mn, mx = float(summed.min()), float(summed.max())
    if mx - mn < 1e-8:
        out = np.zeros_like(summed, dtype="uint8")
    else:
        out = ((summed - mn) / (mx - mn) * 255).astype("uint8")

    if max_side is not None:
        from .io import ensure_max_size
        out = ensure_max_size(out, max_side)

    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)