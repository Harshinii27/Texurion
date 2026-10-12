"""Desktop-application analysis orchestrator (used by ``gui.py``).

Provides a simple, stable API for the Tkinter app: load an image, compute
the LBP map, histogram, texture metrics and heuristic descriptor, and
prepare RGB previews for display.
"""

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from classifier import classify_texture, compute_features, interpret
from core.histogram import get_histogram
from core.io import ensure_max_size, read_image, to_grayscale
from core.lbp import generate_lbp, lbp_visualize, number_of_patterns

_PROCESS_MAX = 1024
_DISPLAY_MAX = 640


@dataclass
class AnalysisResult:
    name: str
    path: str
    radius: int
    points: int
    method: str
    nbins: int
    gray: np.ndarray
    lbp: np.ndarray
    hist: np.ndarray
    original_rgb: np.ndarray = None
    features: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)

    @property
    def label(self):
        return self.classification.get("label", "Unknown")

    @property
    def confidence(self):
        return self.classification.get("confidence", 0.0)

    @property
    def interpretation(self):
        return interpret(self.classification)


def colorize_lbp(lbp, max_side=None):
    """RGB visualisation of an LBP map (coloured)."""

    return lbp_visualize(lbp, max_side=max_side)


def prepare_display(path, radius=3, points=None, method="uniform"):
    """Load an image and return ``(original_rgb, gray_rgb, lbp_rgb)``."""

    image = read_image(path)
    process = ensure_max_size(image, _PROCESS_MAX)

    gray = to_grayscale(process)
    _, lbp = generate_lbp(radius=radius, points=points, method=method, gray=gray)

    original_rgb = cv2.cvtColor(ensure_max_size(process, _DISPLAY_MAX),
                                cv2.COLOR_BGR2RGB)
    gray_rgb = cv2.cvtColor(ensure_max_size(gray, _DISPLAY_MAX),
                            cv2.COLOR_GRAY2RGB)

    return original_rgb, gray_rgb, colorize_lbp(lbp, _DISPLAY_MAX)


def analyze(path, radius=3, points=None, method="uniform"):
    """Run the full desktop analysis pipeline on an image file."""

    image = read_image(path)
    process = ensure_max_size(image, _PROCESS_MAX)

    gray = to_grayscale(process)
    _, lbp = generate_lbp(radius=radius, points=points, method=method, gray=gray)

    actual_points = int(points) if points else 8 * int(radius)
    nbins = number_of_patterns(method, actual_points)

    hist = get_histogram(lbp, bins=nbins)
    features = compute_features(gray, lbp, hist)
    classification = classify_texture(hist, features=features, nbins=nbins)

    return AnalysisResult(
        name=os.path.splitext(os.path.basename(path))[0],
        path=path,
        radius=int(radius),
        points=actual_points,
        method=method,
        nbins=nbins,
        gray=gray,
        lbp=lbp,
        hist=hist,
        original_rgb=cv2.cvtColor(process, cv2.COLOR_BGR2RGB),
        features=features,
        classification=classification,
    )


def previews(result, max_side=_DISPLAY_MAX):
    """Return RGB display previews ``(original, gray, lbp)`` for a result."""

    original = ensure_max_size(result.original_rgb, max_side)
    gray_rgb = cv2.cvtColor(ensure_max_size(result.gray, max_side),
                            cv2.COLOR_GRAY2RGB)
    return original, gray_rgb, colorize_lbp(result.lbp, max_side)