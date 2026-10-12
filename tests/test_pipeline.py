"""Fused pipeline: config roundtrip, dims, ROI, similarity, display views."""

import numpy as np
import pytest

from core.pipeline import (
    FeatureConfig,
    PipelineError,
    crop_roi,
    display_views,
    extract_features,
    similarity_ranks,
    texture_metrics,
)


def test_config_roundtrip_preserves_length():
    cfg = FeatureConfig(include_gabor=True, lbp_scales=((1, 8), (2, 16)))
    restored = FeatureConfig.from_dict(cfg.to_dict())
    assert restored.descriptor_length() == cfg.descriptor_length()
    assert restored.to_dict() == cfg.to_dict()


def test_feature_vector_length_matches_config(tiny_checker):
    for enable_gabor in (False, True):
        cfg = FeatureConfig(include_gabor=enable_gabor, max_side=128)
        extracted = extract_features(tiny_checker, cfg)
        assert extracted["vector"].shape == (cfg.descriptor_length(),)
        assert "lbp" in extracted["blocks"]
        if enable_gabor:
            assert extracted["sizes"]["gabor"] > 0
        if not enable_gabor:
            assert "gabor" not in extracted["blocks"]


def test_feature_vector_colour_opponent_block(tiny_checker):
    cfg = FeatureConfig(include_color_opponent=True, max_side=128)
    before = FeatureConfig(include_color_opponent=False, max_side=128)
    extended = extract_features(tiny_checker, cfg)
    plain = extract_features(tiny_checker, before)
    lbp_len = plain["vector"].size
    assert extended["vector"].size == lbp_len + 3 * lbp_len  # RG, RB, GB pairs
    assert cfg.descriptor_length() == extended["vector"].size
    assert "color_opponent" in extended["blocks"]


def test_roi_crop_clamps_and_validates():
    image = np.zeros((40, 60, 3), dtype="uint8")
    cropped, box = crop_roi(image, (-5, -5, 20, 20))
    assert box == (0, 0, 20, 20)
    assert cropped.shape[:2] == (20, 20)

    with pytest.raises(PipelineError):
        crop_roi(image, (10, 10, 10, 20))  # zero width

    with pytest.raises(PipelineError):
        crop_roi(image, "not a box")


def test_similarity_ranks_closest_first():
    base = np.array([0.5, 0.3, 0.2], dtype="float64")
    gallery = [("b", np.array([0.01, 0.01, 0.98])),
               ("a", base),
               ("c", np.array([0.33, 0.33, 0.34]))]
    ranked = similarity_ranks(base, [(lbl, vec) for lbl, vec in gallery],
                              metric="cosine", top_k=2)
    assert ranked[0][0] == "a"


def test_texture_metrics_are_finite():
    gray = np.random.default_rng(7).integers(0, 256, (64, 64)).astype("uint8")
    metrics = texture_metrics(gray)
    for value in metrics.values():
        assert np.isfinite(value)


def test_tiny_image_clearly_rejected():
    tiny = np.zeros((2, 2, 3), dtype="uint8")
    with pytest.raises(PipelineError):
        extract_features(tiny, FeatureConfig())


def test_display_views_signature(tiny_checker):
    cfg = FeatureConfig(include_gabor=False)
    orig, gray, lbp, gabor = display_views(tiny_checker, cfg)
    assert orig.ndim == 3 and gray.ndim == 3 and lbp.ndim == 3
    assert gabor is None

    cfg2 = FeatureConfig(include_gabor=True)
    _o, _g, _l, gabor2 = display_views(tiny_checker, cfg2)
    assert gabor2 is not None and gabor2.ndim == 3