"""Classifier: real SVM training, model save/load, prediction, data setup."""

import os
import tempfile

import numpy as np
import pytest

from classifier import (
    extract_labeled_dataset,
    load_model,
    predict,
    save_model,
    train_model,
)
from core.pipeline import FeatureConfig


@pytest.fixture
def cfg():
    return FeatureConfig(max_side=64)


@pytest.fixture
def sample_vectors(cfg):
    rng = np.random.default_rng(11)
    d = cfg.descriptor_length()
    X = np.vstack([
        rng.normal(1.0, 0.4, size=(12, d)),
        rng.normal(-1.0, 0.4, size=(12, d)),
    ])
    y = np.array(["brick"] * 12 + ["wood"] * 12)
    return X, y


def test_train_requires_two_classes(cfg):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, cfg.descriptor_length()))
    y = np.array(["only"] * 10)
    with pytest.raises(ValueError):
        train_model(X, y, cfg)


def test_train_requires_enough_samples(cfg):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1, cfg.descriptor_length()))
    with pytest.raises(ValueError):
        train_model(X, np.array(["a"]), cfg)


def test_train_returns_real_metrics(cfg, sample_vectors):
    X, y = sample_vectors
    bundle = train_model(X, y, cfg)
    assert set(bundle["classes"]) == {"brick", "wood"}
    metrics = bundle["metrics"]
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["test_samples"] > 0
    assert bundle["feature_length"] == X.shape[1]


def test_save_load_predict_roundtrip(cfg, sample_vectors):
    X, y = sample_vectors
    bundle = train_model(X, y, cfg)
    with tempfile.TemporaryDirectory(prefix="model_rt_") as tmp:
        path = os.path.join(tmp, "model.pkl")
        model_path, sidecar = save_model(bundle, path)
        assert os.path.isfile(model_path)
        assert os.path.isfile(sidecar)

        loaded = load_model(model_path)
        label, confidence, proba = predict(loaded, X[0])
        assert label in {"brick", "wood"}
        assert 0.0 <= confidence <= 1.0
        assert set(proba) == set(loaded["classes"])
        assert abs(sum(proba.values()) - 1.0) < 1e-6


def test_load_corrupt_model_raises():
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp.write(b"this is not a pickle file at all")
        path = tmp.name
    with pytest.raises(Exception):
        load_model(path)
    os.remove(path)


def test_load_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        load_model(os.path.join("models", "missing.pkl"))


def test_predict_rejects_wrong_feature_length(cfg, sample_vectors):
    X, y = sample_vectors
    bundle = train_model(X, y, cfg)
    with pytest.raises(ValueError):
        predict(bundle, np.zeros(bundle["feature_length"] + 1))


def test_extract_labeled_dataset_skips_bad_files(jpeg_bytes, cfg):
    good = ("brick", "a.jpg", jpeg_bytes)
    bad = ("brick", "broken.jpg", b"not an image")
    X, y, names, errors = extract_labeled_dataset([good, bad, good], cfg)
    assert X.shape[0] == 2
    assert len(errors) == 1
    assert errors[0][0] == "broken.jpg"
    assert X.shape[1] == cfg.descriptor_length()


def test_extract_labeled_dataset_empty_returns_no_data():
    X, y, names, errors = extract_labeled_dataset([], FeatureConfig())
    assert X.shape == (0, 0)