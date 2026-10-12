"""Classification and texture description.

Two clearly separated facilities live here:

1. **Trained classification** (:func:`train_model`, :func:`load_model`,
   :func:`predict`) - a real scikit-learn SVM trained on user-supplied
   labelled data. Probabilities come from the SVM's Platt-calibrated
   ``predict_proba``. Nothing here pretends a model exists when none does.

2. **Heuristic texture description** (:func:`classify_texture`,
   :func:`compute_features`) - a deterministic, rule-based descriptor used
   by the desktop application. It is explicitly *not* a trained classifier
   and is labelled as such wherever it appears.
"""

import json
import os
from datetime import datetime

import numpy as np

from core.io import decode_image
from core.pipeline import FeatureConfig, extract_features, texture_metrics
from core.histogram import histogram_statistics

MODEL_FORMAT_VERSION = 1

TEXTURE_CLASSES = {
    "Smooth": 0.10,
    "Fine": 0.38,
    "Rough": 0.65,
    "Coarse": 0.90,
}

_SIGMA = 0.22


# ---------------------------------------------------------------------------
# 1) Real trained classifier
# ---------------------------------------------------------------------------

def extract_labeled_dataset(files, config):
    """Build feature matrix ``X`` and label vector ``y`` from ``(label, name, bytes)``.

    Files that fail to decode or extract are reported in the returned
    ``errors`` list and skipped, so a partially corrupt upload does not
    abort training.
    """

    X, y, names = [], [], []
    errors = []
    for label, name, data in files:
        try:
            image = decode_image(data, name=name)
            vector = extract_features(image, config)["vector"]
            X.append(vector)
            y.append(str(label))
            names.append(name)
        except Exception as exc:  # noqa: BLE001
            errors.append((name, str(exc)))

    if not X:
        return (np.zeros((0, 0)), np.array([], dtype="<U1"),
                np.array([], dtype="<U1"), errors)

    return (np.vstack(X).astype("float64"), np.asarray(y),
            np.asarray(names), errors)


def train_model(X, y, config, test_size=0.25, random_state=42):
    """Train a real classifier on labelled features.

    Data is split into train/test *after* feature extraction (the same
    config is used for both), stratifying on class when every class has at
    least two samples. The returned bundle holds the model, scaler, classes,
    feature config and measured test metrics - ready to save and reuse.

    Returns a dict (model bundle). Raises ``ValueError`` for degenerate data.
    """

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, classification_report

    X = np.asarray(X, dtype="float64")
    y = np.asarray(y)

    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(
            "Training needs at least two labelled samples per class.")
    if X.shape[1] == 0:
        raise ValueError("Feature vectors are empty.")

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Training requires at least two distinct classes.")

    stratify = y if min(counts) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=stratify)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Real calibrated probabilities: fit calibration on top of an RBF SVM.
    base_svc = SVC(random_state=random_state)
    calib_cv = 2 if min(counts) < 3 else 3
    clf = CalibratedClassifierCV(
        estimator=base_svc, method="sigmoid", ensemble=False,
        cv=calib_cv)
    clf.fit(X_train_s, y_train)

    train_pred = clf.predict(X_train_s)
    test_pred = clf.predict(X_test_s)

    from sklearn.metrics import confusion_matrix

    return {
        "format_version": MODEL_FORMAT_VERSION,
        "model_type": "sklearn-svm-rbf",
        "classifier": clf,
        "scaler": scaler,
        "classes": list(clf.classes_),
        "feature_config": config.to_dict(),
        "feature_length": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "test_samples": int(X_test.shape[0]),
            "test_size": float(test_size),
        },
        "classification_report": classification_report(
            y_test, test_pred, zero_division=0,
            output_dict=False) if len(classes) > 2
        else classification_report(y_test, test_pred, zero_division=0,
                                   output_dict=False),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "confusion_labels": list(classes),
    }


def save_model(bundle, path):
    """Persist a trained model bundle to ``path`` (joblib) plus JSON sidecar."""

    try:
        import joblib
    except ImportError:
        raise ImportError("scikit-learn's joblib is required to save models.")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(bundle, path)

    sidecar = os.path.splitext(path)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(_bundle_metadata(bundle), handle, indent=2, default=str)

    return path, sidecar


_REQUIRED_BUNDLE_KEYS = ("classifier", "scaler", "classes", "feature_config")


def _upgrade_real_bundle(bundle):
    """Normalise a ``training/train_real.py`` bundle to the canonical schema.

    The research training pipeline stores the same core artefacts (scaler,
    classifier, classes, feature config) under different keys and without a
    ``format_version``. Filling in the canonical fields lets the saved model be
    loaded by the normal inference code without retraining or rewriting the
    file.
    """

    upgraded = dict(bundle)
    upgraded["format_version"] = MODEL_FORMAT_VERSION

    counts = (bundle.get("splits") or {}).get("counts") or {}
    real_metrics = bundle.get("metrics_test") or {}

    upgraded.setdefault("model_type", bundle.get("best_model") or "texurion-real")
    upgraded.setdefault("n_samples", counts.get("train"))

    metrics = dict(bundle.get("metrics") or {})
    metrics.setdefault("test_accuracy", real_metrics.get("accuracy"))
    metrics.setdefault("test_samples", counts.get("test"))
    metrics.setdefault("train_accuracy", None)
    metrics.setdefault("test_size", None)
    upgraded["metrics"] = metrics

    upgraded.setdefault("confusion_labels", list(bundle.get("classes", [])))
    return upgraded


def load_model(path):
    """Load a model bundle, validating its structure and version.

    Accepts both the canonical bundle written by :func:`train_model` and the
    real-dataset bundle written by ``training/train_real.py`` (which predates
    the ``format_version`` field), normalising the latter in memory.
    """

    try:
        import joblib
    except ImportError:
        raise ImportError("scikit-learn's joblib is required to load models.")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError(f"Model at {path} has an unknown format.")

    version = bundle.get("format_version")
    if version is not None and version != MODEL_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported model format version "
            f"{version} (expected {MODEL_FORMAT_VERSION}).")

    required = _REQUIRED_BUNDLE_KEYS
    missing = [k for k in required if k not in bundle]
    if missing:
        raise ValueError(f"Model is missing fields: {missing}")

    classes = list(bundle["classes"])
    if len(classes) < 2:
        raise ValueError("Model must have at least two classes.")

    if version != MODEL_FORMAT_VERSION:
        bundle = _upgrade_real_bundle(bundle)

    return bundle


def _bundle_metadata(bundle):
    return {
        "format_version": bundle["format_version"],
        "model_type": bundle["model_type"],
        "classes": bundle["classes"],
        "feature_length": bundle["feature_length"],
        "n_samples": bundle["n_samples"],
        "trained_at": bundle["trained_at"],
        "metrics": bundle["metrics"],
        "feature_config": bundle["feature_config"],
    }


def predict(bundle, vector, keep_probabilities=True):
    """Predict the class and model-derived probabilities for one vector.

    Returns ``(label, confidence, proba_dict)`` where ``confidence`` is the
    probability of the predicted class, drawn from the SVM's calibrated
    ``predict_proba``.
    """

    from sklearn.exceptions import NotFittedError

    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector, dtype="float64")

    required_length = bundle["feature_length"]
    if vector.ndim == 2 and vector.shape[1] == required_length:
        pass
    elif vector.ndim == 1 and vector.size == required_length:
        vector = vector[None, :]
    else:
        raise ValueError(
            f"Feature vector has length {vector.size if vector.ndim == 1 else vector.shape[1]}, "
            f"but the model expects {required_length}. Retrain or reload with "
            "the matching feature configuration.")

    try:
        scaled = bundle["scaler"].transform(vector)
        probabilities = bundle["classifier"].predict_proba(scaled)
        label = bundle["classifier"].predict(scaled)
    except NotFittedError:
        raise ValueError("The loaded model is not fitted.")

    classes = bundle["classes"]
    proba_dict = {
        str(c): float(probabilities[0, i]) for i, c in enumerate(classes)
    }
    confidence = float(probabilities[0, np.argmax(probabilities[0])])
    return str(label[0]), confidence, proba_dict


# ---------------------------------------------------------------------------
# 2) Heuristic texture descriptor (legacy / desktop app) - NOT a classifier
# ---------------------------------------------------------------------------

def compute_features(gray, lbp, hist):
    """Descriptive GLCM + histogram metrics (rule based, not ML)."""

    metrics = texture_metrics(gray, hist)
    stats = histogram_statistics(hist)
    metrics.update(
        {"Entropy": stats["entropy"], "Uniformity": stats["uniformity"],
         "Mean": stats["mean"], "Std Dev": stats["std"]})
    return metrics


def _roughness_score(features, nbins):
    contrast = max(features.get("Contrast", 0.0), 0.0)
    contrast_n = float(np.clip(np.log1p(contrast) / np.log1p(500.0), 0, 1))
    homogeneity = float(np.clip(features.get("Homogeneity", 0.0), 0, 1))
    homog_inv = 1.0 - homogeneity
    entropy = max(features.get("Entropy", 0.0), 0.0)
    entropy_n = float(np.clip(entropy / np.log2(max(nbins, 2)), 0, 1))
    return 0.5 * contrast_n + 0.3 * homog_inv + 0.2 * entropy_n


def classify_texture(hist, features=None, nbins=None):
    """Heuristic texture descriptor.

    NOTE: this is a deterministic rule-based description (Smooth / Fine /
    Rough / Coarse), NOT a trained classifier. It is kept for the desktop
    app and is clearly labelled as a descriptor in the web app.
    """

    if features is None:
        features = {}
    if nbins is None:
        nbins = len(hist) if hist is not None else 26

    roughness = _roughness_score(features, nbins)

    scores = {}
    for label, center in TEXTURE_CLASSES.items():
        scores[label] = float(np.exp(-((roughness - center) ** 2) / (2 * _SIGMA ** 2)))

    total = sum(scores.values()) or 1.0
    probabilities = {k: v / total for k, v in scores.items()}

    label = max(probabilities, key=probabilities.get)
    confidence = probabilities[label] * 100.0

    return {
        "label": label,
        "confidence": confidence,
        "roughness": roughness,
        "scores": probabilities,
        "features": features,
    }


def interpret(result):
    label = result.get("label", "")
    confidence = result.get("confidence", 0.0)
    descriptions = {
        "Smooth": "Uniform surface with little variation between pixels.",
        "Fine": "Subtle, high-frequency detail with small local contrast.",
        "Rough": "Noticeable local variation and irregular micro structure.",
        "Coarse": "Strong, large-scale variation with dominant edges.",
    }
    return f"{descriptions.get(label, '')} ({confidence:.1f}% confidence)".strip()