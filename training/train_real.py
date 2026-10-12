"""Real model training: LBP+GLCM+Gabor features, compare classifiers, save model.

Trains and evaluates on real dataset; saves bundle with preprocessing config.
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from core.io import decode_image
from core.pipeline import FeatureConfig, extract_features

from training.dataset import (
    describe_split,
    extract_valid_samples,
    make_stratified_split,
)

MODEL_VERSION = "texurion-1.0"
MODEL_PATH = os.path.join("models", "texurion_model.pkl")


def _extract_vectors(samples, config):
    X, y, names = [], [], []
    for label, name, data, _ in samples:
        try:
            img = decode_image(data, name=name)
            vec = extract_features(img, config)["vector"]
            X.append(vec)
            y.append(label)
            names.append(name)
        except Exception:
            continue
    if not X:
        return np.zeros((0, 0)), np.array([]), np.array([])
    return np.vstack(X), np.asarray(y), np.asarray(names)


def train_real_model(zip_path="textures.zip", seed=42):
    samples = extract_valid_samples(zip_path)
    if len(samples) < 20:
        raise ValueError(f"Insufficient samples: {len(samples)}")

    train_s, val_s, test_s = make_stratified_split(samples, seed=seed)
    tdesc, tn = describe_split(train_s)
    vdesc, vn = describe_split(val_s)
    xdesc, xn = describe_split(test_s)
    print("Splits:", tdesc, vn, xn)

    # Choose config: include gabor helps directionality
    config = FeatureConfig(
        method="uniform",
        engine="skimage",
        lbp_scales=((1, 8), (2, 16), (3, 24)),
        include_gabor=True,
        include_color_opponent=False,
        max_side=512,
    )

    Xtr, ytr, _ = _extract_vectors(train_s, config)
    Xval, yval, _ = _extract_vectors(val_s, config)
    Xte, yte, _ = _extract_vectors(test_s, config)

    if len(np.unique(ytr)) < 2:
        raise ValueError("Need >=2 classes")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xval_s = scaler.transform(Xval)
    Xte_s = scaler.transform(Xte)

    # Compare SVM (RBF) vs RF
    candidates = []
    # SVM calibrated
    svm = CalibratedClassifierCV(
        estimator=SVC(kernel="rbf", probability=True, random_state=seed),
        method="sigmoid",
        cv=min(3, len(np.unique(ytr))) if len(np.unique(ytr)) >= 2 else 3,
    )
    svm.fit(Xtr_s, ytr)
    candidates.append(("svm_rbf_calibrated", svm, Xval_s, yval))

    rf = RandomForestClassifier(n_estimators=300, random_state=seed)
    rf.fit(Xtr_s, ytr)
    candidates.append(("rf", rf, Xval_s, yval))

    best = None
    best_score = -1
    for name, clf, Xv, yv in candidates:
        yv_pred = clf.predict(Xv)
        acc = accuracy_score(yv, yv_pred)
        if acc > best_score:
            best_score = acc
            best = (name, clf)
        print(f"VAL {name}: acc={acc:.4f}")

    name_best, clf_best = best
    # Evaluate on test
    yte_pred = clf_best.predict(Xte_s)
    acc_te = accuracy_score(yte, yte_pred)
    prec = precision_score(yte, yte_pred, average="weighted", zero_division=0)
    rec = recall_score(yte, yte_pred, average="weighted", zero_division=0)
    f1 = f1_score(yte, yte_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(yte, yte_pred)
    classes = list(clf_best.classes_) if hasattr(clf_best, "classes_") else list(np.unique(ytr))

    bundle = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "classes": classes,
        "feature_config": config.to_dict(),
        "feature_length": int(Xtr.shape[1]),
        "scaler": scaler,
        "classifier": clf_best,
        "best_model": name_best,
        "splits": {
            "train": tdesc,
            "val": vdesc,
            "test": xdesc,
            "counts": {"train": tn, "val": vn, "test": xn},
        },
        "metrics_test": {
            "accuracy": float(acc_te),
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "f1_weighted": float(f1),
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(yte, yte_pred, zero_division=0),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    with open(os.path.join("models", "texurion_model_meta.json"), "w") as f:
        meta = {
            "model_path": MODEL_PATH,
            "model_version": MODEL_VERSION,
            "trained_at": bundle["trained_at"],
            "classes": classes,
            "feature_config": config.to_dict(),
            "splits": bundle["splits"],
            "metrics_test": bundle["metrics_test"],
        }
        json.dump(meta, f, indent=2)
    print("Saved:", MODEL_PATH)
    print("Test:", bundle["metrics_test"])
    print("Classes:", classes)
    return bundle


if __name__ == "__main__":
    train_real_model()
