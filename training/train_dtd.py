"""Train a multi-class texture model on the Describable Textures Dataset (DTD).

Reuses the project's existing LBP feature pipeline (``core.pipeline``) and the
same FeatureConfig contract used at inference time. It NEVER overwrites the
original ``models/texurion_model.pkl``; the new model is written to
``models/texurion_model_dtd.pkl`` plus a JSON sidecar.

Splits come from the dataset's official ``labels/train1.txt``, ``val1.txt``
and ``test1.txt`` files, so validation/test data never leaks into training.

Run:  python -m training.train_dtd
"""

import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import FeatureConfig, extract_features  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "dtd-r1.0.1", "images")
LABEL_DIR = os.path.join(ROOT, "dtd-r1.0.1", "labels")
CACHE = os.path.join(ROOT, "models", "dtd_features.npz")
OUT_MODEL = os.path.join(ROOT, "models", "texurion_model_dtd.pkl")
OUT_META = os.path.join(ROOT, "models", "texurion_model_dtd_meta.json")

CONFIG = FeatureConfig(
    method="uniform",
    engine="skimage",
    lbp_scales=((1, 8), (2, 16), (3, 24)),
    include_gabor=False,
    include_color_opponent=True,
    max_side=256,
)
SPLITS = {"train": "train1.txt", "val": "val1.txt", "test": "test1.txt"}
N_JOBS = 6


def read_split(filename):
    rows = []
    with open(os.path.join(LABEL_DIR, filename), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            label = line.split("/")[0]
            rows.append((label, os.path.join(IMG_DIR, *line.split("/"))))
    return rows


def _extract(item):
    label, path = item
    image = cv2.imread(path)
    if image is None:
        return None
    try:
        vector = extract_features(image, CONFIG)["vector"]
        return label, vector.astype("float32")
    except Exception:  # noqa: BLE001
        return None


def build_split(name, filename):
    items = read_split(filename)
    print(f"[{name}] extracting {len(items)} images ...", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, batch_size=16)(
        delayed(_extract)(item) for item in items)
    results = [r for r in results if r is not None]
    X = np.vstack([r[1] for r in results])
    y = np.asarray([r[0] for r in results])
    print(f"[{name}] done {X.shape} in {time.time() - t0:.0f}s", flush=True)
    return X, y


def load_or_build():
    if os.path.isfile(CACHE):
        print(f"Using cached features: {CACHE}")
        data = np.load(CACHE, allow_pickle=True)
        return (data["Xtr"], data["ytr"], data["Xval"], data["yval"],
                data["Xte"], data["yte"])
    Xtr, ytr = build_split("train", SPLITS["train"])
    Xval, yval = build_split("val", SPLITS["val"])
    Xte, yte = build_split("test", SPLITS["test"])
    np.savez_compressed(CACHE, Xtr=Xtr, ytr=ytr, Xval=Xval, yval=yval,
                        Xte=Xte, yte=yte)
    print(f"Cached features to {CACHE}")
    return Xtr, ytr, Xval, yval, Xte, yte


def main():
    print("Config:", CONFIG.to_dict(), flush=True)
    Xtr, ytr, Xval, yval, Xte, yte = load_or_build()
    print(f"train={Xtr.shape} val={Xval.shape} test={Xte.shape} "
          f"classes={len(np.unique(ytr))}", flush=True)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xval_s = scaler.transform(Xval)
    Xte_s = scaler.transform(Xte)

    candidates = []
    rf = RandomForestClassifier(
        n_estimators=400, n_jobs=-1, random_state=42, class_weight=None)
    t0 = time.time()
    rf.fit(Xtr_s, ytr)
    print(f"RF trained in {time.time() - t0:.0f}s", flush=True)
    candidates.append(("sklearn-random-forest", rf))

    from sklearn.svm import SVC

    t0 = time.time()
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              random_state=42)
    svm.fit(Xtr_s, ytr)
    print(f"SVM trained in {time.time() - t0:.0f}s", flush=True)
    candidates.append(("sklearn-svm-rbf", svm))

    best_name, clf = None, None
    best_val = -1.0
    for name, est in candidates:
        acc = accuracy_score(yval, est.predict(Xval_s))
        print(f"VAL {name}: {acc:.4f}", flush=True)
        if acc > best_val:
            best_val, best_name, clf = acc, name, est
    print(f"Selected: {best_name} (val={best_val:.4f})", flush=True)

    y_pred = clf.predict(Xte_s)
    classes = list(clf.classes_)
    metrics = {
        "train_accuracy": float(accuracy_score(ytr, clf.predict(Xtr_s))),
        "val_accuracy": float(best_val),
        "test_accuracy": float(accuracy_score(yte, y_pred)),
        "test_f1_weighted": float(f1_score(yte, y_pred, average="weighted")),
        "test_samples": int(Xte.shape[0]),
        "train_samples": int(Xtr.shape[0]),
        "val_samples": int(Xval.shape[0]),
        "n_classes": int(len(classes)),
    }

    bundle = {
        "format_version": 1,
        "model_type": best_name,
        "classifier": clf,
        "scaler": scaler,
        "classes": classes,
        "feature_config": CONFIG.to_dict(),
        "feature_length": int(Xtr.shape[1]),
        "n_samples": int(Xtr.shape[0]),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "DTD-r1.0.1 (official split1)",
        "metrics": metrics,
        "confusion_labels": classes,
        "confusion_matrix": confusion_matrix(yte, y_pred, labels=classes).tolist(),
        "classification_report": classification_report(
            yte, y_pred, zero_division=0),
    }

    from classifier import save_model

    save_model(bundle, OUT_MODEL)
    with open(OUT_META, "w", encoding="utf-8") as handle:
        json.dump({"model_path": OUT_MODEL, "model_type": bundle["model_type"],
                   "classes": classes, "feature_config": CONFIG.to_dict(),
                   "metrics": metrics, "trained_at": bundle["trained_at"]},
                  handle, indent=2)
    print("Saved:", OUT_MODEL, flush=True)
    print("Metrics:", json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
