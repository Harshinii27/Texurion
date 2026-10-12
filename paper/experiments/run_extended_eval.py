r"""Real extended evaluation for the paper.

Runs, using ONLY the project's own code and the bundled dataset:

  * a single-split reproduction of the stored model protocol (verification),
  * repeated stratified k-fold cross-validation (5 folds x 10 repeats),
  * a GLCM (Haralick) baseline with a paired test against LBP,
  * a classifier comparison on the LBP features,
  * a feature-block ablation (LBP / +Gabor / +opponent-colour).

Every number is computed live. Nothing is fabricated. Results are written to
paper/experiments/extended_eval.json and figures to paper/figures/.

Run:  & "venv314\Scripts\python.exe" "paper\experiments\run_extended_eval.py"
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from core.io import ensure_max_size, read_image, to_grayscale  # noqa: E402
from core.pipeline import FeatureConfig, extract_features  # noqa: E402
from classifier import train_model  # noqa: E402

FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(ROOT, "textures_upload1")
MAX_SIDE = 512
CLASSES = ("banded", "dotted", "woven")
SEED = 42
N_SPLITS = 5
N_REPEATS = 10


def log(*a):
    print("[eval]", *a, flush=True)


# ---------------------------------------------------------------------------
# 1) Load the dataset once, downscaled to the model's working resolution.
# ---------------------------------------------------------------------------
def load_dataset():
    images, labels, names = [], [], []
    for label in CLASSES:
        folder = os.path.join(DATA, label)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
        for fname in files:
            img = read_image(os.path.join(folder, fname))
            img = ensure_max_size(img, MAX_SIDE)
            images.append(img)
            labels.append(label)
            names.append(f"{label}/{fname}")
    return images, np.asarray(labels), names


# ---------------------------------------------------------------------------
# 2) Feature extractors.
# ---------------------------------------------------------------------------
CONFIGS = {
    "LBP only": FeatureConfig(method="uniform", engine="skimage",
                              lbp_scales=((1, 8), (2, 16), (3, 24)),
                              max_side=MAX_SIDE),
    "LBP + Gabor": FeatureConfig(method="uniform", engine="skimage",
                                 lbp_scales=((1, 8), (2, 16), (3, 24)),
                                 include_gabor=True, max_side=MAX_SIDE),
    "LBP + opponent": FeatureConfig(method="uniform", engine="skimage",
                                    lbp_scales=((1, 8), (2, 16), (3, 24)),
                                    include_color_opponent=True, max_side=MAX_SIDE),
    "LBP + Gabor + opponent": FeatureConfig(method="uniform", engine="skimage",
                                            lbp_scales=((1, 8), (2, 16), (3, 24)),
                                            include_gabor=True,
                                            include_color_opponent=True,
                                            max_side=MAX_SIDE),
}


def extract_block(images, config):
    t0 = time.perf_counter()
    feats = [extract_features(img, config)["vector"] for img in images]
    X = np.vstack(feats).astype("float64")
    log(f"extracted {config.include_gabor=} {config.include_color_opponent=} "
        f"-> {X.shape} in {time.perf_counter() - t0:.1f}s")
    return X


def glcm_features(images):
    """Haralick GLCM baseline: 6 props x 3 distances, averaged over 4 angles."""
    from skimage.feature import graycoprops, graycomatrix

    props = ("contrast", "dissimilarity", "homogeneity", "energy",
             "correlation", "ASM")
    distances = [1, 2, 4]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    rows = []
    for img in images:
        gray = to_grayscale(img).astype("float32")
        if gray.max() <= gray.min():
            q = np.zeros_like(gray, dtype="uint8")
        else:
            q = ((gray - gray.min()) / (gray.max() - gray.min()) * 31).astype("uint8")
        g = graycomatrix(q, distances=distances, angles=angles, levels=32,
                         symmetric=True, normed=True)
        feat = []
        for p in props:
            feat.extend(graycoprops(g, p).mean(axis=1).tolist())
        rows.append(feat)
    return np.asarray(rows, dtype="float64")


# ---------------------------------------------------------------------------
# 3) Cross-validation helpers (same folds reused across experiments).
# ---------------------------------------------------------------------------
def make_cv():
    from sklearn.model_selection import RepeatedStratifiedKFold
    return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=SEED)


def cv_scores(X, y, estimator):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(estimator, X, y, cv=make_cv(),
                             scoring="accuracy", n_jobs=-1)
    return np.asarray(scores, dtype="float64")


def summarise(scores):
    per_repeat = scores.reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "n_folds": int(scores.size),
        "per_repeat_means": per_repeat.tolist(),
        "fold_scores": scores.tolist(),
    }


def ci95(per_repeat_means):
    """95% t-interval over the independent repeat means (df = repeats-1)."""
    from scipy import stats
    arr = np.asarray(per_repeat_means, dtype="float64")
    n = arr.size
    se = arr.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return [float(arr.mean() - t * se), float(arr.mean() + t * se)]


def paired_test(a, b):
    from scipy import stats
    t, p = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(a, b)
    except Exception:
        w, pw = float("nan"), float("nan")
    return {"t": float(t), "p_value": float(p),
            "wilcoxon_stat": float(w), "wilcoxon_p": float(pw),
            "mean_diff": float(np.mean(a) - np.mean(b))}


def main():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    images, y, names = load_dataset()
    log(f"loaded {len(images)} images; classes {sorted(set(y.tolist()))}")

    out = {
        "dataset": {"n_samples": int(len(y)),
                    "classes": sorted(set(y.tolist())),
                    "max_side": MAX_SIDE,
                    "cv": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS,
                           "random_state": SEED}},
        "experiments": {},
    }

    # --- Extract all needed feature matrices -------------------------------
    X_lbp = extract_block(images, CONFIGS["LBP only"])
    X_gabor = extract_block(images, CONFIGS["LBP + Gabor"])
    X_opp = extract_block(images, CONFIGS["LBP + opponent"])
    X_all = extract_block(images, CONFIGS["LBP + Gabor + opponent"])
    X_glcm = glcm_features(images)
    log(f"GLCM features -> {X_glcm.shape}")
    out["feature_dimensions"] = {
        "LBP": int(X_lbp.shape[1]), "GLCM": int(X_glcm.shape[1]),
        "LBP+Gabor": int(X_gabor.shape[1]),
        "LBP+opponent": int(X_opp.shape[1]),
        "LBP+Gabor+opponent": int(X_all.shape[1]),
    }

    svm = Pipeline([("scaler", StandardScaler()), ("clf", SVC())])

    # --- (verification) reproduce the stored single-split model ------------
    bundle = train_model(X_lbp, y, CONFIGS["LBP only"])
    out["experiments"]["single_split_reproduction"] = {
        "train_accuracy": bundle["metrics"]["train_accuracy"],
        "test_accuracy": bundle["metrics"]["test_accuracy"],
        "test_samples": bundle["metrics"]["test_samples"],
        "confusion_matrix": bundle["confusion_matrix"],
        "confusion_labels": bundle["confusion_labels"],
    }
    log("single-split reproduction:", out["experiments"]["single_split_reproduction"])

    # --- Repeated stratified k-fold CV: LBP vs GLCM -----------------------
    lbp_scores = cv_scores(X_lbp, y, svm)
    glcm_scores = cv_scores(X_glcm, y, svm)
    out["experiments"]["repeated_cv"] = {
        "LBP + SVM (RBF)": summarise(lbp_scores),
        "GLCM + SVM (RBF)": summarise(glcm_scores),
    }
    out["experiments"]["repeated_cv"]["LBP + SVM (RBF)"]["ci95"] = ci95(
        out["experiments"]["repeated_cv"]["LBP + SVM (RBF)"]["per_repeat_means"])
    out["experiments"]["repeated_cv"]["GLCM + SVM (RBF)"]["ci95"] = ci95(
        out["experiments"]["repeated_cv"]["GLCM + SVM (RBF)"]["per_repeat_means"])
    out["experiments"]["lbp_vs_glcm_paired_test"] = paired_test(lbp_scores, glcm_scores)
    log("LBP CV mean=%.4f  GLCM CV mean=%.4f  p=%.4g"
        % (lbp_scores.mean(), glcm_scores.mean(),
           out["experiments"]["lbp_vs_glcm_paired_test"]["p_value"]))

    # --- Classifier comparison on LBP features ----------------------------
    classifiers = {
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("clf", SVC())]),
        "SVM (linear)": Pipeline([("scaler", StandardScaler()),
                                  ("clf", SVC(kernel="linear"))]),
        "k-NN (k=5)": Pipeline([("scaler", StandardScaler()),
                                ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "Random forest": RandomForestClassifier(n_estimators=300,
                                                random_state=SEED, n_jobs=-1),
    }
    out["experiments"]["classifier_comparison"] = {}
    score_bank = {}
    for name, est in classifiers.items():
        s = cv_scores(X_lbp, y, est)
        score_bank[name] = s
        out["experiments"]["classifier_comparison"][name] = summarise(s)
        out["experiments"]["classifier_comparison"][name]["ci95"] = ci95(
            out["experiments"]["classifier_comparison"][name]["per_repeat_means"])
        log(f"classifier {name}: {s.mean():.4f} +/- {s.std():.4f}")
    out["experiments"]["classifier_paired_tests_vs_svm_rbf"] = {
        name: paired_test(lbp_scores, s)
        for name, s in score_bank.items() if name != "SVM (RBF)"
    }

    # --- Feature-block ablation -------------------------------------------
    ablation = {
        "LBP only": X_lbp,
        "LBP + Gabor": X_gabor,
        "LBP + opponent": X_opp,
        "LBP + Gabor + opponent": X_all,
    }
    out["experiments"]["ablation"] = {}
    for name, X in ablation.items():
        s = cv_scores(X, y, svm)
        out["experiments"]["ablation"][name] = summarise(s)
        out["experiments"]["ablation"][name]["ci95"] = ci95(
            out["experiments"]["ablation"][name]["per_repeat_means"])
        log(f"ablation {name}: {s.mean():.4f} +/- {s.std():.4f}")

    # --- Persist ----------------------------------------------------------
    jpath = os.path.join(HERE, "extended_eval.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log("wrote", jpath)

    make_figures(out)
    log("DONE")


def make_figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp = out["experiments"]

    # (a) LBP vs GLCM
    names = ["LBP + SVM", "GLCM + SVM"]
    means = [exp["repeated_cv"]["LBP + SVM (RBF)"]["mean"],
             exp["repeated_cv"]["GLCM + SVM (RBF)"]["mean"]]
    stds = [exp["repeated_cv"]["LBP + SVM (RBF)"]["std"],
            exp["repeated_cv"]["GLCM + SVM (RBF)"]["std"]]
    p = exp["lbp_vs_glcm_paired_test"]["p_value"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=["#4f8cff", "#e67e22"], edgecolor="#21365e")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.01, f"{m*100:.1f}%",
                ha="center", fontsize=10, weight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("5x10-fold CV accuracy")
    ax.set_title(f"LBP vs GLCM (paired t-test p = {p:.3g})", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cv_methods.png"), dpi=160, facecolor="white")
    plt.close(fig)

    # (b) classifier comparison
    cc = exp["classifier_comparison"]
    names = list(cc.keys())
    means = [cc[n]["mean"] for n in names]
    stds = [cc[n]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bars = ax.barh(names[::-1], np.array(means)[::-1],
                   xerr=np.array(stds)[::-1], capsize=5, color="#6ab04c",
                   edgecolor="#2f6b23")
    for b, m in zip(bars, np.array(means)[::-1]):
        ax.text(m + 0.01, b.get_y() + b.get_height() / 2, f"{m*100:.1f}%",
                va="center", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("5x10-fold CV accuracy")
    ax.set_title("Classifier comparison on LBP features", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "classifier_comparison.png"), dpi=160,
                facecolor="white")
    plt.close(fig)

    # (c) ablation
    ab = exp["ablation"]
    names = list(ab.keys())
    means = [ab[n]["mean"] for n in names]
    stds = [ab[n]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=["#4f8cff", "#8e7cc3", "#f2a154", "#6ab04c"],
                  edgecolor="#21365e")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.008, f"{m*100:.1f}%",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("5x10-fold CV accuracy")
    ax.set_title("Feature-block ablation", fontsize=10)
    ax.tick_params(axis="x", labelrotation=12, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation.png"), dpi=160, facecolor="white")
    plt.close(fig)
    log("wrote figures: cv_methods.png, classifier_comparison.png, ablation.png")


if __name__ == "__main__":
    main()
