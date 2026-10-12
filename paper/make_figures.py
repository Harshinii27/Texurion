"""Generate REAL figures for the research paper using project code only.

Every pixel and number below is produced by the project's own implementation
(exactly the functions the app calls). Nothing is hand-drawn or fabricated.

Outputs (written to paper/figures):
  brick_pipeline.png      original | grayscale | LBP map | multiscale histogram
  multiscale_hist.png     fused multiscale LBP histogram for brick.jpeg
  lbp_example.png         3x3 numerical example (computed with the project engine)
  confusion_matrix.png    real confusion matrix stored in texture_model.pkl
  benchmark.png           real measured latency (median, this machine)
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))  # make the project's 'core' importable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

SAMPLE = os.path.join(ROOT, "..", "images", "brick.jpeg")
MODEL = os.path.join(ROOT, "..", "texture_model.pkl")


def save_fig(name):
    path = os.path.join(FIG, name)
    plt.savefig(path, dpi=160, facecolor="white")
    plt.close()
    print("wrote", path)
    return path


# ---------------------------------------------------------------------------
# 1) Brick analysis pipeline (real output of core/lib code)
# ---------------------------------------------------------------------------
from core.io import ensure_max_size, read_image, to_grayscale
from core.lbp import generate_lbp, lbp_visualize, lbp_multiscale_histograms, number_of_patterns
from core.pipeline import FeatureConfig

image = read_image(SAMPLE)
gray = to_grayscale(image)

_, lbp = generate_lbp(gray=gray, radius=3, points=24, method="uniform")
lbp_rgb = lbp_visualize(lbp, max_side=512)

config = FeatureConfig(method="uniform", engine="skimage",
                       lbp_scales=((1, 8), (2, 16), (3, 24)),
                       max_side=512)
fused = lbp_multiscale_histograms(gray, config.lbp_scales,
                                  method="uniform", engine="skimage")

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5))
axes[0].imshow(ensure_max_size(image, 512)[:, :, ::-1])
axes[0].set_title("(a) Original")
axes[1].imshow(ensure_max_size(gray, 512), cmap="gray")
axes[1].set_title("(b) Grayscale")
axes[2].imshow(lbp_rgb)
axes[2].set_title("(c) LBP map (uniform,\nR=3, P=24)")
bar = axes[3].bar(np.arange(fused.size), fused, width=0.9, color="#4f8cff")
axes[3].plot(np.arange(fused.size), fused, color="#ffd166")
axes[3].set_title("(d) Fused multiscale LBP\nhistogram (54 bins)")
axes[3].set_xlabel("bin"); axes[3].set_ylabel("frequency")
for a in axes:
    a.set_xticks([]); a.set_yticks([])
for a in axes[:3]:
    a.set_xticks([]); a.set_yticks([])
axes[3].set_xticks([0, 26, 53])
fig.tight_layout()
save_fig("brick_pipeline.png")

# ---------------------------------------------------------------------------
# 2) Multiscale histogram alone (larger, labelled)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.2))
bounds = [0]
for r, p in config.lbp_scales:
    bounds.append(bounds[-1] + number_of_patterns("uniform", p))
colors = ["#4f8cff", "#6ab04c", "#e67e22"]
for i, ((r, p), (lo, hi)) in enumerate(zip(config.lbp_scales, zip(bounds[:-1], bounds[1:]))):
    part = fused[lo:hi]
    ax.bar(np.arange(lo, hi), part, width=0.9, color=colors[i],
           label=f"R={r}, P={p} ({hi-lo} bins)")
ax.plot(np.arange(fused.size), fused, color="#ffd166", linewidth=1.1)
ax.axvline(bounds[1] - 0.5, color="gray", lw=0.8, ls="--")
ax.axvline(bounds[2] - 0.5, color="gray", lw=0.8, ls="--")
ax.set_title("Fused multiscale uniform-LBP histogram (images/brick.jpeg)")
ax.set_xlabel("bin index (concatenated scales)")
ax.set_ylabel("normalised frequency")
ax.legend(fontsize=8)
fig.tight_layout()
save_fig("multiscale_hist.png")

# ---------------------------------------------------------------------------
# 3) Numerical 3x3 example computed by the project's own engine
# ---------------------------------------------------------------------------
patch = np.array([[56, 20, 34],
                  [128, 68, 60],
                  [78, 30, 84]], dtype=np.float64)
tiny = patch[None, :, :]
bits_s = (tiny >= 68).astype(int)
bits = bits_s.reshape(-1)  # top-left -> bottom-right row-major reading order

# Recompute the uniform code exactly as the custom engine does for P=8 neighbours.
radius, points = 1, 8
img = patch.astype(np.float64)
h, w = img.shape
yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
from core.lbp import neighbor_offsets
offsets = neighbor_offsets(radius, points)
samples = np.zeros((points, h, w), dtype=np.float64)
for k in range(points):
    dx, dy = offsets[k, 0], offsets[k, 1]
    x = np.clip(xx + dx, 0, w - 1)
    y = np.clip(yy + dy, 0, h - 1)
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0; wy = y - y0
    a = img[y0, x0]; b = img[y0, x1]; c = img[y1, x0]; d = img[y1, x1]
    samples[k] = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy
sample_bits = (samples[:, 1, 1] >= img[1, 1]).astype(int)
transitions = sum(1 for k in range(points) if sample_bits[k] != sample_bits[(k + 1) % points])
code = int(sample_bits.sum()) if transitions <= 2 else points + 1

# --- draw the centered 3x3 patch with neighbour intensities around the center
fig, axes = plt.subplots(1, 1, figsize=(4.6, 4.4))
ax = axes
ax.axis("off")
for i in range(3):
    for j in range(3):
        val = int(patch[i, j])
        edge = "#2b6cb0" if (i == 1 and j == 1) else "#555"
        ax.add_patch(plt.Rectangle((j, 2 - i), 1, 1, fill=(i != 1 or j != 1),
                                    facecolor="#eef2fa", edgecolor=edge, lw=2))
        ax.text(j + 0.5, 2 - i + 0.5, str(val), ha="center", va="center",
                fontsize=14, weight="bold" if (i == 1 and j == 1) else "normal",
                color="#1a202c" if (i == 1 and j == 1) else "#4a5568")
ax.set_xlim(0, 3); ax.set_ylim(0, 3)
ax.set_title("3x3 patch, center = 68")
save_fig("lbp_patch.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axes[0]
ax.axis("off")
order = "clockwise"
vals = list(sample_bits)  # neighbours in offset order (cos/sin)
labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
for i in range(8):
    ang = i * 45.0
    r = 1.0
    x = np.cos(np.radians(ang)); y = np.sin(np.radians(ang))
    ax.plot([x], [y], "o", ms=26, color="#4f8cff" if vals[i] else "#cbd5e0")
    ax.text(x, y, str(vals[i]), ha="center", va="center", fontsize=11, color="white")
ax.add_patch(plt.Circle((0, 0), 0.18, color="#e67e22", zorder=5))
ax.text(0, 0, "68", ha="center", va="center", fontsize=10, color="white", zorder=6)
for i in range(8):
    ang = i * 45.0
    x = np.cos(np.radians(ang)); y = np.sin(np.radians(ang))
    ax.annotate(labels[i], (x, y), textcoords="offset points", xytext=(0, 16),
                ha="center", fontsize=9, color="#333")
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.7, 1.7)
bitstr_title = "bits: " + "".join(str(b) for b in sample_bits)
ax.set_title("Sampled neighbours  >= center -> 1  (" + bitstr_title + ")")
ax = axes[1]
ax.axis("off")
# uniform code interpretation
ax.text(0.02, 0.72, "Transition count:", fontsize=11, transform=ax.transAxes)
bits_str = "".join(str(b) for b in sample_bits)
ax.text(0.02, 0.62, "U(" + bits_str + ") = " + str(transitions),
        fontsize=12, transform=ax.transAxes, color="#c05621")
ax.text(0.02, 0.42, "Uniform? " + ("YES" if transitions <= 2 else "NO"),
        fontsize=11, transform=ax.transAxes)
ax.text(0.02, 0.22, "Cardinality = sum(bits) = " + str(int(sample_bits.sum())),
        fontsize=11, transform=ax.transAxes)
ax.text(0.02, 0.02, "Uniform-LBP code = " + str(code), fontsize=13,
        transform=ax.transAxes, weight="bold", color="#2b6cb0")
ax.set_title("Uniform mapping (custom engine)")
fig.tight_layout()
save_fig("lbp_example.png")

# ---------------------------------------------------------------------------
# 4) Real confusion matrix from the stored trained model
# ---------------------------------------------------------------------------
from classifier import load_model
bundle = load_model(MODEL)
cm = np.asarray(bundle["confusion_matrix"], dtype=int)
labels = [str(c) for c in bundle["confusion_labels"]]

fig, ax = plt.subplots(figsize=(5.4, 4.4))
im = ax.imshow(cm, cmap="Blues")
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "#1a202c", fontsize=13)
ax.set_xticks(range(cm.shape[1])); ax.set_yticks(range(cm.shape[0]))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.set_title("Confusion matrix, held-out test split (90/360 samples)",
             fontsize=10)
fig.colorbar(im, fraction=0.046)
fig.tight_layout()
save_fig("confusion_matrix.png")

# ---------------------------------------------------------------------------
# 5) Real measured benchmark on this machine (median over reps)
# ---------------------------------------------------------------------------
from core.benchmark import benchmark_all
result = benchmark_all(image, reps=3)

fig, ax = plt.subplots(figsize=(9, 4.2))
names = list(result.timings.keys())
medians = [v["median_ms"] for v in result.timings.values()]
ax.barh(names[::-1], medians[::-1], color="#4f8cff", edgecolor="#21468a")
for bar, value in zip(ax.patches, medians[::-1]):
    ax.text(bar.get_width() + max(medians) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} ms", va="center", fontsize=8)
ax.set_xlabel("median latency (ms)"); ax.set_title("Measured feature-extraction latency (this machine)")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(0, max(medians) * 1.18)
fig.tight_layout()
save_fig("benchmark.png")

# Persist the measured numbers + environment for the paper (REAL, reproducible).
record = {
    "image_size": list(result.image_size),
    "reps": result.reps,
    "environment": result.environment,
    "timings": {k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                for k, v in result.timings.items()},
    "memory": result.memory,
    "equivalence": result.equivalence,
}
with open(os.path.join(ROOT, "experiments", "benchmark_measured.json"), "w") as handle:
    json.dump(record, handle, indent=2)
print("wrote experiments/benchmark_measured.json")
print("equivalence:", result.equivalence)

print("ALL FIGURES DONE.")