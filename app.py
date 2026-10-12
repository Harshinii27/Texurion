"""Texture Analysis Studio - Streamlit web application.

Run with:  streamlit run app.py

Four tabs:

* **Analyzer** - upload an image, choose LBP + ROI, enable advanced feature
  blocks (multiscale LBP, Gabor, opponent colour) and run a fused analysis.
  A trained model (Models tab) enables real classification; a reference
  gallery (built by training on labelled data) enables similarity search.
  The rule-based texture descriptor is always labelled as "not a trained
  classifier".
* **Models** - train a real SVM on a labelled ZIP (folder name = class),
  inspect measured train/test metrics, save/load models.
* **Batch** - process many images/ZIPs concurrently with per-file status,
  timing and CSV export.
* **Performance** - real measured latency/memory per feature block.

Every number shown is computed live in this session; the app never
fabricates results.
"""

import io as _io
import json
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from classifier import (
    _roughness_score,
    extract_labeled_dataset,
    load_model,
    save_model,
    train_model,
)
from core.batch import collect_batch_inputs, process_batch, write_results_csv
from core.benchmark import benchmark_all, build_benchmark_pdf, render_benchmark_chart
from core.io import ensure_max_size, read_image, to_grayscale
from core.histogram import histogram_statistics
from core.lbp import LBP_METHODS
from core.pipeline import (
    DEFAULT_MAX_SIDE,
    FeatureConfig,
    PipelineError,
    crop_roi,
    display_views,
    extract_features,
    similarity_ranks,
    texture_metrics,
)
from report import render_histogram_png

st.set_page_config(
    page_title="Texture Analysis Studio",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

LABEL_ZIP_HINT = (
    "ZIP must contain one folder per class; every image inside the folder "
    "belongs to that class. Example:  textures.zip/tiles/01.jpg  bricks/02.jpg"
)

SCALE_OPTIONS = {
    "R1/P8": (1, 8),
    "R2/P16": (2, 16),
    "R3/P24": (3, 24),
}


def _init_state():
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("gallery", [])
    st.session_state.setdefault("gallery_config", None)


# --------------------------------------------------------------------------
# Theme (gradient backgrounds)
# --------------------------------------------------------------------------

_MAIN_BACKGROUND = (
    "linear-gradient(135deg, #FF77FF 0%, #F068F0 40%, "
    "#E159E1 70%, #FF86FF 100%)"
)
_SIDEBAR_BACKGROUND = "linear-gradient(to bottom, #06C70F, #029A2E, #006D73)"


def _inject_theme():
    """Apply the pink-fuchsia main background and emerald sidebar gradient.

    Only the application chrome is styled; exported images and rendered
    charts/histograms are unaffected.
    """

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: {_MAIN_BACKGROUND};
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stSidebar"] {{
            background: {_SIDEBAR_BACKGROUND};
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
            color: #ffffff;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.45);
        }}
        [data-testid="stSidebar"] [data-testid="stAlert"]
        [data-testid="stMarkdownContainer"] p {{
            color: #1b1b1b;
            text-shadow: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Model loading (cached, with progress feedback)
# --------------------------------------------------------------------------

MODEL_SECURITY_WARNING = (
    "Model files are Python pickle objects. Loading a .pkl/.joblib you did "
    "not create yourself can execute arbitrary code on this machine. Only "
    "load files you trust."
)


@st.cache_resource(show_spinner=False)
def _load_model_from_bytes(model_bytes):
    """Deserialize a model bundle from bytes.

    Cached with ``st.cache_resource`` so re-loading the same file across
    Streamlit reruns/sessions does not repeat the joblib deserialization.
    """

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name
    try:
        return load_model(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _set_model_from_bytes(model_bytes):
    """Load a model with real progress stages, store it, then rerun."""

    with st.status("Loading model…", expanded=True) as status:
        st.write("Reading uploaded bytes and writing a temporary file…")
        st.write("Deserializing model (joblib.load)…")
        try:
            bundle = _load_model_from_bytes(model_bytes)
        except Exception as exc:  # noqa: BLE001
            status.update(label="Model could not be loaded", state="error")
            st.error(f"Model could not be loaded: {exc}")
            return
        st.write("Validating model structure and version…")
        st.session_state["model"] = bundle
        status.update(
            label=f"Model loaded with classes {', '.join(bundle['classes'])}.",
            state="complete")
    st.rerun()


# --------------------------------------------------------------------------
# Shared widgets
# --------------------------------------------------------------------------

def feature_config_widgets(key_prefix, default_max_side=DEFAULT_MAX_SIDE):
    """Build the feature-config group widgets; returns a FeatureConfig."""

    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox(
            "LBP method (extraction)", LBP_METHODS, index=0, key=f"{key_prefix}_method")
    with c2:
        scales = st.multiselect(
            "Multiscale LBP scales",
            list(SCALE_OPTIONS),
            default=list(SCALE_OPTIONS),
            key=f"{key_prefix}_scales",
        )
    with c3:
        engine = st.selectbox(
            "LBP engine",
            ["skimage (reference)", "numpy (custom)", "numba (custom)"],
            index=0,
            help="Custom engines implement a classic circular uniform LBP; "
                 "they are NOT claimed to match or beat scikit-image.",
            key=f"{key_prefix}_engine",
        )

    if not scales:
        st.warning("Select at least one LBP scale; using the default trio.")
        scales = list(SCALE_OPTIONS)

    c4, c5, c6 = st.columns(3)
    with c4:
        include_gabor = st.checkbox(
            "Include Gabor filter bank", value=False,
            help="Adds Gabor mean/std statistics to the fused vector.",
            key=f"{key_prefix}_gabor")
    with c5:
        include_opponent = st.checkbox(
            "Include opponent-colour LBP", value=False,
            help="Adds opponent-colour LBP histograms (requires RGB upload).",
            key=f"{key_prefix}_opponent")
    with c6:
        max_side = st.number_input(
            "Working resolution (max side)", min_value=128, max_value=2048,
            value=int(default_max_side), step=32,
            help="Images are downscaled so their largest side equals this for "
                 "feature extraction. Keep identical for train/predict.",
            key=f"{key_prefix}_max_side")

    engine_value = {"skimage (reference)": "skimage",
                    "numpy (custom)": "numpy",
                    "numba (custom)": "numba"}[engine]

    return FeatureConfig(
        method=method,
        engine=engine_value,
        lbp_scales=tuple(SCALE_OPTIONS[s] for s in scales),
        include_gabor=include_gabor,
        include_color_opponent=include_opponent,
        max_side=int(max_side),
    )


def model_compatibility(config):
    """Return an error message if the loaded model cannot be used, else None."""

    model = st.session_state.get("model")
    if model is None:
        return "No trained model is loaded. Train or load one in the Models tab."
    wanted = config.descriptor_length()
    if model["feature_length"] != wanted:
        return (
            f"Model expects {model['feature_length']} features but the current "
            f"configuration produces {wanted}. Adjust the feature settings to "
            "match the model, or retrain."
        )
    return None


def model_pretrain_note(config):
    """Informational note when classification works but preprocessing differs."""

    model = st.session_state.get("model")
    if model is None:
        return None
    trained_at_max_side = model["feature_config"].get("max_side")
    if trained_at_max_side and trained_at_max_side != config.max_side:
        return (
            f"Note: the model was trained at max_side={trained_at_max_side}, "
            f"but the current setting is {config.max_side}. Feature dimensions "
            "match, so classification runs, yet image preprocessing differs "
            "from training."
        )
    return None


def gallery_compatibility(config):
    gallery_cfg = st.session_state.get("gallery_config")
    gallery = st.session_state.get("gallery", [])
    if not gallery:
        return "No reference gallery. Training a model on labelled data builds one."
    if gallery_cfg is None or gallery_cfg["descriptor_length"] != config.descriptor_length():
        return (
            "Reference gallery was built with a different feature "
            "configuration; reconfigure to match so comparisons are fair."
        )
    return None


# --------------------------------------------------------------------------
# Analyzer tab
# --------------------------------------------------------------------------

@dataclass
class AnalysisRun:
    image: np.ndarray
    gray: np.ndarray
    hist: np.ndarray
    metrics: dict
    config: FeatureConfig
    vector: np.ndarray
    sizes: dict
    box: tuple
    total_pixels: int
    status: str
    error: str | None = None


def _run_analysis(image, config, box):
    """Fused extraction on the ROI box (working-resolution coordinates)."""

    try:
        work = ensure_max_size(image, config.max_side)
        cropped, used_box = crop_roi(work, box)
        gray = to_grayscale(cropped)
        extracted = extract_features(cropped, config)

        lbp_block = np.asarray(extracted["blocks"]["lbp"], dtype="float64")
        total = float(lbp_block.sum())
        fused_hist = lbp_block / total if total > 0 else lbp_block

        metrics = texture_metrics(gray, fused_hist)
        return AnalysisRun(
            image=work, gray=gray, hist=fused_hist, metrics=metrics,
            config=config, vector=extracted["vector"],
            sizes=extracted["sizes"], box=used_box,
            total_pixels=int(gray.size), status="ok",
        )
    except (ValueError, PipelineError, Exception) as exc:  # noqa: BLE001
        return AnalysisRun(
            image=ensure_max_size(image, config.max_side),
            gray=None, hist=None, metrics={}, config=config,
            vector=None, sizes={}, box=box,
            total_pixels=0, status="error", error=str(exc),
        )


def _draw_overlay(work, box):
    """RGB preview of the working image with the ROI rectangle drawn on it."""

    display = ensure_max_size(work, 640)
    sy = display.shape[0] / work.shape[0]
    sx = display.shape[1] / work.shape[1]
    x0, y0, x1, y1 = box
    px0, py0 = int(x0 * sx), int(y0 * sy)
    px1, py1 = int(x1 * sx), int(y1 * sy)
    out = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    cv2.rectangle(out, (px0, py0), (px1, py1), (255, 215, 0), 2,
                  lineType=cv2.LINE_AA)
    return out


def _histogram_figure(hist, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=140)
    hist = np.asarray(hist, dtype="float64")
    x = np.arange(hist.size)
    ax.bar(x, hist, width=0.9, color="#4f8cff", edgecolor="none")
    ax.plot(x, hist, color="#ffd166", linewidth=1.1)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def render_analyzer():
    st.header("Analyzer")

    uploaded = st.file_uploader(
        "Upload an image (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"],
        key="analyzer_upload")

    if uploaded is None:
        st.info("Upload an image to start.")
        if os.path.isfile("images/brick.jpeg"):
            with st.expander("…or use the bundled sample (images/brick.jpeg)"):
                if st.button("Analyse bundled sample"):
                    uploaded = _BundledSample()
        if uploaded is None:
            return

    try:
        image = decode_upload(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Image could not be loaded: {exc}")
        return

    config = feature_config_widgets("analyzer")

    # --- ROI ---
    work = ensure_max_size(image, config.max_side)
    h, w = work.shape[:2]

    st.markdown("**Region of interest (working-resolution pixels)**")

    def _reset_roi(width, height):
        st.session_state["roi_x0"] = 0
        st.session_state["roi_y0"] = 0
        st.session_state["roi_x1"] = width
        st.session_state["roi_y1"] = height

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        x0 = st.number_input("Left x0", 0, w - 1, 0, 1, key="roi_x0")
    with rc2:
        y0 = st.number_input("Top y0", 0, h - 1, 0, 1, key="roi_y0")
    with rc3:
        x1 = st.number_input("Right x1", 1, w, w, 1, key="roi_x1")
    with rc4:
        y1 = st.number_input("Bottom y1", 1, h, h, 1, key="roi_y1")
    st.button("Reset ROI to full image", key="roi_reset",
              on_click=_reset_roi, args=(w, h))
    box = (int(x0), int(y0), int(x1), int(y1))

    run = _run_analysis(image, config, box)

    # --- Visualisation params ---
    cA, cB, cC = st.columns(3)
    with cA:
        viz_radius = st.slider("Visual LBP radius", 1, 10, 3, key="viz_r")
    with cB:
        viz_points = st.slider("Visual LBP points P", 4, 32, 24, step=4, key="viz_p")
    with cC:
        viz_method = st.selectbox("Visual LBP method", LBP_METHODS, index=0,
                                  key="viz_method")

    if run.status == "error":
        st.error(run.error)
        return

    overlay = _draw_overlay(run.image, run.box)
    orig_view, gray_view, lbp_view, gabor_view = display_views(
        run.image, config, viz_radius, viz_points)

    if st.toggle("Show visualisation previews", value=True, key="show_prev"):
        colA, colB, colC, colD = st.columns(4)
        colA.image(overlay, caption="Original + ROI", width="stretch")
        colB.image(gray_view, caption="Grayscale (ROI)", width="stretch")
        colC.image(lbp_view,
                   caption=f"LBP {viz_method} R{viz_radius} P{viz_points}",
                   width="stretch")
        if gabor_view is None:
            placeholder = np.full_like(gray_view, 255, dtype="uint8")
            colD.image(placeholder,
                       caption="Gabor view (enable Gabor in the settings)",
                       width="stretch")
        else:
            colD.image(gabor_view, caption="Gabor magnitude (sum of responses)",
                       width="stretch")

    st.markdown("### Results")
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("ROI pixels", f"{run.total_pixels:,}")
    mcol2.metric("Fused vector length", int(run.vector.size))
    names = ", ".join(sorted(run.sizes.keys()))
    mcol3.metric("Enabled blocks", names or "lbp")

    st.pyplot(_histogram_figure(
        run.hist,
        "Fused multiscale LBP histogram (normalized, concatenated scales)"))

    with st.expander("Texture metrics (rule-based descriptors)"):
        metric_df = pd.DataFrame(
            [{"metric": k, "value": float(v)} for k, v in run.metrics.items()])
        st.dataframe(metric_df, width="stretch")

    # --- Heuristic descriptor (explicitly NOT a trained model) ---
    st.caption("**Rule-based texture descriptor** - not a trained classifier")
    rough = _roughness_score(run.metrics, len(run.hist))
    label = ("Smooth" if rough < 0.24 else "Fine" if rough < 0.48
             else "Rough" if rough < 0.72 else "Coarse")
    dc1, dc2 = st.columns(2)
    dc1.metric("Descriptor label", label)
    dc2.metric("Roughness (rule based)", f"{rough:.3f}")

    # --- Trained classification (only with a real compatible model) ---
    model_error = model_compatibility(run.config)
    if model_error is None:
        model = st.session_state["model"]
        pred_label, confidence, proba = predict_class(model, run.vector)
        st.markdown("#### Trained-model classification")
        note = model_pretrain_note(run.config)
        if note:
            st.caption(note)
        ca, cb = st.columns([1, 2])
        ca.metric("Predicted class", pred_label)
        ca.metric("Confidence (calibrated)", f"{confidence:.1%}")
        proba_df = pd.DataFrame(
            [{"class": c, "probability": v} for c, v in proba.items()])
        cb.bar_chart(proba_df, x="class", y="probability", height=220)
    else:
        st.warning(f"Classification disabled: {model_error}")

    # --- Similarity search against the real reference gallery ---
    gallery_error = gallery_compatibility(run.config)
    if gallery_error is None:
        gallery_vectors = [(g["label"], g["name"], g["vector"])
                           for g in st.session_state["gallery"]]
        ranked = similarity_ranks(
            run.vector,
            [(lbl, vec) for lbl, _n, vec in gallery_vectors],
            metric="cosine", top_k=5)
        name_map = {lbl: _n for lbl, _n, _v in gallery_vectors}
        st.markdown("#### Similarity against reference gallery")
        rows = [[i, lbl, round(d, 4), name_map.get(lbl, "")]
                for i, (lbl, d) in enumerate(ranked, start=1)]
        st.dataframe(pd.DataFrame(
            rows, columns=["rank", "class", "distance (cosine)", "source file"]),
            width="stretch")
    else:
        st.info(f"Similarity search unavailable: {gallery_error}")

    with st.expander("Export results"):
        _offer_exports(
            uploaded_name_of(uploaded), run,
            views={
                "original": orig_view,
                "overlay": overlay,
                "grayscale": gray_view,
                "lbp": lbp_view,
                "gabor": gabor_view,
            },
            viz=(viz_radius, viz_points, viz_method),
        )


def decode_upload(uploaded):
    from core.io import decode_image
    return decode_image(uploaded.getvalue(), name=uploaded_name_of(uploaded))


def uploaded_name_of(uploaded):
    return os.path.basename(str(getattr(uploaded, "name", "image")).replace("\\", "/"))


def _rgb_to_bytes(rgb, fmt="PNG", quality=95):
    """Encode an RGB (or gray) uint8 array into PNG/JPEG bytes."""

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype("uint8")
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    ext = ".jpg" if fmt.upper() in ("JPG", "JPEG") else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext == ".jpg" else []
    ok, buffer = cv2.imencode(ext, arr, params)
    if not ok:
        raise ValueError(f"Could not encode image as {ext}.")
    return buffer.tobytes()


def _apply_smoothing(rgb, kind, kernel):
    """Apply an optional smoothing filter to an exported image.

    This affects only the exported picture, never the LBP/Gabor feature
    extraction, so a smoothed export is never labelled as an LBP result.
    """

    if kind == "None" or kernel <= 1:
        return np.asarray(rgb)

    arr = np.asarray(rgb)
    if kind == "Gaussian blur":
        return cv2.GaussianBlur(arr, (int(kernel), int(kernel)), 0)

    size = int(kernel)
    if arr.ndim == 3:  # median blur supports only k=3/5 for colour images
        size = min(size, 5)
    if size % 2 == 0:
        size += 1
    return cv2.medianBlur(arr, size)


def _offer_exports(name, run, views, viz=None):
    exports_dir = tempfile.mkdtemp(prefix="tex_export_")
    base = os.path.splitext(name)[0]

    hist_png = render_histogram_png(
        run.hist, os.path.join(exports_dir, "histogram.png"),
        title="Fused multiscale LBP histogram")
    with open(hist_png, "rb") as handle:
        hist_bytes = handle.read()

    snapshot = {
        "source": name,
        "config": run.config.to_dict(),
        "roi": list(run.box),
        "roi_pixels": run.total_pixels,
        "metrics": {k: float(v) for k, v in run.metrics.items()},
        "vector_length": int(run.vector.size),
    }
    json_bytes = json.dumps(snapshot, indent=2).encode("utf-8")

    st.markdown("**Image downloads**")
    fmt = st.radio("Image format", ["PNG", "JPEG"], horizontal=True,
                   key="export_fmt",
                   help="PNG is lossless; JPEG is smaller and lets you pick "
                        "an export quality.")
    quality = 95
    if fmt == "JPEG":
        quality = st.slider("JPEG export quality", 1, 100, 95,
                            key="export_quality")

    smooth = st.selectbox(
        "Smoothing filter (image export only - never applied to LBP/Gabor)",
        ["None", "Gaussian blur", "Median blur"], key="export_smooth")
    kernel = 3
    if smooth != "None":
        kernel = st.slider("Smoothing kernel (odd)", 3, 15, 5, step=2,
                           key="export_smooth_k")

    radius, points, _method = viz if viz else (None, None, None)
    lbp_suffix = "lbp" if radius is None else f"lbp_R{int(radius)}_P{int(points)}"

    items = [
        ("Download original image", views.get("original"), "original", "orig"),
        ("Download original + ROI", views.get("overlay"), "original_roi", "roi"),
        ("Download grayscale", views.get("grayscale"), "grayscale", "gray"),
        ("Download LBP image", views.get("lbp"), lbp_suffix, "lbp"),
    ]
    if views.get("gabor") is not None:
        items.append(("Download Gabor view", views.get("gabor"), "gabor", "gabor"))

    ext = "jpg" if fmt.upper() in ("JPG", "JPEG") else "png"
    mime = "image/jpeg" if ext == "jpg" else "image/png"

    download_cols = st.columns(2)
    for index, (label, image, suffix, key) in enumerate(items):
        if image is None:
            continue
        if key in ("orig", "gray"):  # still images may be smoothed; LBP/Gabor not
            image = _apply_smoothing(image, smooth, kernel)
        with download_cols[index % 2]:
            st.download_button(
                label, data=_rgb_to_bytes(image, fmt, quality),
                file_name=f"{base}_{suffix}.{ext}", mime=mime,
                key=f"export_{key}")

    st.markdown("**Other downloads**")
    other_cols = st.columns(2)
    with other_cols[0]:
        st.download_button("Download histogram PNG", data=hist_bytes,
                           file_name=f"{base}_histogram.png", mime="image/png",
                           key="export_histogram")
    with other_cols[1]:
        st.download_button("Download JSON snapshot", data=json_bytes,
                           file_name=f"{base}_analysis.json",
                           mime="application/json", key="export_json")


class _BundledSample:
    """Duck-typed upload for the bundled sample file."""

    name = "brick.jpeg"

    def __init__(self):
        self._path = "images/brick.jpeg"

    def getvalue(self):
        with open(self._path, "rb") as handle:
            return handle.read()


def predict_class(model, vector):
    from classifier import predict
    return predict(model, vector)


# --------------------------------------------------------------------------
# Models tab
# --------------------------------------------------------------------------

def _import_labeled_zip(uploaded):
    """Yield (label, base_name, bytes) from an in-memory labelled ZIP.

    Layout required: <class_folder>/<image>.<ext>. Members with escaping
    paths are skipped.
    """

    import zipfile

    with zipfile.ZipFile(uploaded) as archive:
        for info in archive.infolist():
            norm = info.filename.replace("\\", "/").lstrip("/")
            if info.is_dir() or norm.startswith(("..", "/")) or ":/" in norm:
                continue
            parts = norm.split("/")
            if len(parts) < 2 or not parts[0]:
                continue
            label, rel = parts[0], "/".join(parts[1:])
            if not rel.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            yield label, os.path.basename(rel), archive.read(info)


def render_models():
    st.header("Model Manager")

    model = st.session_state.get("model")

    if model is not None:
        st.success("A trained model is loaded; classification is active.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Classes", ", ".join(model["classes"]))
        m2.metric("Feature length", model["feature_length"])
        m3.metric("Test accuracy (measured)", f"{model['metrics']['test_accuracy']:.1%}")
        st.caption(
            f"Trained {model['trained_at']} on {model['n_samples']} samples "
            f"({model['metrics']['test_samples']} held-out test samples). "
            f"Model type: {model['model_type']}.")
        with st.expander("Per-class classification report"):
            st.text(model.get("classification_report", ""))
        conf = np.asarray(model.get("confusion_matrix", [[]]))
        if conf.size and conf.shape[0] == conf.shape[1]:
            st.caption("Confusion matrix on the held-out test split "
                       "(rows = true, columns = predicted)")
            st.dataframe(pd.DataFrame(conf, index=model["confusion_labels"],
                                      columns=model["confusion_labels"]),
                         width="stretch")
        if st.button("Unload model", key="model_unload"):
            st.session_state["model"] = None
            st.rerun()

    st.subheader("Load a previously trained model")
    st.caption(MODEL_SECURITY_WARNING)

    if os.path.isfile("texture_model.pkl"):
        if st.button("Load bundled model (texture_model.pkl)", key="model_bundled"):
            with open("texture_model.pkl", "rb") as handle:
                _set_model_from_bytes(handle.read())

    loaded = st.file_uploader("Upload a model file (joblib .pkl)",
                              type=["pkl", "joblib"], key="model_upload")
    if loaded is not None:
        _set_model_from_bytes(loaded.getvalue())

    st.divider()
    st.subheader("Train a new model on real labelled data")
    st.caption(LABEL_ZIP_HINT)
    zip_upload = st.file_uploader("Upload labelled ZIP", type=["zip"],
                                  key="train_zip")
    if zip_upload is None:
        return

    config = feature_config_widgets("train", default_max_side=512)

    if st.button("Train classifier", type="primary", key="train_run"):
        files = list(_import_labeled_zip(zip_upload))
        if not files:
            st.error("No labelled images found in the ZIP. Expected "
                     "class-folder/image.jpg layout.")
            return
        st.info(f"Found {len(files)} labelled images. Extracting features...")
        X, y, names, errors = extract_labeled_dataset(files, config)
        if errors:
            with st.expander(f"{len(errors)} file(s) could not be used"):
                for fname, exc in errors[:50]:
                    st.write(f"- **{fname}**: {exc}")
        try:
            st.info("Training the SVM (this may take a moment)...")
            bundle = train_model(X, y, config)
        except ValueError as exc:
            st.error(f"Could not train: {exc}")
            return

        st.session_state["gallery"] = [
            {"label": lbl, "name": nm, "vector": vec}
            for lbl, nm, vec in zip(y, names, X)]
        st.session_state["gallery_config"] = config.to_dict()
        st.session_state["model"] = bundle

        st.success("Model trained on real labelled data.")
        s1, s2, s3 = st.columns(3)
        s1.metric("Train accuracy (fit data)", f"{bundle['metrics']['train_accuracy']:.1%}")
        s2.metric("Test accuracy (held-out)",
                  f"{bundle['metrics']['test_accuracy']:.1%}")
        s3.metric("Test samples", bundle["metrics"]["test_samples"])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            model_path, _sidecar = save_model(bundle, tmp.name)
        with open(model_path, "rb") as handle:
            model_bytes = handle.read()
        st.download_button("Download trained model (.pkl)",
                           data=model_bytes, file_name="texture_model.pkl",
                           mime="application/octet-stream")
        try:
            os.remove(model_path)
        except OSError:
            pass

    st.caption(
        "The reference gallery is rebuilt from the same labelled files, so "
        "similarity search in the Analyzer compares queries against real, "
        "seen samples of known class.")


# --------------------------------------------------------------------------
# Batch tab
# --------------------------------------------------------------------------

def render_batch():
    st.header("Batch Processing")

    files = st.file_uploader(
        "Upload multiple images and/or ZIP archives",
        type=["jpg", "jpeg", "png", "zip"],
        accept_multiple_files=True, key="batch_files")

    with st.expander("ZIP member safety"):
        st.caption(
            "ZIP members are validated against path traversal before "
            "decoding; archives are read in memory and never extracted to "
            "disk. Failed files are reported without stopping the batch.")

    uploaded_as_bytes = [(f.name, f.getvalue()) for f in (files or [])]
    config = feature_config_widgets("batch")

    if not uploaded_as_bytes:
        st.info("Upload files to begin.")
        return

    items, setup_errors = collect_batch_inputs(uploaded_as_bytes)
    if setup_errors:
        with st.expander(f"{len(setup_errors)} archive error(s)"):
            for fname, msg in setup_errors[:100]:
                st.write(f"- **{fname}**: {msg}")

    model = st.session_state.get("model")
    model_ok = model is not None and model["feature_length"] == config.descriptor_length()
    if model is not None and not model_ok:
        st.warning("Loaded model is incompatible with the current feature "
                   "configuration; batch will run without classification.")

    workers = st.slider("Worker threads", 1, 8, 4, key="batch_workers",
                        help="Concurrency for decoding + feature extraction; "
                             "each file is reported independently.")

    if st.button("Process batch", type="primary", key="batch_run"):
        if not items:
            st.warning("No supported images found in the upload.")
            return

        def worker(name, data):
            from core.io import decode_image
            from classifier import predict

            image = decode_image(data, name=name)
            extracted = extract_features(image, config)
            vec = np.asarray(extracted["vector"], dtype="float64")
            total = float(vec.sum())
            entry = {
                "label": "", "confidence": "",
                "fused_entropy": float(histogram_statistics(
                    vec / total if total > 0 else vec)["entropy"]),
            }
            if model_ok:
                pred_label, confidence, _ = predict(model, vec)
                entry["label"] = pred_label
                entry["confidence"] = f"{confidence:.4f}"
            return entry

        results = process_batch(items, worker, workers=workers)
        ok_count = sum(1 for r in results if r["status"] == "ok")
        st.success(f"Processed {ok_count}/{len(results)} files successfully.")

        rows = []
        for r in results:
            result = r.get("result") or {}
            rows.append([
                r["name"], r["status"],
                None if r["duration_ms"] is None else round(r["duration_ms"], 2),
                r["error"] or "", result.get("label", ""),
                result.get("confidence", ""), result.get("fused_entropy", ""),
            ])
        df = pd.DataFrame(
            rows, columns=["file", "status", "duration_ms", "error",
                           "predicted_class", "confidence", "fused_entropy"])
        st.dataframe(df, width="stretch")

        with tempfile.TemporaryDirectory(prefix="tex_batch_") as tmp:
            csv_path = write_results_csv(results, os.path.join(tmp, "batch.csv"))
            with open(csv_path, "rb") as handle:
                csv_bytes = handle.read()
        st.download_button("Download results CSV", data=csv_bytes,
                           file_name="batch_results.csv", mime="text/csv",
                           key="batch_dl")


# --------------------------------------------------------------------------
# Performance tab
# --------------------------------------------------------------------------

def render_performance():
    st.header("Performance Benchmark")
    st.caption(
        "Every figure below is measured on this machine during this session "
        "(time.perf_counter after a warm-up; median over N repetitions). "
        "Memory is the process peak working set.")

    source = st.radio("Input source", ["Bundled sample", "Upload image"],
                      horizontal=True, index=0, key="bench_source")
    image = None
    if source == "Upload image":
        up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"],
                              key="bench_upload")
        if up is not None:
            image = decode_upload(up)
    elif os.path.isfile("images/brick.jpeg"):
        image = read_image("images/brick.jpeg")

    reps = st.slider("Repetitions per component", 1, 10, 3, key="bench_reps")

    if image is None:
        st.info("Choose an image to measure.")
        return

    work_dims = ensure_max_size(image, 768)
    st.caption(f"Input {image.shape[1]}x{image.shape[0]}; the fused pipeline "
               f"benchmark runs at {work_dims.shape[1]}x{work_dims.shape[0]}.")

    if st.button("Run benchmark", type="primary", key="bench_run"):
        with st.spinner("Measuring (this can take a while)..."):
            result = benchmark_all(image, reps=int(reps))

        env_rows = [{"variable": k, "value": str(v)}
                    for k, v in result.environment.items()]
        st.subheader("Environment")
        st.dataframe(pd.DataFrame(env_rows), width="stretch")

        st.subheader(f"Latency (measured, median over {result.reps} runs)")
        rows = [{"component": name, "median_ms": round(s["median_ms"], 3),
                 "mean_ms": round(s["mean_ms"], 3), "std_ms": round(s["std_ms"], 3)}
                for name, s in result.timings.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch")

        chart_path = render_benchmark_chart(
            result, os.path.join(tempfile.gettempdir(), "bench_chart.png"))
        st.image(chart_path, width="stretch")

        st.subheader("Memory")
        mem = result.memory
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline RSS", _fmt_mb(mem["baseline_rss_mb"]))
        m2.metric("Peak during run", _fmt_mb(mem["peak_rss_mb"]))
        m3.metric("Measurement method", mem["method"])

        st.subheader("Custom-engine equivalence")
        eq = result.equivalence
        if eq["equal"]:
            st.success(
                "NumPy and Numba custom engines produced bit-identical output "
                f"on the {result.image_size[0]}x{result.image_size[1]} "
                "benchmark input. The speedup claim, if any, is backed only by "
                "the measured numbers above.")
        else:
            st.error(f"Engines differ ({eq['diff_count']} pixels); latencies are "
                     "informational only.")

        for note in result.notes:
            st.caption(f"* {note}")

        with st.expander("Download benchmark report"):
            try:
                pdf_path = build_benchmark_pdf(
                    result, chart_path,
                    os.path.join(tempfile.gettempdir(), "benchmark_report.pdf"))
                with open(pdf_path, "rb") as handle:
                    pdf_bytes = handle.read()
                st.download_button("Download PDF report", data=pdf_bytes,
                                   file_name="benchmark_report.pdf",
                                   mime="application/pdf", key="bench_pdf")
            except Exception as exc:  # noqa: BLE001
                st.error(f"PDF export failed: {exc}")


def _fmt_mb(value):
    return "n/a" if value is None else f"{value:.1f} MB"


# --------------------------------------------------------------------------
# Sidebar + dispatch
# --------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title(":microscope: Texture Analysis Studio")
        st.caption("Local Binary Pattern + Gabor feature fusion")

        model = st.session_state.get("model")
        if model is not None:
            st.success(f"Model loaded: {', '.join(model['classes'])}")
        else:
            st.warning("No trained model loaded. Trained classification is "
                       "disabled until you train or load one in the Models tab.")
        gallery_n = len(st.session_state.get("gallery", []))
        if gallery_n:
            st.info(f"Reference gallery: {gallery_n} real labelled samples.")
        else:
            st.caption("Reference gallery empty - similarity search disabled.")
        st.caption("All metrics are computed live from real measurements; "
                   "no fabricated results are shown.")


def main():
    _init_state()
    _inject_theme()
    render_sidebar()

    tab = st.tabs(["Analyzer", "Models", "Batch", "Performance"])
    with tab[0]:
        render_analyzer()
    with tab[1]:
        render_models()
    with tab[2]:
        render_batch()
    with tab[3]:
        render_performance()


if __name__ == "__main__":
    main()