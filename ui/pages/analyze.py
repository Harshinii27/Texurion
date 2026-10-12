"""Analyze Image -- upload, configure, run and interpret a real analysis."""

import io
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.pipeline import display_views

from .. import components, runtime, state, store, workflows
from ..brand import APP_NAME, ASSISTANT_NAME

SAMPLE_PATH = os.path.join("images", "brick.jpeg")
SAMPLE_DIR = os.path.join("assets", "samples")

UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


# ---------------------------------------------------------------------------
# Category selection
# ---------------------------------------------------------------------------

def _category_control():
    labels = [f"{wf['icon']} {wf['title']}" for wf in workflows.WORKFLOWS]
    current = st.session_state.get("analyzer_workflow")
    default_index = next(
        (i for i, wf in enumerate(workflows.WORKFLOWS) if wf["key"] == current), 0)
    selected = st.pills(
        "Analysis category", labels, default=labels[default_index],
        key="analyzer_category_pills")
    if selected is None:
        return
    chosen = workflows.WORKFLOWS[labels.index(selected)]["key"]
    if chosen != st.session_state.get("analyzer_workflow"):
        workflows.apply_defaults(chosen, st.session_state)
        st.rerun()


def _workflow_guidance():
    key = st.session_state.get("analyzer_workflow")
    wf = workflows.get(key) if key else None
    if not wf:
        components.note(
            "No category selected. Texurion analyses **texture you provide**; "
            "it does not automatically recognise the object. Pick a category "
            "above for tuned steps, or continue with the general pipeline.")
        return
    st.markdown(f"**{wf['icon']} {wf['title']} workflow**")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Suggested analysis**")
        st.markdown("\n".join(f"- {a}" for a in wf["analyses"]))
    with cols[1]:
        st.markdown("**Limitations**")
        st.caption(wf["limitations"])
    with st.expander("Learn more about this workflow"):
        st.markdown("**Typical use cases**")
        st.markdown("\n".join(f"- {u}" for u in wf["use_cases"]))
        st.caption(f"Tip: {wf['suggestion']}")


# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------

def _set_image(data, name):
    st.session_state["analyzer_image"] = data
    st.session_state["analyzer_image_name"] = name
    st.session_state["analyzer_run"] = None


def _image_input():
    st.markdown("#### 1 · Provide an image")
    tab_upload, tab_camera, tab_sample = st.tabs(
        ["📁 Upload", "📷 Camera", "🧪 Sample"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Drag and drop or browse (JPG, PNG, BMP, TIFF, WebP)",
            type=UPLOAD_TYPES, key="analyzer_upload")
        if uploaded is not None:
            if uploaded.size > MAX_UPLOAD_BYTES:
                st.error(f"File is too large ({uploaded.size / 1e6:.1f} MB). "
                         "Maximum is 200 MB.")
            elif uploaded.getvalue() != st.session_state.get("analyzer_image"):
                _set_image(uploaded.getvalue(), uploaded.name)

    with tab_camera:
        st.caption("Capture a photo with your device camera (if available).")
        shot = st.camera_input("Camera", key="analyzer_camera")
        if shot is not None and shot.getvalue() != st.session_state.get("analyzer_image"):
            _set_image(shot.getvalue(), "camera_capture.jpg")

    with tab_sample:
        _sample_gallery()


def _sample_gallery():
    st.caption("Representative textures from the bundled classes. Click one "
               "to run the full analysis workflow on it.")
    files = []
    if os.path.isdir(SAMPLE_DIR):
        files = sorted(f for f in os.listdir(SAMPLE_DIR)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        if os.path.isfile(SAMPLE_PATH) and st.button(
                "Use bundled sample", key="analyzer_sample"):
            with open(SAMPLE_PATH, "rb") as handle:
                _set_image(handle.read(), "brick.jpeg")
            st.rerun()
        return
    cols = st.columns(4)
    for i, fname in enumerate(files):
        path = os.path.join(SAMPLE_DIR, fname)
        image = cv2.imread(path)
        if image is None:
            continue
        with cols[i % 4]:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
            label = os.path.splitext(fname)[0].replace("_", " ").title()
            if st.button(label, key=f"analyzer_sample_{fname}",
                         width="stretch"):
                with open(path, "rb") as handle:
                    _set_image(handle.read(), fname)
                st.session_state["analyzer_autorun"] = True
                st.rerun()


def _preview(image, name, data):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    cols = st.columns([3, 2], gap="large")
    with cols[0]:
        st.image(rgb, caption=f"Preview · {name}", width="stretch")
    with cols[1]:
        st.markdown("**Image details**")
        st.markdown(
            f"- Dimensions: **{w} × {h} px**\n"
            f"- Channels: **{channels}**\n"
            f"- Size: **{len(data) / 1024:.1f} KB**\n"
            f"- Format hint: **{os.path.splitext(name)[1].lower() or 'unknown'}**")
        if st.button("Remove image", key="analyzer_remove"):
            state.clear_analyzer()
            st.rerun()
        st.caption("Replace it by choosing another file in the Upload tab.")


# ---------------------------------------------------------------------------
# Settings + ROI
# ---------------------------------------------------------------------------

def _settings(config):
    return config


def _roi_controls(work):
    h, w = work.shape[:2]
    st.markdown("**Region of interest (working-resolution pixels)**")
    st.caption("Analysis is restricted to this rectangle. Defaults to the "
               "whole image.")

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
    return (int(x0), int(y0), int(x1), int(y1))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _model_prediction(run):
    if state.model_compatibility(run.config) is not None:
        return None
    model = state.current_model()
    label, confidence, proba = state.predict_class(model, run.vector)
    return {"label": label, "confidence": confidence, "proba": proba}


def _render_overview(run):
    names = ", ".join(sorted(run.sizes.keys())) or "lbp"
    components.metric_row([
        ("Descriptor label", run.label,
         "Rule-based descriptor, NOT a trained classifier."),
        ("Roughness (rule-based)", f"{run.roughness:.3f}",
         runtime.explain_metric("Roughness")["plain"]),
        ("ROI pixels", f"{run.total_pixels:,}", "Pixels analysed in the ROI."),
        ("Feature vector", f"{run.vector.size}", "Length of the fused vector."),
        ("Enabled blocks", names, "Feature blocks used."),
    ])
    st.markdown(f'<div class="tx-note">{runtime.metric_summary(run)}</div>',
                unsafe_allow_html=True)
    st.caption("These are observed image characteristics, not calibrated "
               "physical properties.")
    if st.button(f"🧭 Ask {ASSISTANT_NAME} to explain", key="analyzer_ask_orio"):
        st.session_state["orio_open"] = True
        st.rerun()

    prediction = _model_prediction(run)
    if prediction:
        st.markdown("#### Trained-model classification")
        note = state.model_pretrain_note(run.config)
        if note:
            st.caption(note)
        c1, c2 = st.columns([1, 2])
        c1.metric("Predicted class", prediction["label"])
        c1.metric("Confidence (calibrated)", f"{prediction['confidence']:.1%}")
        df = pd.DataFrame([{"class": k, "probability": v}
                           for k, v in prediction["proba"].items()])
        c2.bar_chart(df, x="class", y="probability", height=200)
    else:
        reason = state.model_compatibility(run.config)
        components.note(
            f"Trained classification is disabled: {reason} "
            "The descriptor label above is a rule-based heuristic, not a model.")


def _render_visual(run):
    cA, cB, cC = st.columns(3)
    viz_radius = cA.slider("Visual LBP radius", 1, 10, 3, key="viz_r")
    viz_points = cB.slider("Visual LBP points P", 4, 32, 24, step=4, key="viz_p")
    if cC.checkbox("Show Gabor view", value=run.gray is not None and
                   bool(run.config.include_gabor), key="viz_gabor") \
            and not run.config.include_gabor:
        st.caption("Enable the Gabor block in Analysis settings to compute it.")

    overlay = runtime.draw_overlay(run.image, run.box)
    original, gray_view, lbp_view, gabor_view = display_views(
        run.image, run.config, viz_radius, viz_points)
    if not run.config.include_gabor:
        gabor_view = None

    r1 = st.columns(2)
    r1[0].image(overlay, caption="Original + ROI", width="stretch")
    r1[1].image(gray_view, caption="Grayscale (ROI)", width="stretch")
    r2 = st.columns(2)
    r2[0].image(lbp_view, caption=f"LBP map (R{viz_radius} P{viz_points})",
                width="stretch")
    if gabor_view is not None:
        r2[1].image(gabor_view, caption="Gabor magnitude (summed)",
                    width="stretch")
    else:
        placeholder = np.full((80, 320, 3), 250, dtype="uint8")
        r2[1].image(placeholder,
                    caption="Gabor view (enable Gabor in settings)",
                    width="stretch")

    st.caption("Visual maps aid interpretation; only the computed metrics are "
               "quantitative. The LBP map is not an AI-generated texture map.")
    st.pyplot(runtime.histogram_figure(
        run.hist, "Fused multiscale LBP histogram (normalised)"))


def _render_metrics(run):
    st.markdown("**Computed texture metrics**")
    st.dataframe(runtime.supported_metric_table(run), width="stretch",
                 hide_index=True)
    st.markdown("**Explain a metric**")
    names = list(run.metrics)
    cols = st.columns(3)
    for i, name in enumerate(names):
        with cols[i % 3]:
            components.explain_button(
                name, f"{run.metrics[name]:.4f}", f"m_{name}")
    st.caption("GLCM metrics come from a 32-level quantised grayscale ROI; "
               "Entropy/Uniformity/Mean/Std Dev come from the fused LBP "
               "histogram.")


def _render_research(run, name):
    st.markdown("**Reproducible settings**")
    st.json(run.config.to_dict())
    st.markdown("**Feature block sizes**")
    st.dataframe(pd.DataFrame([{"block": k, "length": v}
                               for k, v in run.sizes.items()]),
                 width="stretch", hide_index=True)
    vec = np.asarray(run.vector, dtype="float64")
    st.markdown("**Feature-vector statistics**")
    components.metric_row([
        ("Length", str(vec.size), None),
        ("Min", f"{vec.min():.5f}", None),
        ("Max", f"{vec.max():.5f}", None),
        ("Mean", f"{vec.mean():.5f}", None),
        ("Sum", f"{vec.sum():.5f}", None),
        ("Non-zero", str(int(np.count_nonzero(vec))), None),
    ])
    buffer = io.StringIO()
    pd.DataFrame({"index": np.arange(vec.size), "value": vec}).to_csv(
        buffer, index=False)
    st.download_button(
        "Download feature vector (CSV)", data=buffer.getvalue().encode(),
        file_name=f"{os.path.splitext(name)[0]}_features.csv", mime="text/csv",
        key="analyzer_vec_csv")
    st.caption("LBP histogram, GLCM, Gabor and opponent-colour blocks are all "
               "documented above; nothing is summarised into an unverified "
               "score.")


def _render_exports(run, name, config, views, prediction):
    st.markdown("### Save & export")
    left, right = st.columns(2)
    with left:
        if st.button("💾 Save analysis", type="primary", key="analyzer_save"):
            try:
                record_id = store.save_analysis(
                    name, config, run.box, run, views,
                    model_prediction=prediction,
                    workflow=st.session_state.get("analyzer_workflow"))
                st.session_state["last_saved_id"] = record_id
                st.success("Saved locally. Open **Reports** to export it.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not save analysis: {exc}")
    with right:
        st.caption("Saved analyses live in server-local storage "
                   "(`saved_analyses/`). They do not sync across devices.")

    record_id = st.session_state.get("last_saved_id")
    record = store.load_analysis(record_id) if record_id else None
    if record and record.get("name") == name:
        st.markdown("**Downloads**")
        try:
            from .. import reports

            pdf = reports.build_analysis_pdf(record)
            st.download_button("📄 Download PDF report", data=pdf,
                               file_name=f"{os.path.splitext(name)[0]}_texurion.pdf",
                               mime="application/pdf", key="analyzer_pdf")
        except Exception as exc:  # noqa: BLE001
            st.error(f"PDF export unavailable: {exc}")
        st.download_button("📊 Download metrics CSV",
                           data=store.record_csv_bytes(record),
                           file_name=f"{os.path.splitext(name)[0]}_metrics.csv",
                           mime="text/csv", key="analyzer_csv")
        st.download_button("🧾 Download JSON snapshot",
                           data=store.record_json_bytes(record),
                           file_name=f"{os.path.splitext(name)[0]}_analysis.json",
                           mime="application/json", key="analyzer_json")

    st.markdown("**Image downloads**")
    fmt = st.radio("Image format", ["PNG", "JPEG"], horizontal=True,
                   key="analyzer_export_fmt")
    quality = 95
    if fmt == "JPEG":
        quality = st.slider("JPEG quality", 1, 100, 95, key="analyzer_quality")
    for label, key in (("Download original", "original"),
                       ("Download grayscale", "grayscale"),
                       ("Download LBP map", "lbp"),
                       ("Download original + ROI", "overlay")):
        image = views.get(key)
        if image is None:
            continue
        ext = "jpg" if fmt == "JPEG" else "png"
        mime = "image/jpeg" if ext == "jpg" else "image/png"
        st.download_button(
            label, data=runtime.rgb_to_bytes(image, fmt, quality),
            file_name=f"{os.path.splitext(name)[0]}_{key}.{ext}", mime=mime,
            key=f"analyzer_dl_{key}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render():
    st.markdown(f"## 🔍 Analyze Image")
    st.caption(f"{APP_NAME} runs a real LBP / GLCM texture analysis on the "
               "region you choose. Every value shown is computed from your image.")

    _category_control()
    _image_input()

    data = st.session_state.get("analyzer_image")
    if data is None:
        components.empty_state(
            "📤", "No image yet",
            "Provide an image above (upload, camera or sample) to begin. "
            "Supported formats: JPG, PNG, BMP, TIFF, WebP.")
        return

    name = st.session_state.get("analyzer_image_name", "image")
    try:
        image = runtime.decode_bytes(data, name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Image could not be loaded: {exc}")
        if st.button("Clear image", key="analyzer_clear_bad"):
            state.clear_analyzer()
            st.rerun()
        return

    _preview(image, name, data)
    _workflow_guidance()

    with st.expander("⚙️ Analysis settings (advanced)", expanded=False):
        config = state.feature_config_widgets("analyzer")

    work = runtime.ensure_max_size(image, config.max_side)
    with st.expander("🎯 Region of interest", expanded=False):
        box = _roi_controls(work)
    if "roi_x0" not in st.session_state:
        box = (0, 0, work.shape[1], work.shape[0])

    st.write("")
    autorun = st.session_state.pop("analyzer_autorun", False)
    if autorun or st.button("🚀 Analyze Image", type="primary",
                            key="analyzer_run_btn"):
        with st.spinner("Extracting LBP, GLCM and statistics…"):
            run = runtime.run_analysis(image, config, box)
        st.session_state["analyzer_run"] = {"run": run, "name": name}

    stored = st.session_state.get("analyzer_run")
    if not stored:
        return
    run = stored["run"]
    name = stored["name"]
    if run.status == "error":
        st.error(f"Analysis failed: {run.error}")
        st.caption("Try a larger region of interest or a different image.")
        return

    st.write("")
    overview, visual, metrics, research = st.tabs(
        ["Overview", "Visual analysis", "Metrics", "Research mode"])
    with overview:
        _render_overview(run)
    with visual:
        _render_visual(run)
    with metrics:
        _render_metrics(run)
    with research:
        _render_research(run, name)

    views = {}
    original, gray_view, lbp_view, gabor_view = display_views(
        run.image, run.config,
        st.session_state.get("viz_r", 3), st.session_state.get("viz_p", 24))
    views.update({"original": original, "grayscale": gray_view, "lbp": lbp_view,
                  "overlay": runtime.draw_overlay(run.image, run.box)})
    if gabor_view is not None and run.config.include_gabor:
        views["gabor"] = gabor_view

    _render_exports(run, name, run.config, views, _model_prediction(run))
