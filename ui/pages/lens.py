"""Texurion Scanning Lens - separate visual scanning workspace (Google Lens inspired).

Performs the actual image-processing pipeline. Orio explains the results.
"""

import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.pipeline import display_views

from .. import components, runtime, state, store
from ..brand import APP_NAME, ASSISTANT_NAME

LENS_UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]


def _init_state():
    st.session_state.setdefault("lens_image", None)
    st.session_state.setdefault("lens_image_name", None)
    st.session_state.setdefault("lens_run", None)


def _lens_preview(image, name):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    cols = st.columns([3, 2])
    with cols[0]:
        st.image(rgb, caption=f"{name} · {w}×{h}", width="stretch")
    with cols[1]:
        if st.button("Remove", key="lens_remove"):
            st.session_state["lens_image"] = None
            st.session_state["lens_image_name"] = None
            st.session_state["lens_run"] = None
            st.rerun()
        if st.button(f"Ask {ASSISTANT_NAME} about this scan", key="lens_to_orio",
                     disabled=st.session_state.get("lens_run") is None):
            st.session_state["lens_pending_orio"] = True
            st.session_state["orio_open"] = True
            st.switch_page("views/orio_page.py")


def _render_results(run, name, config):
    st.markdown("### Scanning Lens results")
    names = ", ".join(sorted(run.sizes.keys())) or "lbp"
    components.metric_row([
        ("Descriptor label", run.label, "Rule-based descriptor (not trained classifier)"),
        ("Roughness", f"{run.roughness:.3f}", "Computed from GLCM + histogram"),
        ("ROI pixels", f"{run.total_pixels:,}", None),
        ("Feature vector", str(run.vector.size), None),
        ("Blocks", names, None),
    ])
    st.markdown(f'<div class="tx-note">{runtime.metric_summary(run)}</div>', unsafe_allow_html=True)

    prediction = _maybe_predict(run, config)
    if prediction:
        c1, c2 = st.columns([1, 2])
        c1.metric("Predicted class (trained model)", prediction["label"])
        c1.metric("Confidence (calibrated)", f"{prediction['confidence']:.1%}")
        df = pd.DataFrame([{"class": k, "probability": v} for k, v in prediction["proba"].items()])
        c2.bar_chart(df, x="class", y="probability", height=200)

    st.markdown("**Texture metrics**")
    st.dataframe(runtime.supported_metric_table(run), width="stretch", hide_index=True)

    views = {}
    original, gray_view, lbp_view, gabor_view = display_views(run.image, run.config, 3, 24)
    views.update({"original": original, "grayscale": gray_view, "lbp": lbp_view,
                  "overlay": runtime.draw_overlay(run.image, run.box)})
    if gabor_view is not None and run.config.include_gabor:
        views["gabor"] = gabor_view

    a1, a2, a3 = st.columns(3)
    if a1.button("Analyze another", key="lens_another", use_container_width=True):
        st.session_state["lens_image"] = None
        st.session_state["lens_image_name"] = None
        st.session_state["lens_run"] = None
        st.rerun()
    if a2.button("Compare results", key="lens_compare", use_container_width=True):
        st.switch_page("views/compare.py")
    if a3.button(f"Ask {ASSISTANT_NAME}", type="primary", use_container_width=True):
        st.session_state["lens_pending_orio"] = True
        st.session_state["orio_open"] = True
        st.switch_page("views/orio_page.py")

    if st.button("Save this scan", key="lens_save"):
        try:
            record_id = store.save_analysis(
                name, config, run.box, run, views,
                model_prediction=prediction,
                workflow="lens")
            st.success(f"Saved locally (ID: {record_id})")
        except Exception as exc:
            st.error(f"Save failed: {exc}")


def _maybe_predict(run, config):
    if state.model_compatibility(config) is not None:
        return None
    model = state.current_model()
    label, confidence, proba = state.predict_class(model, run.vector)
    return {"label": label, "confidence": confidence, "proba": proba}


def render():
    _init_state()
    st.markdown("## 🔎 Texurion Scanning Lens")
    st.caption("Visual intelligence: upload/scan an image, run the real pipeline, "
               "and view substantiated results. Orio explains — they stay separate.")

    upload_tab, camera_tab = st.tabs(["Upload", "Camera"])
    with upload_tab:
        up = st.file_uploader("Upload image", type=LENS_UPLOAD_TYPES, key="lens_upload")
        if up is not None:
            data = up.getvalue()
            if data != st.session_state.get("lens_image"):
                st.session_state["lens_image"] = data
                st.session_state["lens_image_name"] = up.name
                st.session_state["lens_run"] = None
    with camera_tab:
        shot = st.camera_input("Capture image", key="lens_camera")
        if shot is not None and shot.getvalue() != st.session_state.get("lens_image"):
            st.session_state["lens_image"] = shot.getvalue()
            st.session_state["lens_image_name"] = "camera_capture.jpg"
            st.session_state["lens_run"] = None

    data = st.session_state.get("lens_image")
    if data is None:
        components.empty_state("🔎", "No image selected",
                              "Upload or capture an image to start scanning.")
        return

    try:
        image = runtime.decode_bytes(data, st.session_state.get("lens_image_name", "image"))
    except Exception as exc:
        st.error(f"Invalid image: {exc}")
        return

    _lens_preview(image, st.session_state.get("lens_image_name", "image"))

    with st.expander("Scan settings", expanded=True):
        config = state.feature_config_widgets("lens_scan")

    work = runtime.ensure_max_size(image, config.max_side)
    box = (0, 0, work.shape[1], work.shape[0])

    if st.button("Scan with Texurion Lens", type="primary", key="lens_scan_btn"):
        with st.spinner("Scanning — extracting real texture features..."):
            run = runtime.run_analysis(image, config, box)
        st.session_state["lens_run"] = {"run": run, "name": st.session_state["lens_image_name"]}

    stored = st.session_state.get("lens_run")
    if not stored:
        return
    run = stored["run"]
    if run.status == "error":
        st.error(f"Scan failed: {run.error}")
        return
    _render_results(run, stored["name"], config)
