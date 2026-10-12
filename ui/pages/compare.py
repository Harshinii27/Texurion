"""Compare Images -- two-image texture comparison studio."""

import os

import cv2
import pandas as pd
import streamlit as st

from .. import components, runtime, state
from .. import compare as compare_engine
from .. import reports

UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]


def _uploader(label, slot, key):
    uploaded = st.file_uploader(label, type=UPLOAD_TYPES, key=key)
    if uploaded is not None:
        data = uploaded.getvalue()
        if data != st.session_state.get(f"compare_{slot}"):
            st.session_state[f"compare_{slot}"] = data
            st.session_state[f"compare_{slot}_name"] = uploaded.name
            st.session_state["compare_result"] = None
    data = st.session_state.get(f"compare_{slot}")
    name = st.session_state.get(f"compare_{slot}_name")
    if data is None:
        return None, None
    try:
        image = runtime.decode_bytes(data, name or f"image_{slot}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"{label}: {exc}")
        if st.button(f"Clear {label}", key=f"clear_{slot}"):
            st.session_state[f"compare_{slot}"] = None
            st.session_state["compare_result"] = None
            st.rerun()
        return None, None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=f"{name} · {image.shape[1]}×{image.shape[0]}",
             width="stretch")
    if st.button(f"Remove", key=f"remove_{slot}"):
        st.session_state[f"compare_{slot}"] = None
        st.session_state["compare_result"] = None
        st.rerun()
    return image, name


def _results(result, name_a, name_b):
    st.markdown("### Comparison result")
    st.markdown(f'<div class="tx-note">{compare_engine.summary_text(result)}</div>',
                unsafe_allow_html=True)

    components.metric_row([
        ("Pixel similarity (SSIM)", f"{result['ssim']:.4f}",
         "Structural similarity on aligned grayscale images (1 = identical)."),
        ("Mean abs. difference", f"{result['mae']:.2f} / 255",
         "Average absolute grayscale pixel difference."),
        ("Feature similarity", f"{result['feature_similarity']:.4f}",
         "1 minus the cosine distance between full fused feature vectors."),
        ("LBP chi-square", f"{result['lbp_chi_square']:.4f}",
         "Chi-square distance between normalised LBP histograms."),
    ])

    tabs = st.tabs(["Difference map", "Texture profile", "Metric deltas",
                    "Exports"])
    with tabs[0]:
        cols = st.columns(2)
        cols[0].image(result["gray_a"], caption=f"A · {name_a}",
                      width="stretch", clamp=True)
        cols[1].image(result["diff_image"],
                      caption="Difference (bright = larger change)",
                      width="stretch", clamp=True)
        st.caption("The difference map is a pixel-level aid and is aligned to "
                   "image A's grid. It is not a defect verdict.")

    with tabs[1]:
        fig = compare_engine.radar_figure(result)
        if fig is not None:
            st.pyplot(fig)
            st.caption("Metrics normalised to the maximum across both images. "
                       "Read shape, not absolute area.")
        else:
            st.info("Not enough comparable metrics for a radar chart.")

    with tabs[2]:
        rows = []
        for metric in result["metrics_a"]:
            rows.append({
                "metric": metric,
                "image A": round(result["metrics_a"][metric], 5),
                "image B": round(result["metrics_b"].get(metric, 0), 5),
                "delta (B - A)": round(result["deltas"].get(metric, 0), 5),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Pixel, histogram and feature similarity are distinct "
                   "measures and are reported separately.")

    with tabs[3]:
        try:
            pdf = reports.build_comparison_pdf(result, name_a, name_b)
            st.download_button("📄 Download comparison PDF", data=pdf,
                               file_name="texurion_comparison.pdf",
                               mime="application/pdf", key="compare_pdf")
        except Exception as exc:  # noqa: BLE001
            st.error(f"PDF export unavailable: {exc}")
        st.download_button("📊 Download comparison CSV",
                           data=reports.comparison_csv_bytes(result),
                           file_name="texurion_comparison.csv", mime="text/csv",
                           key="compare_csv")
        fmt = st.radio("Difference image format", ["PNG", "JPEG"],
                       horizontal=True, key="compare_fmt")
        rgb_diff = cv2.cvtColor(result["diff_image"], cv2.COLOR_GRAY2RGB)
        ext = "jpg" if fmt == "JPEG" else "png"
        mime = "image/jpeg" if ext == "jpg" else "image/png"
        st.download_button("🖼️ Download difference image",
                           data=runtime.rgb_to_bytes(rgb_diff, fmt, 95),
                           file_name=f"difference.{ext}", mime=mime,
                           key="compare_diff_dl")


def render():
    st.markdown("## 🆚 Compare Images")
    st.caption("Load two samples and quantify how they differ in pixels and "
               "in texture. Pixel, histogram and feature similarity are "
               "reported separately — they are not interchangeable.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Image A")
        image_a, name_a = _uploader("Image A", "a", "compare_upload_a")
    with right:
        st.markdown("#### Image B")
        image_b, name_b = _uploader("Image B", "b", "compare_upload_b")

    with st.expander("⚙️ Feature settings (applied to both images)"):
        config = state.feature_config_widgets("compare")

    if image_a is None or image_b is None:
        components.note("Provide **both** images to run a comparison. Images of "
                        "different sizes are handled: feature differences are "
                        "scale-normalised and pixel metrics use aligned grids.")
        return

    if st.button("🚀 Run comparison", type="primary", key="compare_run"):
        with st.spinner("Computing pixel, histogram and feature similarity…"):
            try:
                st.session_state["compare_result"] = compare_engine.compare_images(
                    image_a, image_b, config)
                st.session_state["compare_names"] = (name_a, name_b)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Comparison failed: {exc}")
                st.session_state["compare_result"] = None

    result = st.session_state.get("compare_result")
    if result:
        saved_names = st.session_state.get("compare_names", (name_a, name_b))
        _results(result, saved_names[0], saved_names[1])
