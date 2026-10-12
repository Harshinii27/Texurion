"""Research Mode -- reproducible feature vectors and metric tables."""

import io
import os

import numpy as np
import pandas as pd
import streamlit as st

from .. import components, runtime


def _explain_method():
    with st.expander("Methodology (reproducible settings)"):
        st.markdown(
            "**Pipeline:** image → validate & downscale to *working resolution* "
            "→ grayscale → multiscale LBP histograms (+ optional Gabor / "
            "opponent-colour) → fused feature vector → GLCM + histogram "
            "statistics.\n\n"
            "**GLCM:** 32 grey levels, distance 1, 4 orientations "
            "(0, 45, 90, 135°), symmetric and normalised; contrast, "
            "dissimilarity, homogeneity, energy, correlation and ASM are averaged "
            "over orientations.\n\n"
            "**Histogram statistics:** entropy (base 2), uniformity, mean and "
            "standard deviation of the fused LBP histogram.\n\n"
            "**Reproducibility:** the exact :class:`FeatureConfig` used is shown "
            "below. Identical settings reproduce identical vectors.")


def _render_run(run, name):
    st.markdown(f"**Current session analysis:** `{name}`")
    st.markdown("**Feature configuration**")
    st.json(run.config.to_dict())
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Feature block sizes**")
        st.dataframe(pd.DataFrame(
            [{"block": k, "length": v} for k, v in run.sizes.items()]),
            width="stretch", hide_index=True)
    with cols[1]:
        st.markdown("**Texture metrics**")
        st.dataframe(runtime.supported_metric_table(run), width="stretch",
                     hide_index=True)

    vec = np.asarray(run.vector, dtype="float64")
    components.metric_row([
        ("Vector length", str(vec.size), None),
        ("Min", f"{vec.min():.5f}", None),
        ("Max", f"{vec.max():.5f}", None),
        ("Mean", f"{vec.mean():.5f}", None),
        ("Std", f"{vec.std():.5f}", None),
    ])
    buffer = io.StringIO()
    pd.DataFrame({"index": np.arange(vec.size), "value": vec}).to_csv(
        buffer, index=False)
    st.download_button("Download feature vector (CSV)",
                       data=buffer.getvalue().encode(),
                       file_name=f"{os.path.splitext(name)[0]}_features.csv",
                       mime="text/csv", key="research_vec_csv")
    if run.hist is not None:
        st.pyplot(runtime.histogram_figure(
            run.hist, "Fused multiscale LBP histogram"))


def render():
    st.markdown("## 🔬 Research Mode")
    st.caption("Reproducible features, honest statistics and exportable "
               "numbers for teaching and research. No fabricated CNN "
               "embeddings or invented scientific scores.")
    _explain_method()

    stored = st.session_state.get("analyzer_run")
    if stored and stored["run"].status == "ok":
        _render_run(stored["run"], stored["name"])
    else:
        components.empty_state(
            "🔬", "No active analysis",
            "Open **Analyze Image**, run an analysis, then return here for the "
            "full reproducible metric table and feature-vector export.")

    st.divider()
    st.markdown("**Research capabilities**")
    st.markdown(
        "- Local Binary Patterns (multiscale, uniform/NRI/ror/var)\n"
        "- Gray-Level Co-occurrence Matrix features\n"
        "- Contrast, entropy, energy, correlation, homogeneity, ASM\n"
        "- Optional Gabor filter-bank features\n"
        "- Optional opponent-colour LBP histograms\n"
        "- Feature histograms and full feature vectors\n"
        "- CSV export of vectors and metrics")
    st.caption("Deep-learning embeddings and pretrained CNNs are not bundled; "
               "none are simulated.")
