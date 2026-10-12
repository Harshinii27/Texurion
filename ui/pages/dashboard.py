"""Dashboard -- the directed entry point for Texurion."""

import streamlit as st

from .. import components, store, workflows
from ..brand import APP_NAME


def _nav(path):
    st.switch_page(path)


def _explore(workflow_key):
    workflows.apply_defaults(workflow_key, st.session_state)
    if workflow_key == "compare":
        _nav("views/compare.py")
    else:
        _nav("views/analyze.py")


def _render_categories():
    components.section(
        "Pick a starting point",
        "Each card opens a workflow with settings and reporting tuned for it.")
    rows = [workflows.WORKFLOWS[i:i + 4]
            for i in range(0, len(workflows.WORKFLOWS), 4)]
    for row in rows:
        cols = st.columns(len(row), gap="medium")
        for col, wf in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {wf['icon']} {wf['title']}")
                    st.caption(wf["description"])
                    st.markdown(
                        "\n".join(f"- {uc}" for uc in wf["use_cases"]))
                    if st.button("Explore", key=f"cat_{wf['key']}",
                                 use_container_width=True):
                        _explore(wf["key"])


def _render_recent():
    components.section("Recent activity")
    records = store.list_analyses()[:3]
    if not records:
        components.empty_state(
            "🗂️", "No saved analyses yet",
            "Run an analysis on **Analyze Image**, then press **Save analysis**. "
            "Your saved samples and reports will appear here.")
        return
    cols = st.columns(len(records), gap="medium")
    for col, record in zip(cols, records):
        with col:
            with st.container(border=True):
                st.markdown(f"**{record.get('name', 'sample')}**")
                st.caption(record.get("created_at", "").replace("T", " "))
                st.markdown(
                    components.badge(record.get("descriptor_label", "-"), "info")
                    + " " + components.badge(
                        f"roughness {record.get('roughness', 0):.2f}", "muted"),
                    unsafe_allow_html=True)
    if st.button("Open Saved Analyses", key="dash_saved"):
        _nav("views/saved.py")


def render():
    components.brand_mark(f"{APP_NAME} · Texture Intelligence Studio")
    components.hero(
        "Understand Every Texture.",
        "Analyze surfaces, discover patterns, compare images, and turn visual "
        "details into useful insights.",
        eyebrow=f"{APP_NAME} · Texture Intelligence Studio")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📤 Upload an Image", type="primary",
                     use_container_width=True, key="dash_upload"):
            st.session_state["analyzer_open_camera"] = False
            _nav("views/analyze.py")
    with c2:
        if st.button("📷 Scan with Camera", use_container_width=True,
                     key="dash_camera"):
            st.session_state["analyzer_open_camera"] = True
            _nav("views/analyze.py")
    with c3:
        if st.button("🆚 Compare Images", use_container_width=True,
                     key="dash_compare"):
            _nav("views/compare.py")

    components.note(
        "Texture analysis describes **visible surface patterns**. It does not "
        "measure physical roughness, freshness or health unless a validated "
        "model or calibration is provided. Texurion never fabricates results.")

    st.write("")
    _render_categories()
    st.write("")
    _render_recent()

    with st.expander("What can Texurion actually do?"):
        st.markdown(
            "- **Analyze Image** — multiscale LBP (+ optional Gabor and "
            "opponent-colour) with real GLCM/histogram metrics.\n"
            "- **Compare Images** — pixel similarity, LBP histogram distance "
            "and feature-vector similarity, kept separate.\n"
            "- **Models** — optionally train a real SVM on your own labelled "
            "data (no pretrained model is bundled).\n"
            "- **Reports** — PDF / CSV exports built from real computed values.\n"
            "- **Orio** — a guided assistant grounded in your current results.")
