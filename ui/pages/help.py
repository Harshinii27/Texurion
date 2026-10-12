"""Help Center -- guidance, glossary, limitations and accessibility."""

import streamlit as st

from .. import components, runtime
from ..brand import ASSISTANT_NAME, APP_NAME

FAQ = [
    ("What can Texurion actually measure?",
     "Visible texture statistics: multiscale LBP histograms, GLCM metrics "
     "(contrast, homogeneity, energy, correlation, ASM, dissimilarity) and "
     "histogram statistics (entropy, uniformity, mean, std). Optional Gabor "
     "and opponent-colour features can be added."),
    ("Does it recognise the object in my photo?",
     "No. There is no automatic object-recognition model. You choose the "
     "category; Texurion analyses the texture you provide. It will not claim "
     "an image is an apple or a fabric."),
    ("Can it tell if food is fresh, ripe or safe?",
     "No — not from a photo alone. Those require validated data and models. "
     "Texurion reports texture indicators only."),
    ("Where does the class prediction come from?",
     "Only from a model you trained on your own labelled data (Settings → "
     "Model). With no model loaded, classification is disabled and clearly "
     "labelled as a rule-based descriptor instead."),
    ("Is my image uploaded anywhere?",
     "Analysis runs locally on this server using the existing NumPy/OpenCV "
     "pipeline. No third-party image upload occurs. Do not upload sensitive "
     "images to a shared deployment."),
    ("Do saved analyses sync across devices?",
     "No. They are stored server-locally under `saved_analyses/`. They do not "
     "synchronise."),
    ("Can I connect a live AI chatbot?",
     f"{ASSISTANT_NAME} currently answers from a built-in knowledge base. An "
     "external LLM can be added via a server-side backend; no key is bundled."),
]


def _glossary():
    st.markdown("### Metric glossary")
    st.caption("Click any metric for a plain-language explanation.")
    names = ["Contrast", "Dissimilarity", "Homogeneity", "Energy", "ASM",
             "Correlation", "Entropy", "Uniformity", "Mean", "Std Dev",
             "Roughness"]
    cols = st.columns(3)
    for i, name in enumerate(names):
        with cols[i % 3]:
            components.tooltip_info(
                name, runtime.explain_metric(name)["plain"], f"gloss_{name}")


def render():
    st.markdown("## ❓ Help Center")
    st.caption(f"How {APP_NAME} works, what its numbers mean, and its limits.")

    st.markdown("### Frequently asked questions")
    for question, answer_text in FAQ:
        with st.expander(question):
            st.markdown(answer_text)

    _glossary()

    st.markdown("### Limitations & honesty")
    components.note(
        "Every value shown is computed live from your input. Texurion never "
        "fabricates accuracy, confidence, object recognition, defect detection "
        "or performance figures. Texture indicators are not calibrated "
        "physical measurements unless stated.")

    st.markdown("### Accessibility")
    st.markdown(
        "- All controls are keyboard reachable with visible focus rings.\n"
        "- Icon-only controls carry accessible labels/tooltips.\n"
        "- Errors and states are conveyed by text, not colour alone.\n"
        "- Reduced-motion preferences are respected.\n"
        "- Layouts adapt to desktop, tablet and mobile widths.")

    st.markdown("### About")
    st.markdown(
        f"**{APP_NAME}** — Texture Intelligence Studio. Built on the existing "
        "Local Binary Pattern / GLCM / Gabor pipeline. Assistant: "
        f"**{ASSISTANT_NAME}**.")
    if st.button(f"🧭 Open {ASSISTANT_NAME}", key="help_open_orio"):
        st.session_state["orio_open"] = True
        st.rerun()
