"""Orio -- Texurion's guided assistant.

Orio is a deterministic, context-aware assistant. It answers from a curated
knowledge base and from the *actual* current analysis state. It never claims
to have analysed an image or run a model that did not run.

An optional external LLM can be enabled by a server-side integration; because
none is bundled, Texurion clearly labels Orio as a built-in guided assistant
rather than a live conversational model. See ``ORIO_MODEL`` notes in the
README.
"""

import streamlit as st

from . import runtime, state, workflows
from .brand import ASSISTANT_NAME

QUICK_ACTIONS = [
    ("🍎 Fruit freshness", "fruit: How should I analyse fruit freshness?"),
    ("🥖 Food quality", "food: I want to check food quality."),
    ("🧵 Fabric inspection", "fabric: Help me inspect a fabric sample."),
    ("🌱 Soil texture", "soil: How do I analyse soil texture?"),
    ("🧱 Material surfaces", "material: Analyse a material surface."),
    ("🏭 Industrial defects", "industrial: Screen a part for defects."),
    ("🩺 Medical images", "medical: Research analysis on a medical image."),
    ("📷 Compare two images", "compare: How does image comparison work?"),
    ("🔬 Research mode", "research: Tell me about research mode."),
]

_WORKFLOW_ALIASES = {
    "fruit": "food", "food": "food", "ripeness": "food", "freshness": "food",
    "fabric": "textile", "textile": "textile", "weave": "textile",
    "soil": "agriculture", "agriculture": "agriculture", "crop": "agriculture",
    "material": "materials", "surface": "materials", "brick": "materials",
    "industrial": "industrial", "defect": "industrial",
    "medical": "medical", "biological": "medical",
    "research": "research",
}


def greeting():
    return (
        "👋 Hey there!\n\n"
        f"I'm **{ASSISTANT_NAME}**, your texture-analysis guide.\n\n"
        "Not sure where to start? I can help — pick an option below or just "
        "ask.\n\n**What are you analysing today?**"
    )


def _ctx_line(context):
    bits = []
    if context.get("image_name"):
        bits.append(f"image '{context['image_name']}'")
    if context.get("workflow"):
        wf = workflows.get(context["workflow"])
        if wf:
            bits.append(f"the {wf['title']} workflow")
    if context.get("has_model"):
        bits.append("a trained model")
    return ", ".join(bits)


def _with_followups(text, *followups):
    if not followups:
        return text
    return (text + "\n\nYou could ask: "
            + " · ".join(f"*{f}*" for f in followups) + ".")


def _metric_answer(metric_key, advanced):
    info = runtime.explain_metric(metric_key)
    text = f"**{metric_key}.** {info['plain']}"
    if advanced:
        text += (f"\n\nHow it's calculated: {info['method']}\n\n"
                 f"Limitations: {info['limit']}")
    else:
        text += f" Quick read: {info['reading']}"
    return text


def answer(message, context, last_run=None):
    """Return Orio's reply for a message, using real current state."""

    text = (message or "").strip()
    low = text.lower()
    advanced = any(w in low for w in ("advanced", "technical", "formula",
                                      "derive", "equation", "detail"))

    # Metric explanations.
    metrics = {
        "lbp": "LBP turns each pixel into a light/dark code versus its "
               "neighbours; the histogram of those codes is the texture "
               "signature Texurion compares.",
        "glcm": "The grey-level co-occurrence matrix counts how often pairs of "
                "intensities sit next to each other, giving contrast, "
                "homogeneity, energy and correlation.",
        "gabor": "A bank of oriented Gabor filters measures how strongly the "
                 "texture responds at each orientation and frequency.",
        "entropy": None, "homogeneity": None, "contrast": None,
        "correlation": None, "energy": None, "uniformity": None,
        "roughness": None, "ssim": None,
    }
    for key, explain in metrics.items():
        if key in low:
            if explain:
                return explain
            return _metric_answer(key.capitalize(), advanced)

    if "heatmap" in low or "difference map" in low:
        return ("The difference heatmap shows where two images changed the "
                "most; bright spots are pixel-level differences, not a defect "
                "verdict.")

    # Explain current result.
    if any(k in low for k in ("this result", "explain my", "explain the result",
                              "what does this mean", "my analysis")):
        if last_run is not None and last_run.hist is not None:
            return _with_followups(
                "Here's what your current analysis actually shows.\n\n"
                + runtime.metric_summary(last_run),
                "open Research Mode", "explain contrast")
        return ("I don't see a finished analysis yet. Open **Analyze Image**, "
                "upload a photo and press **Analyze Image** — then I can read "
                "back the real values it produced.")

    # Workflow / domain routing.
    for alias, wf_key in _WORKFLOW_ALIASES.items():
        if alias in low:
            wf = workflows.get(wf_key)
            if wf:
                return _with_followups(
                    f"**{wf['title']}.** {wf['description']} "
                    f"Watch out: {wf['limitations'].rstrip('.')}.",
                    wf["suggestion"], "how comparisons work")

    # Core intents.
    if any(k in low for k in ("upload", "how do i start", "get started",
                              "where do i start", "where to start")):
        return ("Open **Analyze Image**, drop in a JPG/PNG, set the region of "
                "interest if you like, then press **Analyze Image**.")

    if "compare" in low:
        return _with_followups(
            "**Compare Images** shows three separate things: pixel similarity "
            "(SSIM/MAE), LBP histogram distance, and full feature-vector "
            "similarity — related, but not interchangeable.",
            "which similarity should I trust", "explain chi-square")

    if any(k in low for k in ("save", "export", "report", "pdf", "csv",
                              "download")):
        return ("Use **Save analysis** to keep a result locally, then "
                "**Reports** to build a branded PDF or export the metrics as "
                "CSV. Saved records live on this machine only.")

    if any(k in low for k in ("confidence", "accurate", "accuracy",
                              "can you diagnose", "guarantee", "safe",
                              "fresh", "ripe", "is it")):
        return ("A photo alone can't confirm ripeness, freshness, safety or a "
                "diagnosis — Texurion only reports what it measures. "
                "Confidence appears only when a trained model produced it.")

    if any(k in low for k in ("limitation", "caveat", "not work", "cannot",
                              "can't", "why can't")):
        return ("Texurion measures visible texture (LBP, GLCM, optional Gabor "
                "and opponent-colour), not physical roughness, porosity, GSM or "
                "health. Anything beyond the computed values would be made up.")

    if any(k in low for k in ("model", "train", "classify", "svm", "classifier")):
        return ("Training is real but optional: open **Settings → Model**, "
                "upload a labelled ZIP (one folder per class) and Texurion "
                "trains an SVM on your features. Without a model it shows a "
                "rule-based descriptor, clearly labelled as not a classifier.")

    if any(k in low for k in ("roi", "region", "crop")):
        return ("The region of interest limits analysis to part of the image. "
                "Set left/top/right/bottom in working-resolution pixels, or "
                "press **Reset ROI to full image**.")

    if any(k in low for k in ("hi", "hello", "hey", "help", "what can you do")):
        ctxt = _ctx_line(context)
        tail = (f"\n\nI can see you're working with {ctxt}." if ctxt else "")
        return (greeting() + tail)

    # Fallback.
    return _with_followups(
        "I can explain any Texurion metric, help you pick an analysis, or read "
        "back your current results.",
        "what can you do", "explain roughness", "how do I start")


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------

def _ensure_greeting():
    if not st.session_state.get("orio_messages"):
        st.session_state["orio_messages"] = [
            {"role": "assistant", "content": greeting()}
        ]


def _push(role, content):
    st.session_state["orio_messages"].append({"role": role, "content": content})


def _render_messages():
    for msg in st.session_state.get("orio_messages", []):
        with st.chat_message(msg["role"],
                             avatar="🧭" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])


def _handle_user_message(message, context, last_run):
    _push("user", message)
    reply = answer(message, context, last_run=last_run)
    _push("assistant", reply)
    return reply


def _quick_action_buttons(context, last_run, key_prefix):
    _ensure_greeting()
    st.caption("Quick actions")
    for row in (QUICK_ACTIONS[:3], QUICK_ACTIONS[3:6], QUICK_ACTIONS[6:]):
        cols = st.columns(len(row))
        for col, (label, prompt) in zip(cols, row):
            if col.button(label, key=f"{key_prefix}_{label}", use_container_width=True):
                _handle_user_message(prompt, context, last_run)
                st.rerun()


def render_chat(context, last_run=None, key_prefix="orio"):
    """Full chat surface, usable on a page or inside a dialog."""

    _ensure_greeting()
    _render_messages()
    _quick_action_buttons(context, last_run, key_prefix)

    with st.form(key=f"{key_prefix}_form", clear_on_submit=True):
        typed = st.text_input("Message Orio", key=f"{key_prefix}_input",
                              placeholder="Ask about a metric, a category, or "
                                          "say 'explain this result'")
        submitted = st.form_submit_button("Send", type="primary")
    if submitted and typed.strip():
        _handle_user_message(typed.strip(), context, last_run)
        st.rerun()


@st.dialog(f"Chat with {ASSISTANT_NAME}", width="large", dismissible=False)
def _orio_dialog(context, last_run):
    top = st.columns([4, 1])
    with top[1]:
        if st.button("Close", key="orio_close", use_container_width=True):
            st.session_state["orio_open"] = False
            st.rerun()
    render_chat(context, last_run=last_run, key_prefix="orio_dialog")


def render_fab():
    """Fixed launcher button shown on every page."""

    with st.container(key="orio_fab"):
        if st.button(f"💬 Ask {ASSISTANT_NAME}", key="orio_launch"):
            st.session_state["orio_open"] = True
            st.rerun()


def maybe_show_dialog(context, last_run=None):
    if st.session_state.get("orio_open"):
        _orio_dialog(context, last_run)


def render_page(context, last_run=None):
    st.markdown(f"## 🧭 {ASSISTANT_NAME} — conversational intelligence")
    st.caption(
        "Orio is a dedicated conversational assistant with its own history and "
        "suggested questions. It explains concepts and interprets actual "
        "analysis outputs, but it never claims to have analysed an image "
        "unless that analysis actually ran.")
    render_chat(context, last_run=last_run, key_prefix="orio_page")
