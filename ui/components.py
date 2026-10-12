"""Reusable presentation components for Texurion."""

import streamlit as st

from . import branding, runtime


def page_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def brand_mark(caption=None):
    """The pixel-T mark used in welcome / loading states."""

    mark = branding.icon_svg()
    if not mark:
        return
    caption_html = f"<span>{caption}</span>" if caption else ""
    st.markdown(
        f'<div class="tx-brand-mark">{mark}{caption_html}</div>',
        unsafe_allow_html=True,
    )


def section(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def badge(text, kind="muted"):
    return f'<span class="tx-badge tx-badge-{kind}">{text}</span>'


def card(icon, title, body, badges=None, meta=None):
    """Return HTML for a category/result card."""

    badge_html = "".join(badges or [])
    meta_html = f'<p style="opacity:.75">{meta}</p>' if meta else ""
    return (
        '<div class="tx-card">'
        f'<div class="tx-icon">{icon}</div>'
        f"<h4>{title}</h4>"
        f"<p>{body}</p>"
        f"{meta_html}{badge_html}"
        "</div>"
    )


def note(text):
    st.markdown(f'<div class="tx-note">{text}</div>', unsafe_allow_html=True)


def hero(title, subtitle, eyebrow=None):
    eyebrow_html = f'<span class="tx-eyebrow">{eyebrow}</span>' if eyebrow else ""
    st.markdown(
        '<div class="tx-hero">'
        f"{eyebrow_html}<h1>{title}</h1>"
        f'<div class="tx-hero-sub">{subtitle}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def metric_row(items):
    """Render a row of metrics. ``items`` = list of (label, value, help)."""

    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        help_text = item[2] if len(item) > 2 else None
        col.metric(label, value, help=help_text)


@st.dialog("Metric explained", width="large")
def _metric_dialog(name, value):
    info = runtime.explain_metric(name)
    st.markdown(f"**{name}** &nbsp; `{value}`")
    st.markdown(f"**What it means.** {info['plain']}")
    st.markdown(f"**How to read it.** {info['reading']}")
    st.markdown(f"**How it is calculated.** {info['method']}")
    st.markdown(f"**Limitations.** {info['limit']}")
    st.caption("Ask Orio for a beginner or advanced walkthrough of any metric.")


def explain_button(name, value, key):
    """A compact 'Explain this result' control beside a metric."""

    if st.button("Explain", key=f"explain_{key}", type="tertiary",
                 help=f"Explain {name}"):
        _metric_dialog(name, value)


def metric_with_explain(label, name, value, key, help_text=None):
    """Metric-like display with an inline Explain button."""

    st.metric(label, value, help=help_text)
    explain_button(name, value, key)


def tooltip_info(label, body, key):
    """A popover that documents an unfamiliar control in plain language."""

    with st.popover(label, icon=":material/help:", key=f"info_{key}"):
        st.markdown(body)


def empty_state(icon, title, body):
    st.markdown(
        f'<div class="tx-card" style="text-align:center">'
        f'<div class="tx-icon" style="margin:0 auto .5rem auto">{icon}</div>'
        f"<h4>{title}</h4><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def stat_badges(has_model, gallery_size):
    parts = []
    parts.append(badge("Trained model active", "ok") if has_model
                 else badge("No trained model", "warn"))
    parts.append(badge(f"Gallery: {gallery_size}", "info")
                 if gallery_size else badge("Gallery empty", "muted"))
    return " ".join(parts)
