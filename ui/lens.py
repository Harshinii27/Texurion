"""Texurion Scanning Lens button (separate from Orio).

A single, always-visible floating action button (FAB) pinned to the
bottom-right of every page. It is visually distinct from the Orio launcher
(which sits just below it) and opens the real Scanning Lens workspace at
``views/lens.py``.
"""

import streamlit as st

try:
    from .brand import ACCENT_GRADIENT
except Exception:
    from ui.brand import ACCENT_GRADIENT  # type: ignore


# ``position: fixed`` keeps the button on screen while the page scrolls.
# ``!important`` defends against Streamlit's own layout rules. The high
# ``z-index`` keeps it above page content and dialogs, and it is offset above
# the Orio FAB (bottom: 22px) so the two never overlap.
CSS = """
.st-key-lens_fab {
    position: fixed !important;
    right: 22px !important;
    bottom: 90px !important;
    z-index: 100001 !important;
    width: auto !important;
}
.st-key-lens_fab button {
    border-radius: 999px !important;
    min-height: 52px !important;
    padding: 0.75rem 1.35rem !important;
    font-size: 1.02rem !important;
    line-height: 1.1 !important;
    white-space: nowrap !important;
    background: @ACCENT_GRADIENT@ !important;
    background-image: @ACCENT_GRADIENT@ !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 800 !important;
    letter-spacing: .2px !important;
    box-shadow: 0 12px 30px rgba(166,30,77,.45) !important;
    transition: transform 120ms ease, filter 120ms ease;
}
.st-key-lens_fab button:hover {
    transform: translateY(-2px);
    filter: brightness(1.06);
}
.st-key-lens_fab button:focus-visible {
    outline: 3px solid #ffffff !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 6px rgba(194,24,91,.45) !important;
}
@media (prefers-reduced-motion: reduce) {
    .st-key-lens_fab button { transition: none; }
    .st-key-lens_fab button:hover { transform: none; }
}
@media (max-width: 640px) {
    .st-key-lens_fab { right: 12px !important; bottom: 84px !important; }
    .st-key-lens_fab button {
        min-height: 46px !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.92rem !important;
    }
}
"""


def inject_css():
    css = CSS.replace("@ACCENT_GRADIENT@", ACCENT_GRADIENT)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_fab():
    """Render the fixed Scanning Lens launcher on every page."""

    inject_css()
    with st.container(key="lens_fab"):
        if st.button(
            "🔎 Scanning Lens",
            key="lens_launch_fab",
            help="Scan an Image",
            use_container_width=False,
        ):
            st.switch_page("views/lens.py")
