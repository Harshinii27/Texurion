"""Theme injection for Texurion.

The CSS below keeps the application's ORIGINAL pink main background and
emerald sidebar gradient untouched (imported from :mod:`ui.brand`) and styles
the surrounding chrome -- cards, buttons, tabs, inputs, the Orio launcher --
to complement them. No design token here overrides those two gradients.
"""

import streamlit as st

from . import branding
from .brand import (
    ACCENT_GRADIENT,
    APP_NAME,
    APP_TAGLINE,
    MAIN_BACKGROUND,
    SIDEBAR_BACKGROUND,
    TOKENS,
)

_CSS = """
@keyframes tx-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont,
                 "Helvetica Neue", Arial, sans-serif;
}

/* ---- Original main background (unchanged) ---- */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: @MAIN_BG@;
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"],
#MainMenu, footer {
    display: none !important;
    visibility: hidden !important;
}

/* ---- Main content typography ---- */
[data-testid="stAppViewContainer"] { color: @INK@; }
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label { color: @INK@; }
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] { color: @INK_SOFT@ !important; }
[data-testid="stAppViewContainer"] a { color: @ACCENT_INK@; }
[data-testid="stAppViewContainer"] code {
    background: rgba(194, 24, 91, 0.08);
    color: @ACCENT_INK@;
    border-radius: 4px;
    padding: 0.05rem 0.3rem;
}

/* ---- Typography scale (readability across audiences) ---- */
[data-testid="stAppViewContainer"] { font-size: 16px; }
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label {
    font-size: 1rem;
    line-height: 1.55;
}
[data-testid="stAppViewContainer"] h1 {
    font-size: 2.1rem; line-height: 1.15; font-weight: 800;
}
[data-testid="stAppViewContainer"] h2 {
    font-size: 1.5rem; line-height: 1.2; font-weight: 750;
}
[data-testid="stAppViewContainer"] h3 { font-size: 1.28rem; font-weight: 700; }
[data-testid="stAppViewContainer"] h4 { font-size: 1.1rem; font-weight: 700; }
[data-testid="stCaptionContainer"] p { font-size: 0.86rem; line-height: 1.45; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 750; }
.stApp [data-testid="stBaseButton-primary"],
.stApp [data-testid="stBaseButton-secondary"],
.stApp [data-testid="stBaseButton-tertiary"] { font-size: 1rem; }

/* ---- Original emerald sidebar gradient (unchanged) ---- */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    background: @SIDEBAR_BG@ !important;
}
[data-testid="stSidebar"] { border-right: 1px solid rgba(16, 46, 36, 0.12); }
[data-testid="stSidebar"] * {
    color: @SIDEBAR_INK@ !important;
    text-shadow: none !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(16, 46, 36, 0.14); }
[data-testid="stSidebar"] a { color: @SIDEBAR_ACCENT@ !important; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    border-radius: 10px;
    padding: 0.4rem 0.6rem;
    margin: 0.1rem 0;
    font-size: 0.98rem;
    font-weight: 600;
    line-height: 1.3;
}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
    background: rgba(255, 255, 255, 0.16) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: rgba(255, 255, 255, 0.22) !important;
    border-left: 3px solid @SIDEBAR_INK@;
    font-weight: 750 !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
    margin-top: 0.35rem;
}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul { gap: 0.12rem; }
[data-testid="stSidebar"] code {
    background: rgba(14, 124, 102, 0.12) !important;
    color: @SIDEBAR_INK@ !important;
}

/* ---- Surfaces (cards / panels) ---- */
[data-testid="stMetric"],
[data-testid="stExpander"] details,
[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stForm"] {
    background-color: @SURFACE@;
    border: 1px solid @BORDER@;
    border-radius: 14px;
    backdrop-filter: blur(6px);
}
[data-testid="stMetric"] { padding: 0.7rem 0.9rem; }
[data-testid="stMetricValue"] { color: @INK@; font-weight: 700; }
[data-testid="stMetricLabel"] p { color: @INK_SOFT@ !important; }
[data-testid="stExpander"] details { overflow: hidden; }
[data-testid="stExpander"] summary { font-weight: 600; }

/* ---- Buttons ---- */
.stApp [data-testid="stBaseButton-primary"],
.stApp button[kind="primary"] {
    background: @ACCENT_GRADIENT@;
    color: #ffffff !important;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    box-shadow: 0 6px 16px rgba(166, 30, 77, 0.28);
    transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
}
.stApp [data-testid="stBaseButton-primary"]:hover,
.stApp button[kind="primary"]:hover {
    transform: translateY(-1px);
    filter: brightness(1.05);
    box-shadow: 0 9px 22px rgba(166, 30, 77, 0.34);
}
.stApp [data-testid="stBaseButton-secondary"],
.stApp button[kind="secondary"] {
    background: @SURFACE@;
    color: @ACCENT_INK@ !important;
    border: 1px solid rgba(194, 24, 91, 0.35);
    border-radius: 12px;
    font-weight: 600;
    transition: background 120ms ease, border-color 120ms ease;
}
.stApp [data-testid="stBaseButton-secondary"]:hover,
.stApp button[kind="secondary"]:hover {
    background: #FDECF3;
    border-color: @ACCENT@;
}
.stApp [data-testid="stBaseButton-tertiary"],
.stApp button[kind="tertiary"] {
    border-radius: 10px;
    font-weight: 600;
    color: @ACCENT_INK@ !important;
}
.stApp button:focus-visible,
.stApp a:focus-visible,
.stApp input:focus-visible,
.stApp textarea:focus-visible {
    outline: 2px solid @FOCUS@ !important;
    outline-offset: 2px;
}

/* ---- Inputs ---- */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: @SURFACE_SOLID@ !important;
    border-color: @BORDER_STRONG@ !important;
    color: @INK@ !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: @SURFACE@ !important;
    border: 1.5px dashed rgba(194, 24, 91, 0.45) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: @INK@ !important; }

/* ---- Tabs + segmented controls ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    background: @SURFACE@;
    border: 1px solid @BORDER@;
    border-radius: 12px;
    padding: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 0.35rem 0.85rem;
    color: @INK_SOFT@;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: @ACCENT_GRADIENT@ !important;
    color: #ffffff !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { background: transparent !important; }
[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: @ACCENT_GRADIENT@ !important;
    color: #ffffff !important;
}

/* ---- Alerts: keep them readable over the gradient ---- */
[data-testid="stAlert"] { border-radius: 12px; }

/* ---- Orio launcher (fixed FAB, secondary to Scanning Lens) ---- */
.st-key-orio_fab {
    position: fixed !important;
    right: 22px !important;
    bottom: 22px !important;
    z-index: 100000 !important;
    width: auto !important;
}
.st-key-orio_fab button {
    border-radius: 999px !important;
    min-height: 44px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.95rem !important;
    background: @ACCENT_GRADIENT@ !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 22px rgba(166, 30, 77, 0.36) !important;
    transition: transform 120ms ease, filter 120ms ease;
}
.st-key-orio_fab button:hover {
    filter: brightness(1.06);
    transform: translateY(-1px);
}
.st-key-orio_fab button:focus-visible {
    outline: 3px solid #ffffff !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 6px rgba(194, 24, 91, 0.45) !important;
}

/* Reserve room at the bottom so the floating shortcuts never cover
   page actions, charts, previews, upload areas or downloads. */
[data-testid="stMainBlockContainer"],
.block-container {
    padding-bottom: 140px !important;
}

/* Contextual toolbar row */
.tx-toolbar { display: flex; gap: 0.5rem; flex-wrap: wrap; }

/* ---- Brand header ---- */
.tx-brand {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.1rem 0 0.4rem 0;
}
.tx-brand .tx-logo {
    flex: 0 0 auto;
    width: 40px; height: 40px;
    display: grid; place-items: center;
}
.tx-brand .tx-logo svg { width: 100%; height: 100%; display: block; }
.tx-brand .tx-name { font-size: 1.25rem; font-weight: 800; letter-spacing: .2px; }
.tx-brand .tx-sub { font-size: .8rem; opacity: .85; }

/* ---- Welcome / loading brand mark ---- */
.tx-brand-mark { display: flex; align-items: center; gap: .75rem; margin-bottom: .35rem; }
.tx-brand-mark svg { width: 46px; height: 46px; display: block; }

/* ---- Hero ---- */
.tx-hero {
    background: @SURFACE@;
    border: 1px solid @BORDER@;
    border-radius: 20px;
    padding: 1.6rem 1.7rem;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(7px);
    animation: tx-fade-in 260ms ease both;
}
.tx-hero h1 {
    font-size: 2.15rem; line-height: 1.12; margin: 0 0 .4rem 0;
    color: @INK@; font-weight: 800;
}
.tx-hero .tx-hero-sub {
    font-size: 1.06rem; color: @INK_SOFT@; max-width: 60ch; margin-bottom: .2rem;
}
.tx-eyebrow {
    display: inline-block; font-size: .72rem; font-weight: 700;
    letter-spacing: .12em; text-transform: uppercase;
    color: @ACCENT_INK@; margin-bottom: .5rem;
}

/* ---- Cards ---- */
.tx-card {
    background: @SURFACE@;
    border: 1px solid @BORDER@;
    border-radius: 16px;
    padding: 1rem 1.05rem;
    height: 100%;
    backdrop-filter: blur(6px);
    animation: tx-fade-in 240ms ease both;
}
.tx-card h4 { margin: .1rem 0 .35rem 0; font-size: 1.12rem; color: @INK@; }
.tx-card p { margin: .15rem 0; color: @INK_SOFT@; font-size: .95rem; line-height: 1.5; }
.tx-card .tx-icon {
    font-size: 1.4rem; width: 42px; height: 42px; border-radius: 12px;
    display: grid; place-items: center; margin-bottom: .5rem;
    background: rgba(194, 24, 91, 0.10);
}
.tx-badge {
    display: inline-block; font-size: .72rem; font-weight: 700;
    padding: .14rem .55rem; border-radius: 999px; margin-right: .25rem;
}
.tx-badge-ok { background: rgba(27,127,90,.14); color: @SUCCESS@; }
.tx-badge-warn { background: rgba(178,106,0,.16); color: @WARNING@; }
.tx-badge-info { background: rgba(42,109,176,.14); color: @INFO@; }
.tx-badge-muted { background: rgba(16,24,32,.08); color: @INK_SOFT@; }
.tx-note {
    border-left: 3px solid @ACCENT@;
    background: rgba(194, 24, 91, 0.06);
    padding: .55rem .8rem; border-radius: 0 10px 10px 0;
    font-size: .88rem; color: @INK_SOFT@;
}

@media (prefers-reduced-motion: reduce) {
    .tx-hero, .tx-card { animation: none; }
    .stApp [data-testid="stBaseButton-primary"]:hover { transform: none; }
}

@media (max-width: 640px) {
    .tx-hero { padding: 1.1rem 1.15rem; }
    .tx-hero h1 { font-size: 1.6rem; }
    .st-key-orio_fab { right: 14px; bottom: 14px; }
}
"""


def _render_css() -> str:
    replacements = {
        "@MAIN_BG@": MAIN_BACKGROUND,
        "@SIDEBAR_BG@": SIDEBAR_BACKGROUND,
        "@ACCENT_GRADIENT@": ACCENT_GRADIENT,
        "@INK@": TOKENS["ink"],
        "@INK_SOFT@": TOKENS["ink_soft"],
        "@SURFACE@": TOKENS["surface"],
        "@SURFACE_SOLID@": TOKENS["surface_solid"],
        "@BORDER@": TOKENS["border"],
        "@BORDER_STRONG@": TOKENS["border_strong"],
        "@ACCENT@": TOKENS["accent"],
        "@ACCENT_INK@": TOKENS["accent_ink"],
        "@FOCUS@": TOKENS["focus"],
        "@SIDEBAR_INK@": TOKENS["sidebar_ink"],
        "@SIDEBAR_ACCENT@": TOKENS["sidebar_accent"],
        "@SUCCESS@": TOKENS["success"],
        "@WARNING@": TOKENS["warning"],
    }
    css = _CSS
    for key, value in replacements.items():
        css = css.replace(key, value)
    return css


def inject_theme() -> None:
    """Inject the Texurion stylesheet into the running Streamlit app."""

    st.markdown(f"<style>{_render_css()}</style>", unsafe_allow_html=True)


def inject_mode(mode: str) -> None:
    """Apply an optional low-light scrim.

    The default ``brand`` mode leaves the original pink/emerald backgrounds
    exactly as they are. ``dim`` keeps the same gradient hue but places a
    translucent scrim over it for low-light use -- the gradient is never
    replaced.
    """

    if mode != "dim":
        return

    scrim = """
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background:
            linear-gradient(135deg, rgba(28,14,22,0.72), rgba(40,16,30,0.72)),
            @MAIN_BG@ !important;
        background-attachment: fixed !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        background:
            linear-gradient(rgba(6,24,18,0.78), rgba(6,24,18,0.78)),
            @SIDEBAR_BG@ !important;
    }
    [data-testid="stSidebar"] * { color: #EAF6F0 !important; }
    [data-testid="stSidebar"] a { color: #9BE3C8 !important; }
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4 { color: #F6EEF3 !important; }
    [data-testid="stCaptionContainer"] p { color: #D9C4D0 !important; }
    [data-testid="stMetric"],
    [data-testid="stExpander"] details,
    [data-testid="stDataFrame"],
    [data-testid="stForm"],
    .tx-card, .tx-hero, .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(30, 18, 26, 0.78) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    """
    scrim = scrim.replace("@MAIN_BG@", MAIN_BACKGROUND).replace(
        "@SIDEBAR_BG@", SIDEBAR_BACKGROUND)
    st.markdown(f"<style>{scrim}</style>", unsafe_allow_html=True)


def brand_header(subtitle: str = APP_TAGLINE) -> str:
    """Return the HTML for the small brand lockup used in the sidebar."""

    mark = branding.icon_svg()
    logo = f'<div class="tx-logo">{mark}</div>' if mark else ""
    return (
        '<div class="tx-brand">'
        f'{logo}'
        f'<div><div class="tx-name">{APP_NAME}</div>'
        f'<div class="tx-sub">{subtitle}</div></div>'
        "</div>"
    )
