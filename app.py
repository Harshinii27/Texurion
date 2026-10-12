"""Texurion - Texture Intelligence Studio (Streamlit entrypoint).

Run with: streamlit run app.py
"""

import streamlit as st

from ui import branding, components, lens as lens_ui, orio, theme
from ui.brand import APP_NAME, APP_TAGLINE, ASSISTANT_NAME
from ui.state import active_context, current_model, init_state

st.set_page_config(
    page_title=f"{APP_NAME} | {APP_TAGLINE}",
    page_icon=branding.favicon_path() or ":material/texture:",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
theme.inject_theme()
theme.inject_mode(st.session_state.get("prefs", {}).get("theme_mode", "brand"))

PAGES = {
    "Workspace": [
        st.Page("views/dashboard.py", title="Dashboard",
                icon=":material/dashboard:", default=True),
        st.Page("views/analyze.py", title="Analyze Image",
                icon=":material/image_search:"),
        st.Page("views/compare.py", title="Compare Images",
                icon=":material/compare:"),
        st.Page("views/saved.py", title="Saved Analyses",
                icon=":material/bookmarks:"),
        st.Page("views/reports.py", title="Reports",
                icon=":material/description:"),
    ],
    "Lens & AI": [
        st.Page("views/lens.py", title="Scanning Lens",
                icon=":material/center_focus_strong:"),
        st.Page("views/orio_page.py", title=f"{ASSISTANT_NAME} AI",
                icon=":material/smart_toy:"),
    ],
    "Explore": [
        st.Page("views/workflows.py", title="Industry Workflows",
                icon=":material/precision_manufacturing:"),
        st.Page("views/research.py", title="Research Mode",
                icon=":material/science:"),
    ],
    "Advanced": [
        st.Page("views/batch.py", title="Batch Processing",
                icon=":material/inventory_2:"),
        st.Page("views/benchmark.py", title="Performance",
                icon=":material/speed:"),
    ],
    "Preferences": [
        st.Page("views/settings.py", title="Settings",
                icon=":material/settings:"),
        st.Page("views/help.py", title="Help Center",
                icon=":material/help:"),
    ],
}

page = st.navigation(PAGES, position="sidebar", expanded=True)

with st.sidebar:
    st.markdown(theme.brand_header(), unsafe_allow_html=True)
    model = current_model()
    gallery_size = len(st.session_state.get("gallery", []))
    st.markdown(components.stat_badges(model is not None, gallery_size),
                unsafe_allow_html=True)

lens_ui.render_fab()
orio.render_fab()
page.run()

stored = st.session_state.get("analyzer_run")
last_run = stored["run"] if stored else None
orio.maybe_show_dialog(active_context(), last_run=last_run)
