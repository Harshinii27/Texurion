"""Full-page Orio assistant."""

import streamlit as st

from .. import orio
from ..state import active_context


def render():
    stored = st.session_state.get("analyzer_run")
    last_run = stored["run"] if stored else None
    orio.render_page(active_context(), last_run=last_run)
