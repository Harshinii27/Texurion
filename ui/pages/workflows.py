"""Workflows -- explore each domain template in full."""

import streamlit as st

from .. import components, workflows


def _workflow_card(wf):
    with st.container(border=True):
        st.markdown(f"### {wf['icon']} {wf['title']}")
        st.caption(wf["description"])
        st.markdown("**Typical use cases**")
        st.markdown("\n".join(f"- {uc}" for uc in wf["use_cases"]))
        st.markdown("**Texurion can compute**")
        st.markdown("\n".join(f"- {a}" for a in wf["analyses"]))
        st.markdown("**Limitations (honest)**")
        st.caption(wf["limitations"])
        st.markdown(f"**Suggested approach:** {wf['suggestion']}")
        if st.button("Start this workflow", key=f"wf_{wf['key']}"):
            workflows.apply_defaults(wf["key"], st.session_state)
            if wf["key"] == "compare":
                st.switch_page("views/compare.py")
            else:
                st.switch_page("views/analyze.py")


def render():
    st.markdown("## 🧩 Industry Workflows")
    st.caption("Pick a template: Texurion tunes settings and highlights "
               "capabilities and limitations for each domain. It does not "
               "download pretrained specialised models.")

    for wf in workflows.WORKFLOWS:
        _workflow_card(wf)
        st.write("")
    components.note(
        "Medical workflows are for research/support only — not diagnostic. "
        "For food, ripeness/safety need validated data; Texurion gives "
        "texture indicators only.")
