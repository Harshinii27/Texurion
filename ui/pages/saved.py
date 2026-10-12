"""Saved Analyses -- review, reopen and delete locally stored records."""

import os

import pandas as pd
import streamlit as st

from .. import components, reports, store


def _table(records):
    rows = []
    for r in records:
        rows.append({
            "Sample": r.get("name", "-"),
            "Saved": r.get("created_at", "").replace("T", " "),
            "Workflow": r.get("workflow") or "General",
            "Descriptor": r.get("descriptor_label", "-"),
            "Roughness": round(r.get("roughness", 0.0), 3),
            "ROI px": r.get("roi_pixels", 0),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _detail(record):
    st.markdown(f"### {record.get('name', 'sample')}")
    st.caption(f"Saved {record.get('created_at', '').replace('T', ' ')} · "
               f"ID {record.get('id')}")
    st.markdown(
        components.badge(record.get("descriptor_label", "-"), "info") + " "
        + components.badge(f"workflow: {record.get('workflow') or 'general'}", "muted"),
        unsafe_allow_html=True)

    cols = st.columns([2, 1], gap="large")
    with cols[0]:
        for key, caption in (("original", "Original"),
                             ("lbp", "LBP map"),
                             ("histogram", "LBP histogram")):
            path = store.asset_path(record, key)
            if path:
                st.image(path, caption=caption, width="stretch")
    with cols[1]:
        if record.get("summary"):
            st.markdown("**Interpretation**")
            st.caption(record["summary"])
        st.markdown("**Texture metrics**")
        metrics = record.get("metrics") or {}
        st.dataframe(pd.DataFrame(
            [{"metric": k, "value": round(v, 5)} for k, v in metrics.items()]),
            width="stretch", hide_index=True)
        if record.get("model_prediction"):
            pred = record["model_prediction"]
            st.markdown(f"**Model prediction:** {pred.get('label')} "
                        f"({pred.get('confidence', 0):.1%})")

    st.markdown("**Downloads**")
    d1, d2, d3 = st.columns(3)
    try:
        d1.download_button(
            "📄 PDF report", data=reports.build_analysis_pdf(record),
            file_name=f"{os.path.splitext(record['name'])[0]}_texurion.pdf",
            mime="application/pdf", key=f"pdf_{record['id']}")
    except Exception as exc:  # noqa: BLE001
        d1.error(f"PDF unavailable: {exc}")
    d2.download_button("📊 Metrics CSV", data=store.record_csv_bytes(record),
                       file_name=f"{os.path.splitext(record['name'])[0]}.csv",
                       mime="text/csv", key=f"csv_{record['id']}")
    d3.download_button("🧾 JSON", data=store.record_json_bytes(record),
                       file_name=f"{os.path.splitext(record['name'])[0]}.json",
                       mime="application/json", key=f"json_{record['id']}")

    confirm_key = f"confirm_delete_{record['id']}"
    if st.session_state.get(confirm_key):
        st.warning("Delete this saved analysis permanently?")
        c1, c2 = st.columns(2)
        if c1.button("Yes, delete", type="primary", key=f"yes_{record['id']}"):
            store.delete_analysis(record["id"])
            st.session_state[confirm_key] = False
            st.success("Deleted.")
            st.rerun()
        if c2.button("Cancel", key=f"no_{record['id']}"):
            st.session_state[confirm_key] = False
            st.rerun()
    else:
        if st.button("🗑️ Delete this analysis", key=f"del_{record['id']}"):
            st.session_state[confirm_key] = True
            st.rerun()


def render():
    st.markdown("## 🗂️ Saved Analyses")
    st.caption("Records are stored locally on this server under "
               "`saved_analyses/`. They do not synchronise across devices.")

    records = store.list_analyses()
    if not records:
        components.empty_state(
            "🗂️", "Nothing saved yet",
            "Analyze an image and press **Save analysis**, then it will appear "
            "here for reopening and export.")
        return

    _table(records)
    st.write("")
    ids = [r["id"] for r in records]
    labels = [f"{r.get('name')} · {r.get('created_at', '').replace('T', ' ')}"
              for r in records]
    choice = st.selectbox("Open a saved analysis", range(len(ids)),
                          format_func=lambda i: labels[i], key="saved_choice")
    record = store.load_analysis(ids[choice])
    if record:
        st.divider()
        _detail(record)
