"""Reports -- build PDF and CSV reports from saved or current analyses."""

import os

import pandas as pd
import streamlit as st

from .. import components, reports, store


def render():
    st.markdown("## 📄 Reports")
    st.caption("Create branded PDF reports using only real, computed values. "
               "Reports are built from saved analyses.")

    records = store.list_analyses()
    if not records:
        components.empty_state(
            "📄", "No reports to generate",
            "Run and save an analysis first (Analyze Image → Save analysis).")
        return

    labels = [f"{r.get('name')} · {r.get('created_at', '').replace('T', ' ')}"
              for r in records]
    idx = st.selectbox("Select a saved analysis", range(len(records)),
                       format_func=lambda i: labels[i], key="report_choice")
    record = records[idx]

    with st.container(border=True):
        st.markdown(f"**{record.get('name')}**")
        st.caption(f"{record.get('descriptor_label')} · roughness "
                   f"{record.get('roughness', 0):.3f}")
        cols = st.columns(3)
        try:
            cols[0].download_button(
                "Download PDF report", data=reports.build_analysis_pdf(record),
                file_name=f"{os.path.splitext(record['name'])[0]}_texurion.pdf",
                mime="application/pdf", key="report_pdf")
        except Exception as exc:  # noqa: BLE001
            cols[0].error(f"PDF unavailable: {exc}")
        cols[1].download_button(
            "Download metrics CSV", data=store.record_csv_bytes(record),
            file_name=f"{os.path.splitext(record['name'])[0]}_metrics.csv",
            mime="text/csv", key="report_csv")
        cols[2].download_button(
            "Download JSON snapshot", data=store.record_json_bytes(record),
            file_name=f"{os.path.splitext(record['name'])[0]}_snapshot.json",
            mime="application/json", key="report_json")

    st.divider()
    st.markdown("**Report contents**")
    st.markdown(
        "- Application name (Texurion) and timestamp\n"
        "- Sample name, workflow, descriptor label, roughness\n"
        "- ROI pixels and feature vector length\n"
        "- Analysis settings (LBP, blocks, working resolution)\n"
        "- Visual thumbnails (original, LBP map) and histogram\n"
        "- All computed texture metrics (table)\n"
        "- Plain-language interpretation and methodology / limitations\n"
        "- Model prediction only if a trained model was loaded")
    st.caption("Nothing is invented. Only computed values from the saved record "
               "are included.")
