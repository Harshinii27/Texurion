"""Batch Processing -- process many images/ZIPs with per-file status."""

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from core.batch import collect_batch_inputs, process_batch, write_results_csv
from core.histogram import histogram_statistics

from .. import state


def render():
    st.markdown("## 🧺 Batch Processing")
    st.caption("Process many images and ZIP archives with per-file status, "
               "timing and CSV export. ZIP members are validated against path "
               "traversal and read in memory.")

    files = st.file_uploader(
        "Upload images and/or ZIP archives",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "zip"],
        accept_multiple_files=True, key="batch_files")

    uploaded = [(f.name, f.getvalue()) for f in (files or [])]
    with st.expander("⚙️ Feature settings"):
        config = state.feature_config_widgets("batch")

    if not uploaded:
        st.info("Upload files to begin.")
        return

    items, setup_errors = collect_batch_inputs(uploaded)
    if setup_errors:
        with st.expander(f"{len(setup_errors)} archive error(s)"):
            for fname, msg in setup_errors[:100]:
                st.write(f"- **{fname}**: {msg}")

    model = state.current_model()
    model_ok = model is not None and model["feature_length"] == config.descriptor_length()
    if model is not None and not model_ok:
        st.warning("Loaded model is incompatible with these settings; batch "
                   "runs without classification.")

    workers = st.slider("Worker threads", 1, 8, 4, key="batch_workers")

    if st.button("Process batch", type="primary", key="batch_run"):
        if not items:
            st.warning("No supported images found in the upload.")
            return

        def worker(name, data):
            from core.io import decode_image
            from core.pipeline import extract_features

            image = decode_image(data, name=name)
            extracted = extract_features(image, config)
            vec = np.asarray(extracted["vector"], dtype="float64")
            total = float(vec.sum())
            entry = {
                "label": "", "confidence": "",
                "fused_entropy": float(histogram_statistics(
                    vec / total if total > 0 else vec)["entropy"]),
            }
            if model_ok:
                label, confidence, _ = state.predict_class(model, vec)
                entry["label"] = label
                entry["confidence"] = f"{confidence:.4f}"
            return entry

        results = process_batch(items, worker, workers=workers)
        ok = sum(1 for r in results if r["status"] == "ok")
        st.success(f"Processed {ok}/{len(results)} files successfully.")

        rows = []
        for r in results:
            result = r.get("result") or {}
            rows.append([
                r["name"], r["status"],
                None if r["duration_ms"] is None else round(r["duration_ms"], 2),
                r["error"] or "", result.get("label", ""),
                result.get("confidence", ""), result.get("fused_entropy", ""),
            ])
        df = pd.DataFrame(rows, columns=[
            "file", "status", "duration_ms", "error", "predicted_class",
            "confidence", "fused_entropy"])
        st.dataframe(df, width="stretch", hide_index=True)

        with tempfile.TemporaryDirectory(prefix="tex_batch_") as tmp:
            csv_path = write_results_csv(results, os.path.join(tmp, "batch.csv"))
            with open(csv_path, "rb") as handle:
                csv_bytes = handle.read()
        st.download_button("Download results CSV", data=csv_bytes,
                           file_name="batch_results.csv", mime="text/csv",
                           key="batch_dl")
