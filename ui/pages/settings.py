"""Settings -- preferences, model manager, storage and assistant info."""

import os
import tempfile

import streamlit as st

from classifier import extract_labeled_dataset, save_model, train_model

from .. import components, state, store
from ..brand import ASSISTANT_NAME, APP_NAME
from ..state import ENGINE_OPTIONS, MODEL_SECURITY_WARNING

LABEL_ZIP_HINT = (
    "ZIP must contain one folder per class; every image inside the folder "
    "belongs to that class. Example: textures.zip/tiles/01.jpg bricks/02.jpg")


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

def _preferences():
    st.markdown("### Preferences")
    prefs = st.session_state.setdefault("prefs", {})
    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio(
            "Theme", ["Brand pink (default)", "Dim (low-light)"],
            index=0 if prefs.get("theme_mode", "brand") == "brand" else 1,
            key="pref_theme_mode",
            help="The original pink gradient is preserved in both modes; Dim "
                 "adds a translucent scrim for low-light use.")
        prefs["theme_mode"] = "brand" if mode.startswith("Brand") else "dim"
    with c2:
        st.toggle("Show contextual hints", value=prefs.get("show_hints", True),
                  key="pref_show_hints",
                  help="Inline explanations for unfamiliar controls.")

    st.markdown("**Default analysis settings** (applied to new sessions)")
    c3, c4, c5 = st.columns(3)
    with c3:
        method = st.selectbox(
            "Default LBP method",
            ["uniform", "nri_uniform", "ror", "var", "default"],
            index=["uniform", "nri_uniform", "ror", "var", "default"].index(
                prefs.get("default_method", "uniform")),
            key="pref_method")
    with c4:
        engine_labels = list(ENGINE_OPTIONS)
        current_engine = next(
            (l for l, v in ENGINE_OPTIONS.items()
             if v == prefs.get("default_engine", "skimage")), engine_labels[0])
        engine = st.selectbox("Default LBP engine", engine_labels,
                              index=engine_labels.index(current_engine),
                              key="pref_engine")
    with c5:
        max_side = st.number_input(
            "Default working resolution", 128, 2048,
            int(prefs.get("default_max_side", 512)), 32, key="pref_max_side")

    c6, c7 = st.columns(2)
    gabor = c6.checkbox("Enable Gabor by default",
                        value=prefs.get("include_gabor", False), key="pref_gabor")
    opponent = c7.checkbox("Enable opponent-colour by default",
                           value=prefs.get("include_opponent", False),
                           key="pref_opponent")

    if st.button("Save preferences", type="primary", key="pref_save"):
        prefs.update({
            "default_method": method,
            "default_engine": ENGINE_OPTIONS[engine],
            "default_max_side": int(max_side),
            "include_gabor": gabor,
            "include_opponent": opponent,
        })
        for widget_key in ("analyzer_method", "analyzer_scales",
                           "analyzer_engine", "analyzer_gabor",
                           "analyzer_opponent", "analyzer_max_side"):
            st.session_state.pop(widget_key, None)
        st.success("Preferences saved.")


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------

def _import_labeled_zip(uploaded):
    import zipfile

    with zipfile.ZipFile(uploaded) as archive:
        for info in archive.infolist():
            norm = info.filename.replace("\\", "/").lstrip("/")
            if info.is_dir() or norm.startswith(("..", "/")) or ":/" in norm:
                continue
            parts = norm.split("/")
            if len(parts) < 2 or not parts[0]:
                continue
            label, rel = parts[0], "/".join(parts[1:])
            if not rel.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            yield label, os.path.basename(rel), archive.read(info)


def _model_manager():
    st.markdown("### Model (optional trained classifier)")
    model = state.current_model()
    if model is not None:
        st.success("A trained model is loaded; classification is active.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Classes", ", ".join(model["classes"]))
        m2.metric("Feature length", model["feature_length"])
        m3.metric("Test accuracy (measured)",
                  f"{model['metrics']['test_accuracy']:.1%}")
        st.caption(f"Trained {model['trained_at']} on {model['n_samples']} "
                   f"samples ({model['metrics']['test_samples']} held-out). "
                   f"Type: {model['model_type']}.")
        with st.expander("Per-class classification report"):
            st.text(model.get("classification_report", ""))
        if st.button("Unload model", key="model_unload"):
            st.session_state["model"] = None
            st.rerun()

    st.caption(MODEL_SECURITY_WARNING)

    if os.path.isfile("texture_model.pkl"):
        if st.button("Load bundled sample model (texture_model.pkl)",
                     key="model_bundled"):
            with open("texture_model.pkl", "rb") as handle:
                state.set_model_from_bytes(handle.read())
            st.rerun()

    dtd_model = os.path.join("models", "texurion_model_dtd.pkl")
    if os.path.isfile(dtd_model):
        st.caption("Bundled DTD model: a 47-class texture classifier trained on "
                   "the Describable Textures Dataset using LBP features.")
        if st.button("Load bundled DTD model (47 classes)",
                     key="model_bundled_dtd"):
            with open(dtd_model, "rb") as handle:
                state.set_model_from_bytes(handle.read())
            st.rerun()

    loaded = st.file_uploader("Upload a model file (.pkl / .joblib)",
                              type=["pkl", "joblib"], key="model_upload")
    if loaded is not None and st.button("Load uploaded model",
                                        key="model_load_btn"):
        if state.set_model_from_bytes(loaded.getvalue()):
            st.rerun()

    st.divider()
    st.markdown("#### Train a new model on your labelled data")
    st.caption(LABEL_ZIP_HINT)
    zip_upload = st.file_uploader("Upload labelled ZIP", type=["zip"],
                                  key="train_zip")
    if zip_upload is None:
        return

    config = state.feature_config_widgets("train", default_max_side=512)

    if st.button("Train classifier", type="primary", key="train_run"):
        files = list(_import_labeled_zip(zip_upload))
        if not files:
            st.error("No labelled images found. Expected "
                     "class-folder/image.jpg layout.")
            return
        st.info(f"Found {len(files)} labelled images. Extracting features…")
        X, y, names, errors = extract_labeled_dataset(files, config)
        if errors:
            with st.expander(f"{len(errors)} file(s) skipped"):
                for fname, exc in errors[:50]:
                    st.write(f"- **{fname}**: {exc}")
        try:
            bundle = train_model(X, y, config)
        except ValueError as exc:
            st.error(f"Could not train: {exc}")
            return

        st.session_state["gallery"] = [
            {"label": lbl, "name": nm, "vector": vec}
            for lbl, nm, vec in zip(y, names, X)]
        st.session_state["gallery_config"] = config.to_dict()
        st.session_state["model"] = bundle

        st.success("Model trained on real labelled data.")
        s1, s2, s3 = st.columns(3)
        s1.metric("Train accuracy", f"{bundle['metrics']['train_accuracy']:.1%}")
        s2.metric("Test accuracy (held-out)",
                  f"{bundle['metrics']['test_accuracy']:.1%}")
        s3.metric("Test samples", bundle["metrics"]["test_samples"])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            model_path, _ = save_model(bundle, tmp.name)
        with open(model_path, "rb") as handle:
            st.download_button("Download trained model (.pkl)",
                               data=handle.read(), file_name="texture_model.pkl",
                               mime="application/octet-stream", key="model_dl")
        try:
            os.remove(model_path)
        except OSError:
            pass
        st.caption("The reference gallery is rebuilt from the same labelled "
                   "files for similarity search.")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _storage():
    st.markdown("### Storage")
    root = store.data_dir()
    records = store.list_analyses()
    st.markdown(f"- Data directory: `{root}`\n"
                f"- Saved analyses: **{len(records)}**")
    st.caption("Storage is server-local. Records do not synchronise across "
               "devices or deployments.")
    if records:
        confirm = st.checkbox("I understand this deletes all saved analyses",
                              key="clear_confirm")
        if st.button("Delete all saved analyses", disabled=not confirm,
                     type="secondary", key="clear_all"):
            for record in records:
                store.delete_analysis(record["id"])
            st.success("All saved analyses deleted.")
            st.rerun()


# ---------------------------------------------------------------------------
# Assistant info
# ---------------------------------------------------------------------------

def _assistant_info():
    st.markdown(f"### {ASSISTANT_NAME} assistant")
    st.markdown(
        f"{ASSISTANT_NAME} is Texurion's built-in guided assistant. It answers "
        "from a curated knowledge base and your current analysis state. It "
        "never claims to have analysed an image it did not process.")
    with st.expander("External AI integration (optional)"):
        st.markdown(
            "No external LLM is bundled, so no API key is required. To connect "
            "a hosted model, add the key as a **server-side** environment "
            "variable (never in client code) and route requests through a "
            "backend. Example variable name:\n\n"
            "```\nTEXURION_LLM_API_KEY=...\n```\n\n"
            "Until a backend integration is added, Orio uses deterministic, "
            "honest answers.")


def render():
    st.markdown(f"## ⚙️ Settings")
    st.caption(f"{APP_NAME} preferences, model management and storage.")
    tab_pref, tab_model, tab_storage, tab_assistant = st.tabs(
        ["Preferences", "Model", "Storage", f"{ASSISTANT_NAME}"])
    with tab_pref:
        _preferences()
    with tab_model:
        _model_manager()
    with tab_storage:
        _storage()
    with tab_assistant:
        _assistant_info()
