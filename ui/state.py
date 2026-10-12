"""Session state, shared analysis controls and model helpers for Texurion."""

import os
import tempfile

import numpy as np
import streamlit as st

from classifier import load_model
from core.lbp import LBP_METHODS
from core.pipeline import DEFAULT_MAX_SIDE, FeatureConfig

SCALE_OPTIONS = {
    "R1/P8": (1, 8),
    "R2/P16": (2, 16),
    "R3/P24": (3, 24),
}

ENGINE_OPTIONS = {
    "scikit-image (reference)": "skimage",
    "NumPy (custom)": "numpy",
    "Numba (custom, JIT)": "numba",
}

MODEL_SECURITY_WARNING = (
    "Model files are Python pickle objects. Loading a .pkl/.joblib you did not "
    "create yourself can execute arbitrary code on this machine. Only load "
    "files you trust."
)

DEFAULT_SCALES = tuple(SCALE_OPTIONS)


def init_state() -> None:
    """Initialise every session key the application relies on."""

    defaults = {
        "model": None,
        "gallery": [],
        "gallery_config": None,
        "orio_messages": [],
        "orio_open": False,
        "dismissed_hints": set(),
        "saved_index_cache": None,
        # Analyze workflow state
        "analyzer_image": None,
        "analyzer_image_name": None,
        "analyzer_image_meta": None,
        "analyzer_workflow": None,
        "analyzer_run": None,
        # Compare workflow state
        "compare_a": None,
        "compare_a_name": None,
        "compare_b": None,
        "compare_b_name": None,
        "compare_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    st.session_state.setdefault(
        "prefs",
        {
            "theme_mode": "brand",
            "show_hints": True,
            "default_max_side": 512,
            "default_method": "uniform",
            "default_engine": "skimage",
            "include_gabor": False,
            "include_opponent": False,
        },
    )


def _pref(key, fallback):
    return st.session_state.get("prefs", {}).get(key, fallback)


# ---------------------------------------------------------------------------
# Feature configuration widgets (shared by Analyze / Compare / Batch / Models)
# ---------------------------------------------------------------------------

def feature_config_widgets(key_prefix, default_max_side=None, compact=False):
    """Render feature controls and return a :class:`FeatureConfig`."""

    if default_max_side is None:
        default_max_side = _pref("default_max_side", DEFAULT_MAX_SIDE)

    method_default = _pref("default_method", "uniform")
    method_index = LBP_METHODS.index(method_default) if method_default in LBP_METHODS else 0
    engine_default = _pref("default_engine", "skimage")
    engine_labels = list(ENGINE_OPTIONS)
    engine_label_default = next(
        (label for label, value in ENGINE_OPTIONS.items() if value == engine_default),
        engine_labels[0],
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox(
            "LBP method", LBP_METHODS, index=method_index,
            key=f"{key_prefix}_method",
            help="Local Binary Pattern variant used for the histogram block.")
    with c2:
        scales = st.multiselect(
            "Multiscale LBP scales", list(SCALE_OPTIONS), default=list(DEFAULT_SCALES),
            key=f"{key_prefix}_scales",
            help="Each scale concatenates a normalised LBP histogram.")
    with c3:
        engine_label = st.selectbox(
            "LBP engine", engine_labels, index=engine_labels.index(engine_label_default),
            key=f"{key_prefix}_engine",
            help="Custom engines implement a classic circular uniform LBP; they "
                 "are not claimed to match or beat scikit-image.")

    if not scales:
        st.warning("Select at least one LBP scale; using the default trio.")
        scales = list(SCALE_OPTIONS)

    c4, c5, c6 = st.columns(3)
    with c4:
        include_gabor = st.checkbox(
            "Gabor filter bank", value=_pref("include_gabor", False),
            key=f"{key_prefix}_gabor",
            help="Adds Gabor mean/std statistics to the fused feature vector.")
    with c5:
        include_opponent = st.checkbox(
            "Opponent-colour LBP", value=_pref("include_opponent", False),
            key=f"{key_prefix}_opponent",
            help="Adds opponent-colour LBP histograms (needs a colour image).")
    with c6:
        max_side = st.number_input(
            "Working resolution (max side)", min_value=128, max_value=2048,
            value=int(default_max_side), step=32, key=f"{key_prefix}_max_side",
            help="Images are downscaled so their largest side equals this before "
                 "feature extraction. Keep identical between train and predict.")

    return FeatureConfig(
        method=method,
        engine=ENGINE_OPTIONS[engine_label],
        lbp_scales=tuple(SCALE_OPTIONS[s] for s in scales),
        include_gabor=include_gabor,
        include_color_opponent=include_opponent,
        max_side=int(max_side),
    )


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_model_bytes(model_bytes):
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name
    try:
        return load_model(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def set_model_from_bytes(model_bytes) -> bool:
    """Load a model bundle into session state. Returns True on success."""

    try:
        bundle = _load_model_bytes(model_bytes)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model could not be loaded: {exc}")
        return False
    st.session_state["model"] = bundle
    st.success(f"Model loaded with classes: {', '.join(bundle['classes'])}.")
    return True


def current_model():
    return st.session_state.get("model")


def model_compatibility(config):
    """Return a blocking message if the loaded model cannot run, else None."""

    model = st.session_state.get("model")
    if model is None:
        return "No trained model is loaded. Train or load one in Settings."
    wanted = config.descriptor_length()
    if model["feature_length"] != wanted:
        return (
            f"Model expects {model['feature_length']} features but the current "
            f"configuration produces {wanted}. Match the settings or retrain."
        )
    return None


def model_pretrain_note(config):
    model = st.session_state.get("model")
    if model is None:
        return None
    trained_at = model["feature_config"].get("max_side")
    if trained_at and trained_at != config.max_side:
        return (
            f"Note: the model was trained at max_side={trained_at}, current "
            f"setting is {config.max_side}. Dimensions match so classification "
            "runs, but preprocessing differs from training."
        )
    return None


def gallery_compatibility(config):
    gallery = st.session_state.get("gallery", [])
    if not gallery:
        return "No reference gallery. Training a model on labelled data builds one."
    cfg = st.session_state.get("gallery_config")
    if cfg is None or cfg["descriptor_length"] != config.descriptor_length():
        return ("Reference gallery used a different feature configuration; "
                "reconfigure to match.")
    return None


def predict_class(model, vector):
    from classifier import predict
    return predict(model, vector)


def active_context():
    """Small, serialisable summary of the current UI state for Orio."""

    return {
        "has_image": st.session_state.get("analyzer_image") is not None,
        "image_name": st.session_state.get("analyzer_image_name"),
        "workflow": st.session_state.get("analyzer_workflow"),
        "has_model": st.session_state.get("model") is not None,
        "gallery_size": len(st.session_state.get("gallery", [])),
    }


def clear_analyzer():
    st.session_state["analyzer_image"] = None
    st.session_state["analyzer_image_name"] = None
    st.session_state["analyzer_image_meta"] = None
    st.session_state["analyzer_run"] = None
