"""Industry workflow templates.

Each workflow selects appropriate analysis options and explains what Texurion
can and cannot measure for that domain. Specialised models (e.g. ripeness or
defect CNNs) are not bundled, so those workflows state the capability
limitation and route the user to the general texture analysis that is
implemented.
"""

WORKFLOWS = [
    {
        "key": "food",
        "title": "Fruits & Food",
        "icon": "🍎",
        "description": "Screen visible surface appearance and texture of produce and food.",
        "use_cases": [
            "Visible surface defect screening",
            "Texture consistency between batches",
            "Ripeness proxy from surface pattern",
        ],
        "analyses": [
            "Multiscale LBP texture profile",
            "GLCM contrast / homogeneity",
            "Reference-gallery similarity",
        ],
        "limitations": (
            "Ripeness, freshness and safety cannot be established from a "
            "photograph alone. These need a validated model trained on "
            "labelled samples or physical testing. Texurion reports texture "
            "indicators only."),
        "defaults": {"method": "uniform", "include_gabor": False,
                     "include_opponent": True, "max_side": 512},
        "suggestion": "Compare two samples to rank visual texture dissimilarity.",
    },
    {
        "key": "textile",
        "title": "Textiles & Fabric",
        "icon": "🧵",
        "description": "Analyse weave regularity, surface uniformity and visible defects.",
        "use_cases": [
            "Weave / repeated-pattern analysis",
            "Surface uniformity scoring",
            "Comparative fabric inspection",
        ],
        "analyses": [
            "Multiscale LBP (weave spacing)",
            "Gabor bank (directional structure)",
            "GLCM correlation / energy",
        ],
        "limitations": (
            "Thread count, GSM and fibre composition are not measured; these "
            "require physical instruments. Texurion characterises the visible "
            "pattern."),
        "defaults": {"method": "uniform", "include_gabor": True,
                     "include_opponent": False, "max_side": 768},
        "suggestion": "Use Gabor features to capture directional weave structure.",
    },
    {
        "key": "agriculture",
        "title": "Agriculture & Soil",
        "icon": "🌱",
        "description": "Describe soil and crop surface texture from imagery.",
        "use_cases": [
            "Visual texture / clod size patterns",
            "Surface colour-appearance features",
            "Sample-to-sample comparison",
        ],
        "analyses": [
            "Multiscale LBP",
            "GLCM contrast / entropy",
            "Opponent-colour LBP",
        ],
        "limitations": (
            "Particle-size distribution and porosity need calibrated imaging "
            "or sieving; a single photo is a proxy only."),
        "defaults": {"method": "uniform", "include_gabor": False,
                     "include_opponent": True, "max_side": 512},
        "suggestion": "Compare field samples captured under the same lighting.",
    },
    {
        "key": "materials",
        "title": "Materials & Surfaces",
        "icon": "🧱",
        "description": "Characterise surface pattern variation and consistency.",
        "use_cases": [
            "Texture variation across a surface",
            "Pattern consistency checks",
            "Visible crack / scratch screening",
        ],
        "analyses": [
            "Multiscale LBP",
            "GLCM homogeneity / contrast",
            "Gabor orientation responses",
        ],
        "limitations": (
            "Cracks and scratches are highlighted as high-contrast texture "
            "regions, not detected by a trained defect model. Confirm "
            "physically."),
        "defaults": {"method": "uniform", "include_gabor": True,
                     "include_opponent": False, "max_side": 768},
        "suggestion": "Run a reference-image comparison for repeat inspections.",
    },
    {
        "key": "industrial",
        "title": "Industrial Inspection",
        "icon": "🏭",
        "description": "Repeatable batch inspection with reference matching.",
        "use_cases": [
            "Batch image comparison",
            "Reference-image matching",
            "Repeatable inspection settings",
        ],
        "analyses": [
            "Fixed-configuration LBP + GLCM",
            "Feature-vector similarity to reference",
            "CSV / PDF QC report",
        ],
        "limitations": (
            "Without a trained model, this is a consistency check relative to "
            "a reference, not an automated pass/fail grader."),
        "defaults": {"method": "uniform", "include_gabor": False,
                     "include_opponent": False, "max_side": 768},
        "suggestion": "Lock one FeatureConfig and reuse it for every sample.",
    },
    {
        "key": "medical",
        "title": "Medical & Biological",
        "icon": "🩺",
        "description": "Research-oriented texture feature extraction and visualisation.",
        "use_cases": [
            "Texture feature extraction",
            "Feature / histogram visualisation",
            "Experimental anomaly screening",
        ],
        "analyses": [
            "Multiscale LBP",
            "GLCM + histogram statistics",
            "Gabor bank",
        ],
        "limitations": (
            "This is a research and support tool, NOT a diagnostic system. "
            "Images alone do not establish a diagnosis. Do not use for "
            "clinical decisions."),
        "defaults": {"method": "uniform", "include_gabor": True,
                     "include_opponent": False, "max_side": 512},
        "suggestion": "Export the research metrics for statistical analysis.",
    },
    {
        "key": "research",
        "title": "Research & Education",
        "icon": "🔬",
        "description": "Reproducible feature vectors, histograms and exports.",
        "use_cases": [
            "Reproducible experiments",
            "Teaching LBP / GLCM concepts",
            "Feature-vector export for ML",
        ],
        "analyses": [
            "Multiscale LBP + Gabor + opponent colour",
            "GLCM metrics",
            "Feature-vector + histogram CSV export",
        ],
        "limitations": (
            "Every parameter is exposed and reported so results can be "
            "reproduced; nothing is summarised into an unverified score."),
        "defaults": {"method": "uniform", "include_gabor": True,
                     "include_opponent": True, "max_side": 1024},
        "suggestion": "Open Research Mode for the full metric table and CSV.",
    },
    {
        "key": "compare",
        "title": "Compare Two Images",
        "icon": "📷",
        "description": "Quantify how two samples differ in pixels and texture.",
        "use_cases": [
            "Before / after inspection",
            "Reference vs sample",
            "Batch-to-batch variation",
        ],
        "analyses": [
            "Pixel similarity (MAE, SSIM)",
            "LBP histogram distance (chi-square, cosine)",
            "Feature-vector cosine similarity",
        ],
        "limitations": (
            "Pixel, feature and semantic similarity are different things and "
            "kept separate in the report."),
        "defaults": {"method": "uniform", "include_gabor": False,
                     "include_opponent": False, "max_side": 512},
        "suggestion": "Open the Compare page to load both images.",
    },
]


WORKFLOW_BY_KEY = {w["key"]: w for w in WORKFLOWS}


def get(key):
    return WORKFLOW_BY_KEY.get(key)


def apply_defaults(key, session_state, prefix="analyzer"):
    """Seed feature controls from a workflow and clear existing widget state."""

    workflow = get(key)
    if workflow is None:
        return
    prefs = session_state.setdefault("prefs", {})
    defaults = workflow["defaults"]
    prefs["default_method"] = defaults.get("method", "uniform")
    prefs["include_gabor"] = defaults.get("include_gabor", False)
    prefs["include_opponent"] = defaults.get("include_opponent", False)
    prefs["default_max_side"] = defaults.get("max_side", 512)
    session_state["analyzer_workflow"] = key
    for widget_key in (
        f"{prefix}_method", f"{prefix}_scales", f"{prefix}_engine",
        f"{prefix}_gabor", f"{prefix}_opponent", f"{prefix}_max_side",
    ):
        session_state.pop(widget_key, None)
