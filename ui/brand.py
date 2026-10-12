"""Brand constants and design tokens for Texurion.

IMPORTANT: the two background gradients below are the *original* gradients
from the pre-existing application (see the old ``app.py``). They are the
brand's identity and must not be changed. Every other colour token here was
selected to harmonise with those backgrounds while keeping AA contrast.
"""

APP_NAME = "Texurion"
APP_TAGLINE = "Texture Intelligence Studio"
ASSISTANT_NAME = "Orio"

# ---------------------------------------------------------------------------
# Original brand backgrounds -- DO NOT CHANGE
# ---------------------------------------------------------------------------

MAIN_BACKGROUND = (
    "linear-gradient(135deg, #FFE2F0 0%, #FFD0E4 40%, "
    "#FFB8D2 70%, #FF9CBD 100%)"
)
SIDEBAR_BACKGROUND = (
    "linear-gradient(to bottom, #E8F7F0 0%, #D5EFE4 50%, #C2E6D6 100%)"
)

# ---------------------------------------------------------------------------
# Supporting palette (harmonises with the pink gradient / emerald sidebar)
# ---------------------------------------------------------------------------

TOKENS = {
    "ink": "#101820",
    "ink_soft": "#3b4553",
    "ink_muted": "#67707e",
    "surface": "rgba(255, 255, 255, 0.86)",
    "surface_solid": "#ffffff",
    "surface_alt": "#FDF3F8",
    "border": "rgba(16, 24, 32, 0.10)",
    "border_strong": "rgba(16, 24, 32, 0.18)",
    # Stronger magenta used only for interactive surfaces (AA with white text)
    "accent": "#C2185B",
    "accent_soft": "#E75480",
    "accent_ink": "#8E1746",
    "focus": "#B5179E",
    "sidebar_ink": "#12352a",
    "sidebar_accent": "#0E7C66",
    "success": "#1B7F5A",
    "warning": "#B26A00",
    "danger": "#C0392B",
    "info": "#2A6DB0",
}

# Gradient used on primary buttons/hero chip (derived from the brand pink).
ACCENT_GRADIENT = "linear-gradient(135deg, #D6336C 0%, #A61E4D 100%)"
