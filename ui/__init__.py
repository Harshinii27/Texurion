"""Texurion UI package.

The desktop/CLI ``core`` package holds the (unchanged) image-processing
algorithms. Everything in ``ui`` is presentation and orchestration for the
Streamlit "Texurion" web application, with the "Orio" assistant.
"""

from .brand import APP_NAME, ASSISTANT_NAME  # noqa: F401

__all__ = ["APP_NAME", "ASSISTANT_NAME"]
