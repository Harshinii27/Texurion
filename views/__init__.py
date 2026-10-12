"""Texurion Streamlit page entrypoints.

Each module here is a thin wrapper that imports the project root and then
delegates to the matching ``ui.pages`` implementation. Keeping ``views`` a
real package (rather than relying on Streamlit putting this directory on
``sys.path``) lets the wrappers use the same absolute, root-based imports as
the rest of the application.
"""
