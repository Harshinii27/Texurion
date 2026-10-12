"""Bootstrap so view scripts can import the ``ui`` package regardless of cwd.

Streamlit may place a page script's own directory on ``sys.path``; this adds
the project root (the parent of ``views/``) as well.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
