"""Repository-wide Python startup guard.

Automatically strips incompatible PYTHONPATH entries when commands are run from
this repo inside a virtualenv. This avoids HPC module pollution such as Python
3.10 site-packages being injected into a Python 3.11 venv.
"""

from __future__ import annotations

try:
    from src.utils.pythonpath_guard import sanitize_incompatible_pythonpath

    sanitize_incompatible_pythonpath()
except Exception:
    # Startup hooks must never block the actual program entrypoint.
    pass
