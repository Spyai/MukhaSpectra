"""Which build produced a result (IEC 62304 configuration management, risk H16).

``software_version`` = backend ``__version__`` + ``+`` + build id, e.g. ``0.1.0+3f2c1ab``. The build
id comes from ``MUKHA_BUILD_ID`` (baked into the Docker image at build time), else from
``git rev-parse --short HEAD`` (``.dirty`` appended when tracked files have uncommitted changes),
else ``unknown``. It is resolved once per process.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache

from app import __version__
from app.settings import BACKEND_DIR

UNKNOWN_BUILD = "unknown"
_GIT_TIMEOUT_S = 3


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return (done.stdout or "").strip()


@lru_cache(maxsize=8)
def resolve_build_id(configured: str | None) -> str:
    """The configured build id, else the git commit (``.dirty`` if modified), else 'unknown'."""
    if configured:
        return configured
    commit = _git("rev-parse", "--short", "HEAD")
    if not commit:
        return UNKNOWN_BUILD
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return f"{commit}.dirty" if dirty else commit


def software_version(build_id: str) -> str:
    return f"{__version__}+{build_id}"
