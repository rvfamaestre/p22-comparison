from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import MutableMapping, Sequence


_PYTHON_VERSION_PATTERN = re.compile(
    r"(?i)(?:^|[\\/])python(?P<major>\d+)\.(?P<minor>\d+)(?:[\\/]|$)"
)


def _normalize_path(path: str) -> str:
    try:
        return str(Path(path).resolve(strict=False))
    except OSError:
        return os.path.abspath(path)


def _extract_python_version(path: str) -> tuple[int, int] | None:
    match = _PYTHON_VERSION_PATTERN.search(path)
    if match is None:
        return None
    return int(match.group("major")), int(match.group("minor"))


def _is_within(path: str, root: str | None) -> bool:
    if not root:
        return False
    try:
        normalized_path = _normalize_path(path)
        normalized_root = _normalize_path(root)
        return os.path.commonpath([normalized_path, normalized_root]) == normalized_root
    except (OSError, ValueError):
        return False


def _running_inside_venv(
    env: MutableMapping[str, str], prefix: str, base_prefix: str
) -> bool:
    return bool(env.get("VIRTUAL_ENV")) or prefix != base_prefix


def _should_drop_path(
    path: str,
    current_version: tuple[int, int],
    venv_root: str | None,
) -> bool:
    if not path:
        return False
    if _is_within(path, venv_root):
        return False
    path_version = _extract_python_version(path)
    if path_version is None:
        return False
    return path_version != current_version


def sanitize_incompatible_pythonpath(
    sys_path: list[str] | None = None,
    env: MutableMapping[str, str] | None = None,
    *,
    prefix: str | None = None,
    base_prefix: str | None = None,
    version_info: object | None = None,
    pathsep: str | None = None,
) -> dict[str, list[str]]:
    """Remove PYTHONPATH entries that target another Python major/minor version.

    This is aimed at HPC/module environments that export site-packages from a
    system Python into a virtualenv-backed interpreter. The most common failure
    mode is importing NumPy extensions built for Python 3.10 from a Python 3.11
    virtualenv.
    """

    sys_path = sys.path if sys_path is None else sys_path
    env = os.environ if env is None else env
    prefix = sys.prefix if prefix is None else prefix
    base_prefix = sys.base_prefix if base_prefix is None else base_prefix
    version_info = sys.version_info if version_info is None else version_info
    pathsep = os.pathsep if pathsep is None else pathsep

    if not _running_inside_venv(env, prefix, base_prefix):
        return {"sys_path": [], "pythonpath": []}

    current_version = (int(version_info.major), int(version_info.minor))
    venv_root = env.get("VIRTUAL_ENV") or prefix

    removed_sys_path: list[str] = []
    cleaned_sys_path: list[str] = []
    for entry in sys_path:
        if isinstance(entry, str) and _should_drop_path(entry, current_version, venv_root):
            removed_sys_path.append(entry)
            continue
        cleaned_sys_path.append(entry)
    sys_path[:] = cleaned_sys_path

    raw_pythonpath = env.get("PYTHONPATH", "")
    removed_pythonpath: list[str] = []
    if raw_pythonpath:
        cleaned_entries: list[str] = []
        for entry in raw_pythonpath.split(pathsep):
            if _should_drop_path(entry, current_version, venv_root):
                removed_pythonpath.append(entry)
                continue
            cleaned_entries.append(entry)
        if cleaned_entries:
            env["PYTHONPATH"] = pathsep.join(cleaned_entries)
        else:
            env.pop("PYTHONPATH", None)

    return {"sys_path": removed_sys_path, "pythonpath": removed_pythonpath}
