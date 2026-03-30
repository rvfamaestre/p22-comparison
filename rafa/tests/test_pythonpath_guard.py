from __future__ import annotations

from types import SimpleNamespace

from src.utils.pythonpath_guard import sanitize_incompatible_pythonpath


def make_version(major: int, minor: int) -> SimpleNamespace:
    return SimpleNamespace(major=major, minor=minor)


def test_noop_outside_virtualenv() -> None:
    sys_path = [
        "/repo",
        "/softwares/spack/python3.10/site-packages",
    ]
    env = {}

    removed = sanitize_incompatible_pythonpath(
        sys_path,
        env,
        prefix="/usr/bin/python3.11",
        base_prefix="/usr/bin/python3.11",
        version_info=make_version(3, 11),
        pathsep=":",
    )

    assert removed == {"sys_path": [], "pythonpath": []}
    assert sys_path == [
        "/repo",
        "/softwares/spack/python3.10/site-packages",
    ]


def test_drops_mismatched_pythonpath_entries_inside_virtualenv() -> None:
    sys_path = [
        "/repo",
        "/usr/users/me/project/.venv-hpc/lib/python3.11/site-packages",
        "/softwares/spack/python3.10/site-packages",
        "/shared/tools",
    ]
    env = {
        "VIRTUAL_ENV": "/usr/users/me/project/.venv-hpc",
        "PYTHONPATH": "/softwares/spack/python3.10/site-packages:/repo/src:/shared/tools",
    }

    removed = sanitize_incompatible_pythonpath(
        sys_path,
        env,
        prefix="/usr/users/me/project/.venv-hpc",
        base_prefix="/usr/bin/python3.11",
        version_info=make_version(3, 11),
        pathsep=":",
    )

    assert removed["sys_path"] == ["/softwares/spack/python3.10/site-packages"]
    assert removed["pythonpath"] == ["/softwares/spack/python3.10/site-packages"]
    assert sys_path == [
        "/repo",
        "/usr/users/me/project/.venv-hpc/lib/python3.11/site-packages",
        "/shared/tools",
    ]
    assert env["PYTHONPATH"] == "/repo/src:/shared/tools"


def test_keeps_same_version_external_entries() -> None:
    sys_path = [
        "/repo",
        "/softwares/spack/python3.11/site-packages",
    ]
    env = {
        "VIRTUAL_ENV": "/usr/users/me/project/.venv-hpc",
        "PYTHONPATH": "/softwares/spack/python3.11/site-packages:/repo/src",
    }

    removed = sanitize_incompatible_pythonpath(
        sys_path,
        env,
        prefix="/usr/users/me/project/.venv-hpc",
        base_prefix="/usr/bin/python3.11",
        version_info=make_version(3, 11),
        pathsep=":",
    )

    assert removed == {"sys_path": [], "pythonpath": []}
    assert sys_path == [
        "/repo",
        "/softwares/spack/python3.11/site-packages",
    ]
    assert env["PYTHONPATH"] == "/softwares/spack/python3.11/site-packages:/repo/src"
