"""Executes native HRX tools shipped in the wheel."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, NoReturn, Optional


_LIBHSA_ENV = "IREE_HAL_AMDGPU_LIBHSA_PATH"
_LIBAQLPROFILE_ENV = "IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH"


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "HRX wheel is missing a required runtime library: " + ", ".join(names)
    )


def native_environment(
    environ: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Returns the environment used for a bundled native executable."""
    result = dict(os.environ if environ is None else environ)
    native_root = Path(__file__).resolve().parent / "_native"
    library_dir = native_root / "lib"

    if _LIBHSA_ENV not in result:
        result[_LIBHSA_ENV] = str(
            _first_existing(
                library_dir,
                ("libhsa-runtime64.so.1", "libhsa-runtime64.so"),
            )
        )
        aqlprofile_names = (
            "libhsa-amd-aqlprofile64.so.1",
            "libhsa-amd-aqlprofile64.so",
        )
        for name in aqlprofile_names:
            candidate = library_dir / name
            if candidate.is_file():
                result.setdefault(_LIBAQLPROFILE_ENV, str(candidate))
                break

        search_paths = [
            str(library_dir),
            str(library_dir / "rocm_sysdeps" / "lib"),
        ]
        existing_search_path = result.get("LD_LIBRARY_PATH")
        if existing_search_path:
            search_paths.append(existing_search_path)
        result["LD_LIBRARY_PATH"] = os.pathsep.join(search_paths)

    return result


def exec_native(tool_name: str) -> NoReturn:
    """Replaces this process with a bundled HRX executable."""
    executable = Path(__file__).resolve().parent / "_native" / "bin" / tool_name
    if not executable.is_file():
        raise RuntimeError(f"HRX wheel is missing native executable: {tool_name}")
    os.execve(
        str(executable),
        [str(executable), *sys.argv[1:]],
        native_environment(),
    )
