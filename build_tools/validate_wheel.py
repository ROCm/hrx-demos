#!/usr/bin/env python3
"""Validates the native runtime layout of an HRX Demos wheel."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from collections import defaultdict
from pathlib import Path


NATIVE_LIBRARY_PREFIX = "hrx_demos/_native/lib/"
REQUIRED_RUNTIME_LIBRARIES = {
    NATIVE_LIBRARY_PREFIX + "libhsa-runtime64.so.1",
    NATIVE_LIBRARY_PREFIX + "libhsa-amd-aqlprofile64.so.1",
}
STABLE_RUNTIME_NAMES = {
    "libhsa-runtime64": "libhsa-runtime64.so.1",
    "libhsa-amd-aqlprofile64": "libhsa-amd-aqlprofile64.so.1",
}
ROCM_LIBRARY_PREFIXES = ("libamd", "libhsa", "libroc")


def validate_wheel(path: Path) -> tuple[int, int]:
    duplicates: dict[str, list[str]] = defaultdict(list)
    native_libraries: list[str] = []
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        missing = sorted(REQUIRED_RUNTIME_LIBRARIES - names)
        if missing:
            raise RuntimeError(
                "wheel is missing runtime libraries: " + ", ".join(missing)
            )

        for info in wheel.infolist():
            if info.is_dir():
                continue
            name = info.filename
            basename = Path(name).name
            if name.startswith(NATIVE_LIBRARY_PREFIX) and ".so" in basename:
                native_libraries.append(name)
                digest = hashlib.sha256(wheel.read(info)).hexdigest()
                duplicates[digest].append(name)
            if ".libs/" in name and basename.startswith(ROCM_LIBRARY_PREFIXES):
                raise RuntimeError(
                    f"auditwheel grafted a duplicate ROCm library: {name}"
                )

    for prefix, allowed_name in STABLE_RUNTIME_NAMES.items():
        aliases = sorted(
            name
            for name in native_libraries
            if Path(name).name.startswith(prefix + ".so")
            and Path(name).name != allowed_name
        )
        if aliases:
            raise RuntimeError(
                f"wheel contains unstable {prefix} aliases: " + ", ".join(aliases)
            )

    duplicate_groups = [names for names in duplicates.values() if len(names) > 1]
    if duplicate_groups:
        formatted = "; ".join(", ".join(names) for names in duplicate_groups)
        raise RuntimeError(f"wheel contains duplicate native libraries: {formatted}")

    return len(native_libraries), path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    count, size = validate_wheel(args.wheel)
    print(f"Validated {args.wheel}: {count} native libraries, {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
