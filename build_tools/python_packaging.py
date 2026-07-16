"""Build and staging helpers for the HRX Demos Python wheel."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from setuptools import Distribution
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


REPO_ROOT = Path(__file__).resolve().parents[1]
BAZEL_TARGETS = (
    "//binding/cli:id4",
    "@hrx_system//libhrx/tools:hrx-info",
)
ROCM_LIBRARY_GLOBS = (
    "lib/libhsa-runtime64.so*",
    "lib/libhsa-amd-aqlprofile64.so*",
    "lib/libhsakmt.so*",
    "lib/librocprofiler-register.so*",
    "lib/rocm_sysdeps/lib/*.so*",
)
ROCM_DOCUMENTS = (
    "share/doc/rocr/LICENSE.md",
    "share/doc/hsa-amd-aqlprofile/LICENSE.md",
    "share/doc/rocprofiler-register/LICENSE.md",
    "share/therock/therock_manifest.json",
)


def discover_bazel(explicit: Optional[str] = None) -> Path:
    """Finds Bazel from an override, PATH, or a containing workspace."""
    requested = explicit or os.environ.get("BAZEL")
    if requested:
        candidate = shutil.which(requested) or requested
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise RuntimeError(f"Bazel is not executable: {requested}")

    path_candidate = shutil.which("bazel")
    if path_candidate:
        return Path(path_candidate).resolve()

    for parent in (REPO_ROOT, *REPO_ROOT.parents):
        candidate = parent / "programs" / "bazel" / "bin" / "bazel"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    raise RuntimeError(
        "Bazel was not found. Run `python build_tools/setup_python.py --help` "
        "for the pinned Bazelisk installation instructions."
    )


def read_bazel_repo_env(path: Path) -> dict[str, str]:
    """Reads --repo_env assignments from a Bazel rc file."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for token in shlex.split(line):
            prefix = "--repo_env="
            if not token.startswith(prefix):
                continue
            assignment = token[len(prefix) :]
            if "=" in assignment:
                name, value = assignment.split("=", 1)
                values[name] = value
    return values


def discover_rocm_root() -> Path:
    """Finds the ROCm root used by the Bazel repository configuration."""
    configured = os.environ.get("IREE_ROCM_PATH")
    if not configured:
        configured = read_bazel_repo_env(REPO_ROOT / ".bazelrc.local").get(
            "IREE_ROCM_PATH"
        )
    if not configured:
        raise RuntimeError(
            "IREE_ROCM_PATH is not configured; run "
            "`python build_tools/setup_python.py --rocm /path/to/rocm`."
        )
    root = Path(configured).expanduser().resolve()
    if not (root / "include").is_dir():
        raise RuntimeError(f"ROCm root has no include directory: {root}")
    if not (root / "lib" / "libhsa-runtime64.so.1").is_file():
        raise RuntimeError(f"ROCm root has no HSA runtime: {root}")
    return root


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _query_bazel_output(bazel: Path, target: str) -> Path:
    result = subprocess.run(
        [str(bazel), "cquery", "-c", "opt", "--output=files", target],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    output_paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(output_paths) != 1:
        raise RuntimeError(
            f"Expected one Bazel output for {target}, got {output_paths!r}"
        )
    output = Path(output_paths[0])
    if not output.is_absolute():
        output = REPO_ROOT / output
    if not output.is_file():
        raise RuntimeError(f"Bazel output for {target} does not exist: {output}")
    return output.resolve()


def build_native_executables(bazel: Path) -> dict[str, Path]:
    """Builds and locates the native executables included in the wheel."""
    _run([str(bazel), "build", "-c", "opt", *BAZEL_TARGETS])
    return {
        "hrx-id4": _query_bazel_output(bazel, BAZEL_TARGETS[0]),
        "hrx-info": _query_bazel_output(bazel, BAZEL_TARGETS[1]),
    }


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(), destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entries(native_root: Path) -> Iterable[dict[str, object]]:
    for path in sorted(native_root.rglob("*")):
        if path.is_file():
            yield {
                "path": path.relative_to(native_root).as_posix(),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }


def stage_native_payload(
    native_root: Path,
    executables: dict[str, Path],
    rocm_root: Path,
) -> list[Path]:
    """Stages Bazel tools and the redistributable ROCm runtime payload."""
    if native_root.exists():
        shutil.rmtree(native_root)
    native_root.mkdir(parents=True)

    for installed_name, source in executables.items():
        destination = native_root / "bin" / installed_name
        _copy_file(source, destination)
        destination.chmod(destination.stat().st_mode | 0o111)

    copied_rocm_paths: set[str] = set()
    for pattern in ROCM_LIBRARY_GLOBS:
        for source in sorted(rocm_root.glob(pattern)):
            relative_path = source.relative_to(rocm_root)
            _copy_file(source, native_root / relative_path)
            copied_rocm_paths.add(relative_path.as_posix())

    if not any(path.startswith("lib/libhsa-runtime64.so") for path in copied_rocm_paths):
        raise RuntimeError(f"No HSA runtime libraries found under {rocm_root}")

    for relative_name in ROCM_DOCUMENTS:
        source = rocm_root / relative_name
        if not source.is_file():
            continue
        if relative_name.endswith("therock_manifest.json"):
            destination = native_root / "rocm-manifest.json"
        else:
            component = Path(relative_name).parent.name
            destination = native_root / "licenses" / f"{component}-LICENSE.md"
        _copy_file(source, destination)

    manifest_path = native_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "files": list(_manifest_entries(native_root)),
                "rocm_paths": sorted(copied_rocm_paths),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [path for path in native_root.rglob("*") if path.is_file()]


def validate_wheel_platform() -> None:
    machine = platform.machine().lower()
    if os.name != "posix" or platform.system() != "Linux":
        raise RuntimeError("HRX Demos wheels currently require Linux")
    if machine not in ("x86_64", "amd64"):
        raise RuntimeError(f"HRX Demos wheels currently require x86-64, got {machine}")


class BinaryDistribution(Distribution):
    """Marks the distribution as platform-specific without a Python extension."""

    def has_ext_modules(self) -> bool:
        return True


class Py3PlatformWheel(bdist_wheel):
    """Tags native executable wheels as Python-minor independent."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


class BazelBuildPy(build_py):
    """Builds and stages the native payload into setuptools' build tree."""

    def run(self) -> None:
        validate_wheel_platform()
        bazel = discover_bazel()
        rocm_root = discover_rocm_root()
        executables = build_native_executables(bazel)
        package_root = Path(self.build_lib) / "hrx_demos"
        if package_root.exists():
            shutil.rmtree(package_root)
        super().run()
        native_root = package_root / "_native"
        self._native_outputs = stage_native_payload(
            native_root, executables, rocm_root
        )

    def get_outputs(self, include_bytecode: int = 1) -> list[str]:
        outputs = super().get_outputs(include_bytecode)
        outputs.extend(str(path) for path in getattr(self, "_native_outputs", ()))
        return outputs
