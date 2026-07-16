#!/usr/bin/env python3
"""Standalone HRX Demos presubmit entry point."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_TEST_TRIGGERS = (
    ".bazelrc",
    ".bazelversion",
    "BUILD.bazel",
    "MODULE.bazel",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mutation = parser.add_mutually_exclusive_group()
    mutation.add_argument("--fix", action="store_true", help="Accepted for symmetry.")
    mutation.add_argument("--check", action="store_true", help="Accepted for symmetry.")
    parser.add_argument("--tests", action="store_true", help="Run repository tests.")
    parser.add_argument(
        "--files-from",
        help="Path to a newline-separated repo-relative changed-file list.",
    )
    return parser.parse_args()


def run_command(command: list[str], description: str) -> bool:
    print(f"[hrx-demos] {description}: {' '.join(command)}")
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode == 0


def selected_files(files_from: str | None) -> list[str]:
    if not files_from:
        return []
    with open(files_from, encoding="utf-8") as file_list:
        return [line.strip() for line in file_list if line.strip()]


def should_run_tests(files_from: str | None) -> bool:
    paths = selected_files(files_from)
    if not paths:
        return files_from is None
    return any(
        path in GLOBAL_TEST_TRIGGERS
        or path.startswith("build_tools/")
        or not path.startswith("docs/")
        for path in paths
    )


def bazel_build_command() -> list[str]:
    return ["bazel", "build", "//..."]


def bazel_test_command() -> list[str]:
    return [
        "bazel",
        "test",
        "--test_tag_filters=-manual,-requires-gpu",
        "//...",
    ]


def run_bazel_tests() -> bool:
    ok = run_command(bazel_build_command(), "Bazel build")
    return run_command(bazel_test_command(), "Bazel tests") and ok


def main() -> int:
    args = parse_arguments()
    if not args.tests:
        return 0
    if not should_run_tests(args.files_from):
        print("hrx-demos presubmit: no build-affecting files")
        return 0
    return 0 if run_bazel_tests() else 1


if __name__ == "__main__":
    raise SystemExit(main())
