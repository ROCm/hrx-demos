# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def load_launcher():
    launcher_path = (
        Path(__file__).resolve().parents[1]
        / "binding"
        / "python"
        / "_launcher.py"
    )
    spec = importlib.util.spec_from_file_location("hrx_demos._launcher", launcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {launcher_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = load_launcher()

    def test_defaults_to_bundled_hsa(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_root = Path(temporary_directory) / "hrx_demos"
            library_dir = package_root / "_native" / "lib"
            library_dir.mkdir(parents=True)
            hsa_path = library_dir / "libhsa-runtime64.so.1"
            hsa_path.write_bytes(b"")
            aql_path = library_dir / "libhsa-amd-aqlprofile64.so.1"
            aql_path.write_bytes(b"")
            with mock.patch.object(
                self.launcher, "__file__", str(package_root / "_launcher.py")
            ):
                result = self.launcher.native_environment({})
            self.assertEqual(
                result["IREE_HAL_AMDGPU_LIBHSA_PATH"], str(hsa_path)
            )
            self.assertEqual(
                result["IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH"], str(aql_path)
            )
            self.assertTrue(result["LD_LIBRARY_PATH"].startswith(str(library_dir)))

    def test_preserves_explicit_hsa_without_bundled_search_path(self):
        result = self.launcher.native_environment(
            {
                "IREE_HAL_AMDGPU_LIBHSA_PATH": "/custom/libhsa.so",
                "LD_LIBRARY_PATH": "/custom/lib",
            }
        )
        self.assertEqual(
            result,
            {
                "IREE_HAL_AMDGPU_LIBHSA_PATH": "/custom/libhsa.so",
                "LD_LIBRARY_PATH": "/custom/lib",
            },
        )

    def test_exec_forwards_arguments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_root = Path(temporary_directory) / "hrx_demos"
            executable = package_root / "_native" / "bin" / "hrx-info"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"")
            with (
                mock.patch.object(
                    self.launcher, "__file__", str(package_root / "_launcher.py")
                ),
                mock.patch.object(self.launcher, "native_environment", return_value={"A": "B"}),
                mock.patch.object(self.launcher.sys, "argv", ["hrx-info", "--help"]),
                mock.patch.object(self.launcher.os, "execve") as execve,
            ):
                self.launcher.exec_native("hrx-info")
            execve.assert_called_once_with(
                str(executable), [str(executable), "--help"], {"A": "B"}
            )


if __name__ == "__main__":
    unittest.main()
