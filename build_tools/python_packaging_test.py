# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools import python_packaging


class PythonPackagingTest(unittest.TestCase):
    def test_package_version_defaults_to_project_version(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(python_packaging.package_version(), "0.1.0")

    def test_package_version_accepts_ci_override(self):
        with mock.patch.dict(
            os.environ, {"HRX_DEMOS_PACKAGE_VERSION": "0.2.0.dev1"}, clear=True
        ):
            self.assertEqual(python_packaging.package_version(), "0.2.0.dev1")

    def test_reads_last_repo_env_value(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            bazelrc = Path(temporary_directory) / ".bazelrc.local"
            bazelrc.write_text(
                "common --repo_env=IREE_ROCM_PATH=/first\n"
                "common --repo_env=CC='/path with spaces/clang'\n"
                "common --repo_env=IREE_ROCM_PATH=/second\n",
                encoding="utf-8",
            )
            self.assertEqual(
                python_packaging.read_bazel_repo_env(bazelrc),
                {
                    "IREE_ROCM_PATH": "/second",
                    "CC": "/path with spaces/clang",
                },
            )

    def test_stages_executables_rocm_layout_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rocm_root = root / "rocm"
            hsa_versioned = rocm_root / "lib" / "libhsa-runtime64.so.1.2.3"
            hsa_versioned.parent.mkdir(parents=True)
            hsa_versioned.write_bytes(b"hsa")
            (rocm_root / "lib" / "libhsa-runtime64.so.1").symlink_to(
                hsa_versioned.name
            )
            sysdep = rocm_root / "lib" / "rocm_sysdeps" / "lib" / "libdep.so.1"
            sysdep.parent.mkdir(parents=True)
            sysdep.write_bytes(b"dep")
            license_path = rocm_root / "share" / "doc" / "rocr" / "LICENSE.md"
            license_path.parent.mkdir(parents=True)
            license_path.write_text("license", encoding="utf-8")

            id4 = root / "id4"
            info = root / "info"
            id4.write_bytes(b"id4")
            info.write_bytes(b"info")
            native_root = root / "wheel" / "_native"
            outputs = python_packaging.stage_native_payload(
                native_root,
                {"hrx-id4": id4, "hrx-info": info},
                rocm_root,
            )

            self.assertTrue((native_root / "bin" / "hrx-id4").is_file())
            staged_hsa = native_root / "lib" / "libhsa-runtime64.so.1"
            self.assertTrue(staged_hsa.is_file())
            self.assertFalse(staged_hsa.is_symlink())
            self.assertEqual(staged_hsa.read_bytes(), b"hsa")
            self.assertTrue(
                (native_root / "lib" / "rocm_sysdeps" / "lib" / "libdep.so.1").is_file()
            )
            self.assertTrue((native_root / "licenses" / "rocr-LICENSE.md").is_file())
            self.assertIn(native_root / "manifest.json", outputs)
            manifest = json.loads(
                (native_root / "manifest.json").read_text(encoding="utf-8")
            )
            manifest_paths = {entry["path"] for entry in manifest["files"]}
            self.assertIn("bin/hrx-id4", manifest_paths)
            self.assertIn("lib/libhsa-runtime64.so.1", manifest_paths)


if __name__ == "__main__":
    unittest.main()
