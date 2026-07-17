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

    def test_preserves_llvm_readelf_multicall_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = Path(temporary_directory)
            binary_dir = rocm_root / "lib" / "llvm" / "bin"
            binary_dir.mkdir(parents=True)
            readobj = binary_dir / "llvm-readobj"
            readobj.write_bytes(b"")
            readobj.chmod(0o755)
            readelf = binary_dir / "llvm-readelf"
            readelf.symlink_to(readobj.name)

            self.assertEqual(python_packaging._find_llvm_readelf(rocm_root), readelf)

    def test_parses_needed_libraries(self):
        output = """
  0x0000000000000001 (NEEDED) Shared library: [libone.so.1]
  0x0000000000000001 (NEEDED) Shared library: [libtwo.so.2]
"""
        with mock.patch.object(
            python_packaging.subprocess,
            "run",
            return_value=python_packaging.subprocess.CompletedProcess(
                [], 0, stdout=output, stderr=""
            ),
        ):
            self.assertEqual(
                python_packaging._read_needed_libraries(
                    Path("llvm-readelf"), Path("library.so")
                ),
                ("libone.so.1", "libtwo.so.2"),
            )

    def test_stages_only_rocm_runtime_dependency_closure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rocm_root = root / "rocm"
            hsa_versioned = rocm_root / "lib" / "libhsa-runtime64.so.1.2.3"
            hsa_versioned.parent.mkdir(parents=True)
            hsa_versioned.write_bytes(b"hsa")
            (rocm_root / "lib" / "libhsa-runtime64.so.1").symlink_to(hsa_versioned.name)
            aql_versioned = rocm_root / "lib" / "libhsa-amd-aqlprofile64.so.1.0.0"
            aql_versioned.write_bytes(b"aql")
            (rocm_root / "lib" / "libhsa-amd-aqlprofile64.so.1").symlink_to(
                aql_versioned.name
            )
            profiler = rocm_root / "lib" / "librocprofiler-register.so.0"
            profiler.write_bytes(b"profiler")
            sysdep = (
                rocm_root / "lib" / "rocm_sysdeps" / "lib" / "librocm_sysdeps_dep.so.1"
            )
            sysdep.parent.mkdir(parents=True)
            sysdep.write_bytes(b"dep")
            unrelated = sysdep.parent / "librocm_sysdeps_unrelated.so.1"
            unrelated.write_bytes(b"unrelated")
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
                needed_libraries=lambda path: {
                    "libhsa-runtime64.so.1.2.3": (
                        "librocprofiler-register.so.0",
                        "librocm_sysdeps_dep.so.1",
                        "libc.so.6",
                    ),
                }.get(path.name, ()),
            )

            self.assertTrue((native_root / "bin" / "hrx-id4").is_file())
            staged_hsa = native_root / "lib" / "libhsa-runtime64.so.1"
            self.assertTrue(staged_hsa.is_file())
            self.assertFalse(staged_hsa.is_symlink())
            self.assertEqual(staged_hsa.read_bytes(), b"hsa")
            self.assertTrue(
                (
                    native_root
                    / "lib"
                    / "rocm_sysdeps"
                    / "lib"
                    / "librocm_sysdeps_dep.so.1"
                ).is_file()
            )
            self.assertFalse((native_root / "lib" / hsa_versioned.name).exists())
            self.assertFalse((native_root / "lib" / aql_versioned.name).exists())
            self.assertFalse(
                (native_root / "lib" / "rocm_sysdeps" / "lib" / unrelated.name).exists()
            )
            self.assertTrue((native_root / "licenses" / "rocr-LICENSE.md").is_file())
            self.assertIn(native_root / "manifest.json", outputs)
            manifest = json.loads(
                (native_root / "manifest.json").read_text(encoding="utf-8")
            )
            manifest_paths = {entry["path"] for entry in manifest["files"]}
            self.assertIn("bin/hrx-id4", manifest_paths)
            self.assertIn("lib/libhsa-runtime64.so.1", manifest_paths)

    def test_rejects_missing_rocm_dependency(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            rocm_root = Path(temporary_directory)
            library_dir = rocm_root / "lib"
            library_dir.mkdir(parents=True)
            (library_dir / "libhsa-runtime64.so.1").write_bytes(b"hsa")
            (library_dir / "libhsa-amd-aqlprofile64.so.1").write_bytes(b"aql")

            with self.assertRaisesRegex(RuntimeError, "librocm_missing.so.1"):
                python_packaging.resolve_rocm_runtime_payload(
                    rocm_root,
                    needed_libraries=lambda path: (
                        ("librocm_missing.so.1",)
                        if path.name == "libhsa-runtime64.so.1"
                        else ()
                    ),
                )


if __name__ == "__main__":
    unittest.main()
