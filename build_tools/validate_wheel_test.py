# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from build_tools import validate_wheel


class ValidateWheelTest(unittest.TestCase):
    def _write_wheel(self, root: Path, files: dict[str, bytes]) -> Path:
        wheel = root / "test.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, contents in files.items():
                archive.writestr(name, contents)
        return wheel

    def _valid_files(self) -> dict[str, bytes]:
        prefix = validate_wheel.NATIVE_LIBRARY_PREFIX
        return {
            "hrx_demos-0.1.0.dist-info/METADATA": (
                "Metadata-Version: 2.4\n"
                "Name: hrx-demos\n"
                "Version: 0.1.0\n"
                "Description-Content-Type: text/markdown\n"
                "\n"
                + validate_wheel.README_PATH.read_text(encoding="utf-8")
            ).encode(),
            prefix + "libhsa-runtime64.so.1": b"hsa",
            prefix + "libhsa-amd-aqlprofile64.so.1": b"aql",
            prefix + "librocprofiler-register.so.0": b"profiler",
            prefix + "rocm_sysdeps/lib/librocm_sysdeps_elf.so.1": b"elf",
        }

    def test_accepts_unique_soname_payload(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            wheel = self._write_wheel(Path(temporary_directory), self._valid_files())
            count, _ = validate_wheel.validate_wheel(wheel)
            self.assertEqual(count, 4)

    def test_rejects_duplicate_library_contents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            files = self._valid_files()
            prefix = validate_wheel.NATIVE_LIBRARY_PREFIX
            files[prefix + "rocm_sysdeps/lib/librocm_sysdeps_copy.so.1"] = b"elf"
            wheel = self._write_wheel(Path(temporary_directory), files)
            with self.assertRaisesRegex(RuntimeError, "duplicate native libraries"):
                validate_wheel.validate_wheel(wheel)

    def test_rejects_unstable_runtime_alias(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            files = self._valid_files()
            prefix = validate_wheel.NATIVE_LIBRARY_PREFIX
            files[prefix + "libhsa-runtime64.so.1.2.3"] = b"versioned"
            wheel = self._write_wheel(Path(temporary_directory), files)
            with self.assertRaisesRegex(RuntimeError, "unstable libhsa-runtime64"):
                validate_wheel.validate_wheel(wheel)

    def test_rejects_auditwheel_rocm_graft(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            files = self._valid_files()
            files["hrx_demos.libs/librocm_sysdeps_elf-deadbeef.so.1"] = b"copy"
            wheel = self._write_wheel(Path(temporary_directory), files)
            with self.assertRaisesRegex(RuntimeError, "auditwheel grafted"):
                validate_wheel.validate_wheel(wheel)

    def test_rejects_stale_long_description(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            files = self._valid_files()
            files["hrx_demos-0.1.0.dist-info/METADATA"] = (
                b"Metadata-Version: 2.4\n"
                b"Name: hrx-demos\n"
                b"Version: 0.1.0\n"
                b"Description-Content-Type: text/markdown\n"
                b"\n"
                b"stale README\n"
            )
            wheel = self._write_wheel(Path(temporary_directory), files)
            with self.assertRaisesRegex(RuntimeError, "does not match README.md"):
                validate_wheel.validate_wheel(wheel)


if __name__ == "__main__":
    unittest.main()
