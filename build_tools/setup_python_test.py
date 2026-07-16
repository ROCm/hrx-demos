# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_tools import setup_python


class SetupPythonTest(unittest.TestCase):
    def test_generated_bazelrc_is_compatible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rocm_root = root / "rocm"
            rocm_root.mkdir()
            cc = root / "clang"
            cxx = root / "clang++"
            cc.write_text("", encoding="utf-8")
            cxx.write_text("", encoding="utf-8")
            values = setup_python.configured_values(rocm_root, cc, cxx)
            bazelrc = root / ".bazelrc.local"
            setup_python.write_bazelrc(
                bazelrc, setup_python.render_bazelrc(values)
            )
            self.assertTrue(setup_python.bazelrc_is_compatible(bazelrc, values))

    def test_different_rocm_root_is_incompatible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            cc = root / "clang"
            cxx = root / "clang++"
            cc.write_text("", encoding="utf-8")
            cxx.write_text("", encoding="utf-8")
            bazelrc = root / ".bazelrc.local"
            first_values = setup_python.configured_values(first_root, cc, cxx)
            setup_python.write_bazelrc(
                bazelrc, setup_python.render_bazelrc(first_values)
            )
            second_values = setup_python.configured_values(second_root, cc, cxx)
            self.assertFalse(
                setup_python.bazelrc_is_compatible(bazelrc, second_values)
            )


if __name__ == "__main__":
    unittest.main()
