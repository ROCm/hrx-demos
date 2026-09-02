#!/usr/bin/env python3
"""Release the ABI-fixed packed workitem register before the GEMM loop.

The incumbent moves packed_tid out of v0 before reusing v0:v47 for WMMA
accumulators. Prepared Low otherwise keeps the ABI live-in SSA value alive
through the output epilogue and makes that legal register reuse impossible.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def rewrite(text: str) -> str:
    anchor = (
        "  %packed_tid = low.live_in<amdgpu.workitem_id.packed.xy> : "
        "reg<amdgpu.vgpr>"
    )
    if text.count(anchor) != 1:
        raise ValueError("expected exactly one packed workitem live-in")
    saved = (
        anchor
        + "\n  %packed_tid_saved = low.copy %packed_tid : "
        "reg<amdgpu.vgpr> -> reg<amdgpu.vgpr>"
    )
    text = text.replace(anchor, saved, 1)
    head, tail = text.split(saved, 1)
    return head + saved + tail.replace("%packed_tid", "%packed_tid_saved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(rewrite(args.input.read_text()))


if __name__ == "__main__":
    main()
