#!/usr/bin/env python3
"""Replace the scalar fp16 GEMM epilogue with StoreVectorWidth=4 packets.

The gfx12 WMMA accumulator mapping is not linear in accumulator number.  For
each element index, accumulator groups (q, q+4, q+8, q+12) describe four
adjacent rows.  Preserve the existing address calculation and pack those
groups into one 64-bit store, matching the incumbent's advertised SVW=4.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


STORE_RE = re.compile(
    r"  %native_d_a(?P<a>\d+)_e(?P<e>\d+) = low\.slice "
    r"%tail31_compute_y(?P=a)\[(?P=e)\] : reg<amdgpu\.vgpr x8> -> reg<amdgpu\.vgpr>\n"
    r"  %native_d_a(?P=a)_e(?P=e)_f16 = low\.op<amdgpu\.v_cvt_f16_f32>"
    r"\(%native_d_a(?P=a)_e(?P=e)\) : \(reg<amdgpu\.vgpr>\) -> reg<amdgpu\.vgpr>\n"
    r"  low\.op<amdgpu\.global_store_b16_saddr>"
    r"\(%native_d_base, %native_d_a(?P=a)_e(?P=e)_f16, %d\) "
    r"\{offset = (?P<offset>\d+)\} : "
    r"\(reg<amdgpu\.vgpr>, reg<amdgpu\.vgpr>, reg<amdgpu\.sgpr x2>\)\n"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text()
    matches = list(STORE_RE.finditer(source))
    if len(matches) != 128:
        raise ValueError(f"expected 128 scalar stores, found {len(matches)}")

    records: dict[tuple[int, int], tuple[int, str]] = {}
    for match in matches:
        a = int(match.group("a"))
        e = int(match.group("e"))
        records[a, e] = (int(match.group("offset")), match.group(0))

    lines: list[str] = []
    for q in range(4):
        accumulators = (q, q + 4, q + 8, q + 12)
        for e in range(8):
            offsets = [records[a, e][0] for a in accumulators]
            if offsets != list(range(offsets[0], offsets[0] + 8, 2)):
                raise ValueError(
                    f"non-contiguous output group q={q} e={e}: {offsets}"
                )
            stem = f"native_d_q{q}_e{e}"
            for lane, a in enumerate(accumulators):
                lines.extend(
                    (
                        f"  %{stem}_v{lane} = low.slice %tail31_compute_y{a}[{e}] : "
                        "reg<amdgpu.vgpr x8> -> reg<amdgpu.vgpr>",
                        f"  %{stem}_h{lane} = low.op<amdgpu.v_cvt_f16_f32>(%{stem}_v{lane}) : "
                        "(reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                    )
                )
            lines.extend(
                (
                    f"  %{stem}_p01 = low.op<amdgpu.v_pack_b32_f16>(%{stem}_h0, %{stem}_h1) : "
                    "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                    f"  %{stem}_p23 = low.op<amdgpu.v_pack_b32_f16>(%{stem}_h2, %{stem}_h3) : "
                    "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>",
                    f"  %{stem}_packed = low.concat(%{stem}_p01, %{stem}_p23) : "
                    "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr x2>",
                    f"  low.op<amdgpu.global_store_b64_saddr>(%native_d_base, %{stem}_packed, %d) "
                    f"{{offset = {offsets[0]}}} : "
                    "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr x2>, reg<amdgpu.sgpr x2>)",
                )
            )

    begin = matches[0].start()
    end = matches[-1].end()
    output = source[:begin] + "\n".join(lines) + "\n" + source[end:]
    if "global_store_b16_saddr" in output:
        raise ValueError("scalar output stores remain")
    args.output.write_text(output)


if __name__ == "__main__":
    main()
