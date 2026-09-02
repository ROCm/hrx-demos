#!/usr/bin/env python3
"""Remove explicit copies used only as tied-operation destinations in Low IR.

Prepared Low currently spells destructive WMMA updates as a copy followed by a
tied result. For hand-scheduled accumulator rings the copy is unnecessary when
the original SSA value has no later use: the operation can tie directly to that
value. This transformer preserves the original artifact and emits the direct-
tie experiment as a separate motif.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


COPY_RE = re.compile(
    r"^(?P<indent>\s*)%(?P<copy>[A-Za-z0-9_]+) = low\.copy %(?P<source>[A-Za-z0-9_]+) : "
    r"(?P<type>reg<amdgpu\.vgpr x8>) -> (?P=type)\n$"
)


def rewrite(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    removed = 0
    index = 0
    while index < len(lines):
        match = COPY_RE.match(lines[index])
        if match is None or index + 1 >= len(lines):
            output.append(lines[index])
            index += 1
            continue
        copy_name = match.group("copy")
        source_name = match.group("source")
        next_line = lines[index + 1]
        if f"%{copy_name}" not in next_line or "low.op<amdgpu.v_wmma_" not in next_line:
            output.append(lines[index])
            index += 1
            continue
        output.append(next_line.replace(f"%{copy_name}", f"%{source_name}"))
        removed += 1
        index += 2
    return "".join(output), removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rewritten, removed = rewrite(args.input.read_text())
    if removed == 0:
        raise RuntimeError("no tied-operation copies matched")
    args.output.write_text(rewritten)
    print(f"removed {removed} tied-operation copies")


if __name__ == "__main__":
    main()
