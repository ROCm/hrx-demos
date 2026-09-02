#!/usr/bin/env python3
"""Extract a bounded AMDGPU instruction schedule from an ELF code object."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


INSTRUCTION_RE = re.compile(
    r"^\s*(?P<mnemonic>[^\s:]+)(?:\s+(?P<operands>.*?))?\s*//\s+"
    r"(?P<address>[0-9A-Fa-f]+):"
)


@dataclass(frozen=True)
class Instruction:
    address: int
    mnemonic: str
    operands: str


def category(mnemonic: str) -> str:
    if mnemonic.startswith(("v_wmma", "v_mfma", "v_swmmac", "v_dot")):
        return "matrix"
    if mnemonic.startswith(("buffer_load", "global_load", "flat_load")):
        return "global_load"
    if mnemonic.startswith(("buffer_store", "global_store", "flat_store")):
        return "global_store"
    if mnemonic.startswith(("ds_load", "ds_read")):
        return "lds_load"
    if mnemonic.startswith(("ds_store", "ds_write")):
        return "lds_store"
    if mnemonic.startswith("s_wait"):
        return "wait"
    if mnemonic.startswith("s_barrier"):
        return "barrier"
    if mnemonic.startswith(("s_branch", "s_cbranch")):
        return "branch"
    return "other"


def parse_address(value: str) -> int:
    return int(value, 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit a normalized schedule for a half-open address interval"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--start", required=True, type=parse_address)
    parser.add_argument("--stop", required=True, type=parse_address)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--llvm-objdump",
        type=Path,
        default=Path.home() / "rocm/llvm/bin/llvm-objdump",
    )
    args = parser.parse_args()

    if args.stop <= args.start:
        raise ValueError("--stop must be greater than --start")
    command = [
        str(args.llvm_objdump),
        "--disassemble",
        f"--start-address={args.start}",
        f"--stop-address={args.stop}",
        str(args.input),
    ]
    disassembly = subprocess.run(
        command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ).stdout

    instructions: list[Instruction] = []
    for line in disassembly.splitlines():
        match = INSTRUCTION_RE.match(line)
        if not match:
            continue
        address = int(match.group("address"), 16)
        if args.start <= address < args.stop:
            instructions.append(
                Instruction(
                    address=address,
                    mnemonic=match.group("mnemonic"),
                    operands=(match.group("operands") or "").strip(),
                )
            )
    if not instructions:
        raise ValueError("no instructions found in interval")

    categories = [category(item.mnemonic) for item in instructions]
    significant = [
        {
            "ordinal": ordinal,
            "byte_offset": item.address - args.start,
            "category": item_category,
            "mnemonic": item.mnemonic,
            "operands": item.operands,
        }
        for ordinal, (item, item_category) in enumerate(zip(instructions, categories))
        if item_category != "other"
    ]
    result = {
        "schema": "loom-blas.amdgpu-interval.v1",
        "label": args.label,
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "bounds": {"start": hex(args.start), "stop": hex(args.stop)},
        "instruction_count": len(instructions),
        "category_counts": dict(sorted(Counter(categories).items())),
        "significant_schedule": significant,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
