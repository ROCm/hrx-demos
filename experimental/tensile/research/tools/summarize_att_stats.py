#!/usr/bin/env python3
"""Reduce rocprof-compute ATT instruction statistics to reviewable JSON."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Totals:
    static_instructions: int = 0
    hitcount: int = 0
    latency: int = 0
    stall: int = 0
    idle: int = 0

    def add(self, hitcount: int, latency: int, stall: int, idle: int) -> None:
        self.static_instructions += 1
        self.hitcount += hitcount
        self.latency += latency
        self.stall += stall
        self.idle += idle


def category(instruction: str) -> str:
    mnemonic = instruction.split(maxsplit=1)[0] if instruction else ""
    if mnemonic.startswith("v_wmma") or mnemonic.startswith("v_mfma"):
        return "matrix"
    if mnemonic.startswith("v_perm"):
        return "permute"
    if mnemonic.startswith("ds_read") or mnemonic.startswith("ds_load"):
        return "lds_read"
    if mnemonic.startswith("ds_write") or mnemonic.startswith("ds_store"):
        return "lds_write"
    if mnemonic.startswith("buffer_load") or mnemonic.startswith("global_load"):
        return "global_load"
    if mnemonic.startswith("buffer_store") or mnemonic.startswith("global_store"):
        return "global_store"
    if mnemonic in {
        "s_waitcnt",
        "s_waitcnt_vscnt",
        "s_wait_loadcnt",
        "s_wait_dscnt",
        "s_wait_kmcnt",
    }:
        return "wait"
    if mnemonic == "s_barrier":
        return "barrier"
    if mnemonic in {"s_nop", "s_delay_alu"}:
        return "delay"
    if mnemonic.startswith("s_branch") or mnemonic.startswith("s_cbranch"):
        return "branch"
    if mnemonic.startswith("v_"):
        return "vector_other"
    if mnemonic.startswith("s_"):
        return "scalar_other"
    return "other"


def summarize(path: Path, label: str) -> dict[str, object]:
    totals = Totals()
    categories: dict[str, Totals] = {}
    hot: list[dict[str, object]] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            instruction = row["Instruction"].strip()
            if not instruction or instruction.startswith(";"):
                continue
            hitcount = int(row["Hitcount"])
            latency = int(row["Latency"])
            stall = int(row["Stall"])
            idle = int(row["Idle"])
            totals.add(hitcount, latency, stall, idle)
            kind = category(instruction)
            categories.setdefault(kind, Totals()).add(
                hitcount, latency, stall, idle
            )
            hot.append(
                {
                    "address": int(row["Vaddr"]),
                    "instruction": instruction,
                    "hitcount": hitcount,
                    "latency": latency,
                    "stall": stall,
                    "idle": idle,
                }
            )
    hot.sort(key=lambda item: int(item["stall"]), reverse=True)
    return {
        "label": label,
        "source": str(path),
        "totals": asdict(totals),
        "categories": {
            name: asdict(value) for name, value in sorted(categories.items())
        },
        "highest_stall_instructions": hot[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="ATT stats CSV files produced by rocprof-compute",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Stable label corresponding to each input",
    )
    args = parser.parse_args()
    if len(args.input) != len(args.label):
        parser.error("provide exactly one --label per input")
    for path in args.input:
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"input must be a nonempty file: {path}")

    document = {
        "schema": "loom-blas.att-instruction-summary.v1",
        "captures": [
            summarize(path, label)
            for path, label in zip(args.input, args.label, strict=True)
        ],
    }
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
