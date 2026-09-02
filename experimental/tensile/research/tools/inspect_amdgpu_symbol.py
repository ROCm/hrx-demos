#!/usr/bin/env python3
"""Bound and summarize one AMDGPU kernel in a size-zero-symbol HSACO."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Options:
    input_path: Path
    symbol: str
    output_path: Path | None
    llvm_nm: Path
    llvm_objdump: Path


@dataclass(frozen=True)
class Symbol:
    address: int
    kind: str
    name: str


def parse_options() -> Options:
    default_bin = Path.home() / "rocm" / "llvm" / "bin"
    parser = argparse.ArgumentParser(
        description=(
            "Summarize one AMDGPU kernel, using the next external text symbol "
            "to bound code objects whose ELF function sizes are zero"
        )
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--llvm-nm", type=Path, default=default_bin / "llvm-nm")
    parser.add_argument(
        "--llvm-objdump", type=Path, default=default_bin / "llvm-objdump"
    )
    args = parser.parse_args()
    return Options(
        input_path=args.input_path,
        symbol=args.symbol,
        output_path=args.output,
        llvm_nm=args.llvm_nm,
        llvm_objdump=args.llvm_objdump,
    )


def run(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout


def load_external_symbols(options: Options) -> list[Symbol]:
    output = run(
        [
            str(options.llvm_nm),
            "-n",
            "--defined-only",
            "--extern-only",
            str(options.input_path),
        ]
    )
    symbols: list[Symbol] = []
    for line in output.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        address_text, kind, name = fields
        try:
            address = int(address_text, 16)
        except ValueError:
            continue
        symbols.append(Symbol(address=address, kind=kind, name=name))
    return symbols


def kernel_bounds(symbols: list[Symbol], name: str) -> tuple[int, int]:
    matches = [item for item in symbols if item.kind == "T" and item.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one external text symbol named {name!r}; got {len(matches)}")
    start = matches[0].address
    following = sorted(
        {item.address for item in symbols if item.kind == "T" and item.address > start}
    )
    if not following:
        raise ValueError("cannot bound the final text symbol without an ELF symbol size")
    return start, following[0]


INSTRUCTION_RE = re.compile(
    r"^\s*(?P<mnemonic>[^\s:]+)(?:\s+.*?)?\s*//\s+[0-9A-Fa-f]+:"
)
TARGET_RE = re.compile(r'^\s*\.amdgcn_target\s+"(?P<target>[^"]+)"')


def instruction_categories(histogram: Counter[str]) -> dict[str, int]:
    categories = {
        "accelerated_multiply": ("v_wmma", "v_mfma", "v_dot"),
        "global_load": ("buffer_load", "global_load", "flat_load"),
        "global_store": ("buffer_store", "global_store", "flat_store"),
        "lds_load": ("ds_load", "ds_read"),
        "lds_store": ("ds_store", "ds_write"),
        "lane_exchange": ("ds_bpermute", "ds_swizzle", "v_permlane"),
        "barrier": ("s_barrier",),
        "wait": ("s_wait",),
        "branch": ("s_branch", "s_cbranch"),
    }
    return {
        category: sum(
            count
            for mnemonic, count in histogram.items()
            if mnemonic.startswith(prefixes)
        )
        for category, prefixes in categories.items()
    }


def summarize(options: Options) -> dict[str, object]:
    encoded = options.input_path.read_bytes()
    symbols = load_external_symbols(options)
    start, stop = kernel_bounds(symbols, options.symbol)
    disassembly = run(
        [
            str(options.llvm_objdump),
            "--disassemble",
            f"--start-address={start}",
            f"--stop-address={stop}",
            str(options.input_path),
        ]
    )

    histogram: Counter[str] = Counter()
    target: str | None = None
    labels = 0
    for line in disassembly.splitlines():
        target_match = TARGET_RE.match(line)
        if target_match:
            target = target_match.group("target")
        instruction_match = INSTRUCTION_RE.match(line)
        if instruction_match:
            histogram[instruction_match.group("mnemonic")] += 1
        elif re.match(r"^[0-9A-Fa-f]+ <[^>]+>:$", line):
            labels += 1

    if not histogram:
        raise ValueError("objdump produced no recognized instructions")
    if target is None:
        raise ValueError("objdump did not report an AMDGPU target")

    ordered_histogram = dict(
        sorted(histogram.items(), key=lambda item: (-item[1], item[0]))
    )
    return {
        "schema": "loom-blas.amdgpu-symbol.v1",
        "source": str(options.input_path.resolve()),
        "encoded_size": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "target": target,
        "symbol": options.symbol,
        "bounds": {
            "start": f"0x{start:x}",
            "stop": f"0x{stop:x}",
            "span_bytes": stop - start,
            "method": "next_external_text_symbol",
        },
        "instruction_count": sum(histogram.values()),
        "label_count": labels,
        "instruction_categories": instruction_categories(histogram),
        "instruction_histogram": ordered_histogram,
    }


def main() -> None:
    options = parse_options()
    output = json.dumps(summarize(options), indent=2, sort_keys=True) + "\n"
    if options.output_path is None:
        print(output, end="")
    else:
        options.output_path.parent.mkdir(parents=True, exist_ok=True)
        options.output_path.write_text(output)


if __name__ == "__main__":
    main()
