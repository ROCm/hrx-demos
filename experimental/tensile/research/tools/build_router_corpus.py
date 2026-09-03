#!/usr/bin/env python3
"""Build a deterministic bounded FP16 NN router/topology corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from analyze_tensile_solution_space import family_id, mechanism_schedule, schedule


def classify(m: int, n: int, k: int) -> str:
    if m >= 4 * n:
        return "m_tall"
    if n >= 4 * m:
        return "n_wide"
    if k <= min(m, n) // 4:
        return "shallow_k"
    if k >= 4 * max(m, n):
        return "deep_k"
    return "balanced"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logic", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--tile-m", type=int, required=True)
    parser.add_argument("--tile-n", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
    parser.add_argument("--anchor", action="append", default=[])
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--max-dimension", type=int, default=4096)
    parser.add_argument(
        "--align-router-cells",
        action="store_true",
        help="round router dimensions up to motif multiples while retaining source-row provenance",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.coverage <= 1 or args.limit < 1:
        parser.error("invalid coverage or limit")

    document = yaml.load(args.logic.read_bytes(), Loader=yaml.CSafeLoader)
    solutions = document[5]
    rows = document[7]
    by_index = {
        int(value.get("SolutionIndex", ordinal)): value
        for ordinal, value in enumerate(solutions)
    }
    candidates: list[dict[str, object]] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, list) or len(row) < 2 or not isinstance(row[0], list):
            continue
        size, choice = row[0], row[1]
        if len(size) < 4 or not isinstance(choice, list) or not choice:
            continue
        m, n, batch, k = (int(size[0]), int(size[1]), int(size[2]), int(size[3]))
        solution_index = int(choice[0])
        solution = by_index.get(solution_index)
        if solution is None or batch != 1:
            continue
        if max(m, n, k) > args.max_dimension:
            continue
        source_shape = [m, n, k]
        if args.align_router_cells:
            m = (m + args.tile_m - 1) // args.tile_m * args.tile_m
            n = (n + args.tile_n - 1) // args.tile_n * args.tile_n
            k = (k + args.tile_k - 1) // args.tile_k * args.tile_k
        elif m % args.tile_m or n % args.tile_n or k % args.tile_k:
            continue
        if k < 2 * args.tile_k:
            continue
        normalized = schedule(solution)
        mechanism = mechanism_schedule(normalized)
        candidates.append(
            {
                "row": ordinal,
                "source_shape": source_shape,
                "m": m,
                "n": n,
                "k": k,
                "solution": solution_index,
                "family": family_id(mechanism),
                "configuration": family_id(normalized),
                "geometry": classify(m, n, k),
                "mechanism": mechanism,
                "schedule": normalized,
            }
        )
    if not candidates:
        raise ValueError("no legal aligned interior rows")

    counts = Counter(str(item["family"]) for item in candidates)
    selected_families: list[str] = []
    covered = 0
    for identifier, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        selected_families.append(identifier)
        covered += count
        if covered / len(candidates) >= args.coverage:
            break

    selected: dict[tuple[int, int, int], dict[str, object]] = {}
    reasons: defaultdict[tuple[int, int, int], list[str]] = defaultdict(list)

    def add(item: dict[str, object], reason: str) -> None:
        key = (int(item["m"]), int(item["n"]), int(item["k"]))
        selected.setdefault(key, item)
        reasons[key].append(reason)

    for raw in args.anchor:
        values = tuple(int(value) for value in raw.lower().replace("x", ",").split(","))
        if len(values) != 3:
            raise ValueError(f"invalid anchor: {raw}")
        exact = next(
            (item for item in candidates if tuple(item[key] for key in ("m", "n", "k")) == values),
            None,
        )
        if exact is None:
            # Anchors need not be explicit router points. Associate the nearest
            # row only for topology metadata while retaining the exact request.
            exact = min(
                candidates,
                key=lambda item: sum(abs(int(item[key]) - value) for key, value in zip(("m", "n", "k"), values)),
            ).copy()
            exact.update({"m": values[0], "n": values[1], "k": values[2], "row": None})
        add(exact, "anchor")

    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for item in candidates:
        grouped[str(item["family"])].append(item)
    for identifier in selected_families:
        values = sorted(grouped[identifier], key=lambda item: (int(item["m"]) * int(item["n"]) * int(item["k"]), int(item["m"]), int(item["n"]), int(item["k"])))
        for position, label in ((0, "small"), (len(values) // 2, "median"), (-1, "large")):
            add(values[position], f"dominant_family_{label}")

    # Cover geometry strata with the most frequently occurring family first.
    family_rank = {value: ordinal for ordinal, value in enumerate(selected_families)}
    for geometry in ("balanced", "m_tall", "n_wide", "shallow_k", "deep_k"):
        values = [item for item in candidates if item["geometry"] == geometry]
        if values:
            values.sort(key=lambda item: (family_rank.get(str(item["family"]), 9999), int(item["m"]) * int(item["n"]) * int(item["k"]), int(item["m"]), int(item["n"]), int(item["k"])))
            add(values[len(values) // 2], f"geometry_{geometry}")

    # Sample transitions in sorted router order on each side of a family change.
    ordered = sorted(candidates, key=lambda item: (int(item["m"]), int(item["n"]), int(item["k"]), int(item["row"])))
    for left, right in zip(ordered, ordered[1:]):
        if left["family"] != right["family"]:
            add(left, "router_family_boundary_left")
            add(right, "router_family_boundary_right")
        if len(selected) >= args.limit:
            break

    reason_rank = {
        "anchor": 0,
        "dominant_family_median": 1,
        "dominant_family_small": 2,
        "dominant_family_large": 3,
    }
    ordered_selected = sorted(
        selected.items(),
        key=lambda pair: (
            min(reason_rank.get(reason, 4) for reason in reasons[pair[0]]),
            pair[0],
        ),
    )
    cells = []
    for key, item in ordered_selected:
        value = {name: item[name] for name in ("m", "n", "k", "row", "source_shape", "solution", "family", "configuration", "geometry", "mechanism", "schedule") if name in item}
        value["reasons"] = sorted(set(reasons[key]))
        cells.append(value)
    cells = cells[: args.limit]
    output = {
        "schema": "loom-blas.router-corpus.v1",
        "target": args.target,
        "vendor_backend": args.backend,
        "semantic_scope": "FP16 NN, FP32 accumulation, FP16 output, alpha=1, beta=0, no bias",
        "weighting": "unweighted; router-row frequency is topology evidence, not demand weight",
        "source": str(args.logic),
        "source_sha256": hashlib.sha256(args.logic.read_bytes()).hexdigest(),
        "legal_aligned_router_rows": len(candidates),
        "dominant_family_coverage": covered / len(candidates),
        "dominant_families": [
            {"id": value, "rows": counts[value], "fraction": counts[value] / len(candidates)}
            for value in selected_families
        ],
        "cell_count": len(cells),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if args.output.stat().st_size == 0:
        raise RuntimeError("empty output")


if __name__ == "__main__":
    main()
