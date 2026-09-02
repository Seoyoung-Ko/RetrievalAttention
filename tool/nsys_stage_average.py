#!/usr/bin/env python3
"""Summarize nested RetroInfer NVTX ranges from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"decode_step_(\d+)$")
LAYER_RE = re.compile(r"layer_(\d+)$")


def stats(values: list[float]) -> tuple[float, float, float, float]:
    return (
        statistics.fmean(values),
        statistics.pstdev(values),
        min(values),
        max(values),
    )


def rounded(value: float) -> float:
    return round(value, 6)


def contained(child: dict, parent: dict) -> bool:
    return child["start"] >= parent["start"] and child["end"] <= parent["end"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path, help="SQLite file exported by nsys")
    parser.add_argument("output_prefix", type=Path, help="Output path prefix")
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite)
    rows = connection.execute(
        """
        SELECT n.start, n.end, COALESCE(n.text, s.value) AS name
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON s.id = n.textId
        WHERE n.end IS NOT NULL
        ORDER BY n.start
        """
    ).fetchall()
    connection.close()

    events = [
        {"start": start, "end": end, "name": name, "ms": (end - start) / 1e6}
        for start, end, name in rows
    ]
    tokens = [event for event in events if TOKEN_RE.fullmatch(event["name"])]
    if not tokens:
        raise SystemExit("No decode_step_<N> NVTX ranges found")

    per_token: dict[int, dict[tuple[str, str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_layer: dict[tuple[int, str, str], list[float]] = defaultdict(list)

    for token in tokens:
        token_id = int(TOKEN_RE.fullmatch(token["name"]).group(1))
        token_events = [event for event in events if event is not token and contained(event, token)]
        per_token[token_id][("token", "token_wall", "wall")].append(token["ms"])

        layers = [event for event in token_events if LAYER_RE.fullmatch(event["name"])]
        for layer in layers:
            layer_id = int(LAYER_RE.fullmatch(layer["name"]).group(1))
            per_token[token_id][("all_layers", "layer_wall", "wall")].append(layer["ms"])
            per_layer[(layer_id, "layer_wall", "wall")].append(layer["ms"])

            layer_events = [
                event
                for event in token_events
                if event is not layer and contained(event, layer)
            ]
            for event in layer_events:
                name = event["name"]
                if name.startswith("stage/"):
                    stage = name.removeprefix("stage/")
                    category = "layer_stage"
                elif name.startswith("ri/"):
                    stage = "decode_attention/" + name.removeprefix("ri/")
                    category = "decode_substage"
                else:
                    continue
                per_token[token_id][("all_layers", stage, category)].append(event["ms"])
                per_layer[(layer_id, stage, category)].append(event["ms"])

        # Stages outside layer ranges (embedding, final norm, LM head, sampling).
        for event in token_events:
            if not event["name"].startswith("stage/"):
                continue
            if any(contained(event, layer) for layer in layers):
                continue
            stage = event["name"].removeprefix("stage/")
            per_token[token_id][("token", stage, "token_stage")].append(event["ms"])

    # Convert each token's repeated layer calls into a per-token total, then
    # average those totals. This weights every generated token equally.
    keys = sorted({key for values in per_token.values() for key in values})
    summary_rows = []
    for scope, stage, category in keys:
        token_totals = [sum(per_token[token_id].get((scope, stage, category), [])) for token_id in sorted(per_token)]
        nonempty_counts = [
            len(per_token[token_id].get((scope, stage, category), []))
            for token_id in sorted(per_token)
        ]
        mean_total, std_total, min_total, max_total = stats(token_totals)
        total_calls = sum(nonempty_counts)
        summary_rows.append(
            {
                "scope": scope,
                "stage": stage,
                "category": category,
                "profiled_tokens": len(tokens),
                "calls_per_token": rounded(total_calls / len(tokens)),
                "avg_ms_per_call": rounded(sum(token_totals) / total_calls),
                "avg_total_ms_per_token": rounded(mean_total),
                "stddev_total_ms_per_token": rounded(std_total),
                "min_total_ms_per_token": rounded(min_total),
                "max_total_ms_per_token": rounded(max_total),
            }
        )

    layer_rows = []
    for (layer, stage, category), values in sorted(per_layer.items()):
        mean, stddev, minimum, maximum = stats(values)
        layer_rows.append(
            {
                "layer": layer,
                "stage": stage,
                "category": category,
                "profiled_tokens": len(tokens),
                "calls": len(values),
                "avg_ms": rounded(mean),
                "stddev_ms": rounded(stddev),
                "min_ms": rounded(minimum),
                "max_ms": rounded(maximum),
            }
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    token_path = args.output_prefix.with_name(args.output_prefix.name + "_per_token.csv")
    layer_path = args.output_prefix.with_name(args.output_prefix.name + "_per_layer.csv")
    with token_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with layer_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)

    print(token_path)
    print(layer_path)


if __name__ == "__main__":
    main()
