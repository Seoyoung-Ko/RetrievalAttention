#!/usr/bin/env python3
"""Plot a binary decode-step x KV-position heatmap from a top-k KV trace."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def topk_csvs(path, all_sessions=False):
    input_path = Path(path)
    if not input_path.is_dir():
        return [input_path]

    paths = []
    for candidate in input_path.rglob("*_topk_kv.csv"):
        with candidate.open() as trace_file:
            if trace_file.readline() and trace_file.readline():
                paths.append(candidate)
    if not paths:
        return []
    paths.sort(key=lambda candidate: (candidate.stat().st_mtime_ns, str(candidate)))
    return paths if all_sessions else [paths[-1]]


def compact_position(position):
    if position >= 1024 * 1024:
        return f"{position / (1024 * 1024):g}M"
    if position >= 1024:
        return f"{position / 1024:g}K"
    return str(position)


def evenly_spaced_indices(length, maximum):
    if length <= maximum:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=int))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="*_topk_kv.csv file or trace directory")
    parser.add_argument("--output", required=True, help="output PNG path")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--request", type=int, default=0)
    parser.add_argument("--batch-group", type=int, default=0)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--top-k", type=int, default=None, help="Use the first k attention-ranked positions (cannot exceed the traced k)")
    parser.add_argument("--max-kv-pos", type=int, default=None, help="Exclusive upper KV-token-position bound")
    parser.add_argument("--bucket-size", type=int, default=1, help="KV positions per displayed row; a bucket is 1 when any selected token falls in it")
    parser.add_argument("--all-sessions", action="store_true", help="Combine every matching trace instead of using the newest one")
    args = parser.parse_args()

    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.max_kv_pos is not None and args.max_kv_pos <= 0:
        parser.error("--max-kv-pos must be positive")
    if args.bucket_size <= 0:
        parser.error("--bucket-size must be positive")

    csv_paths = topk_csvs(args.input, args.all_sessions)
    if not csv_paths:
        raise SystemExit(f"No non-empty *_topk_kv.csv files matched: {args.input}")
    print("reading " + ", ".join(str(path) for path in csv_paths))

    selections = {}
    position_spaces = set()
    selection_scopes = set()
    traced_top_ks = set()
    selected_heads = set()

    for csv_path in csv_paths:
        with csv_path.open(newline="") as input_file:
            reader = csv.DictReader(input_file)
            required = {
                "request_idx", "batch_group_idx", "layer", "decode_step",
                "query_head", "top_k", "selected_kv_positions",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise SystemExit(f"{csv_path}: missing CSV columns: {sorted(missing)}")

            for row in reader:
                if int(row["request_idx"]) != args.request:
                    continue
                if int(row["batch_group_idx"]) != args.batch_group:
                    continue
                if int(row["layer"]) != args.layer:
                    continue
                if args.task is not None and row.get("task") != args.task:
                    continue
                if args.head is not None and int(row["query_head"]) != args.head:
                    continue

                position_spaces.add(row.get("position_space") or "legacy_unknown")
                selection_scopes.add(row.get("selection_scope") or "legacy_unknown")
                traced_top_k = int(row["top_k"])
                traced_top_ks.add(traced_top_k)
                selected_heads.add(int(row["query_head"]))
                output_top_k = traced_top_k if args.top_k is None else args.top_k
                if output_top_k > traced_top_k:
                    raise SystemExit(
                        f"Requested --top-k {output_top_k}, but trace contains only top-{traced_top_k}."
                    )

                positions = [
                    int(value)
                    for value in row["selected_kv_positions"].split(";")[:output_top_k]
                    if value
                ]
                if args.max_kv_pos is not None:
                    positions = [position for position in positions if position < args.max_kv_pos]
                key = (
                    row.get("trace_session_id", ""),
                    int(row["decode_step"]),
                    int(row["query_head"]),
                )
                selections[key] = positions

    if not selections:
        raise SystemExit("No top-k rows matched the requested filters.")
    if position_spaces != {"token"}:
        raise SystemExit("Trace does not contain original token positions; rerun with the updated tracer.")
    if selection_scopes != {"full_context_attention_topk"}:
        raise SystemExit(f"Unsupported selection scope: {sorted(selection_scopes)}")
    if len(selected_heads) > 1:
        raise SystemExit(
            f"Matched query heads {sorted(selected_heads)}. Panel (b) represents one head; pass --head."
        )

    decode_steps = sorted({step for _, step, _ in selections})
    decode_steps = list(range(decode_steps[0], decode_steps[-1] + 1))
    observed_positions = [
        position
        for positions in selections.values()
        for position in positions
    ]
    if not observed_positions and args.max_kv_pos is None:
        raise SystemExit("No selected KV position remains after filtering.")
    kv_limit = args.max_kv_pos or (max(observed_positions) + 1)
    bucket_count = math.ceil(kv_limit / args.bucket_size)
    image_data = np.zeros((bucket_count, len(decode_steps)), dtype=np.uint8)

    step_index = {step: index for index, step in enumerate(decode_steps)}
    for (_, step, _), positions in selections.items():
        column = step_index[step]
        for position in positions:
            image_data[position // args.bucket_size, column] = 1

    output_top_k = args.top_k or min(traced_top_ks)
    figure, axis = plt.subplots(figsize=(12, 6.2))
    image = axis.imshow(
        image_data,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(-0.5, len(decode_steps) - 0.5, kv_limit, 0),
        cmap="viridis",
        norm=Normalize(vmin=0, vmax=1),
    )
    axis.set_title(
        f"Retrieved top-{output_top_k} KVs over time "
        f"(layer {args.layer}, head {next(iter(selected_heads))})"
    )
    axis.set_xlabel("Decoding step $t$")
    axis.set_ylabel("KV position (token index)")

    x_ticks = evenly_spaced_indices(len(decode_steps), 12)
    axis.set_xticks(x_ticks, [decode_steps[index] for index in x_ticks])
    y_ticks = np.linspace(0, kv_limit, 9, dtype=int)
    axis.set_yticks(y_ticks, [compact_position(position) for position in y_ticks])

    colorbar = figure.colorbar(image, ax=axis, pad=0.02, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["0", "1"])
    colorbar.set_label("Selected (in top-k)")
    figure.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(
        f"wrote {output_path} "
        f"({len(decode_steps)} steps x {bucket_count} KV buckets, bucket={args.bucket_size})"
    )


if __name__ == "__main__":
    main()
