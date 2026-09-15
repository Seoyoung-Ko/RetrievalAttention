#!/usr/bin/env python3
"""Plot decode-step x original KV-token-position attention weights."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def attention_csvs(path, all_sessions=False):
    input_path = Path(path)
    if not input_path.is_dir():
        return [input_path]

    paths = []
    for candidate in input_path.rglob("*_attention.csv"):
        with candidate.open() as trace_file:
            if trace_file.readline() and trace_file.readline():
                paths.append(candidate)
    if not paths:
        return []
    paths.sort(key=lambda candidate: (candidate.stat().st_mtime_ns, str(candidate)))
    # A trace directory commonly contains retries and traces made with older
    # schemas. Mixing them creates fictitious steps/heads and double-buckets
    # positions. The most recent run is the useful default.
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
    parser.add_argument("--input", required=True, help="*_attention.csv file or directory")
    parser.add_argument("--output", required=True, help="output PNG path")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--request", type=int, default=None)
    parser.add_argument("--head", type=int, default=None, help="Query-head index to plot. If unset, average across traced heads.")
    parser.add_argument("--task", default=None)
    parser.add_argument("--max-kv-pos", type=int, default=None, help="Exclusive upper bound of the KV token positions to visualize")
    parser.add_argument("--bucket-size", type=int, default=None, help="Output KV bucket size. Defaults to the bucket size stored in the trace.")
    parser.add_argument("--vmin", type=float, default=None, help="Log color-scale minimum (automatic by default)")
    parser.add_argument("--vmax", type=float, default=None, help="Log color-scale maximum (automatic by default)")
    parser.add_argument("--all-sessions", action="store_true", help="Combine every trace file under an input directory (normally undesirable)")
    args = parser.parse_args()

    if args.bucket_size is not None and args.bucket_size <= 0:
        parser.error("--bucket-size must be positive")
    if args.max_kv_pos is not None and args.max_kv_pos <= 0:
        parser.error("--max-kv-pos must be positive")

    csv_paths = attention_csvs(args.input, args.all_sessions)
    if not csv_paths:
        raise SystemExit(f"No *_attention.csv files matched: {args.input}")
    print("reading " + ", ".join(str(path) for path in csv_paths))

    # First sum token/source-bucket weights into an output bucket for each
    # individual traced head. Only then average heads; averaging raw token
    # rows would incorrectly divide a bucket's attention by its token count.
    per_series = defaultdict(float)
    series = set()
    source_bucket_sizes = set()
    position_spaces = set()
    attention_scopes = set()
    output_bucket_size = args.bucket_size

    for csv_path in csv_paths:
        with csv_path.open(newline="") as input_file:
            reader = csv.DictReader(input_file)
            required = {"layer", "request_idx", "decode_step", "kv_position", "query_head", "attention_weight"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise SystemExit(f"{csv_path}: missing CSV columns: {sorted(missing)}")

            for row in reader:
                if args.layer is not None and int(row["layer"]) != args.layer:
                    continue
                if args.request is not None and int(row["request_idx"]) != args.request:
                    continue
                if args.task is not None and row.get("task") != args.task:
                    continue
                if args.head is not None and int(row["query_head"]) != args.head:
                    continue

                source_bucket_size = int(row.get("kv_bucket_size") or 1)
                source_bucket_sizes.add(source_bucket_size)
                position_spaces.add(row.get("position_space") or "legacy_unknown")
                attention_scopes.add(row.get("attention_scope") or "legacy_unknown")
                if output_bucket_size is None:
                    output_bucket_size = source_bucket_size
                if output_bucket_size < source_bucket_size or output_bucket_size % source_bucket_size:
                    raise SystemExit(
                        f"Cannot convert stored bucket size {source_bucket_size} to requested "
                        f"bucket size {output_bucket_size}; choose an equal or integer-multiple size."
                    )

                kv_position = int(row["kv_position"])
                if args.max_kv_pos is not None and kv_position >= args.max_kv_pos:
                    continue
                step = int(row["decode_step"])
                bucket = kv_position // output_bucket_size
                series_id = (
                    row.get("trace_session_id", ""),
                    int(row["request_idx"]),
                    int(row.get("batch_group_idx") or 0),
                    int(row["query_head"]),
                )
                series.add(series_id)
                per_series[(step, bucket, series_id)] += float(row["attention_weight"])

    if not per_series:
        raise SystemExit("No attention-weight rows matched the requested filters.")
    if position_spaces != {"token"} or attention_scopes != {"full_context"}:
        raise SystemExit(
            "This is a legacy attention trace whose kv_position is the compact "
            "execution-buffer slot, not the original token position. Rerun "
            "simple_test.py with the updated tracer before plotting."
        )
    if len(source_bucket_sizes) > 1:
        print(f"warning: input contains multiple stored bucket sizes: {sorted(source_bucket_sizes)}")

    decode_steps = sorted({step for step, _, _ in per_series})
    min_step, max_step = decode_steps[0], decode_steps[-1]
    # Preserve missing decode steps as dark columns instead of compressing time.
    decode_steps = list(range(min_step, max_step + 1))

    observed_bucket_count = max(bucket for _, bucket, _ in per_series) + 1
    if args.max_kv_pos is None:
        kv_limit = observed_bucket_count * output_bucket_size
    else:
        kv_limit = args.max_kv_pos
    bucket_count = max(observed_bucket_count, math.ceil(kv_limit / output_bucket_size))

    image_data = np.zeros((bucket_count, len(decode_steps)), dtype=np.float64)
    for step_idx, step in enumerate(decode_steps):
        for bucket in range(bucket_count):
            image_data[bucket, step_idx] = np.mean([
                per_series.get((step, bucket, series_id), 0.0)
                for series_id in series
            ])

    positive = image_data[image_data > 0]
    if positive.size == 0:
        raise SystemExit("Matched rows contain no positive attention weights.")
    vmax = args.vmax if args.vmax is not None else float(positive.max())
    vmin = args.vmin if args.vmin is not None else vmax * 1e-4
    if not 0 < vmin < vmax:
        raise SystemExit(f"Color scale must satisfy 0 < vmin < vmax (got {vmin:g}, {vmax:g}).")

    plot_data = np.maximum(image_data, vmin)
    figure, axis = plt.subplots(figsize=(12, 6.2))
    image = axis.imshow(
        plot_data,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(-0.5, len(decode_steps) - 0.5, kv_limit, 0),
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )

    title = "Attention weights over time"
    qualifiers = []
    if args.layer is not None:
        qualifiers.append(f"layer {args.layer}")
    if len(series) == 1:
        qualifiers.append(f"head {next(iter(series))[3]}")
    elif args.head is not None:
        qualifiers.append(f"head {args.head}, mean of {len(series)} trace series")
    else:
        qualifiers.append(f"mean of {len(series)} trace series")
    axis.set_title(f"{title} ({', '.join(qualifiers)})")
    axis.set_xlabel("Decoding step $t$")
    axis.set_ylabel("KV position (token index)")

    x_ticks = evenly_spaced_indices(len(decode_steps), 12)
    axis.set_xticks(x_ticks, [decode_steps[index] for index in x_ticks])
    y_ticks = np.linspace(0, kv_limit, 9, dtype=int)
    axis.set_yticks(y_ticks, [compact_position(position) for position in y_ticks])

    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Attention weight")
    figure.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(
        f"wrote {output_path} "
        f"({len(decode_steps)} steps x {bucket_count} KV buckets, bucket={output_bucket_size})"
    )


if __name__ == "__main__":
    main()
