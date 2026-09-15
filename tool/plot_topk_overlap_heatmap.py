#!/usr/bin/env python3
"""Plot overlap@k between top-k KV sets at decode steps t and t-delta."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def parse_deltas(value):
    try:
        deltas = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("deltas must be comma-separated integers") from error
    if not deltas or any(delta <= 0 for delta in deltas):
        raise argparse.ArgumentTypeError("deltas must contain positive integers")
    return deltas


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
    parser.add_argument("--top-k", type=int, default=None, help="Use the first k attention-ranked positions")
    parser.add_argument("--deltas", type=parse_deltas, default=parse_deltas("1,2,4,8,16,32,64,128"))
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--all-sessions", action="store_true", help="Average matching sessions instead of using the newest one")
    args = parser.parse_args()

    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.min_step is not None and args.max_step is not None and args.min_step > args.max_step:
        parser.error("--min-step cannot exceed --max-step")

    csv_paths = topk_csvs(args.input, args.all_sessions)
    if not csv_paths:
        raise SystemExit(f"No non-empty *_topk_kv.csv files matched: {args.input}")
    print("reading " + ", ".join(str(path) for path in csv_paths))

    streams = defaultdict(dict)
    position_spaces = set()
    selection_scopes = set()
    selected_heads = set()
    traced_top_ks = set()

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
                query_head = int(row["query_head"])
                if args.head is not None and query_head != args.head:
                    continue

                step = int(row["decode_step"])
                if args.min_step is not None and step < args.min_step:
                    continue
                if args.max_step is not None and step > args.max_step:
                    continue

                traced_top_k = int(row["top_k"])
                output_top_k = traced_top_k if args.top_k is None else args.top_k
                if output_top_k > traced_top_k:
                    raise SystemExit(
                        f"Requested --top-k {output_top_k}, but trace contains only top-{traced_top_k}."
                    )
                positions = {
                    int(value)
                    for value in row["selected_kv_positions"].split(";")[:output_top_k]
                    if value
                }
                stream = (row.get("trace_session_id", ""), query_head)
                if step in streams[stream]:
                    raise SystemExit(f"Duplicate top-k row for stream={stream}, step={step}")
                streams[stream][step] = positions
                position_spaces.add(row.get("position_space") or "legacy_unknown")
                selection_scopes.add(row.get("selection_scope") or "legacy_unknown")
                selected_heads.add(query_head)
                traced_top_ks.add(traced_top_k)

    if not streams:
        raise SystemExit("No top-k rows matched the requested filters.")
    if position_spaces != {"token"}:
        raise SystemExit("Trace does not contain original token positions; rerun with the updated tracer.")
    if selection_scopes != {"full_context_attention_topk"}:
        raise SystemExit(f"Unsupported selection scope: {sorted(selection_scopes)}")
    if len(selected_heads) > 1:
        raise SystemExit(
            f"Matched query heads {sorted(selected_heads)}. Panel (c) represents one head; pass --head."
        )

    output_top_k = args.top_k or min(traced_top_ks)
    all_steps = sorted({step for by_step in streams.values() for step in by_step})
    decode_steps = list(range(all_steps[0], all_steps[-1] + 1))
    overlap_values = defaultdict(list)
    for by_step in streams.values():
        for step, current_set in by_step.items():
            for delta in args.deltas:
                previous_set = by_step.get(step - delta)
                if previous_set is None:
                    continue
                overlap = len(current_set & previous_set) / output_top_k
                overlap_values[(delta, step)].append(overlap)

    usable_deltas = [
        delta for delta in args.deltas
        if any(key_delta == delta for key_delta, _ in overlap_values)
    ]
    skipped_deltas = sorted(set(args.deltas) - set(usable_deltas))
    if skipped_deltas:
        print(f"skipping deltas with no valid step pair: {skipped_deltas}")
    if not usable_deltas:
        raise SystemExit("No valid (t, t-delta) pairs. Trace more decode steps or request smaller deltas.")

    image_data = np.full((len(usable_deltas), len(decode_steps)), np.nan)
    step_index = {step: index for index, step in enumerate(decode_steps)}
    for delta_idx, delta in enumerate(usable_deltas):
        for step in decode_steps:
            values = overlap_values.get((delta, step))
            if values:
                image_data[delta_idx, step_index[step]] = float(np.mean(values))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    figure, axis = plt.subplots(figsize=(12, 5.2))
    image = axis.imshow(
        np.ma.masked_invalid(image_data),
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    axis.set_title(
        f"Overlap of top-{output_top_k} retrieved KVs "
        f"(step $t$ vs. $t-\\Delta$; layer {args.layer}, head {next(iter(selected_heads))})"
    )
    axis.set_xlabel("Decoding step $t$")
    axis.set_ylabel("Step distance $\\Delta$")

    x_ticks = evenly_spaced_indices(len(decode_steps), 12)
    axis.set_xticks(x_ticks, [decode_steps[index] for index in x_ticks])
    axis.set_yticks(range(len(usable_deltas)), usable_deltas)

    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(f"Overlap@{output_top_k}")
    figure.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(
        f"wrote {output_path} "
        f"({len(usable_deltas)} deltas x {len(decode_steps)} decode steps)"
    )


if __name__ == "__main__":
    main()
