#!/usr/bin/env python3
"""Simulate LRU HBM caches and plot top-k KV hit ratio over decode time."""

import argparse
import csv
from collections import OrderedDict, defaultdict
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


def parse_cache_sizes(value):
    try:
        sizes = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("cache sizes must be comma-separated numbers") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("cache sizes must be positive")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("cache sizes must be unique")
    return sizes


def evenly_spaced_indices(length, maximum):
    if length <= maximum:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=int))


def cache_size_label(size):
    return str(int(size)) if size.is_integer() else f"{size:g}"


def simulate_lru(selections, capacity):
    """Return pre-admission hit ratios; higher-attention entries stay most recent."""
    cache = OrderedDict()
    hit_ratios = {}
    for step in sorted(selections):
        selected = selections[step]
        cached_before_access = set(cache)
        hits = sum(position in cached_before_access for position in selected)
        hit_ratios[step] = hits / len(selected) if selected else 0.0

        # CSV positions are strongest-attention first. Process in reverse so
        # the strongest item becomes the most-recently-used entry.
        for position in reversed(selected):
            if position in cache:
                cache.move_to_end(position)
            else:
                cache[position] = None
        while len(cache) > capacity:
            cache.popitem(last=False)
    return hit_ratios


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
    parser.add_argument(
        "--cache-sizes-gb",
        type=parse_cache_sizes,
        default=parse_cache_sizes("64,32,16,8,4,2,1,0.5"),
        help="Comma-separated HBM capacities in GiB, displayed in the given order",
    )
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype-bytes", type=int, default=2, help="Bytes per K/V scalar; bf16/fp16=2")
    parser.add_argument("--all-sessions", action="store_true", help="Average matching sessions instead of using the newest one")
    args = parser.parse_args()

    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    for name in ("num_layers", "num_kv_heads", "head_dim", "dtype_bytes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

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

                traced_top_k = int(row["top_k"])
                output_top_k = traced_top_k if args.top_k is None else args.top_k
                if output_top_k > traced_top_k:
                    raise SystemExit(
                        f"Requested --top-k {output_top_k}, but trace contains only top-{traced_top_k}."
                    )
                # De-duplicate without losing attention-rank order.
                positions = list(dict.fromkeys(
                    int(value)
                    for value in row["selected_kv_positions"].split(";")[:output_top_k]
                    if value
                ))
                step = int(row["decode_step"])
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
            f"Matched query heads {sorted(selected_heads)}. Panel (d) represents one head; pass --head."
        )

    # One cached token reserves its K and V at every layer/KV head. This turns
    # model-level HBM capacity into the number of token positions it can hold.
    kv_bytes_per_token = (
        args.num_layers
        * args.num_kv_heads
        * 2
        * args.head_dim
        * args.dtype_bytes
    )
    capacities = [
        max(1, int(size * (1024 ** 3) // kv_bytes_per_token))
        for size in args.cache_sizes_gb
    ]
    print(f"model-wide KV bytes/token: {kv_bytes_per_token}")
    print("cache token capacities: " + ", ".join(
        f"{cache_size_label(size)} GiB={capacity} tokens"
        for size, capacity in zip(args.cache_sizes_gb, capacities)
    ))

    all_steps = sorted({step for selections in streams.values() for step in selections})
    decode_steps = list(range(all_steps[0], all_steps[-1] + 1))
    step_index = {step: index for index, step in enumerate(decode_steps)}
    hit_values = defaultdict(list)
    for selections in streams.values():
        for capacity_idx, capacity in enumerate(capacities):
            for step, hit_ratio in simulate_lru(selections, capacity).items():
                hit_values[(capacity_idx, step)].append(hit_ratio)

    image_data = np.full((len(capacities), len(decode_steps)), np.nan)
    for (capacity_idx, step), values in hit_values.items():
        image_data[capacity_idx, step_index[step]] = float(np.mean(values))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    output_top_k = args.top_k or min(traced_top_ks)
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
        f"HBM cache hit map over time "
        f"(LRU, top-{output_top_k}; layer {args.layer}, head {next(iter(selected_heads))})"
    )
    axis.set_xlabel("Decoding step $t$")
    axis.set_ylabel("Cache size (GiB)")

    x_ticks = evenly_spaced_indices(len(decode_steps), 12)
    axis.set_xticks(x_ticks, [decode_steps[index] for index in x_ticks])
    axis.set_yticks(
        range(len(args.cache_sizes_gb)),
        [cache_size_label(size) for size in args.cache_sizes_gb],
    )

    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("HBM hit ratio")
    figure.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(
        f"wrote {output_path} "
        f"({len(capacities)} cache sizes x {len(decode_steps)} decode steps)"
    )


if __name__ == "__main__":
    main()
