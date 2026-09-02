#!/usr/bin/env python3
"""Plot median recent-window selection coverage by layer and decode step."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="lookback_coverage_samples.csv",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Number of previous decode tokens included in the history union",
    )
    parser.add_argument(
        "--task",
        help="Optional task filter, e.g. VT",
    )
    parser.add_argument(
        "--request",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min-step",
        type=int,
        help="Optional minimum decode step",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        help="Optional maximum decode step",
    )
    args = parser.parse_args()

    grouped = defaultdict(list)

    with open(args.input, newline="") as input_file:
        for row in csv.DictReader(input_file):
            if int(row["window_size"]) != args.window_size:
                continue

            if args.task is not None and row["task"] != args.task:
                continue

            if int(row["request_idx"]) != args.request:
                continue

            decode_step = int(row["decode_step"])

            if args.min_step is not None and decode_step < args.min_step:
                continue

            if args.max_step is not None and decode_step > args.max_step:
                continue

            key = (
                int(row["layer"]),
                decode_step,
            )

            grouped[key].append(
                float(row["lookback_coverage"])
            )

    if not grouped:
        raise SystemExit(
            "No lookback-coverage rows matched the requested filters."
        )

    layers = sorted({
        layer
        for layer, _ in grouped
    })

    steps = sorted({
        step
        for _, step in grouped
    })

    layer_index = {
        layer: index
        for index, layer in enumerate(layers)
    }

    step_index = {
        step: index
        for index, step in enumerate(steps)
    }

    image_data = np.full(
        (len(layers), len(steps)),
        np.nan,
    )

    for (layer, step), values in grouped.items():
        image_data[
            layer_index[layer],
            step_index[step],
        ] = median(values)

    figure, axis = plt.subplots(
        figsize=(
            max(9, len(steps) * 0.14),
            max(5, len(layers) * 0.16),
        )
    )

    image = axis.imshow(
        np.ma.masked_invalid(image_data),
        aspect="auto",
        interpolation="none",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    axis.set_title(
        f"Median recent-window selection coverage (W={args.window_size})"
    )
    axis.set_xlabel("Decode step")
    axis.set_ylabel("Layer")

    x_stride = max(1, len(steps) // 12)
    y_stride = max(1, len(layers) // 16)

    axis.set_xticks(
        range(0, len(steps), x_stride),
        steps[::x_stride],
    )
    axis.set_yticks(
        range(0, len(layers), y_stride),
        layers[::y_stride],
    )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Recent-window coverage")

    figure.tight_layout()

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        args.output,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()