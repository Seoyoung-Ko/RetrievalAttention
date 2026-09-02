#!/usr/bin/env python3
"""Plot selected-cluster residency over decode tokens for one trace stream."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One *_retrieval.csv file")
    parser.add_argument("--output", required=True, help="PNG output path")
    parser.add_argument("--request", type=int, default=0)
    parser.add_argument("--batch-group", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=50, help="Most frequently selected clusters to display")
    args = parser.parse_args()

    rows = []
    with open(args.input, newline="") as trace_file:
        reader = csv.DictReader(trace_file)
        compact = "selected_cluster_ids" in (reader.fieldnames or [])
        for row in reader:
            if (
                int(row["request_idx"]) == args.request
                and int(row["batch_group_idx"]) == args.batch_group
                and int(row["layer"]) == args.layer
            ):
                if compact:
                    for cluster_id, residency in zip(
                        row["selected_cluster_ids"].split(";"),
                        row["selected_residency"].split(";"),
                    ):
                        rows.append({
                            "decode_step": row["decode_step"],
                            "cluster_id": cluster_id,
                            "residency": "hbm" if residency == "H" else "cpu",
                        })
                else:
                    rows.append(row)
    if not rows:
        raise SystemExit("No rows matched the requested request/batch-group/layer.")

    steps = sorted({int(row["decode_step"]) for row in rows})
    counts = Counter(int(row["cluster_id"]) for row in rows)
    clusters = [cluster for cluster, _ in counts.most_common(args.top_k)]
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    step_index = {step: index for index, step in enumerate(steps)}
    matrix = np.full((len(clusters), len(steps)), np.nan)
    for row in rows:
        cluster = int(row["cluster_id"])
        if cluster in cluster_index:
            matrix[cluster_index[cluster], step_index[int(row["decode_step"])]] = (
                1 if row["residency"] == "hbm" else 0
            )

    cmap = ListedColormap(["#d95f02", "#1b9e77"])
    cmap.set_bad("#f4f4f4")
    figure, axis = plt.subplots(figsize=(max(10, len(steps) * 0.22), max(6, len(clusters) * 0.18)))
    image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", interpolation="none", cmap=cmap, vmin=0, vmax=1)
    axis.set_title(f"Selected-cluster residency: request={args.request}, group={args.batch_group}, layer={args.layer}")
    axis.set_xlabel("Decode step")
    axis.set_ylabel("Cluster ID (top by selection frequency)")
    tick_stride = max(1, len(steps) // 12)
    axis.set_xticks(range(0, len(steps), tick_stride), steps[::tick_stride])
    axis.set_yticks(range(len(clusters)), clusters)
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["CPU", "HBM"])
    figure.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
