#!/usr/bin/env python3
"""Create Experiment-2 PNGs from analyze_retrieval_trace.py output."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def rows(path):
    with open(path, newline="") as input_file:
        return list(csv.DictReader(input_file))


def cdf(values):
    values = sorted(values)
    return values, [(index + 1) / len(values) for index in range(len(values))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # (a) Byte hit ratio, p05--p95 band by context.
    hit_rows = rows(input_dir / "hit_ratio_by_context.csv")
    contexts = [float(row["context_bucket_start"]) for row in hit_rows]
    p05 = [float(row["byte_hit_ratio_p05"]) for row in hit_rows]
    p50 = [float(row["byte_hit_ratio_p50"]) for row in hit_rows]
    p95 = [float(row["byte_hit_ratio_p95"]) for row in hit_rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.fill_between(contexts, p05, p95, alpha=0.25, label="p05--p95")
    axis.plot(contexts, p50, marker="o", label="median")
    axis.set(xlabel="Context length", ylabel="Byte hit ratio", ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "byte_hit_ratio_by_context.png", dpi=200)
    plt.close(figure)

    # (b) Miss-byte CDF; one curve per task.
    event_rows = rows(input_dir / "miss_bytes_samples.csv")
    miss_by_task = defaultdict(list)
    hit_ratio_by_condition = defaultdict(list)
    for row in event_rows:
        miss_by_task[row["task"]].append(float(row["miss_bytes"]) / (1024 * 1024))
        hit_ratio_by_condition[
            (row["task"], float(row["cache_ratio"]))
        ].append(float(row["byte_hit_ratio"]))
    figure, axis = plt.subplots(figsize=(7, 4))
    for task, values in sorted(miss_by_task.items()):
        x_values, y_values = cdf(values)
        axis.plot(x_values, y_values, label=task)
    axis.set(xlabel="Miss bytes (MiB)", ylabel="CDF", ylim=(0, 1))
    axis.legend(title="Task")
    figure.tight_layout()
    figure.savefig(output_dir / "miss_bytes_cdf.png", dpi=200)
    plt.close(figure)

    # Optional comparison view: event-level byte-hit-ratio distribution.
    conditions = sorted(hit_ratio_by_condition.items())
    figure, axis = plt.subplots(figsize=(max(7, len(conditions) * 1.25), 4))
    violin = axis.violinplot(
        [values for _, values in conditions], showmedians=True, showextrema=False
    )
    for body in violin["bodies"]:
        body.set_facecolor("#4c78a8")
        body.set_edgecolor("black")
        body.set_alpha(0.7)
    axis.set_xticks(
        range(1, len(conditions) + 1),
        [f"{task}\ncache={ratio:g}" for (task, ratio), _ in conditions],
    )
    axis.set(ylabel="Byte hit ratio", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(output_dir / "byte_hit_ratio_violin.png", dpi=200)
    plt.close(figure)

    # (c) Selection-retention CDF; one curve per delta.
    retention_by_delta = defaultdict(list)
    for row in rows(input_dir / "retention_samples.csv"):
        retention_by_delta[int(row["delta"])].append(float(row["retention"]))
    figure, axis = plt.subplots(figsize=(7, 4))
    for delta, values in sorted(retention_by_delta.items()):
        x_values, y_values = cdf(values)
        axis.plot(x_values, y_values, label=f"Δ={delta}")
    axis.set(xlabel="Selection retention", ylabel="CDF", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "retention_cdf.png", dpi=200)
    plt.close(figure)

    print(f"wrote PNGs to {output_dir}")


if __name__ == "__main__":
    main()
