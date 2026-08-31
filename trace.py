import argparse
import glob
import os

import numpy as np
import pandas as pd

TRACE_GLOB = "./traces/ruler_cache_ratio/0.025/**/*.csv"
OUT_DIR = "./analysis/result/cache_ratio/0.025"



def coefficient_of_variation(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean()
    return x.std(ddof=1) / mean if mean > 0 else np.nan


def main():
    files = sorted(glob.glob(TRACE_GLOB, recursive=True))
    if not files:
        raise FileNotFoundError(TRACE_GLOB)

    os.makedirs(OUT_DIR, exist_ok=True)

    frames = []
    empty_files = []
    for path in files:
        frame = pd.read_csv(path)
        if frame.empty:
            empty_files.append(path)
        else:
            frames.append(frame)

    if not frames:
        raise ValueError(
            f"All {len(files)} matched trace files are empty: {TRACE_GLOB}"
        )

    if empty_files:
        print(f"Skipping {len(empty_files)} empty trace file(s).")

    df = pd.concat(frames, ignore_index=True)

    df["D_MiB"] = df["selected_bytes"] / (1024 ** 2)
    df["H_MiB"] = df["hit_bytes"] / (1024 ** 2)
    df["R_MiB"] = df["remote_bytes"] / (1024 ** 2)

    assert np.allclose(
        df["selected_bytes"],
        df["hit_bytes"] + df["remote_bytes"],
    )

    df.to_parquet(
        os.path.join(OUT_DIR, "trace.parquet"),
        index=False,
    )

    # Per-context distributions.
    summary = (
        df.groupby(["task", "input_context_len"])
        .agg(
            samples=("D_MiB", "size"),
            D_mean=("D_MiB", "mean"),
            D_p05=("D_MiB", lambda x: x.quantile(0.05)),
            D_p50=("D_MiB", "median"),
            D_p95=("D_MiB", lambda x: x.quantile(0.95)),
            R_mean=("R_MiB", "mean"),
            R_p05=("R_MiB", lambda x: x.quantile(0.05)),
            R_p50=("R_MiB", "median"),
            R_p95=("R_MiB", lambda x: x.quantile(0.95)),
            h_mean=("hit_ratio", "mean"),
            h_p05=("hit_ratio", lambda x: x.quantile(0.05)),
            h_p50=("hit_ratio", "median"),
            h_p95=("hit_ratio", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )

    summary.to_csv(
        os.path.join(OUT_DIR, "distribution_summary.csv"),
        index=False,
    )

    cv_rows = []
    for (task, length), group in df.groupby(
        ["task", "input_context_len"]
    ):
        cv_rows.append({
            "task": task,
            "input_context_len": length,
            "D_cv": coefficient_of_variation(group["D_MiB"]),
            "R_cv": coefficient_of_variation(group["R_MiB"]),
            "h_std": group["hit_ratio"].std(ddof=1),
        })

    pd.DataFrame(cv_rows).to_csv(
        os.path.join(OUT_DIR, "variation.csv"),
        index=False,
    )

    # Warm-up versus steady state.
    df["phase"] = np.where(
        df["decode_step"] < 32,
        "warmup",
        "steady",
    )

    phase = (
        df.groupby(
            ["task", "input_context_len", "phase"]
        )
        .agg(
            D_mean=("D_MiB", "mean"),
            R_mean=("R_MiB", "mean"),
            hit_mean=("hit_ratio", "mean"),
        )
        .reset_index()
    )

    phase.to_csv(
        os.path.join(OUT_DIR, "phase_summary.csv"),
        index=False,
    )

    print(summary.to_string(index=False))
    print()
    print("Wrote:", OUT_DIR)


if __name__ == "__main__":
    main()
