#!/usr/bin/env python3
"""Summarize compact RetroInfer retrieval traces for Experiment 2."""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean


EVENT_COLUMNS = {
    "trace_session_id", "request_idx", "batch_group_idx", "kv_head_group",
    "decode_step", "layer", "current_context_len", "selected_clusters",
}
COMPACT_COLUMNS = EVENT_COLUMNS | {
    "selected_cluster_ids", "selected_residency", "selected_bytes",
    "hit_bytes", "miss_bytes", "byte_hit_ratio", "cache_state",
}


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summary(values):
    return {
        "count": len(values),
        "mean": fmean(values) if values else None,
        "p05": quantile(values, 0.05),
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def trace_paths(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(Path(item).rglob("*_retrieval.csv"))
        else:
            paths.extend(Path(path) for path in glob.glob(item))
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise FileNotFoundError("No *_retrieval.csv files matched --input.")
    return paths


def event_key(row):
    return (
        row["trace_session_id"], row.get("task", ""), row.get("sample_id", ""),
        int(row["request_idx"]), int(row["batch_group_idx"]),
        int(row["kv_head_group"]), int(row["decode_step"]), int(row["layer"]),
    )


def make_event(row, path, line, clusters, residency):
    selected_vectors = int(row.get("selected_vectors") or (
        int(row.get("hit_vectors", 0)) + int(row.get("miss_vectors", 0))
    ))
    hit_vectors = int(row.get("hit_vectors", 0))
    miss_vectors = int(row.get("miss_vectors", 0))
    selected_bytes = int(row.get("selected_bytes") or selected_vectors)
    hit_bytes = int(row.get("hit_bytes") or hit_vectors)
    miss_bytes = int(row.get("miss_bytes") or miss_vectors)
    return {
        "source": str(path), "session": row["trace_session_id"],
        "task": row.get("task", ""), "sample": row.get("sample_id", ""),
        "request": int(row["request_idx"]),
        "batch_group": int(row["batch_group_idx"]),
        "kv_head_group": int(row["kv_head_group"]),
        "step": int(row["decode_step"]), "layer": int(row["layer"]),
        "context": int(row["current_context_len"]),
        "batch_size": int(row.get("batch_size") or 1),
        "cache_ratio": float(row.get("cache_ratio") or 0.0),
        "cache_state": row.get("cache_state") or "unknown",
        "selected_count": int(row["selected_clusters"]),
        "clusters": clusters,
        "residency": residency,
        "selected_vectors": selected_vectors,
        "hit_vectors": hit_vectors,
        "miss_vectors": miss_vectors,
        "selected_bytes": selected_bytes,
        "hit_bytes": hit_bytes,
        "miss_bytes": miss_bytes,
        "byte_hit_ratio": float(row.get("byte_hit_ratio") or (
            hit_bytes / selected_bytes if selected_bytes else 0.0
        )),
    }


def load_events(paths):
    events = {}
    malformed = []
    for path in paths:
        with path.open(newline="") as trace_file:
            reader = csv.DictReader(trace_file)
            fields = set(reader.fieldnames or [])
            missing = EVENT_COLUMNS - fields
            if missing:
                raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
            compact = COMPACT_COLUMNS <= fields
            for line, row in enumerate(reader, start=2):
                key = event_key(row)
                if compact:
                    clusters = [int(value) for value in row["selected_cluster_ids"].split(";") if value]
                    residency = [value for value in row["selected_residency"].split(";") if value]
                    event = make_event(row, path, line, clusters, residency)
                    if key in events:
                        raise ValueError(f"{path}:{line}: duplicate compact event {key}")
                    events[key] = event
                else:
                    event = events.setdefault(key, make_event(row, path, line, [], []))
                    event["clusters"].append(int(row["cluster_id"]))
                    event["residency"].append(row.get("residency", ""))
    for event in events.values():
        if len(event["clusters"]) != event["selected_count"]:
            malformed.append(event)
    return list(events.values()), malformed


def bucket(context, width):
    return context if width == 0 else (context // width) * width


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(events, context_bin, deltas, windows):
    by_context = defaultdict(list)
    streams = defaultdict(list)
    frequency = defaultdict(lambda: {"steps": set(), "ranks": []})
    for event in events:
        by_context[bucket(event["context"], context_bin)].append(event)
        stream = (
            event["session"], event["task"], event["sample"], event["request"],
            event["batch_group"], event["kv_head_group"], event["layer"],
        )
        streams[stream].append(event)
        for rank, cluster in enumerate(event["clusters"]):
            stats = frequency[stream + (cluster,)]
            stats["steps"].add(event["step"])
            stats["ranks"].append(rank)

    context_rows = []
    for context, group in sorted(by_context.items()):
        ratios = [event["byte_hit_ratio"] for event in group]
        misses = [event["miss_bytes"] for event in group]
        context_rows.append({
            "context_bucket_start": context, "events": len(group),
            **{f"byte_hit_ratio_{name}": value for name, value in summary(ratios).items()},
            **{f"miss_bytes_{name}": value for name, value in summary(misses).items()},
        })

    miss_rows = [{
        "trace_session_id": event["session"], "task": event["task"],
        "sample_id": event["sample"], "request_idx": event["request"],
        "batch_group_idx": event["batch_group"], "kv_head_group": event["kv_head_group"],
        "decode_step": event["step"], "layer": event["layer"],
        "current_context_len": event["context"], "cache_ratio": event["cache_ratio"],
        "cache_state": event["cache_state"], "selected_bytes": event["selected_bytes"],
        "hit_bytes": event["hit_bytes"], "miss_bytes": event["miss_bytes"],
        "byte_hit_ratio": event["byte_hit_ratio"],
    } for event in events]

    # Reconstruct the unit that the GPU attention path executes together:
    # one request/token/layer across all KV-head groups.
    layer_groups = defaultdict(list)
    for event in events:
        layer_groups[(
            event["session"], event["task"], event["sample"], event["request"],
            event["step"], event["layer"], event["cache_ratio"], event["cache_state"],
            event["batch_size"],
        )].append(event)
    layer_rows = []
    for key, group in sorted(layer_groups.items()):
        group.sort(key=lambda event: event["kv_head_group"])
        selected_bytes = sum(event["selected_bytes"] for event in group)
        hit_bytes = sum(event["hit_bytes"] for event in group)
        layer_rows.append({
            "trace_session_id": key[0], "task": key[1], "sample_id": key[2],
            "request_idx": key[3], "decode_step": key[4], "layer": key[5],
            "cache_ratio": key[6], "cache_state": key[7], "batch_size": key[8],
            "kv_head_groups": len(group),
            "selected_vectors_total": sum(event["selected_vectors"] for event in group),
            "hit_vectors_total": sum(event["hit_vectors"] for event in group),
            "miss_vectors_total": sum(event["miss_vectors"] for event in group),
            "selected_bytes_total": selected_bytes,
            "hit_bytes_total": hit_bytes,
            "miss_bytes_total": sum(event["miss_bytes"] for event in group),
            "byte_hit_ratio": hit_bytes / selected_bytes if selected_bytes else 0.0,
            "selected_vectors_by_kv_head": ";".join(
                str(event["selected_vectors"]) for event in group
            ),
            "hit_vectors_by_kv_head": ";".join(
                str(event["hit_vectors"]) for event in group
            ),
        })

        # Exact-lag overlap:
    # |S_t ∩ S_{t-delta}| / |S_t|
    retention_rows = []

    # Recent-window coverage:
    # |S_t ∩ (S_{t-1} ∪ ... ∪ S_{t-window})| / |S_t|
    lookback_rows = []

    window_set = set(windows)
    max_window = max(windows, default=0)

    for stream, stream_events in streams.items():
        stream_events.sort(key=lambda event: event["step"])
        by_step = {event["step"]: event for event in stream_events}

        for current in stream_events:
            current_set = set(current["clusters"])

            # ----------------------------------------------------------
            # 1. 기존 exact-lag selection overlap
            # ----------------------------------------------------------
            for delta in deltas:
                previous = by_step.get(current["step"] - delta)
                if previous is None:
                    continue

                previous_set = set(previous["clusters"])
                retained = len(current_set & previous_set)

                retention_rows.append({
                    "trace_session_id": stream[0],
                    "task": stream[1],
                    "sample_id": stream[2],
                    "request_idx": stream[3],
                    "batch_group_idx": stream[4],
                    "kv_head_group": stream[5],
                    "layer": stream[6],
                    "decode_step": current["step"],
                    "previous_decode_step": previous["step"],
                    "current_context_len": current["context"],
                    "delta": delta,
                    "retained_clusters": retained,
                    "current_selected_clusters": len(current_set),
                    "retention": (
                        retained / len(current_set)
                        if current_set else 0.0
                    ),
                })

            # ----------------------------------------------------------
            # 2. 새로운 recent-window selection coverage
            # ----------------------------------------------------------
            if max_window == 0:
                continue

            # 모든 window가 정확히 같은 current event를 사용하도록
            # 최대 window만큼의 연속 history가 모두 있는 경우만 사용합니다.
            previous_events = []
            complete_history = True

            for lag in range(1, max_window + 1):
                previous = by_step.get(current["step"] - lag)

                if previous is None:
                    complete_history = False
                    break

                previous_events.append(previous)

            if not complete_history:
                continue

            history_union = set()

            for lag, previous in enumerate(previous_events, start=1):
                history_union.update(previous["clusters"])

                if lag not in window_set:
                    continue

                covered_set = current_set & history_union
                uncovered_set = current_set - history_union

                lookback_rows.append({
                    "trace_session_id": stream[0],
                    "task": stream[1],
                    "sample_id": stream[2],
                    "request_idx": stream[3],
                    "batch_group_idx": stream[4],
                    "kv_head_group": stream[5],
                    "layer": stream[6],
                    "decode_step": current["step"],
                    "oldest_history_step": current["step"] - lag,
                    "newest_history_step": current["step"] - 1,
                    "current_context_len": current["context"],
                    "window_size": lag,
                    "history_unique_clusters": len(history_union),
                    "covered_clusters": len(covered_set),
                    "uncovered_clusters": len(uncovered_set),
                    "current_selected_clusters": len(current_set),
                    "lookback_coverage": (
                        len(covered_set) / len(current_set)
                        if current_set else 0.0
                    ),
                })

    frequency_rows = []
    for key, stats in sorted(frequency.items()):
        stream_events = streams[key[:7]]
        steps = sorted(stats["steps"])
        frequency_rows.append({
            "trace_session_id": key[0], "task": key[1], "sample_id": key[2],
            "request_idx": key[3], "batch_group_idx": key[4], "kv_head_group": key[5],
            "layer": key[6], "cluster_id": key[7], "selected_events": len(steps),
            "stream_events": len(stream_events),
            "selection_frequency": len(steps) / len(stream_events),
            "first_decode_step": steps[0], "last_decode_step": steps[-1],
            "mean_selection_rank": fmean(stats["ranks"]),
        })

    return context_rows, miss_rows, layer_rows, retention_rows, lookback_rows, frequency_rows, {
        "byte_hit_ratio": summary([event["byte_hit_ratio"] for event in events]),
        "miss_bytes": summary([event["miss_bytes"] for event in events]),
        "retention": {
            str(delta): summary([
                row["retention"]
                for row in retention_rows
                if row["delta"] == delta
            ])
            for delta in deltas
        },
        "lookback_coverage": {
            str(window): summary([
                row["lookback_coverage"]
                for row in lookback_rows
                if row["window_size"] == window
            ])
            for window in windows
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-bin", type=int, default=4096)
    parser.add_argument("--deltas", default="1,2,4,8")
    parser.add_argument(
        "--windows",
        default="1,2,4,8",
        help="Recent lookback-window sizes, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--cache-state", choices=("steady", "all"), default="steady",
        help="Use only steady events by default; 'all' retains cold/warming events.",
    )
    args = parser.parse_args()
    if args.context_bin < 0:
        parser.error("--context-bin must be >= 0")
    deltas = [int(value) for value in args.deltas.split(",") if int(value) > 0]
    windows = sorted({
        int(value)
        for value in args.windows.split(",")
        if int(value) > 0
    })

    paths = trace_paths(args.input)
    events, malformed = load_events(paths)
    if args.cache_state == "steady":
        events = [event for event in events if event["cache_state"] == "steady"]
    if not events:
        raise SystemExit("No events remain. Use a longer decode run or --cache-state all.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        context_rows,
        miss_rows,
        layer_rows,
        retention_rows,
        lookback_rows,
        frequency_rows,
        overall,
    ) = analyze(
        events,
        args.context_bin,
        deltas,
        windows,
    )
    write_csv(output_dir / "hit_ratio_by_context.csv", list(context_rows[0]), context_rows)
    write_csv(output_dir / "miss_bytes_samples.csv", list(miss_rows[0]), miss_rows)
    write_csv(output_dir / "layer_events.csv", list(layer_rows[0]), layer_rows)
    retention_fields = [
        "trace_session_id",
        "task",
        "sample_id",
        "request_idx",
        "batch_group_idx",
        "kv_head_group",
        "layer",
        "decode_step",
        "previous_decode_step",
        "current_context_len",
        "delta",
        "retained_clusters",
        "current_selected_clusters",
        "retention",
    ]
    write_csv(output_dir / "retention_samples.csv", retention_fields, retention_rows)
    lookback_fields = [
        "trace_session_id",
        "task",
        "sample_id",
        "request_idx",
        "batch_group_idx",
        "kv_head_group",
        "layer",
        "decode_step",
        "oldest_history_step",
        "newest_history_step",
        "current_context_len",
        "window_size",
        "history_unique_clusters",
        "covered_clusters",
        "uncovered_clusters",
        "current_selected_clusters",
        "lookback_coverage",
    ]

    write_csv(
        output_dir / "lookback_coverage_samples.csv",
        lookback_fields,
        lookback_rows,
    )
    write_csv(output_dir / "cluster_frequency.csv", list(frequency_rows[0]), frequency_rows)
    report = {
         "input_files": [str(path) for path in paths],
        "events": len(events),
        "malformed_events": len(malformed),
        "cache_state_filter": args.cache_state,
        "context_bin": args.context_bin,
        "deltas": deltas,
        "windows": windows,
        "overall": overall,
    }
    with open(output_dir / "summary.json", "w") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
