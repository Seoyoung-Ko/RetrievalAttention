#!/usr/bin/env python3
"""GPU primitive microbenchmark using RetroInfer's attention/gather kernels.

This intentionally bypasses the model, selector, CPU list, and cache admission.
It measures only HBM-resident gather and exact attention with the same tensor
layout/GQA shape used by retroinfer_cache.py.
"""

import argparse
import csv
import time
from pathlib import Path
from statistics import median

import torch
from retroinfer_kernels import gather_copy_vectors
from weighted_flash_decoding import weighted_flash_decoding


def values(text):
    return [int(item) for item in text.split(",") if item]


def timed(callable_, warmup, iterations):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    host_start = time.perf_counter_ns()
    start.record()
    for _ in range(iterations):
        callable_()
    end.record()
    end.synchronize()
    host_us = (time.perf_counter_ns() - host_start) / iterations / 1000
    return start.elapsed_time(end) * 1000 / iterations, host_us


def trace_points(path, quantiles):
    with open(path, newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    selected = sorted(rows, key=lambda row: int(row["selected_vectors_total"]))
    points = []
    for quantile in quantiles:
        row = selected[round((len(selected) - 1) * quantile)]
        heads = int(row["kv_head_groups"])
        points.append((
            max(1, int(row["selected_vectors_total"]) // heads),
            max(1, int(row["hit_vectors_total"]) // heads),
            f"trace_p{int(quantile * 100):02d}",
        ))
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--batch", default="1,2,4")
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--selected-vectors", default="512,1024,2048,4096")
    parser.add_argument("--hit-ratios", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--layout", choices=("contiguous", "random"), default="contiguous")
    parser.add_argument("--cache-pages", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--layer-events", help="Optional analysis/.../layer_events.csv")
    args = parser.parse_args()
    if args.query_heads % args.kv_heads:
        parser.error("--query-heads must be divisible by --kv-heads")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    points = [(size, None, "controlled") for size in values(args.selected_vectors)]
    if args.layer_events:
        points.extend(trace_points(args.layer_events, [0.05, 0.25, 0.5, 0.75, 0.95]))
    rows = []
    torch.manual_seed(0)

    for batch in values(args.batch):
        groups, gqa = batch * args.kv_heads, args.query_heads // args.kv_heads
        query = torch.randn((groups, 1, gqa, args.head_dim), device=device, dtype=dtype)
        for selected, trace_hit, source in points:
            selected = min(selected, args.cache_pages * 8)
            kv = torch.randn((groups, selected, 1, args.head_dim), device=device, dtype=dtype)
            vv = torch.randn_like(kv)
            lengths = torch.full((groups,), selected, device=device, dtype=torch.int32)
            attention = lambda: weighted_flash_decoding(
                query, kv, vv, cache_seqlens=lengths, return_softmax_lse=False
            )
            for run in range(args.runs):
                gpu_us, host_us = timed(attention, args.warmup, args.iterations)
                rows.append({"primitive": "attention", "source": source, "layout": "contiguous", "batch": batch, "kv_heads": args.kv_heads, "query_heads": args.query_heads, "head_dim": args.head_dim, "dtype": args.dtype, "selected_vectors_per_kv_head": selected, "hit_vectors_per_kv_head": selected, "gpu_event_us": gpu_us, "host_completion_us": host_us, "run": run})

            ratios = [trace_hit / selected] if trace_hit is not None else [float(x) for x in args.hit_ratios.split(",")]
            for ratio in ratios:
                hit = max(1, min(selected, round(selected * ratio)))
                pages = (hit + 7) // 8
                cache_k = torch.randn((groups, args.cache_pages * 8, args.head_dim), device=device, dtype=dtype)
                cache_v = torch.randn_like(cache_k)
                out_k = torch.empty((groups, hit, args.head_dim), device=device, dtype=dtype)
                out_v = torch.empty_like(out_k)
                meta_k = torch.empty_like(cache_k)
                meta_out = torch.empty_like(out_k)
                page_slots = torch.arange(pages, device=device, dtype=torch.int64)
                if args.layout == "random":
                    page_slots = torch.randperm(args.cache_pages, device=device)[:pages]
                offsets = (page_slots[:, None] * 8 + torch.arange(8, device=device)).flatten()[:hit]
                offsets = offsets.repeat(groups, 1)
                gather = lambda: gather_copy_vectors(cache_k, out_k, cache_v, out_v, meta_k, meta_out, offsets, groups, args.cache_pages * 8, hit, hit, 0, hit)
                for run in range(args.runs):
                    gpu_us, host_us = timed(gather, args.warmup, args.iterations)
                    rows.append({"primitive": "gather", "source": source, "layout": args.layout, "batch": batch, "kv_heads": args.kv_heads, "query_heads": args.query_heads, "head_dim": args.head_dim, "dtype": args.dtype, "selected_vectors_per_kv_head": selected, "hit_vectors_per_kv_head": hit, "gpu_event_us": gpu_us, "host_completion_us": host_us, "run": run})
                hit_lengths = torch.full((groups,), hit, device=device, dtype=torch.int32)
                hit_attention = lambda: weighted_flash_decoding(
                    query, out_k.view(groups, hit, 1, args.head_dim),
                    out_v.view(groups, hit, 1, args.head_dim),
                    cache_seqlens=hit_lengths, return_softmax_lse=False,
                )
                for run in range(args.runs):
                    gpu_us, host_us = timed(hit_attention, args.warmup, args.iterations)
                    rows.append({"primitive": "attention_hit", "source": source, "layout": args.layout, "batch": batch, "kv_heads": args.kv_heads, "query_heads": args.query_heads, "head_dim": args.head_dim, "dtype": args.dtype, "selected_vectors_per_kv_head": selected, "hit_vectors_per_kv_head": hit, "gpu_event_us": gpu_us, "host_completion_us": host_us, "run": run})

    fields = list(rows[0])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} measurements)")


if __name__ == "__main__":
    main()
