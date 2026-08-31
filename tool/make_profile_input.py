#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert RULER JSONL data to simple_test.py JSON format."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="RULER validation.jsonl path",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output JSON path for simple_test.py",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to select",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting sample offset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        help="Tokenizer used to verify actual input lengths",
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

    return rows


def main():
    args = parse_args()

    rows = load_jsonl(args.src)

    begin = args.offset
    end = begin + args.num_samples

    if begin < 0:
        raise ValueError("--offset must be non-negative")

    if end > len(rows):
        raise ValueError(
            f"Requested rows [{begin}:{end}], "
            f"but source contains only {len(rows)} rows"
        )

    selected = rows[begin:end]
    converted = []

    for row in selected:
        if "input" not in row:
            raise KeyError("RULER row does not contain 'input'")

        if "outputs" not in row:
            raise KeyError("RULER row does not contain 'outputs'")

        outputs = row["outputs"]

        # Preserve the official RULER representation: list[str].
        if isinstance(outputs, str):
            outputs = [outputs]

        converted.append(
            {
                "input": row["input"],
                "outputs": outputs,
            }
        )

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    with args.dst.open("w", encoding="utf-8") as f:
        json.dump(
            converted,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    lengths = []
    hashes = []

    for sample_id, sample in enumerate(converted):
        input_ids = tokenizer(
            sample["input"],
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        token_length = len(input_ids)
        input_hash = hashlib.sha256(
            sample["input"].encode("utf-8")
        ).hexdigest()[:16]

        lengths.append(token_length)
        hashes.append(input_hash)

        print(
            f"sample={sample_id}, "
            f"source_index={selected[sample_id].get('index')}, "
            f"source_length={selected[sample_id].get('length')}, "
            f"tokenized_length={token_length}, "
            f"sha256={input_hash}, "
            f"outputs={sample['outputs']}"
        )

    if len(set(hashes)) != len(hashes):
        raise RuntimeError("Duplicate input samples were selected")

    print()
    print(f"source: {args.src}")
    print(f"output: {args.dst}")
    print(f"number of samples: {len(converted)}")
    print(f"minimum token length: {min(lengths)}")
    print(f"maximum token length: {max(lengths)}")
    print(f"length difference: {max(lengths) - min(lengths)}")
    print(f"padded batch input length: {max(lengths)}")


if __name__ == "__main__":
    main()