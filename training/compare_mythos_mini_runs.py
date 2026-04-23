#!/usr/bin/env python3
"""
Compare OpenMythos-Mini JSONL metric files.

Examples:
    python training/compare_mythos_mini_runs.py runs/baseline.jsonl runs/deep_loops.jsonl
    python training/compare_mythos_mini_runs.py runs/*.jsonl --sort-by best_val_loss
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", help="One or more JSONL metric files.")
    parser.add_argument(
        "--sort-by",
        choices=[
            "label",
            "best_val_loss",
            "last_val_loss",
            "best_train_loss",
            "last_step",
            "last_tokens_per_sec",
        ],
        default="best_val_loss",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Human-readable table or JSON output.",
    )
    parser.add_argument(
        "--write-csv",
        type=str,
        default=None,
        help="Optional CSV file to write the summarized results to.",
    )
    return parser.parse_args()


def load_metrics_file(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")
    return rows


def summarize_run(path: str | Path) -> dict:
    entries = load_metrics_file(path)
    last = entries[-1]
    path = Path(path)

    def minimum(key: str) -> float:
        values = [entry[key] for entry in entries if key in entry]
        return min(values)

    return {
        "label": path.stem,
        "path": str(path),
        "steps_logged": len(entries),
        "last_step": int(last["step"]),
        "best_train_loss": minimum("train_loss"),
        "best_val_loss": minimum("val_loss"),
        "last_train_loss": float(last["train_loss"]),
        "last_val_loss": float(last["val_loss"]),
        "last_tokens_per_sec": last.get("tokens_per_sec"),
        "variant": last.get("variant"),
        "preset": last.get("preset"),
        "attn_type": last.get("attn_type"),
        "recurrent_use_moe": last.get("recurrent_use_moe"),
        "use_act": last.get("use_act"),
        "n_loops": last.get("n_loops"),
        "corpus_files": last.get("corpus_files"),
        "corpus_chars": last.get("corpus_chars"),
    }


def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def format_table(rows: list[dict]) -> str:
    columns = [
        "label",
        "preset",
        "variant",
        "attn_type",
        "n_loops",
        "recurrent_use_moe",
        "use_act",
        "best_val_loss",
        "last_val_loss",
        "last_tokens_per_sec",
        "corpus_files",
        "best_train_loss",
        "last_step",
    ]
    widths = {
        col: max(len(col), max(len(format_value(row.get(col))) for row in rows))
        for col in columns
    }

    def render(row: dict) -> str:
        return "  ".join(format_value(row.get(col)).ljust(widths[col]) for col in columns)

    header = render({col: col for col in columns})
    divider = "  ".join("-" * widths[col] for col in columns)
    body = "\n".join(render(row) for row in rows)
    return f"{header}\n{divider}\n{body}"


def write_csv(path: str | Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = [summarize_run(path) for path in args.metrics]

    rows.sort(key=lambda row: row[args.sort_by])
    if args.write_csv is not None:
        write_csv(args.write_csv, rows)

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        print(format_table(rows))


if __name__ == "__main__":
    main()
