#!/usr/bin/env python3
"""Merge disjoint trajectory-generation shards into one canonical JSONL pool."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args()

    rows: list[dict] = []
    for part in args.parts:
        with part.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise SystemExit(f"{part}:{line_no}: invalid JSON: {exc}") from exc

    keys = [(row.get("task_name"), row.get("task_index")) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate (task_name, task_index) rows across shards")
    if len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(rows)}")
    rows.sort(key=lambda row: (str(row.get("task_name", "")), int(row["task_index"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    print(f"wrote {len(rows)} ordered rows to {args.output}")


if __name__ == "__main__":
    main()
