#!/usr/bin/env python3
"""
Create a tiny Parquet subset for fast local debugging.

Why: the RL train dataset can be ~1.8M rows (repeated problems), and
RLHFDataset's prompt-length filtering tokenizes every row, which is slow.

This tool writes a small Parquet file (e.g. 64 rows) that keeps the same schema
as the source file so the online training/eval pipeline stays unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        required=True,
        help="Source parquet path (large).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output parquet path (small).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=64,
        help="Number of rows to keep (default: 64).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be > 0")

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_file():
        raise SystemExit(f"Source parquet not found: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Use pyarrow.dataset to read only the first N rows (avoid scanning 1.8M rows).
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    dataset = ds.dataset(str(src), format="parquet")
    table = dataset.head(args.n)
    if table.num_rows <= 0:
        raise SystemExit(f"Empty table read from {src}")

    pq.write_table(table, str(out), compression="zstd")
    print(f"OK: wrote {table.num_rows} rows -> {out}")


if __name__ == "__main__":
    main()

