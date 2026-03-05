#!/usr/bin/env python3
"""
Filter Big-Math-RL-Verified dataset by llama8b_solve_rate.

Usage:
    python filter_big_math_dataset.py --output_dir <output_dir> [--min_rate <min>] [--max_rate <max>]

Example:
    python filter_big_math_dataset.py --output_dir ./filtered_data --min_rate 0.05 --max_rate 0.5

Requirements:
    - Install datasets: pip install datasets
    - Set HuggingFace token: export HF_TOKEN=<your_token> or pass via --token
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Filter Big-Math-RL-Verified dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for filtered data",
    )
    parser.add_argument(
        "--min_rate",
        type=float,
        default=0.05,
        help="Minimum llama8b_solve_rate (default: 0.05)",
    )
    parser.add_argument(
        "--max_rate",
        type=float,
        default=0.5,
        help="Maximum llama8b_solve_rate (default: 0.5)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SynthLabsAI/Big-Math-RL-Verified",
        help="Dataset name on HuggingFace",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json", "parquet"],
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=None,
        help="Shard size (number of samples per file). If None, save to single file.",
    )
    return parser.parse_args()


def load_and_filter_dataset(
    dataset_name: str,
    split: str,
    min_rate: float,
    max_rate: float,
    token: str = None,
) -> List[Dict[str, Any]]:
    """
    Load dataset and filter by llama8b_solve_rate.

    Returns:
        List of filtered samples
    """
    print(f"Loading dataset: {dataset_name} ({split} split)")
    print(f"Filter condition: {min_rate} <= llama8b_solve_rate <= {max_rate}")

    # Load dataset
    load_kwargs = {"path": dataset_name, "split": split}
    if token:
        load_kwargs["token"] = token

    ds = load_dataset(**load_kwargs)

    print(f"Dataset size: {len(ds):,}")
    print(f"Columns: {ds.column_names}")

    # Check if llama8b_solve_rate exists
    if 'llama8b_solve_rate' not in ds.column_names:
        print(f"\nError: 'llama8b_solve_rate' not found in dataset!")
        print(f"Available columns: {ds.column_names}")
        raise ValueError("Column 'llama8b_solve_rate' not found")

    # Get statistics before filtering
    rates = ds['llama8b_solve_rate']
    # Filter out None values
    valid_rates = [r for r in rates if r is not None]
    print(f"\nllama8b_solve_rate statistics (before filtering):")
    print(f"  Total samples: {len(rates):,}")
    print(f"  Valid samples (non-None): {len(valid_rates):,}")
    print(f"  None samples: {len(rates) - len(valid_rates):,}")
    if valid_rates:
        print(f"  Min: {min(valid_rates):.4f}")
        print(f"  Max: {max(valid_rates):.4f}")
        print(f"  Mean: {sum(valid_rates)/len(valid_rates):.4f}")

    # Filter dataset (handle None values)
    print(f"\nFiltering dataset...")
    filtered_ds = ds.filter(lambda x: x['llama8b_solve_rate'] is not None and min_rate <= x['llama8b_solve_rate'] <= max_rate)

    print(f"Filtered dataset size: {len(filtered_ds):,}")
    print(f"Filtered out: {len(ds) - len(filtered_ds):,} samples")

    # Get statistics after filtering
    if len(filtered_ds) > 0:
        filtered_rates = filtered_ds['llama8b_solve_rate']
        print(f"\nllama8b_solve_rate statistics (after filtering):")
        print(f"  Min: {min(filtered_rates):.4f}")
        print(f"  Max: {max(filtered_rates):.4f}")
        print(f"  Mean: {sum(filtered_rates)/len(filtered_rates):.4f}")

    # Convert to list of dicts
    print(f"\nConverting to list...")
    filtered_list = filtered_ds.to_list()

    return filtered_list


def save_filtered_data(
    data: List[Dict[str, Any]],
    output_dir: str,
    output_format: str,
    shard_size: int = None,
):
    """
    Save filtered data to disk.

    Args:
        data: List of samples
        output_dir: Output directory
        output_format: Output format (jsonl, json, or parquet)
        shard_size: Number of samples per shard (if None, save to single file)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(data):,} samples to {output_dir}")

    if shard_size is None:
        # Save to single file
        if output_format == "jsonl":
            output_file = output_path / "filtered_data.jsonl"
            print(f"Writing to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in tqdm(data, desc="Writing JSONL"):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif output_format == "json":
            output_file = output_path / "filtered_data.json"
            print(f"Writing to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif output_format == "parquet":
            output_file = output_path / "filtered_data.parquet"
            print(f"Writing to {output_file}")
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.table(data)
            pq.write_table(table, output_file)
    else:
        # Save to multiple shards
        n_shards = (len(data) + shard_size - 1) // shard_size
        print(f"Splitting into {n_shards} shards ({shard_size:,} samples each)")

        for i in tqdm(range(n_shards), desc="Writing shards"):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(data))
            shard_data = data[start_idx:end_idx]

            if output_format == "jsonl":
                output_file = output_path / f"filtered_data_{i:04d}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in shard_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            elif output_format == "json":
                output_file = output_path / f"filtered_data_{i:04d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(shard_data, f, ensure_ascii=False, indent=2)
            elif output_format == "parquet":
                output_file = output_path / f"filtered_data_{i:04d}.parquet"
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.table(shard_data)
                pq.write_table(table, output_file)

    print(f"\nDone! Data saved to {output_dir}")

    # Save metadata
    metadata = {
        "total_samples": len(data),
        "filter_min_rate": args.min_rate,
        "filter_max_rate": args.max_rate,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "format": output_format,
        "num_shards": n_shards if shard_size else 1,
        "shard_size": shard_size,
    }
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to {metadata_file}")


def main():
    args = parse_args()

    # Get token from args or env
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: No HF_TOKEN provided. Trying to load without authentication...")
        print("If the dataset is gated, please set HF_TOKEN environment variable or use --token")

    try:
        # Load and filter dataset
        filtered_data = load_and_filter_dataset(
            dataset_name=args.dataset_name,
            split=args.split,
            min_rate=args.min_rate,
            max_rate=args.max_rate,
            token=token,
        )

        if len(filtered_data) == 0:
            print("\nNo samples matched the filter condition!")
            return

        # Save filtered data
        save_filtered_data(
            data=filtered_data,
            output_dir=args.output_dir,
            output_format=args.format,
            shard_size=args.shard_size,
        )

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    args = parse_args()
    main()
