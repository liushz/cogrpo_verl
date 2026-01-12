#!/usr/bin/env python3
"""
Step 1: Extract and classify rollout data into Pool A (correct) and Pool B (incorrect).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_parser import load_rollout_files, split_by_accuracy


def main():
    parser = argparse.ArgumentParser(description='Extract and classify rollout data')
    parser.add_argument('--rollout_dir', type=str, required=True,
                       help='Directory containing rollout JSONL files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for classified data')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (None = all)')
    parser.add_argument('--test_mode', type=int, default=0,
                       help='Test mode: only process first N items (0 = all)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading rollout files from: {args.rollout_dir}")
    items = load_rollout_files(args.rollout_dir, max_files=args.max_files)
    
    if args.test_mode > 0:
        items = items[:args.test_mode]
        print(f"Test mode: Processing only first {args.test_mode} items")
    
    print(f"Total items loaded: {len(items)}")
    
    # Split by accuracy
    correct_items, incorrect_items = split_by_accuracy(items)
    
    print(f"Pool A (correct): {len(correct_items)} items")
    print(f"Pool B (incorrect): {len(incorrect_items)} items")
    
    # Save Pool A
    pool_a_path = os.path.join(args.output_dir, 'pool_a_correct.jsonl')
    with open(pool_a_path, 'w', encoding='utf-8') as f:
        for item in correct_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved Pool A to: {pool_a_path}")
    
    # Save Pool B
    pool_b_path = os.path.join(args.output_dir, 'pool_b_incorrect.jsonl')
    with open(pool_b_path, 'w', encoding='utf-8') as f:
        for item in incorrect_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved Pool B to: {pool_b_path}")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Correct ratio: {len(correct_items) / len(items) * 100:.1f}%")
    print(f"Incorrect ratio: {len(incorrect_items) / len(items) * 100:.1f}%")
    
    print("\n✅ Step 1 completed successfully!")


if __name__ == '__main__':
    main()

