#!/usr/bin/env python3
"""
Step 4: Generate positive samples (GO) from correct responses.
Uses semantic splitting to create multiple context samples per response.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.semantic_splitter import SemanticSplitter


def generate_positive_samples(item: Dict, splitter: SemanticSplitter, num_samples_per_item: int = 3) -> List[Dict]:
    """
    Generate positive samples from a correct response.
    
    Args:
        item: Correct item with question and output
        splitter: SemanticSplitter instance
        num_samples_per_item: Number of different context samples to generate
        
    Returns:
        List of positive samples in messages format
    """
    question = item.get('question', '')
    output = item.get('output', '')
    
    if not question or not output:
        return []
    
    # Get semantic split contexts
    contexts = splitter.get_split_contexts(output, num_samples=num_samples_per_item)
    
    samples = []
    for context in contexts:
        if len(context) < 50:  # Too short
            continue
        if len(context) > 2048:  # Too long
            context = context[:2048]
        
        sample = {
            "messages": [
                {"role": "user", "content": f"Question: {question}\n\n{context}"},
                {"role": "assistant", "content": "<GO>"}
            ],
            "intervention_type": "GO"
        }
        samples.append(sample)
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate positive samples (GO)')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input file: pool_a_correct.jsonl')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file: positive_samples_go.jsonl')
    parser.add_argument('--samples_per_item', type=int, default=3,
                       help='Number of samples to generate per item')
    parser.add_argument('--min_context_length', type=int, default=50,
                       help='Minimum context length in characters')
    parser.add_argument('--max_context_length', type=int, default=2048,
                       help='Maximum context length in characters')
    parser.add_argument('--target_count', type=int, default=None,
                       help='Target number of samples (None = generate from all items)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize splitter
    splitter = SemanticSplitter()
    
    # Load items
    print(f"Loading items from: {args.input_file}")
    items = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    
    print(f"Total items: {len(items)}")
    
    # Generate samples
    all_samples = []
    
    for i, item in enumerate(items):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(items)}...")
        
        samples = generate_positive_samples(item, splitter, args.samples_per_item)
        
        # Filter by length
        valid_samples = []
        for sample in samples:
            context = sample['messages'][0]['content']
            if args.min_context_length <= len(context) <= args.max_context_length:
                valid_samples.append(sample)
        
        all_samples.extend(valid_samples)
        
        # Check if we've reached target count
        if args.target_count and len(all_samples) >= args.target_count:
            all_samples = all_samples[:args.target_count]
            break
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Save
    print(f"Saving {len(all_samples)} positive samples to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Statistics
    print("\n=== Statistics ===")
    print(f"Total input items: {len(items)}")
    print(f"Generated samples: {len(all_samples)}")
    print(f"Average samples per item: {len(all_samples) / len(items):.2f}")
    
    if all_samples:
        avg_len = sum(len(s['messages'][0]['content']) for s in all_samples) / len(all_samples)
        print(f"Average context length: {avg_len:.0f} chars")
    
    print("\n✅ Step 4 completed successfully!")


if __name__ == '__main__':
    main()

