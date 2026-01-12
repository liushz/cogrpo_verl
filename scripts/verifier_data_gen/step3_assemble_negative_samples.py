#!/usr/bin/env python3
"""
Step 3: Assemble negative samples (Warning and Correction) based on intervention points.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.semantic_splitter import SemanticSplitter


def assemble_negative_sample(item: Dict, splitter: SemanticSplitter) -> Optional[Dict]:
    """
    Assemble a negative sample from intervention data.
    
    Args:
        item: Item with intervention data
        splitter: SemanticSplitter instance
        
    Returns:
        Assembled sample in messages format, or None if failed
    """
    question = item.get('question', '')
    wrong_response = item.get('output', '')
    insert_after_snippet = item.get('insert_after_snippet', '')
    verifier_content = item.get('verifier_content', '')
    intervention_type = item.get('intervention_type', 'Correction')
    
    if not question or not wrong_response or not insert_after_snippet or not verifier_content:
        return None
    
    # Find insertion point
    insertion_point = splitter.find_insertion_point(wrong_response, insert_after_snippet)
    if insertion_point is None:
        # Try to find a similar match (fuzzy matching)
        # For now, skip if exact match fails
        return None
    
    # Extract context before insertion
    context = splitter.extract_context_before_insertion(wrong_response, insertion_point)
    
    if len(context) < 50:  # Too short
        return None
    if len(context) > 2048:  # Too long
        context = context[:2048]
    
    # Assemble messages format
    user_content = f"Question: {question}\n\n{context}"
    
    sample = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": verifier_content}
        ],
        "intervention_type": intervention_type
    }
    
    return sample


def main():
    parser = argparse.ArgumentParser(description='Assemble negative samples')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input file: pool_b_with_interventions.jsonl')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--min_context_length', type=int, default=50,
                       help='Minimum context length in characters')
    parser.add_argument('--max_context_length', type=int, default=2048,
                       help='Maximum context length in characters')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Process items
    warning_samples = []
    correction_samples = []
    failed_count = 0
    anchor_match_failed = 0
    
    for i, item in enumerate(items):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(items)}...")
        
        sample = assemble_negative_sample(item, splitter)
        if sample is None:
            failed_count += 1
            # Check if it's an anchor matching issue
            if item.get('insert_after_snippet') and item.get('output'):
                anchor_match_failed += 1
            continue
        
        # Validate context length
        context = sample['messages'][0]['content']
        if len(context) < args.min_context_length or len(context) > args.max_context_length:
            failed_count += 1
            continue
        
        # Separate by intervention type
        if sample['intervention_type'] == 'Warning':
            warning_samples.append(sample)
        else:
            correction_samples.append(sample)
    
    # Save results
    warning_path = os.path.join(args.output_dir, 'negative_samples_warning.jsonl')
    with open(warning_path, 'w', encoding='utf-8') as f:
        for sample in warning_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved {len(warning_samples)} Warning samples to: {warning_path}")
    
    correction_path = os.path.join(args.output_dir, 'negative_samples_correction.jsonl')
    with open(correction_path, 'w', encoding='utf-8') as f:
        for sample in correction_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Saved {len(correction_samples)} Correction samples to: {correction_path}")
    
    # Statistics
    print("\n=== Statistics ===")
    print(f"Total input items: {len(items)}")
    print(f"Warning samples: {len(warning_samples)}")
    print(f"Correction samples: {len(correction_samples)}")
    print(f"Failed: {failed_count}")
    print(f"Anchor match failed: {anchor_match_failed}")
    
    if warning_samples:
        avg_len = sum(len(s['messages'][0]['content']) for s in warning_samples) / len(warning_samples)
        print(f"Average Warning context length: {avg_len:.0f} chars")
    
    if correction_samples:
        avg_len = sum(len(s['messages'][0]['content']) for s in correction_samples) / len(correction_samples)
        print(f"Average Correction context length: {avg_len:.0f} chars")
    
    print("\n✅ Step 3 completed successfully!")


if __name__ == '__main__':
    main()

