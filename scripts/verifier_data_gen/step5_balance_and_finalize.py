#!/usr/bin/env python3
"""
Step 5: Balance and finalize training data.
Mixes positive (GO), warning, and correction samples according to specified ratios.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict


def load_samples(filepath: str) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def balance_samples(
    positive_samples: List[Dict],
    warning_samples: List[Dict],
    correction_samples: List[Dict],
    positive_ratio: float = 0.80,
    warning_ratio: float = 0.10,
    correction_ratio: float = 0.10
) -> List[Dict]:
    """
    Balance samples according to specified ratios.
    
    Args:
        positive_samples: List of GO samples
        warning_samples: List of Warning samples
        correction_samples: List of Correction samples
        positive_ratio: Target ratio for positive samples
        warning_ratio: Target ratio for warning samples
        correction_ratio: Target ratio for correction samples
        
    Returns:
        Balanced list of samples
    """
    # Normalize ratios
    total_ratio = positive_ratio + warning_ratio + correction_ratio
    positive_ratio /= total_ratio
    warning_ratio /= total_ratio
    correction_ratio /= total_ratio
    
    # Determine target counts based on available negative samples
    total_negative = len(warning_samples) + len(correction_samples)
    if total_negative == 0:
        print("Warning: No negative samples available!")
        return positive_samples
    
    # Calculate target counts
    # If we have N negative samples, we want positive_ratio / (1 - positive_ratio) * N positive samples
    target_positive = int(total_negative * positive_ratio / (1 - positive_ratio))
    target_warning = int(total_negative * warning_ratio / (1 - positive_ratio))
    target_correction = int(total_negative * correction_ratio / (1 - positive_ratio))
    
    # Sample from available pools
    sampled_positive = random.sample(positive_samples, min(target_positive, len(positive_samples)))
    sampled_warning = random.sample(warning_samples, min(target_warning, len(warning_samples)))
    sampled_correction = random.sample(correction_samples, min(target_correction, len(correction_samples)))
    
    # Combine
    all_samples = sampled_positive + sampled_warning + sampled_correction
    
    # Shuffle
    random.shuffle(all_samples)
    
    return all_samples


def validate_sample(sample: Dict) -> bool:
    """Validate a sample has correct format."""
    if 'messages' not in sample:
        return False
    if not isinstance(sample['messages'], list) or len(sample['messages']) < 2:
        return False
    if sample['messages'][0].get('role') != 'user':
        return False
    if sample['messages'][1].get('role') != 'assistant':
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Balance and finalize training data')
    parser.add_argument('--positive_file', type=str, required=True,
                       help='Input file: positive_samples_go.jsonl')
    parser.add_argument('--warning_file', type=str, required=True,
                       help='Input file: negative_samples_warning.jsonl')
    parser.add_argument('--correction_file', type=str, required=True,
                       help='Input file: negative_samples_correction.jsonl')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file: verifier_sft_train_data.jsonl')
    parser.add_argument('--positive_ratio', type=float, default=0.80,
                       help='Target ratio for positive samples')
    parser.add_argument('--warning_ratio', type=float, default=0.10,
                       help='Target ratio for warning samples')
    parser.add_argument('--correction_ratio', type=float, default=0.10,
                       help='Target ratio for correction samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load samples
    print("Loading samples...")
    positive_samples = load_samples(args.positive_file)
    warning_samples = load_samples(args.warning_file)
    correction_samples = load_samples(args.correction_file)
    
    print(f"Positive (GO): {len(positive_samples)}")
    print(f"Warning: {len(warning_samples)}")
    print(f"Correction: {len(correction_samples)}")
    
    # Balance
    print("\nBalancing samples...")
    balanced_samples = balance_samples(
        positive_samples,
        warning_samples,
        correction_samples,
        args.positive_ratio,
        args.warning_ratio,
        args.correction_ratio
    )
    
    # Validate
    print("Validating samples...")
    valid_samples = []
    invalid_count = 0
    for sample in balanced_samples:
        if validate_sample(sample):
            valid_samples.append(sample)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"Warning: {invalid_count} invalid samples removed")
    
    # Save
    print(f"Saving {len(valid_samples)} samples to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in valid_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Statistics
    print("\n=== Final Statistics ===")
    print(f"Total samples: {len(valid_samples)}")
    
    go_count = sum(1 for s in valid_samples if s.get('intervention_type') == 'GO')
    warning_count = sum(1 for s in valid_samples if s.get('intervention_type') == 'Warning')
    correction_count = sum(1 for s in valid_samples if s.get('intervention_type') == 'Correction')
    
    print(f"GO samples: {go_count} ({go_count/len(valid_samples)*100:.1f}%)")
    print(f"Warning samples: {warning_count} ({warning_count/len(valid_samples)*100:.1f}%)")
    print(f"Correction samples: {correction_count} ({correction_count/len(valid_samples)*100:.1f}%)")
    
    # Sample examples
    print("\n=== Sample Examples ===")
    for intervention_type in ['GO', 'Warning', 'Correction']:
        examples = [s for s in valid_samples if s.get('intervention_type') == intervention_type]
        if examples:
            example = random.choice(examples)
            print(f"\n{intervention_type} example:")
            print(json.dumps(example, ensure_ascii=False, indent=2))
    
    print("\n✅ Step 5 completed successfully!")


if __name__ == '__main__':
    main()

