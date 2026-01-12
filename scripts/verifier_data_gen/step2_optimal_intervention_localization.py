#!/usr/bin/env python3
"""
Step 2: Use Oracle to locate optimal intervention points.
Outputs JSON with intervention_type, insert_after_snippet, and verifier_content.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.api_client import OracleAPIClient
from utils.json_validator import parse_oracle_response


def load_prompt_template(template_path: str) -> str:
    """Load prompt template from file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_prompt(template: str, question: str, wrong_response: str, gold_answer: str) -> str:
    """Format prompt template with actual data."""
    prompt = template.replace('{{question}}', question)
    prompt = prompt.replace('{{wrong_response}}', wrong_response)
    prompt = prompt.replace('{{gold_answer}}', str(gold_answer) if gold_answer else 'N/A')
    return prompt


def main():
    parser = argparse.ArgumentParser(description='Locate optimal intervention points using Oracle')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input file: pool_b_incorrect.jsonl')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file: pool_b_with_interventions.jsonl')
    parser.add_argument('--prompt_template', type=str, required=True,
                       help='Path to prompt template file')
    parser.add_argument('--oracle_urls', type=str, required=True,
                       help='Comma-separated list of Oracle API URLs')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Batch size for parallel API calls')
    parser.add_argument('--max_workers', type=int, default=10,
                       help='Maximum number of parallel workers')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing output file')
    parser.add_argument('--test_mode', type=int, default=0,
                       help='Test mode: only process first N items (0 = all)')
    parser.add_argument('--verifier_content_max_words', type=int, default=100,
                       help='Maximum words in verifier_content (default: 100)')
    parser.add_argument('--anchor_snippet_min_len', type=int, default=10,
                       help='Minimum length of anchor snippet (default: 10)')
    parser.add_argument('--anchor_snippet_max_len', type=int, default=50,
                       help='Maximum length of anchor snippet (default: 50)')

    args = parser.parse_args()
    
    # Load prompt template
    template = load_prompt_template(args.prompt_template)
    
    # Initialize API client
    urls = [url.strip() for url in args.oracle_urls.split(',')]
    api_client = OracleAPIClient(urls, max_retries=3, timeout=300)  # 300s for thinking model
    
    # Load input items
    print(f"Loading items from: {args.input_file}")
    items = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    
    if args.test_mode > 0:
        items = items[:args.test_mode]
        print(f"Test mode: Processing only first {args.test_mode} items")
    
    print(f"Total items to process: {len(items)}")
    
    # Check for resume
    processed_ids = set()
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming from existing file: {args.output_file}")
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Use a combination of fields as ID
                    item_id = f"{item.get('question', '')[:50]}_{item.get('output', '')[:50]}"
                    processed_ids.add(item_id)
    
    # Process items in batches
    results = []
    failed_count = 0
    
    for i in range(0, len(items), args.batch_size):
        batch = items[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(items) + args.batch_size - 1) // args.batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} items)...")
        
        # Filter out already processed items
        batch_to_process = []
        batch_indices = []
        for idx, item in enumerate(batch):
            item_id = f"{item.get('question', '')[:50]}_{item.get('output', '')[:50]}"
            if item_id not in processed_ids:
                batch_to_process.append(item)
                batch_indices.append(i + idx)
        
        if not batch_to_process:
            print(f"  All items in this batch already processed, skipping...")
            continue
        
        # Prepare prompts
        prompts = []
        for item in batch_to_process:
            question = item.get('question', '')
            wrong_response = item.get('output', '')
            gold_answer = item.get('ground_truth', item.get('pred', ''))
            prompt = format_prompt(template, question, wrong_response, gold_answer)
            prompts.append(prompt)
        
        # Call API in parallel
        responses = api_client.call_batch(prompts, max_workers=args.max_workers)
        
        # Process responses
        batch_results = []
        batch_failed = 0
        
        for item, response_text in zip(batch_to_process, responses):
            if response_text is None:
                batch_failed += 1
                failed_count += 1
                continue
            
            # Parse response
            intervention_data, error = parse_oracle_response(
                response_text,
                verifier_content_max_words=args.verifier_content_max_words,
                anchor_snippet_min_len=args.anchor_snippet_min_len,
                anchor_snippet_max_len=args.anchor_snippet_max_len
            )
            if intervention_data is None:
                print(f"  Warning: Failed to parse response: {error}")
                batch_failed += 1
                failed_count += 1
                continue
            
            # Add intervention data
            result = item.copy()
            result['intervention_type'] = intervention_data['intervention_type']
            result['insert_after_snippet'] = intervention_data.get('insert_after_snippet', '')
            result['verifier_content'] = intervention_data['verifier_content']
            result['oracle_response'] = response_text  # Keep raw response for debugging
            batch_results.append(result)
            results.append(result)
        
        # Save intermediate results
        if batch_results:
            with open(args.output_file, 'a', encoding='utf-8') as f:
                for result in batch_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"  Batch {batch_num} completed. Success: {len(batch_results)}, Failed: {batch_failed}")
    
    # Final statistics
    print("\n=== Statistics ===")
    print(f"Total processed: {len(items)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {failed_count}")
    
    if results:
        warning_count = sum(1 for r in results if r.get('intervention_type') == 'Warning')
        correction_count = sum(1 for r in results if r.get('intervention_type') == 'Correction')
        print(f"Warning type: {warning_count}")
        print(f"Correction type: {correction_count}")
    
    print(f"\n✅ Step 2 completed! Results saved to: {args.output_file}")


if __name__ == '__main__':
    main()

