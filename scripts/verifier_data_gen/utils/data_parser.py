"""
Data parsing utilities for GRPO rollout data.
"""
import json
import re
from typing import Dict, List, Optional, Tuple


def parse_rollout_item(item: Dict) -> Dict:
    """
    Parse a single rollout data item.
    
    Args:
        item: Raw rollout JSON item
        
    Returns:
        Parsed item with extracted fields
    """
    parsed = {
        'input': item.get('input', ''),
        'output': item.get('output', ''),
        'score': item.get('score', 0.0),
        'acc': item.get('acc', False),
        'pred': item.get('pred', ''),
        'ground_truth': item.get('ground_truth', None),
        'step': item.get('step', 0),
        'reward': item.get('reward', 0.0),
    }
    
    # Extract question from input
    # Input format: "system\n[system prompt]\nuser\n[question]\nassistant\n"
    # We need to extract only the content between "user" and "assistant"
    input_text = parsed['input']
    user_pattern = re.search(r'\nuser\n(.*?)(?=\nassistant\n|$)', input_text, re.DOTALL)
    if user_pattern:
        parsed['question'] = user_pattern.group(1).strip()
    else:
        # Fallback: try to find content after "user" tag
        input_parts = input_text.split('\n')
        try:
            user_idx = input_parts.index('user')
            assistant_idx = input_parts.index('assistant') if 'assistant' in input_parts else len(input_parts)
            if user_idx + 1 < assistant_idx:
                parsed['question'] = '\n'.join(input_parts[user_idx + 1:assistant_idx]).strip()
            else:
                parsed['question'] = input_text
        except ValueError:
            parsed['question'] = input_text
    
    return parsed


def load_rollout_files(directory: str, max_files: Optional[int] = None) -> List[Dict]:
    """
    Load all rollout JSONL files from a directory.
    
    Args:
        directory: Directory containing .jsonl files
        max_files: Maximum number of files to load (None = all)
        
    Returns:
        List of parsed rollout items
    """
    import os
    import glob
    
    jsonl_files = sorted(glob.glob(os.path.join(directory, '*.jsonl')))
    
    if max_files:
        jsonl_files = jsonl_files[:max_files]
    
    all_items = []
    for filepath in jsonl_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        parsed = parse_rollout_item(item)
                        all_items.append(parsed)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line in {filepath}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    return all_items


def split_by_accuracy(items: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Split items into correct and incorrect pools based on accuracy.
    
    Args:
        items: List of parsed rollout items
        
    Returns:
        (correct_items, incorrect_items)
    """
    correct = []
    incorrect = []
    
    for item in items:
        if item.get('acc', False):
            correct.append(item)
        else:
            incorrect.append(item)
    
    return correct, incorrect


def extract_reasoning_steps(text: str) -> List[str]:
    """
    Extract reasoning steps from response text.
    Handles various formats: <think>, <think>, numbered steps, etc.
    
    Args:
        text: Full response text
        
    Returns:
        List of reasoning step strings
    """
    steps = []
    
    # Try to extract from <think> tags
    reasoning_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1)
        # Split by common patterns
        # Try numbered steps: "Step 1:", "Step 2:", etc.
        step_pattern = r'Step\s+\d+[:\-]\s*(.*?)(?=Step\s+\d+[:\-]|$)'
        matches = re.finditer(step_pattern, reasoning_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            step_text = match.group(1).strip()
            if step_text:
                steps.append(step_text)
        
        # If no numbered steps found, try splitting by double newlines
        if not steps:
            potential_steps = reasoning_text.split('\n\n')
            steps = [s.strip() for s in potential_steps if s.strip() and len(s.strip()) > 20]
    
    # If no reasoning tags, try to extract from the full text
    if not steps:
        # Try numbered steps in full text
        step_pattern = r'(?:Step\s+\d+[:\-]|^\d+[\.\)]\s+)(.*?)(?=(?:Step\s+\d+[:\-]|^\d+[\.\)]\s+|$))'
        matches = re.finditer(step_pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        for match in matches:
            step_text = match.group(1).strip()
            if step_text and len(step_text) > 20:
                steps.append(step_text)
    
    # Fallback: return the full text as a single step
    if not steps:
        steps = [text.strip()]
    
    return steps

