"""
JSON validation utilities for Oracle API responses.
"""
import json
import re
from typing import Dict, Optional, Tuple


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON from text that may contain extra content.
    Handles cases where Oracle returns JSON wrapped in markdown or other text.
    
    Args:
        text: Raw text that may contain JSON
        
    Returns:
        Extracted JSON string, or None if not found
    """
    # Try to find JSON object
    # Pattern: { ... }
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            # Try to parse it
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue
    
    # If no valid JSON found, try parsing the whole text
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    
    return None


def validate_intervention_json(data: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate the intervention JSON structure from Oracle.
    
    Args:
        data: Parsed JSON dictionary
        
    Returns:
        (is_valid, error_message)
    """
    required_fields = ['intervention_type', 'insert_after_snippet', 'verifier_content']
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate intervention_type
    if data['intervention_type'] not in ['Warning', 'Correction']:
        return False, f"Invalid intervention_type: {data['intervention_type']}. Must be 'Warning' or 'Correction'"
    
    # Validate insert_after_snippet
    snippet = data['insert_after_snippet']
    if not isinstance(snippet, str):
        return False, "insert_after_snippet must be a string"
    if len(snippet) < 5 or len(snippet) > 50:
        return False, f"insert_after_snippet length must be between 5 and 50 chars, got {len(snippet)}"
    
    # Validate verifier_content
    content = data['verifier_content']
    if not isinstance(content, str):
        return False, "verifier_content must be a string"
    if not content.startswith('<WAIT>'):
        return False, "verifier_content must start with '<WAIT>'"
    
    # Check word count (should be <= 15 words)
    words = content.replace('<WAIT>', '').strip().split()
    if len(words) > 15:
        return False, f"verifier_content has {len(words)} words, should be <= 15"
    
    return True, None


def parse_oracle_response(response_text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Parse and validate Oracle API response.
    
    Args:
        response_text: Raw response from Oracle API
        
    Returns:
        (parsed_data, error_message)
        If successful: (dict, None)
        If failed: (None, error_message)
    """
    # Extract JSON from response
    json_str = extract_json_from_text(response_text)
    if json_str is None:
        return None, "No valid JSON found in response"
    
    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {str(e)}"
    
    # Validate structure
    is_valid, error = validate_intervention_json(data)
    if not is_valid:
        return None, error
    
    return data, None

