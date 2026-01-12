"""
Semantic splitter for dynamic granularity based on semantic atoms.
Supports splitting at: periods, commas, logical connectors, before equals signs.
"""
import re
from typing import List, Tuple, Optional


class SemanticSplitter:
    """Split text based on semantic boundaries rather than fixed rules."""
    
    # Logical connectors that indicate semantic boundaries
    LOGICAL_CONNECTORS = [
        r'Therefore\s*,?\s*',
        r'So\s*,?\s*',
        r'Given that\s*,?\s*',
        r'Thus\s*,?\s*',
        r'Hence\s*,?\s*',
        r'Consequently\s*,?\s*',
        r'As a result\s*,?\s*',
        r'It follows that\s*,?\s*',
        r'We have\s*,?\s*',
        r'Note that\s*,?\s*',
        r'Observe that\s*,?\s*',
    ]
    
    def __init__(self):
        # Compile regex patterns for logical connectors
        self.logical_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.LOGICAL_CONNECTORS]
    
    def find_insertion_point(self, text: str, anchor_snippet: str) -> Optional[int]:
        """
        Find the unique insertion point in text based on anchor_snippet.
        
        Args:
            text: The full text to search in
            anchor_snippet: The anchor text snippet (5-10 chars) to find
            
        Returns:
            The index position after the anchor snippet, or None if not found/not unique
        """
        # Find all occurrences of the anchor snippet
        matches = []
        start = 0
        while True:
            pos = text.find(anchor_snippet, start)
            if pos == -1:
                break
            matches.append(pos)
            start = pos + 1
        
        if len(matches) == 0:
            return None
        
        if len(matches) > 1:
            # Multiple matches - try to find the most reasonable one
            # Prefer matches that are not at the very beginning or end
            for match_pos in matches:
                # Check if it's not at the very start (first 10 chars) or very end (last 10 chars)
                if 10 < match_pos < len(text) - 10:
                    return match_pos + len(anchor_snippet)
            # If all are at edges, use the first one
            return matches[0] + len(anchor_snippet)
        
        # Unique match
        return matches[0] + len(anchor_snippet)
    
    def extract_context_before_insertion(self, text: str, insertion_point: int) -> str:
        """
        Extract all content before the insertion point as context.
        
        Args:
            text: The full text
            insertion_point: The index where insertion should happen
            
        Returns:
            The context string (everything before insertion_point)
        """
        if insertion_point < 0 or insertion_point > len(text):
            return text
        return text[:insertion_point].strip()
    
    def semantic_split(self, text: str, max_chunks: Optional[int] = None) -> List[Tuple[int, str]]:
        """
        Split text at semantic boundaries.
        
        Args:
            text: The text to split
            max_chunks: Maximum number of chunks to return (None = all)
            
        Returns:
            List of (position, boundary_type) tuples indicating split points
            boundary_type can be: 'period', 'comma', 'logical', 'equals'
        """
        split_points = []
        
        # 1. Find periods (sentence endings)
        for match in re.finditer(r'\.\s+', text):
            split_points.append((match.end(), 'period'))
        
        # 2. Find logical connectors
        for pattern in self.logical_patterns:
            for match in pattern.finditer(text):
                split_points.append((match.end(), 'logical'))
        
        # 3. Find commas followed by long derivations (heuristic: comma + space + capital letter or number)
        for match in re.finditer(r',\s+(?=[A-Z0-9])', text):
            # Check if what follows is substantial (at least 20 chars)
            following_text = text[match.end():match.end()+50]
            if len(following_text.strip()) > 20:
                split_points.append((match.end(), 'comma'))
        
        # 4. Find equals signs (before calculations)
        for match in re.finditer(r'\s+=\s+', text):
            split_points.append((match.start(), 'equals'))
        
        # Sort by position and remove duplicates
        split_points = sorted(set(split_points), key=lambda x: x[0])
        
        # Filter out split points that are too close to each other (< 20 chars)
        filtered_points = []
        last_pos = 0
        for pos, boundary_type in split_points:
            if pos - last_pos >= 20:
                filtered_points.append((pos, boundary_type))
                last_pos = pos
        
        # Limit number of chunks if specified
        if max_chunks and len(filtered_points) > max_chunks:
            # Select evenly distributed points
            step = len(filtered_points) // max_chunks
            filtered_points = [filtered_points[i * step] for i in range(max_chunks)]
        
        return filtered_points
    
    def get_split_contexts(self, text: str, num_samples: int = 3) -> List[str]:
        """
        Get multiple context samples from a text by splitting at different semantic boundaries.
        Used for generating positive samples (GO).
        
        Args:
            text: The full text
            num_samples: Number of different context samples to generate
            
        Returns:
            List of context strings (each ending at a different semantic boundary)
        """
        split_points = self.semantic_split(text, max_chunks=num_samples * 2)
        
        if not split_points:
            # No split points found, return the full text
            return [text]
        
        # Select evenly distributed split points
        contexts = []
        step = max(1, len(split_points) // num_samples)
        for i in range(0, len(split_points), step):
            if len(contexts) >= num_samples:
                break
            pos, _ = split_points[i]
            context = text[:pos].strip()
            if len(context) > 50:  # Only include substantial contexts
                contexts.append(context)
        
        # If we don't have enough, add the full text as the last one
        if len(contexts) < num_samples and text.strip():
            contexts.append(text.strip())
        
        return contexts[:num_samples]

