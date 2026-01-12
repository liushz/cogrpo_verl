"""
Oracle API client for calling gpt-oss-120b model (thinking model).
Handles batch processing, retries, and error handling.
"""
import os
import time
import random
from typing import List, Dict, Optional, Callable
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Disable proxy for internal Oracle services
_internal_ips = "100.101.167.1,100.96.167.1,100.98.194.1,100.100.107.1,100.99.37.1,localhost,127.0.0.1"
existing_no_proxy = os.environ.get('NO_PROXY') or os.environ.get('no_proxy') or ''
if existing_no_proxy:
    _internal_ips = existing_no_proxy + ',' + _internal_ips
os.environ['NO_PROXY'] = _internal_ips
os.environ['no_proxy'] = _internal_ips


class OracleAPIClient:
    """Client for calling Oracle model APIs."""

    def __init__(self, urls: List[str], max_retries: int = 3, timeout: int = 300):
        """
        Initialize Oracle API client.

        Args:
            urls: List of API endpoint URLs
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds (increased for thinking model)
        """
        self.urls = urls if isinstance(urls, list) else urls.split()
        self.max_retries = max_retries
        self.timeout = timeout
        self.current_url_index = 0
        # Create OpenAI clients for each URL
        self.clients = [OpenAI(base_url=url.strip(), api_key="EMPTY") for url in self.urls]

    def _get_next_client(self) -> OpenAI:
        """Round-robin client selection."""
        client = self.clients[self.current_url_index]
        self.current_url_index = (self.current_url_index + 1) % len(self.clients)
        return client, self.urls[self.current_url_index - 1]

    def _call_api(self, prompt: str, client: Optional[OpenAI] = None, url: Optional[str] = None) -> Optional[str]:
        """
        Call Oracle API with a single prompt.
        Uses gpt-oss-120b thinking model with max_completion_tokens and reasoning_effort.

        Args:
            prompt: The prompt text
            client: Specific OpenAI client to use (if None, uses round-robin)
            url: URL for logging purposes

        Returns:
            Response text, or None if failed
        """
        if client is None:
            client, url = self._get_next_client()

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_completion_tokens=32768,  # For gpt-oss-120b thinking model
                    reasoning_effort="high",  # Enable high reasoning for better analysis
                    timeout=self.timeout
                )
                if response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content
                else:
                    print(f"Warning: Unexpected response format from {url}")
                    return None

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                    # Try next client on retry
                    client, url = self._get_next_client()
                else:
                    print(f"Error calling {url} after {self.max_retries} attempts: {e}")
                    return None

        return None
    
    def call_batch(self, prompts: List[str], max_workers: int = 10) -> List[Optional[str]]:
        """
        Call API for a batch of prompts in parallel.
        
        Args:
            prompts: List of prompts
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of responses (same order as prompts, None for failed calls)
        """
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._call_api, prompt): i
                for i, prompt in enumerate(prompts)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Error processing prompt {index}: {e}")
                    results[index] = None
        
        return results
    
    def call_with_retry(self, prompt: str, post_process_fn: Optional[Callable] = None) -> Optional[str]:
        """
        Call API with retry and post-processing.
        
        Args:
            prompt: The prompt text
            post_process_fn: Optional function to process the response
            
        Returns:
            Response text, or None if failed
        """
        response = self._call_api(prompt)
        if response is None:
            return None
        
        if post_process_fn:
            return post_process_fn(response)
        
        return response

