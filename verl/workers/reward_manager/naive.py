# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import os
from openai import OpenAI
from typing import Dict, Any


from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

# Filter reward model URLs to valid http(s) endpoints; ignore blanks
_raw_reward_urls = os.environ.get("REWARD_MODEL_URLS", "")
reward_model_urls = [u.strip() for u in _raw_reward_urls.split(",") if u.strip().startswith(("http://", "https://"))]
reward_model_key = os.environ.get("REWARD_MODEL_KEY", "NONE")


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_model_clients = [OpenAI(base_url=url, api_key=reward_model_key) for url in reward_model_urls] if reward_model_urls else None

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # ========== DEBUG: Print data structure info ==========
        import sys
        print(f"[DEBUG NaiveRewardManager] batch keys: {list(data.batch.keys())}", file=sys.stderr)
        print(f"[DEBUG NaiveRewardManager] non_tensor_batch keys: {list(data.non_tensor_batch.keys())}", file=sys.stderr)
        print(f"[DEBUG NaiveRewardManager] reward_model_clients: {self.reward_model_clients}", file=sys.stderr)
        print(f"[DEBUG NaiveRewardManager] len(data): {len(data)}", file=sys.stderr)
        print(f"[DEBUG NaiveRewardManager] data.batch['responses'].shape: {data.batch['responses'].shape}", file=sys.stderr)
        if hasattr(data.batch, 'get') and 'attention_mask' in data.batch:
            print(f"[DEBUG NaiveRewardManager] data.batch['attention_mask'].shape: {data.batch['attention_mask'].shape}", file=sys.stderr)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Thread-safe dict for tracking printed data sources
        print_lock = threading.Lock()
        def process_row(args):
            i, data_item, already_print_data_sources = args
            # NOTE: Changed from defaultdict(list) to dict to avoid wrapping values in lists
            # Previously, reward_extra_info[key].append(value) would create lists like ["204"]
            # which caused "unhashable type: 'list'" error in metric_utils.calc_maj_val
            reward_extra_info = {}

            # ========== DEBUG: Check if 'prompts' exists ==========
            if 'prompts' not in data_item.batch:
                print(f"[DEBUG] Row {i}: 'prompts' key not in data_item.batch! Available keys: {list(data_item.batch.keys())}", file=sys.stderr)
                # Try to construct prompts from input_ids if available
                if 'input_ids' in data_item.batch:
                    print(f"[DEBUG] Row {i}: Using 'input_ids' as prompts fallback", file=sys.stderr)
                    prompt_ids = data_item.batch['input_ids']
                else:
                    print(f"[DEBUG] Row {i}: ERROR - No 'input_ids' either!", file=sys.stderr)
                    return i, 0, 1, {}
            else:
                prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            attention_mask = data_item.batch['attention_mask']

            # ========== DEBUG: Print tensor shapes and values ==========
            print(f"[DEBUG Row {i}] prompt_ids.shape: {prompt_ids.shape}", file=sys.stderr)
            print(f"[DEBUG Row {i}] prompt_length: {prompt_length}", file=sys.stderr)
            print(f"[DEBUG Row {i}] attention_mask.shape: {attention_mask.shape}", file=sys.stderr)
            print(f"[DEBUG Row {i}] response_ids.shape: {response_ids.shape}", file=sys.stderr)
            print(f"[DEBUG Row {i}] attention_mask sum: {attention_mask.sum()}", file=sys.stderr)
            print(f"[DEBUG Row {i}] attention_mask[prompt_length:] sum: {attention_mask[prompt_length:].sum()}", file=sys.stderr)
            print(f"[DEBUG Row {i}] response_ids (first 100): {response_ids[:100]}", file=sys.stderr)
            print(f"[DEBUG Row {i}] response_ids (min value): {response_ids.min()}", file=sys.stderr)
            print(f"[DEBUG Row {i}] response_ids (max value): {response_ids.max()}", file=sys.stderr)
            print(f"[DEBUG Row {i}] tokenizer vocab_size: {self.tokenizer.vocab_size}", file=sys.stderr)

            valid_response_length = attention_mask[prompt_length:].sum()
            valid_response_ids = response_ids[:int(valid_response_length)]

            print(f"[DEBUG Row {i}] valid_response_length: {valid_response_length}", file=sys.stderr)
            print(f"[DEBUG Row {i}] valid_response_ids.shape: {valid_response_ids.shape}", file=sys.stderr)
            print(f"[DEBUG Row {i}] valid_response_ids (first 100): {valid_response_ids[:100]}", file=sys.stderr)

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # ========== Co-GRPO Diagnostic: Stream Detection ==========
            # Both control and exp batches contain the same metadata structure.
            # Heuristic: Check if exp_hints/exp_critiques lists are populated
            exp_hints_val = data_item.non_tensor_batch.get('exp_hints', [])
            exp_critiques_val = data_item.non_tensor_batch.get('exp_critiques', [])
            has_exp_metadata = (isinstance(exp_hints_val, list) and any(exp_hints_val)) or \
                               (isinstance(exp_critiques_val, list) and any(c and c != 'No interventions' for c in exp_critiques_val))
            is_exp_stream = has_exp_metadata

            # Get ground_truth before if-else
            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "")

            if is_exp_stream:
                # Experimental stream
                print(f"\n{'='*70}", file=sys.stderr)
                print(f"[EXP_STREAM] Sample {i}", file=sys.stderr)
                print(f"{'='*70}", file=sys.stderr)
                print(f"[Full Response]:\n{response_str}\n", file=sys.stderr)

                # Dump to file
                import json, os
                dump_dir = '/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/verifier_outputs'
                os.makedirs(dump_dir, exist_ok=True)

                exp_data = {
                    'sample_idx': i,
                    'stream_type': 'exp',
                    'full_response': response_str,
                    'exp_hints': data_item.non_tensor_batch.get('exp_hints', ''),
                    'exp_critiques': data_item.non_tensor_batch.get('exp_critiques', ''),
                    'ground_truth': ground_truth,
                    'data_source': data_source,
                }

                dump_file = os.path.join(dump_dir, f'exp_sample_{i}.json')
                with open(dump_file, 'w', encoding='utf-8') as f:
                    json.dump(exp_data, f, ensure_ascii=False, indent=2)

                print(f"[DUMP] Saved to: {dump_file}\n", file=sys.stderr)
                print(f"{'='*70}\n", file=sys.stderr)
            else:
                # Control stream (or couldn't determine)
                print(f"\n[CONTROL_STREAM] Sample {i}", file=sys.stderr)
                print(f"[Response]:\n{response_str}\n", file=sys.stderr)

                # Dump to file
                import json, os
                dump_dir = '/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/control_outputs'
                os.makedirs(dump_dir, exist_ok=True)

                control_data = {
                    'sample_idx': i,
                    'stream_type': 'control',
                    'response': response_str,
                    'ground_truth': ground_truth,
                    'data_source': data_source,
                }

                dump_file = os.path.join(dump_dir, f'control_sample_{i}.json')
                with open(dump_file, 'w', encoding='utf-8') as f:
                    json.dump(control_data, f, ensure_ascii=False, indent=2)

                print(f"[DUMP] Saved to: {dump_file}\n", file=sys.stderr)

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # ========== DEBUG: Print compute_score inputs ==========
            print(f"[DEBUG Row {i}] data_source: {data_source}", file=sys.stderr)
            print(f"[DEBUG Row {i}] response_str (first 200 chars): {response_str[:200]}", file=sys.stderr)
            print(f"[DEBUG Row {i}] ground_truth: {ground_truth}", file=sys.stderr)
            print(f"[DEBUG Row {i}] reward_model_clients: {self.reward_model_clients}", file=sys.stderr)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                reward_model_clients=self.reward_model_clients
            )

            # ========== DEBUG: Print compute_score output ==========
            print(f"[DEBUG Row {i}] score result: {score} (type: {type(score)})", file=sys.stderr)
            
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                # NOTE: Changed from append(value) to direct assignment to avoid wrapping in lists
                for key, value in score.items():
                    reward_extra_info[key] = value
            else:
                reward = score
            
            with print_lock:
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print(f"[score]", score)  
            return i, reward, valid_response_length, reward_extra_info

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_row, args))

        # Fill reward tensor with results
        reward_extra_info = {}
        for i, score, valid_response_length, sample_reward_extra_info in results:
            reward_tensor[i, valid_response_length - 1] = score
            # Update extra info at the list index
            for key, value in sample_reward_extra_info.items():
                if key not in reward_extra_info:
                    reward_extra_info[key] = [None] * len(data)
                reward_extra_info[key][i] = value
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
