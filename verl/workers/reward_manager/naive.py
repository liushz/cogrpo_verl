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
import numpy as np
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

        import sys
        import os
        # NOTE: Disabled ad-hoc debug printing for stability/clean logs.
        # If needed again, re-enable by restoring the env-driven flags.
        debug_enabled = False
        debug_limit = 0

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # ========== DEBUG: Print data structure info ==========
        if debug_enabled:
            print(f"[DEBUG NaiveRewardManager] batch keys: {list(data.batch.keys())}", file=sys.stderr)
            print(f"[DEBUG NaiveRewardManager] non_tensor_batch keys: {list(data.non_tensor_batch.keys())}", file=sys.stderr)
            print(f"[DEBUG NaiveRewardManager] reward_model_clients: {self.reward_model_clients}", file=sys.stderr)
            print(f"[DEBUG NaiveRewardManager] len(data): {len(data)}", file=sys.stderr)
            print(f"[DEBUG NaiveRewardManager] data.batch['responses'].shape: {data.batch['responses'].shape}", file=sys.stderr)
            if hasattr(data.batch, 'get') and 'attention_mask' in data.batch:
                print(f"[DEBUG NaiveRewardManager] data.batch['attention_mask'].shape: {data.batch['attention_mask'].shape}", file=sys.stderr)

        # Use a single timestamp per batch to avoid creating a directory per sample
        from datetime import datetime

        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Prefer upstream-provided experiment id, then env override, else timestamp-based.
        experiment_id = data.non_tensor_batch.get("experiment_id") or os.environ.get("VERL_EXPERIMENT_ID") or f"exp_{batch_timestamp}"
        # Batch tag to avoid filename collisions; prefer upstream batch ids/steps.
        batch_tag = data.non_tensor_batch.get("batch_id") or data.non_tensor_batch.get("step") or data.non_tensor_batch.get("global_step") or batch_timestamp

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Thread-safe dict for tracking printed data sources
        print_lock = threading.Lock()
        def process_row(args):
            i, data_item, already_print_data_sources = args
            def _to_jsonable_scalar(v):
                # json.dump cannot serialize numpy scalar types reliably.
                if isinstance(v, (np.generic,)):
                    return v.item()
                # Be defensive: DataProtoItem should yield per-sample scalars, but handle arrays/tensors too.
                if isinstance(v, np.ndarray):
                    if v.size == 1:
                        return v.reshape(-1)[0].item() if isinstance(v.reshape(-1)[0], np.generic) else v.reshape(-1)[0]
                    return v.tolist()
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        return v.item()
                    return v.detach().cpu().tolist()
                return v
            # NOTE: Changed from defaultdict(list) to dict to avoid wrapping values in lists
            # Previously, reward_extra_info[key].append(value) would create lists like ["204"]
            # which caused "unhashable type: 'list'" error in metric_utils.calc_maj_val
            reward_extra_info = {}

            # ========== DEBUG: Check if 'prompts' exists ==========
            if 'prompts' not in data_item.batch:
                if debug_enabled and (debug_limit <= 0 or i < debug_limit):
                    print(f"[DEBUG] Row {i}: 'prompts' key not in data_item.batch! Available keys: {list(data_item.batch.keys())}", file=sys.stderr)
                # Try to construct prompts from input_ids if available
                if 'input_ids' in data_item.batch:
                    if debug_enabled and (debug_limit <= 0 or i < debug_limit):
                        print(f"[DEBUG] Row {i}: Using 'input_ids' as prompts fallback", file=sys.stderr)
                    prompt_ids = data_item.batch['input_ids']
                else:
                    if debug_enabled and (debug_limit <= 0 or i < debug_limit):
                        print(f"[DEBUG] Row {i}: ERROR - No 'input_ids' either!", file=sys.stderr)
                    # Return dummy dump_record to match return signature
                    dummy_dump_record = {
                        'sample_idx': i,
                        'full_response': '',
                        'ground_truth': '',
                        'data_source': '',
                        'experiment_id': '',
                        'timestamp': '',
                        'stream_type': 'error',
                        'error': 'No input_ids or prompts found'
                    }
                    return i, 0, 1, {}, dummy_dump_record
            else:
                prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            # Convert to Python int for indexing (torch scalars can't be used as slice indices)
            valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            attention_mask = data_item.batch['attention_mask']

            # ========== DEBUG: Print tensor shapes and values ==========
            if debug_enabled and (debug_limit <= 0 or i < debug_limit):
                print(f"[DEBUG Row {i}] prompt_ids.shape: {prompt_ids.shape}", file=sys.stderr)
                print(f"[DEBUG Row {i}] prompt_length: {prompt_length}", file=sys.stderr)
                print(f"[DEBUG Row {i}] valid_prompt_length: {valid_prompt_length}", file=sys.stderr)
                print(f"[DEBUG Row {i}] attention_mask.shape: {attention_mask.shape}", file=sys.stderr)
                print(f"[DEBUG Row {i}] response_ids.shape: {response_ids.shape}", file=sys.stderr)
                print(f"[DEBUG Row {i}] attention_mask sum: {attention_mask.sum()}", file=sys.stderr)
                print(f"[DEBUG Row {i}] attention_mask[prompt_length:] sum: {attention_mask[prompt_length:].sum()}", file=sys.stderr)
                print(f"[DEBUG Row {i}] response_ids (first 100): {response_ids[:100]}", file=sys.stderr)
                print(f"[DEBUG Row {i}] response_ids (min value): {response_ids.min()}", file=sys.stderr)
                print(f"[DEBUG Row {i}] response_ids (max value): {response_ids.max()}", file=sys.stderr)
                print(f"[DEBUG Row {i}] tokenizer vocab_size: {self.tokenizer.vocab_size}", file=sys.stderr)

            # Convert to Python int for indexing and tensor indexing
            valid_response_length = int(attention_mask[prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            if debug_enabled and (debug_limit <= 0 or i < debug_limit):
                print(f"[DEBUG Row {i}] valid_response_length: {valid_response_length}", file=sys.stderr)
                print(f"[DEBUG Row {i}] valid_response_ids.shape: {valid_response_ids.shape}", file=sys.stderr)
                print(f"[DEBUG Row {i}] valid_response_ids (first 100): {valid_response_ids[:100]}", file=sys.stderr)

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            # Decode full response for dump (including hints)
            # This preserves the complete intervention flow for exp streams
            full_response_with_hints = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # [CORRECT FIX] Always use the full response string.
            # The Verifier (Compass) handles hints and extracts the final answer.
            response_str = full_response_with_hints

            # ========== Co-GRPO: Stream Type Detection ==========
            # Trainer passes stream_type via non_tensor_batch (see ray_trainer.py:1311, 1328)
            # Fallback to heuristic detection if not set (for backward compatibility)
            stream_type_from_trainer = data_item.non_tensor_batch.get('stream_type', None)

            if stream_type_from_trainer:
                # Trainer explicitly marked stream type (CoGRPO case)
                is_exp_stream = (stream_type_from_trainer == 'exp')
            else:
                # Fallback: Heuristic detection (for non-CoGRPO or old code)
                # NOTE: DataProtoItem extracts single elements from np.ndarray, so hints/critiques are strings or None
                hints_val = data_item.non_tensor_batch.get('hints', '')
                critiques_val = data_item.non_tensor_batch.get('critiques', '')

                # Check for non-empty hints or critiques (string or array)
                has_exp_metadata = False
                if hints_val:
                    # Could be string (DataProtoItem) or np.ndarray (raw batch)
                    if isinstance(hints_val, str):
                        has_exp_metadata = hints_val.strip() != ''
                    elif isinstance(hints_val, np.ndarray):
                        has_exp_metadata = hints_val.size > 0 and any(hints_val)

                if not has_exp_metadata and critiques_val:
                    if isinstance(critiques_val, str):
                        has_exp_metadata = critiques_val.strip() != '' and critiques_val != 'No interventions'
                    elif isinstance(critiques_val, np.ndarray):
                        has_exp_metadata = critiques_val.size > 0 and any(c and c != 'No interventions' for c in critiques_val)

                # Also check num_interventions as additional signal
                num_interventions = data_item.non_tensor_batch.get('num_interventions', None)
                if num_interventions is not None:
                    if isinstance(num_interventions, (int, np.integer)):
                        has_exp_metadata = has_exp_metadata or (num_interventions > 0)

                is_exp_stream = has_exp_metadata

            if debug_enabled and (debug_limit <= 0 or i < debug_limit):
                print(f"[DEBUG Row {i}] stream_type_from_trainer: {stream_type_from_trainer}", file=sys.stderr)
                print(f"[DEBUG Row {i}] is_exp_stream: {is_exp_stream}", file=sys.stderr)

            # Get ground_truth before if-else
            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "")

            # Build dump record; batch-level write happens after processing all rows
            dump_record = {
                'sample_idx': i,
                'full_response': full_response_with_hints if is_exp_stream else response_str,  # Use full response with hints for exp stream
                'ground_truth': ground_truth,
                'data_source': data_source,
                'experiment_id': experiment_id,
                'timestamp': batch_timestamp,
                # Basic token-length diagnostics (works even if rollout diagnostics missing)
                'valid_prompt_length': int(valid_prompt_length),
                'valid_response_length': int(valid_response_length),
            }
            if is_exp_stream:
                if debug_enabled and (debug_limit <= 0 or i < debug_limit):
                    print(f"\n{'='*70}", file=sys.stderr)
                    print(f"[EXP_STREAM] Sample {i}", file=sys.stderr)
                    print(f"{'='*70}", file=sys.stderr)
                    print(f"[Full Response (with hints)]:\n{full_response_with_hints}\n", file=sys.stderr)
                    print(f"[Policy-only Response (for reward)]:\n{response_str}\n", file=sys.stderr)
                dump_record['stream_type'] = 'exp'
                dump_record['hints'] = data_item.non_tensor_batch.get('hints', '')
                dump_record['critiques'] = data_item.non_tensor_batch.get('critiques', '')
                dump_record['policy_only_response'] = response_str  # Also save policy-only for comparison
                # Rollout-side diagnostics (best-effort; may be absent depending on rollout implementation)
                for k in (
                    'prompt_len',
                    'response_len',
                    'gen_len',
                    'hint_len',
                    'last_finish_reason',
                    'context_exhausted',
                    'first_step_tokens_len',
                    'error',
                ):
                    if k in data_item.non_tensor_batch:
                        dump_record[k] = _to_jsonable_scalar(data_item.non_tensor_batch.get(k))
            else:
                if debug_enabled and (debug_limit <= 0 or i < debug_limit):
                    print(f"\n[CONTROL_STREAM] Sample {i}", file=sys.stderr)
                    print(f"[Response]:\n{response_str}\n", file=sys.stderr)
                dump_record['stream_type'] = 'control'
                dump_record['response'] = response_str
                for k in ('error',):
                    if k in data_item.non_tensor_batch:
                        dump_record[k] = _to_jsonable_scalar(data_item.non_tensor_batch.get(k))

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # ========== DEBUG: Print compute_score inputs ==========
            if debug_enabled and (debug_limit <= 0 or i < debug_limit):
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
            if debug_enabled and (debug_limit <= 0 or i < debug_limit):
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
            return i, reward, valid_response_length, reward_extra_info, dump_record

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_row, args))

        # Fill reward tensor with results
        reward_extra_info = {}
        exp_dump_records = []
        control_dump_records = []
        for i, score, valid_response_length, sample_reward_extra_info, dump_record in results:
            # Handle edge case: zero-length response
            # Write reward to the last valid position (or -1 if response is empty)
            # CRITICAL FIX: Clamp to tensor size to prevent IndexError when hints extend beyond max_response_length
            # This can happen in by_step mode where hints are inserted during generation
            reward_position = max(0, valid_response_length - 1)  # Clamp to 0 if response is empty
            reward_position = min(reward_position, reward_tensor.size(1) - 1)  # Clamp to max tensor size
            reward_tensor[i, reward_position] = score
            # Update extra info at the list index
            for key, value in sample_reward_extra_info.items():
                if key not in reward_extra_info:
                    reward_extra_info[key] = [None] * len(data)
                reward_extra_info[key][i] = value

            # Collect dump records for batch-level write
            if dump_record.get('stream_type') == 'exp':
                exp_dump_records.append(dump_record)
            elif dump_record.get('stream_type') == 'control':
                control_dump_records.append(dump_record)

        # Write dump files with organized directory structure:
        # experiment_id/
        #   ├── control/
        #   │   └── batch_YYYYMMDD_HHMMSS.json (filename now uses batch_tag)
        #   └── exp/
        #       └── batch_YYYYMMDD_HHMMSS.json
        import json, os
        dump_base_dir = os.environ.get("VERL_DUMP_DIR", '/mnt/shared-storage-user/liuhongwei/main_works/temp_debug')

        # Create experiment directory structure
        experiment_dir = os.path.join(dump_base_dir, 'outputs', experiment_id)

        # Write control stream batch
        if control_dump_records:
            control_subdir = os.path.join(experiment_dir, 'control')
            os.makedirs(control_subdir, exist_ok=True)
            control_file = os.path.join(control_subdir, f'batch_{batch_tag}.json')

            # Add batch metadata
            control_batch_metadata = {
                'experiment_id': experiment_id,
                'stream_type': 'control',
                'batch_timestamp': batch_timestamp,
                'batch_tag': batch_tag,
                'total_samples': len(control_dump_records),
                'samples': control_dump_records
            }

            with open(control_file, 'w', encoding='utf-8') as f:
                json.dump(control_batch_metadata, f, ensure_ascii=False, indent=2)
            # NOTE: Disabled per-batch dump logging (the files are still written).

        # Write exp stream batch
        if exp_dump_records:
            exp_subdir = os.path.join(experiment_dir, 'exp')
            os.makedirs(exp_subdir, exist_ok=True)
            exp_file = os.path.join(exp_subdir, f'batch_{batch_tag}.json')

            # Add batch metadata
            exp_batch_metadata = {
                'experiment_id': experiment_id,
                'stream_type': 'exp',
                'batch_timestamp': batch_timestamp,
                'batch_tag': batch_tag,
                'total_samples': len(exp_dump_records),
                'samples': exp_dump_records
            }

            with open(exp_file, 'w', encoding='utf-8') as f:
                json.dump(exp_batch_metadata, f, ensure_ascii=False, indent=2)
            # NOTE: Disabled per-batch dump logging (the files are still written).

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
