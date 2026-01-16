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
from typing import Dict, Any

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("compassverifier")
class CompassVerifierRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # Debug verbosity (opt-in via env). Keep default runs quiet.
        debug_enabled = os.environ.get("VERL_RM_DEBUG", "").lower() in ("1", "true", "yes")
        should_print_examine = debug_enabled and self.num_examine > 0

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

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
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                # NOTE: Changed from append(value) to direct assignment to avoid wrapping in lists
                for key, value in score.items():
                    reward_extra_info[key] = value
            else:
                reward = score
            
            if should_print_examine:
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
