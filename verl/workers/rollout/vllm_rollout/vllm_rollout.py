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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from verl import DataProto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


class InterventionPolicy:
    """
    控制 Verifier 干预策略，避免过度干预或资源浪费。
    """
    def __init__(self, max_interventions=3, confidence_threshold=0.7):
        self.max_interventions = max_interventions
        self.confidence_threshold = confidence_threshold
    
    def should_intervene(self, state, decision):
        """
        判断是否应该进行干预。
        
        Args:
            state: 样本状态字典
            decision: Verifier 决策字典
        
        Returns:
            bool: 是否应该干预
        """
        # 1. 达到最大干预次数
        if len(state['hints']) >= self.max_interventions:
            return False
        
        # 2. 决策不是 Intervene
        if decision.get("action") != "Intervene":
            return False
        
        # 3. 没有 hint
        if not decision.get("hint"):
            return False
        
        # 简单阈值判断，可根据实际 verifier 输出调整
        return True


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer  # Store tokenizer for dual_stream_rollout
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = int(self.config.get("max_num_batched_tokens", 8192))

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp")
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        
        # System prompt and task instruction for Verifier (aligned with format_training_data.py)
        self.verifier_system_prompt = "You are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within ```...``` tags."
        self.verifier_task_instruction = """You are a **Socratic Reasoning Supervisor** and **Logic Auditor** for a large language model.
Your goal is to monitor student's (model's) reasoning process step-by-step.

**Your Workflow:**
1.  **Analyze**: Read the provided `Question` and student's `Current Reasoning Trace`.
2.  **Diagnose (System 2 Thought)**:
    - Generate a ``` ... ``` block.
    - Inside, perform a **Shadow Verification**:
        - If step involves calculation, re-calculate it independently.
        - If step involves logic, verify rule application.
        - If you detect a pattern of error (e.g., hallucination, logic gap), diagnose it.
3.  **Act**:
    - If reasoning is sound: Output `<GO>`.
    - If there is a critical error or high-risk pattern: Output `<WAIT>` followed by a brief, fuzzy guidance (imperative mood, do not leak answer).

**Output Format:**
```
[Target]: ...
[Analysis/Calculation]: ...
[Verdict]: ...
```

<GO> or <WAIT> [Guidance]
**Question & Student Response:**"""
        
        # Verifier LoRA configuration for Co-GRPO
        self.verifier_lora_name = kwargs.pop("verifier_lora_name", "verifier_lora")
        self.verifier_lora_int_id = None

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)


    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            # self.inference_engine.llm_engine.list_loras
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path=getattr(self, "verifier_lora_path", None)  # Use actual Verifier LoRA path)] * batch_size
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            response = output[0].to(idx.device)
            log_probs = output[1].to(idx.device)

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

            # utilize current sampling params
            if self.sampling_params.n > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                batch_size = batch_size * self.sampling_params.n
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)

    def _register_verifier_lora(self, verifier_lora_path=None):
        """
        Register or load Verifier LoRA adapter.
        
        Note: This is a placeholder implementation. In practice, you should:
        1. Pre-train a Verifier LoRA using SFT on critique data
        2. Save it to a path and pass via verifier_lora_config
        3. Load it here for inference
        
        For now, this method attempts to find an existing LoRA or signals
        that LoRA loading is needed.
        """
        if verifier_lora_path is None:
            # Try to find existing LoRA by name
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                # Find verifier LoRA by name
                for lora_id in lora_int_ids:
                    lora_info = self.inference_engine.llm_engine.get_lora_info(lora_id)
                    if lora_info and self.verifier_lora_name in str(lora_info):
                        self.verifier_lora_int_id = lora_id
                        logger.info(f"Found existing Verifier LoRA with ID: {lora_id}")
                        return
                
                # If not found by name, check if we have multiple LoRAs
                if len(lora_int_ids) > 1:
                    self.verifier_lora_int_id = lora_int_ids[1]
                    logger.warning(f"Verifier LoRA not found by name, using LoRA ID: {lora_int_ids[1]}")
                else:
                    logger.warning("No Verifier LoRA found. Verifier will use base model or policy LoRA.")
                    # Set to None or same as policy - Verifier will use base model
                    self.verifier_lora_int_id = None
            else:
                logger.warning("No LoRAs registered in vLLM engine. Verifier will use base model.")
                self.verifier_lora_int_id = None
        else:
            # TODO: Implement dynamic LoRA loading from path
            # This requires vLLM engine support for runtime LoRA registration
            logger.warning(f"Dynamic LoRA loading from {verifier_lora_path} not yet implemented.")
            self.verifier_lora_int_id = None

    @GPUMemoryLogger(role="vllm dual stream rollout", logger=logger)
    @torch.no_grad()
    def dual_stream_rollout(
        self,
        prompts: DataProto,
        intervention_mode: str = "by_response",
        token_check_interval: int = 5,
        entropy_threshold: float = 0.5,
        use_entropy_filter: bool = True,
        **kwargs
    ) -> DataProto:
        """
        Perform dual-stream rollout for Co-GRPO:
        - Control Stream: Policy generates without Verifier guidance
        - Experimental Stream: Policy generates with Verifier intervention based on mode

        Args:
            prompts: DataProto containing prompts
            intervention_mode: Verifier intervention mode. Options:
                - "by_response": Intervene after complete response (default, current implementation)
                - "by_token": Intervene at token-level with fixed interval and entropy filtering
                - "by_step": Intervene at step boundaries (line breaks, think tokens, etc.)
            token_check_interval: For by_token mode, check every N tokens (default: 5)
            entropy_threshold: For by_token mode, entropy threshold for filtering (default: 0.5)
            use_entropy_filter: For by_token mode, whether to use entropy filtering (default: True)

        Returns:
            DataProto with control_responses, exp_responses, hints, critiques, and associated metadata
        """
        try:
            timing_info = {}
            start_time = time.time()

            tokenizer = self.tokenizer  # Use stored tokenizer

            # Rebuild vllm cache engine
            if self.config.free_cache_engine:
                self.inference_engine.init_cache_engine()

            idx = prompts.batch["input_ids"]  # (bs, prompt_length)
            attention_mask = prompts.batch["attention_mask"]
            position_ids = prompts.batch["position_ids"]
            eos_token_id = prompts.meta_info["eos_token_id"]
            batch_size = idx.size(0)

            # Handle empty batch
            if batch_size == 0:
                from tensordict import TensorDict
                empty_batch = TensorDict({}, batch_size=0)
                timing_info["total"] = time.time() - start_time
                meta_info = prompts.meta_info.copy()
                meta_info["timing"] = timing_info
                return DataProto(batch=empty_batch, non_tensor_batch={}, meta_info=meta_info)

            # Route to appropriate intervention mode
            import sys
            print(f"[DUAL_STREAM_DEBUG] intervention_mode={intervention_mode}, batch_size={batch_size}", file=sys.stderr, flush=True)

            if intervention_mode == "by_response":
                result = self._dual_stream_rollout_by_response(
                    prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                    position_ids, eos_token_id, batch_size, **kwargs
                )
                print(f"[DUAL_STREAM_DEBUG] by_response returned result type: {type(result)}", file=sys.stderr, flush=True)
                return result
            elif intervention_mode == "by_token":
                result = self._dual_stream_rollout_by_token(
                    prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                    position_ids, eos_token_id, batch_size, token_check_interval,
                    entropy_threshold, use_entropy_filter, **kwargs
                )
                print(f"[DUAL_STREAM_DEBUG] by_token returned result type: {type(result)}", file=sys.stderr, flush=True)
                return result
            elif intervention_mode == "by_step":
                print(f"[DUAL_STREAM_DEBUG] About to call _dual_stream_rollout_by_step", file=sys.stderr, flush=True)
                result = self._dual_stream_rollout_by_step(
                    prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                    position_ids, eos_token_id, batch_size, **kwargs
                )
                print(f"[DUAL_STREAM_DEBUG] by_step returned result type: {type(result)}", file=sys.stderr, flush=True)
                return result
            else:
                raise ValueError(f"Unknown intervention_mode: {intervention_mode}. Must be one of: by_response, by_token, by_step")
        except Exception as e:
            import sys
            print(f"[DUAL_STREAM_ERROR] Exception in dual_stream_rollout: {type(e).__name__}: {str(e)}", file=sys.stderr, flush=True)
            import traceback
            print(f"[DUAL_STREAM_ERROR] Traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            logger.error(f"Exception in dual_stream_rollout: {type(e).__name__}: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise  # Re-raise the exception so it propagates up
    
    def _dual_stream_rollout_by_response(
        self, prompts, timing_info, start_time, tokenizer, idx, attention_mask,
        position_ids, eos_token_id, batch_size, **kwargs
    ) -> DataProto:
        """
        Original by_response mode: Generate complete response, then Verifier intervenes.
        This is the default implementation.
        """
        # ========== Control Stream: Policy generates without guidance ==========
        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        
        if not do_sample:
            control_kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        elif is_validate:
            control_kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        else:
            control_kwargs = {}

        # Generate control responses (no LoRA)
        control_start = time.time()
        with self.update_sampling_params(**control_kwargs):
            control_output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                lora_request=None,  # No LoRA for control stream
                use_tqdm=False,
            )
        
        control_responses = control_output[0].to(idx.device)
        control_log_probs = control_output[1].to(idx.device)
        timing_info["control_gen"] = time.time() - control_start
        
        if control_responses.shape[1] < self.config.response_length:
            control_responses = pad_sequence_to_length(control_responses, self.config.response_length, self.pad_token_id)
            control_log_probs = pad_sequence_to_length(control_log_probs, self.config.response_length, self.pad_token_id)

        # ========== Experimental Stream: Policy -> Verifier -> Policy ==========
        # Step 1: Policy generates initial draft (CoT)
        draft_start = time.time()
        draft_kwargs = control_kwargs.copy()
        with self.update_sampling_params(**draft_kwargs):
            draft_output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                lora_request=None,  # No LoRA for initial draft
                use_tqdm=False,
            )
        
        draft_responses = draft_output[0].to(idx.device)
        draft_log_probs = draft_output[1].to(idx.device)
        
        if draft_responses.shape[1] < self.config.response_length:
            draft_responses = pad_sequence_to_length(draft_responses, self.config.response_length, self.pad_token_id)
            draft_log_probs = pad_sequence_to_length(draft_log_probs, self.config.response_length, self.pad_token_id)
        timing_info["draft_gen"] = time.time() - draft_start

        # Step 2: Decode draft responses and construct Verifier input
        # Format: System prompt + Task instruction + Prompt + Draft CoT -> Verifier generates Critique + Hint
        draft_texts = []
        prompt_texts = []
        for i in range(batch_size):
            prompt_text = tokenizer.decode(idx[i][attention_mask[i].bool()], skip_special_tokens=True)
            draft_text = tokenizer.decode(draft_responses[i], skip_special_tokens=True)
            prompt_texts.append(prompt_text)
            draft_texts.append(draft_text)
        
        # Construct Verifier input with system prompt + task instruction (aligned with format_training_data.py)
        verifier_inputs = []
        for prompt, draft in zip(prompt_texts, draft_texts):
            # Format: system + user (with task instruction + question + draft)
            verifier_prompt = f"<|system|>\n{self.verifier_system_prompt}\n\n<|user|>\n{self.verifier_task_instruction}\n\nQuestion: {prompt}\n\nStudent Response: {draft}\n\n"
            verifier_inputs.append(verifier_prompt)
        
        # Tokenize Verifier inputs
        verifier_encoded = tokenizer(
            verifier_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.prompt_length + self.config.response_length,
        )
        verifier_input_ids = verifier_encoded["input_ids"].to(idx.device)
        verifier_attention_mask = verifier_encoded["attention_mask"].to(idx.device)
        
        # Convert to list format for vLLM
        verifier_idx_list = []
        for i in range(batch_size):
            verifier_idx_list.append(_pre_process_inputs(self.pad_token_id, verifier_input_ids[i]))

        # Step 3: Verifier generates Critique + Hint (using Verifier LoRA)
        verifier_start = time.time()
        if self.verifier_lora_int_id is None:
            self._register_verifier_lora()
        
        verifier_lora_requests = None
        if self.verifier_lora_int_id is not None:
            verifier_lora_requests = [
                LoRARequest(
                    lora_name=self.verifier_lora_name,
                    lora_int_id=self.verifier_lora_int_id,
                    lora_path=getattr(self, "verifier_lora_path", None)  # Use actual Verifier LoRA path
                )
            ] * batch_size
        
        with self.update_sampling_params(**control_kwargs):
            verifier_output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=verifier_idx_list,
                lora_request=verifier_lora_requests,
                use_tqdm=False,
            )
        
        verifier_responses = verifier_output[0].to(idx.device)
        verifier_log_probs = verifier_output[1].to(idx.device)
        
        if verifier_responses.shape[1] < self.config.response_length:
            verifier_responses = pad_sequence_to_length(verifier_responses, self.config.response_length, self.pad_token_id)
            verifier_log_probs = pad_sequence_to_length(verifier_log_probs, self.config.response_length, self.pad_token_id)

        # Decode Verifier outputs (Critique + Hint)
        verifier_texts = []
        hints = []  # Store extracted actual hints (<GO> or <WAIT> content only)
        for i in range(batch_size):
            verifier_text = tokenizer.decode(verifier_responses[i], skip_special_tokens=True)
            verifier_texts.append(verifier_text)  # Keep full output for logging
            
            # Extract actual hint (only <GO> or <WAIT> content, stripped)
            decision = self._parse_verifier_decision(verifier_text)
            actual_hint = decision["hint"].strip() if decision["hint"] else ""
            hints.append(actual_hint)
        timing_info["verifier_gen"] = time.time() - verifier_start
        
        # Step 4: Policy generates final answer with Verifier guidance
        # Use extracted hints only (stripped, without think blocks)
        final_inputs = []
        for prompt, hint in zip(prompt_texts, hints):
            # Only use the extracted hint content (already stripped)
            final_prompt = f"Prompt: {prompt}\n\nHint: {hint}\n\nFinal Answer:"
            final_inputs.append(final_prompt)
        
        final_encoded = tokenizer(
            final_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.prompt_length + self.config.response_length,
        )
        final_input_ids = final_encoded["input_ids"].to(idx.device)
        final_attention_mask = final_encoded["attention_mask"].to(idx.device)
        
        final_idx_list = []
        for i in range(batch_size):
            final_idx_list.append(_pre_process_inputs(self.pad_token_id, final_input_ids[i]))

        # Generate final responses (no LoRA, back to Policy)
        final_start = time.time()
        with self.update_sampling_params(**control_kwargs):
            final_output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=final_idx_list,
                lora_request=None,  # No LoRA for final generation
                use_tqdm=False,
            )
        
        exp_responses = final_output[0].to(idx.device)
        exp_log_probs = final_output[1].to(idx.device)
        
        if exp_responses.shape[1] < self.config.response_length:
            exp_responses = pad_sequence_to_length(exp_responses, self.config.response_length, self.pad_token_id)
            exp_log_probs = pad_sequence_to_length(exp_log_probs, self.config.response_length, self.pad_token_id)
        timing_info["final_gen"] = time.time() - final_start

        # Store the actual prompts used for experimental stream
        # This is important for correct response mask calculation later
        exp_prompt_ids = final_input_ids
        exp_prompt_attention_mask = final_attention_mask
        exp_prompt_length = exp_prompt_attention_mask.sum(dim=1)

        # ========== Prepare output DataProto ==========
        # Store hints and critiques in non_tensor_batch
        hints = np.array(verifier_texts, dtype=object)
        critiques = np.array(verifier_texts, dtype=object)  # For now, same as hints (can be separated later)
        
        # Store intervention statistics for penalty calculation
        # In by_response mode, each sample has 1 intervention (the verifier output)
        num_interventions = np.ones(batch_size, dtype=np.int32)
        hint_token_counts = np.array([
            len(tokenizer.encode(text, add_special_tokens=False)) for text in verifier_texts
        ], dtype=np.int32)
        
        # Combine control and experimental responses
        # Helper function to get last valid position id
        def get_last_valid_position_id(position_ids, attention_mask):
            """Get the last valid position id based on attention mask."""
            batch_size = position_ids.size(0)
            valid_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
            if position_ids.dim() == 3:  # multi-dimensional position (e.g., qwen2vl)
                last_pos = position_ids[torch.arange(batch_size), :, valid_lengths]
            else:
                last_pos = position_ids[torch.arange(batch_size), valid_lengths]
            return last_pos
        
        last_valid_pos = get_last_valid_position_id(position_ids, attention_mask)
        
        # Control stream data
        control_seq = torch.cat([idx, control_responses], dim=-1)
        control_response_length = control_responses.size(1)
        control_delta_position_id = torch.arange(1, control_response_length + 1, device=position_ids.device)
        control_delta_position_id = control_delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:
            control_delta_position_id = control_delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
            last_valid_pos_expanded = last_valid_pos.unsqueeze(-1)  # [batch_size, 3, 1]
        else:
            last_valid_pos_expanded = last_valid_pos.unsqueeze(-1)  # [batch_size, 1]
        control_response_position_ids = last_valid_pos_expanded + control_delta_position_id
        control_position_ids = torch.cat([position_ids, control_response_position_ids], dim=-1)
        control_response_attention_mask = get_response_mask(response_id=control_responses, eos_token=eos_token_id, dtype=attention_mask.dtype)
        control_attention_mask = torch.cat((attention_mask, control_response_attention_mask), dim=-1)

        # Experimental stream data
        # Use the actual prompt used (final_input_ids) instead of original idx
        exp_seq = torch.cat([exp_prompt_ids, exp_responses], dim=-1)
        exp_response_length = exp_responses.size(1)
        exp_delta_position_id = torch.arange(1, exp_response_length + 1, device=position_ids.device)
        exp_delta_position_id = exp_delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
        # Create position_ids for exp_prompt from scratch
        exp_prompt_len = exp_prompt_ids.size(1)
        if position_ids.dim() == 3:
            # Multi-dimensional position (e.g., qwen2vl)
            exp_prompt_position_ids = torch.arange(exp_prompt_len, device=exp_prompt_ids.device)
            exp_prompt_position_ids = exp_prompt_position_ids.unsqueeze(0).unsqueeze(1).expand(
                batch_size, position_ids.size(1), -1
            )
            # Prepare delta for 3D
            exp_delta_position_id = exp_delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)
        else:
            # Standard 2D position
            exp_prompt_position_ids = torch.arange(exp_prompt_len, device=exp_prompt_ids.device)
            exp_prompt_position_ids = exp_prompt_position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get last valid position from exp_prompt
        exp_last_valid_pos = get_last_valid_position_id(
            exp_prompt_position_ids,
            exp_prompt_attention_mask
        )
        
        # Calculate response position_ids based on last valid position
        if position_ids.dim() == 3:
            exp_last_valid_pos_expanded = exp_last_valid_pos.unsqueeze(-1)  # [batch_size, 3, 1]
        else:
            exp_last_valid_pos_expanded = exp_last_valid_pos.unsqueeze(-1)  # [batch_size, 1]
        exp_response_position_ids = exp_last_valid_pos_expanded + exp_delta_position_id
        
        # Concatenate prompt and response position_ids
        exp_position_ids = torch.cat([exp_prompt_position_ids, exp_response_position_ids], dim=-1)
        
        exp_response_attention_mask = get_response_mask(response_id=exp_responses, eos_token=eos_token_id, dtype=attention_mask.dtype)
        exp_attention_mask = torch.cat((exp_prompt_attention_mask, exp_response_attention_mask), dim=-1)

        # Create batch with both streams
        batch = TensorDict(
            {
                "prompts": idx,  # Original prompts (for control stream)
                "exp_prompts": exp_prompt_ids,  # Actual prompts used for experimental stream (with critique)
                "control_responses": control_responses,
                "exp_responses": exp_responses,
                "control_input_ids": control_seq,
                "exp_input_ids": exp_seq,
                "control_rollout_log_probs": control_log_probs,
                "exp_rollout_log_probs": exp_log_probs,
                "control_attention_mask": control_attention_mask,
                "exp_attention_mask": exp_attention_mask,
                "control_position_ids": control_position_ids,
                "exp_position_ids": exp_position_ids,
                "control_response_mask": control_response_attention_mask,  # Store response masks
                "exp_response_mask": exp_response_attention_mask,
            },
            batch_size=batch_size,
        )

        # Store hints and critiques in non_tensor_batch
        non_tensor_batch = {
            "hints": hints,
            "critiques": critiques,
            "draft_responses": np.array([tokenizer.decode(draft_responses[i], skip_special_tokens=True) for i in range(batch_size)], dtype=object),
            "num_interventions": num_interventions,
            "hint_token_counts": hint_token_counts,
        }

        # Free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        # Add timing information to meta_info
        timing_info["total"] = time.time() - start_time
        meta_info = prompts.meta_info.copy()
        meta_info["timing"] = timing_info

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    
    def _get_step_boundary_stop_config(self, tokenizer, eos_token_id=None):
        """
        获取 step boundary 的停止配置。
        
        Args:
            tokenizer: Tokenizer instance
            eos_token_id: EOS token ID (optional)
        
        Returns:
            dict: 包含 stop_token_ids 和 stop_sequences 的配置
        """
        stop_token_ids = set()
        stop_sequences = set()
        
        # 1. 思考结束标记 & 换行
        for seq in ["</think>", "<｜end of thought｜>", "\n\n", ".\n\n"]:
            stop_sequences.add(seq)
        
        # 2. 显式添加 \n 的 Token ID (防止 tokenizer 差异)
        try:
            nl_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
            stop_token_ids.add(nl_id)
        except:
            pass
            
        if eos_token_id is not None:
            stop_token_ids.add(eos_token_id)
            
        return {
            'stop_token_ids': list(stop_token_ids), 
            'stop_sequences': list(stop_sequences)
        }


    def _run_verifier_inference(self, verifier_inputs, tokenizer, idx_device, exp_kwargs):
        """
        批量调用 Verifier 进行推理。
        
        Args:
            verifier_inputs: List[str] - Verifier 输入文本列表
            tokenizer: Tokenizer instance
            idx_device: Device for tensors
            exp_kwargs: Sampling parameters
        
        Returns:
            List[dict]: 每个输入的决策结果，包含 'action', 'hint', 'critique'
        """
        if not verifier_inputs:
            return []
        
        try:
            # Register Verifier LoRA if needed
            if self.verifier_lora_int_id is None:
                self._register_verifier_lora()
            
            # 批量 tokenize
            verifier_encoded = tokenizer(
                verifier_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.prompt_length + self.config.response_length,
            )
            verifier_input_ids = verifier_encoded["input_ids"].to(idx_device)
            
            # 转换格式
            verifier_idx_list = []
            for i in range(len(verifier_inputs)):
                verifier_idx_list.append(_pre_process_inputs(self.pad_token_id, verifier_input_ids[i]))
            
            # 准备 LoRA requests
            verifier_lora_requests = None
            if self.verifier_lora_int_id is not None:
                verifier_lora_requests = [
                    LoRARequest(
                        lora_name=self.verifier_lora_name,
                        lora_int_id=self.verifier_lora_int_id,
                        lora_path=getattr(self, "verifier_lora_path", None)  # Use actual Verifier LoRA path
                    )
                ] * len(verifier_inputs)
            
            # 批量生成 Verifier 响应
            with self.update_sampling_params(**exp_kwargs):
                verifier_output = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=verifier_idx_list,
                    lora_request=verifier_lora_requests,
                    use_tqdm=False,
                )
            
            verifier_responses = verifier_output[0].to(idx_device)
            
            # 解析所有 Verifier 输出
            decisions = []
            for i in range(len(verifier_inputs)):
                verifier_response = verifier_responses[i]
                verifier_text = tokenizer.decode(verifier_response, skip_special_tokens=True)
                decision = self._parse_verifier_decision(verifier_text)
                decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Verifier inference failed: {e}")
            # 返回默认决策（Pass）
            return [{'action': 'Pass', 'hint': None, 'critique': f'[Error: {str(e)}]'} for _ in verifier_inputs]
    
    def _parse_verifier_decision(self, verifier_text: str) -> dict:
        """
        Parse Verifier output to extract decision (Pass/Intervene) and hint.
        
        New format support:
        - Verifier output: ```...\n...[GO> or <WAIT> XXX
        - We extract only the actual hint (<GO> or <WAIT> XXX), removing think blocks
        
        Returns:
            dict with keys: "action" (str), "hint" (str, extracted <GO>/<WAIT> content)
        """
        verifier_text = verifier_text.strip()
        
        # Step 1: Remove think blocks (```\n...\n```)
        text_without_think = re.sub(
            r'```\n(.*?)\n```',
            '',
            verifier_text,
            flags=re.DOTALL
        )
        
        # Step 2: Extract <GO> or <WAIT> content from the remaining text
        go_match = re.search(r'<GO>\s*(.*)', text_without_think, re.IGNORECASE)
        wait_match = re.search(r'<WAIT>\s*(.*)', text_without_think, re.IGNORECASE)
        
        if go_match:
            action = "Intervene"
            hint = go_match.group(1).strip()  # Extract and strip the content
        elif wait_match:
            action = "Intervene"
            hint = wait_match.group(1).strip()  # Extract and strip the content
        else:
            # Fallback: try old format
            if verifier_text.lower().startswith("intervene"):
                action = "Intervene"
                hint_match = re.search(r'intervene[:\s]+(.+)', verifier_text, re.IGNORECASE | re.DOTALL)
                hint = hint_match.group(1).strip() if hint_match else ""
            elif verifier_text.lower().startswith("pass"):
                action = "Pass"
                hint = ""
            else:
                action = "Pass"  # Default: treat as Pass
                hint = ""
        
        return {
            "action": action,
            "hint": hint,  # Only contains the actual <GO> or <WAIT> content (stripped)
            "critique": verifier_text  # Keep full output for logging
        }
    
    def _find_step_boundaries(self, response_tokens: torch.Tensor, tokenizer, device) -> torch.Tensor:
        """
        Find step boundaries in response tokens using REPRO logic.
        
        Args:
            response_tokens: [batch_size, seq_len] tensor of token IDs
            tokenizer: Tokenizer instance
            device: Device for tensors
            
        Returns:
            step_boundaries: [batch_size, max_steps] tensor of boundary indices
        """
        batch_size, seq_len = response_tokens.shape
        
        # Encode step boundary markers (similar to REPRO)
        try:
            end_think_token_id = tokenizer.encode("</think>", add_special_tokens=False)
            if len(end_think_token_id) > 0:
                end_think_token_id = end_think_token_id[0]
            else:
                end_think_token_id = None
        except:
            end_think_token_id = None
        
        # Line break tokens
        line_break_ids = []
        for pattern in [".\n", "?\n", "\n\n", ".\n\n", "?\n\n"]:
            try:
                encoded = tokenizer.encode(pattern, add_special_tokens=False)
                if len(encoded) > 0:
                    line_break_ids.append(encoded[0])
            except:
                pass
        
        # Handle empty batch or zero sequence length
        if batch_size == 0 or seq_len == 0:
            return torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # Find boundaries
        boundaries_list = []
        for b in range(batch_size):
            boundaries = []
            tokens = response_tokens[b]
            
            # Create think mask (similar to REPRO)
            if end_think_token_id is not None:
                think_mask = ~(tokens == end_think_token_id).cumsum(0).bool()
            else:
                think_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
            
            # Find line breaks
            line_break_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            for lb_id in line_break_ids:
                line_break_mask = line_break_mask | (tokens == lb_id)
            
            # Shift line break mask (boundary is after the line break)
            line_break_mask = torch.roll(line_break_mask, 1, 0)
            line_break_mask[0] = False
            
            # Combine with think mask
            valid_boundaries = think_mask & line_break_mask
            
            # Get boundary indices
            boundary_indices = torch.where(valid_boundaries)[0].tolist()
            boundaries.extend(boundary_indices)
            
            # Always include the end (if seq_len > 0)
            if seq_len > 0:
                boundaries.append(seq_len - 1)
            boundaries = sorted(set(boundaries))
            
            # Filter out invalid boundaries (negative or out of range)
            boundaries = [b for b in boundaries if 0 <= b < seq_len]
            
            # Ensure at least one boundary (the end)
            if not boundaries and seq_len > 0:
                boundaries = [seq_len - 1]
            
            boundaries_list.append(boundaries)
        
        # Handle empty boundaries_list
        if not boundaries_list:
            return torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # Pad to same length
        max_boundaries = max(len(b) for b in boundaries_list) if boundaries_list else 1
        boundaries_tensor = torch.zeros(batch_size, max_boundaries, dtype=torch.long, device=device)
        for b, boundaries in enumerate(boundaries_list):
            if boundaries:
                boundaries_tensor[b, :len(boundaries)] = torch.tensor(boundaries, device=device)
        
        return boundaries_tensor
    
    def _dual_stream_rollout_by_token(
        self, prompts, timing_info, start_time, tokenizer, idx, attention_mask,
        position_ids, eos_token_id, batch_size, token_check_interval,
        entropy_threshold, use_entropy_filter, **kwargs
    ) -> DataProto:
        """
        by_token mode: Incremental generation with token-level Verifier intervention.
        Checks every N tokens or at high-entropy tokens.
        """
        # For now, fall back to by_response mode with a warning
        # Full incremental generation requires vLLM API changes
        logger.warning(
            "by_token mode is not fully implemented yet due to vLLM API limitations. "
            "Falling back to by_response mode. "
            "Full implementation requires incremental generation support."
        )
        
        # Use by_response as fallback
        return self._dual_stream_rollout_by_response(
            prompts, timing_info, start_time, tokenizer, idx, attention_mask,
            position_ids, eos_token_id, batch_size, **kwargs
        )
    
    def _dual_stream_rollout_by_step(
        self, prompts, timing_info, start_time, tokenizer, idx, attention_mask,
        position_ids, eos_token_id, batch_size, **kwargs
    ) -> DataProto:
        """
        by_step mode: 使用 stop_sequences 实现批量并行增量打断生成（修正版）。
        
        关键修正：
        1. 统一 sampling_params（计算 max_remaining_tokens）
        2. 引入 loss_mask 区分 Hint（0）和生成内容（1）
        3. Token ID + String Suffix 双重停止检测
        4. 完整的 DataProto 构建
        """
        # ========== Control Stream: Policy generates without guidance ==========
        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        
        if not do_sample:
            control_kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        elif is_validate:
            control_kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        else:
            control_kwargs = {}

        # Generate control responses (no LoRA)
        control_start = time.time()
        with self.update_sampling_params(**control_kwargs):
            control_output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                lora_request=None,
                use_tqdm=False,
            )
        
        control_responses = control_output[0].to(idx.device)
        control_log_probs = control_output[1].to(idx.device)
        timing_info["control_gen"] = time.time() - control_start
        
        if control_responses.shape[1] < self.config.response_length:
            control_responses = pad_sequence_to_length(control_responses, self.config.response_length, self.pad_token_id)
            control_log_probs = pad_sequence_to_length(control_log_probs, self.config.response_length, self.pad_token_id)

        # ========== Experimental Stream: 批量并行增量打断生成 ==========
        # 1. 获取停止配置
        stop_config = self._get_step_boundary_stop_config(tokenizer, eos_token_id)
        if not stop_config['stop_sequences'] and not stop_config['stop_token_ids']:
            logger.warning("No stop sequences found, falling back to by_response mode")
            return self._dual_stream_rollout_by_response(
                prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                position_ids, eos_token_id, batch_size, **kwargs
            )
        
        logger.info(f"Stop config: token_ids={stop_config['stop_token_ids']}, sequences={stop_config['stop_sequences']}")
        
        # 2. 初始化 InterventionPolicy
        intervention_policy = InterventionPolicy(
            max_interventions=kwargs.get('max_interventions', 3),
            confidence_threshold=kwargs.get('confidence_threshold', 0.7)
        )
        
        # 3. 准备采样参数
        if not do_sample:
            exp_kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        elif is_validate:
            exp_kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        else:
            exp_kwargs = {}
        
        # 4. 初始化所有样本的状态（批量管理）
        sample_states = []
        for b in range(batch_size):
            prompt_tokens = idx[b][attention_mask[b].bool()].tolist()
            sample_states.append({
                'prompt_tokens': prompt_tokens,
                'prompt_text': tokenizer.decode(prompt_tokens, skip_special_tokens=True),
                'response_tokens': [],  # 关键：hint 追加到这里
                'loss_masks': [],  # 关键新增：1 for Model Gen, 0 for Hint
                'hints': [],
                'critiques': [],
                'step_count': 0,
                'is_complete': False,
                'error': None,
            })
        
        # 5. 监控指标
        metrics = {
            'total_steps': 0,
            'total_interventions': 0,
            'stop_sequence_hits': 0,
            'errors': 0,
        }
        
        exp_start = time.time()
        max_steps = kwargs.get('max_steps', 20)
        
        # 6. 批量增量生成循环
        for global_step in range(max_steps):
            metrics['total_steps'] += 1
            
            # 6.1 收集所有活跃样本（未完成的样本）
            active_indices = [b for b in range(batch_size) if not sample_states[b]['is_complete']]
            
            if not active_indices:
                break
            
            # 6.2 批量构建当前输入
            batch_inputs = []
            batch_indices = []
            current_response_lens = []
            
            for b in active_indices:
                state = sample_states[b]
                
                # Input = Prompt + Response (Hint included)
                current_input = state['prompt_tokens'] + state['response_tokens']
                batch_inputs.append(current_input)
                batch_indices.append(b)
                current_response_lens.append(len(state['response_tokens']))
            
            if not batch_inputs:
                break
            
            # 6.3 计算统一的 max_tokens（关键修正）
            max_remaining = max([
                self.config.response_length - l 
                for l in current_response_lens
            ])
            
            if max_remaining <= 0:
                for b in batch_indices:
                    sample_states[b]['is_complete'] = True
                break
            
            # 6.4 批量生成到下一个 boundary
            try:
                # 转换格式
                batch_inputs_processed = []
                for inp in batch_inputs:
                    batch_inputs_processed.append(
                        _pre_process_inputs(self.pad_token_id, torch.tensor(inp, device=idx.device))
                    )
                
                # 使用统一的 sampling_params（修正）
                with self.update_sampling_params(
                    stop=stop_config['stop_sequences'],
                    stop_token_ids=stop_config['stop_token_ids'],
                    max_tokens=max_remaining,
                    **exp_kwargs
                ):
                    step_output = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=batch_inputs_processed,
                        lora_request=None,
                        use_tqdm=False,
                    )
            except Exception as e:
                logger.error(f"Generation error at step {global_step}: {e}")
                metrics['errors'] += len(batch_indices)
                # 标记失败样本为完成
                for b in batch_indices:
                    sample_states[b]['is_complete'] = True
                    sample_states[b]['error'] = str(e)
                continue
            
            # 6.5 提取新生成的 tokens（批量处理）
            step_responses = step_output[0].to(idx.device)
            
            # 6.6 更新每个样本的状态并检测停止
            verifier_inputs = []
            verifier_input_map = []  # (batch_idx, step_idx)
            
            for i, b in enumerate(batch_indices):
                state = sample_states[b]
                step_tokens = step_responses[i]
                
                # 提取新生成的部分（去掉 prompt 部分）
                input_len = len(batch_inputs[i])
                new_tokens = step_tokens[input_len:].tolist()
                
                # 调试日志
                logger.debug(f"Step {global_step}, Sample {b}: input_len={input_len}, "
                            f"step_tokens_len={len(step_tokens)}, new_tokens_len={len(new_tokens)}")
                
                # 检查是否生成了新内容
                if len(new_tokens) == 0:
                    state['is_complete'] = True
                    continue
                
                # 关键：追加到 response_tokens（不是 prompt_tokens）
                state['response_tokens'].extend(new_tokens)
                state['loss_masks'].extend([1] * len(new_tokens))  # Model Gen = 1
                state['step_count'] += 1
                
                # 检查是否应该结束
                if eos_token_id is not None and eos_token_id in new_tokens:
                    state['is_complete'] = True
                    continue
                if len(state['response_tokens']) >= self.config.response_length:
                    state['is_complete'] = True
                    continue
                
                # 增强版停止检测（Token ID + String Suffix 双重检测）
                hit_stop = False
                
                # 方法1：检查最后一个 token ID
                if new_tokens and new_tokens[-1] in stop_config['stop_token_ids']:
                    hit_stop = True
                    metrics['stop_sequence_hits'] += 1
                
                # 方法2：检查字符串后缀
                if not hit_stop:
                    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    for seq in stop_config['stop_sequences']:
                        if new_text.endswith(seq):
                            hit_stop = True
                            metrics['stop_sequence_hits'] += 1
                            break
                
                if hit_stop:
                    # 在 boundary 处停止，需要 Verifier 判断
                    current_reasoning = tokenizer.decode(state['response_tokens'], skip_special_tokens=True)
                    verifier_input = (
                        f"Prompt: {state['prompt_text']}\n\n"
                        f"Current Reasoning: {current_reasoning}\n\n"
                        f"Should I intervene? (Pass/Intervene):"
                    )
                    verifier_inputs.append(verifier_input)
                    verifier_input_map.append((b, state['step_count']))
            
            # 6.7 批量调用 Verifier
            if verifier_inputs:
                decisions = self._run_verifier_inference(
                    verifier_inputs, tokenizer, idx.device, exp_kwargs
                )
                
                for i, (b, step_idx) in enumerate(verifier_input_map):
                    state = sample_states[b]
                    decision = decisions[i]
                    
                    # 记录 critique
                    critique = decision.get('critique', decision.get('hint', ''))
                    state['critiques'].append(critique)
                    
                    # 使用 InterventionPolicy 判断是否应该干预
                    if intervention_policy.should_intervene(state, decision):
                        hint = decision['hint']
                        state['hints'].append(hint)
                        metrics['total_interventions'] += 1
                        
                        # 关键：将 hint 追加到 response_tokens，并设置 loss_mask 为 0
                        hint_text = f"\n\n[Guide]: {hint}\n\n"
                        hint_tokens = tokenizer.encode(hint_text, add_special_tokens=False)
                        
                        # 验证编码正确性
                        decoded = tokenizer.decode(hint_tokens, skip_special_tokens=True)
                        if decoded != hint_text:
                            logger.warning(f"Hint encoding mismatch: expected '{hint_text}', got '{decoded}'")
                        
                        state['response_tokens'].extend(hint_tokens)
                        state['loss_masks'].extend([0] * len(hint_tokens))  # Hint = 0 (No Loss)
        
        timing_info["exp_gen"] = time.time() - exp_start
        
        # 7. 记录监控指标
        timing_info["exp_metrics"] = {
            'avg_steps': sum(s['step_count'] for s in sample_states) / batch_size if batch_size > 0 else 0,
            'intervention_rate': metrics['total_interventions'] / batch_size if batch_size > 0 else 0,
            'stop_sequence_hit_rate': metrics['stop_sequence_hits'] / metrics['total_steps'] if metrics['total_steps'] > 0 else 0,
            'error_rate': metrics['errors'] / batch_size if batch_size > 0 else 0,
        }
        logger.info(f"Generation metrics: {timing_info['exp_metrics']}")
        
        # 8. 数据后处理：构建完整的 DataProto（关键修正）
        exp_input_ids_list = []
        exp_attention_mask_list = []
        exp_position_ids_list = []
        exp_loss_mask_list = []  # 新增字段
        exp_responses_list = []
        all_hints = []
        all_critiques = []
        
        for b in range(batch_size):
            state = sample_states[b]
            p_len = len(state['prompt_tokens'])
            r_len = len(state['response_tokens'])
            
            # 8.1 构建完整序列
            full_ids = torch.tensor(
                state['prompt_tokens'] + state['response_tokens'], 
                dtype=torch.long, 
                device=idx.device
            )
            
            # 8.2 构建 Attention Mask (1 for Valid)
            att_mask = torch.ones_like(full_ids)
            
            # 8.3 构建 Loss Mask (Prompt=0, Hint=0, Gen=1)
            loss_mask = torch.zeros_like(full_ids)
            if r_len > 0:
                r_mask = torch.tensor(state['loss_masks'], dtype=torch.long, device=idx.device)
                loss_mask[p_len:p_len+r_len] = r_mask
            
            # 8.4 构建 response tensor（用于 reward 计算）
            if r_len > 0:
                response_tensor = torch.tensor(state['response_tokens'], dtype=torch.long, device=idx.device)
            else:
                response_tensor = torch.tensor([], dtype=torch.long, device=idx.device)
            
            # 8.5 Padding 到最大长度
            total_len = self.config.prompt_length + self.config.response_length
            if full_ids.size(0) < total_len:
                pad_len = total_len - full_ids.size(0)
                full_ids = torch.cat([full_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long, device=idx.device)])
                att_mask = torch.cat([att_mask, torch.zeros((pad_len,), dtype=att_mask.dtype, device=idx.device)])
                loss_mask = torch.cat([loss_mask, torch.zeros((pad_len,), dtype=torch.long, device=idx.device)])
            elif full_ids.size(0) > total_len:
                full_ids = full_ids[:total_len]
                att_mask = att_mask[:total_len]
                loss_mask = loss_mask[:total_len]
            
            # Padding response
            if response_tensor.size(0) < self.config.response_length:
                response_tensor = pad_sequence_to_length(
                    response_tensor.unsqueeze(0),
                    self.config.response_length,
                    self.pad_token_id
                ).squeeze(0)
            elif response_tensor.size(0) > self.config.response_length:
                response_tensor = response_tensor[:self.config.response_length]
            
            # 8.6 Position IDs（简化版，根据实际需求调整）
            pos_ids = torch.arange(total_len, dtype=torch.long, device=idx.device)
            
            exp_input_ids_list.append(full_ids)
            exp_attention_mask_list.append(att_mask)
            exp_loss_mask_list.append(loss_mask)
            exp_position_ids_list.append(pos_ids)
            exp_responses_list.append(response_tensor)
            
            all_hints.append("\n".join(state['hints']))
            all_critiques.append("\n---\n".join(state['critiques']) if state['critiques'] else "No interventions")
        
        # 9. Stack Tensors
        exp_input_ids = torch.stack(exp_input_ids_list)  # [batch_size, total_len]
        exp_attention_mask = torch.stack(exp_attention_mask_list)
        exp_loss_mask = torch.stack(exp_loss_mask_list)  # 关键新增
        exp_position_ids = torch.stack(exp_position_ids_list)
        exp_responses = torch.stack(exp_responses_list)  # [batch_size, response_length]
        
        # 10. 构建完整的 DataProto（参考现有实现）
        # Control stream data
        control_seq = torch.cat([idx, control_responses], dim=-1)
        control_response_length = control_responses.size(1)
        control_delta_position_id = torch.arange(1, control_response_length + 1, device=position_ids.device)
        control_delta_position_id = control_delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
        # 计算 last_valid_pos
        def get_last_valid_position_id(position_ids, attention_mask):
            batch_size = position_ids.size(0)
            valid_lengths = attention_mask.sum(dim=1) - 1
            if position_ids.dim() == 3:
                last_pos = position_ids[torch.arange(batch_size), :, valid_lengths]
            else:
                last_pos = position_ids[torch.arange(batch_size), valid_lengths]
            return last_pos
        
        last_valid_pos = get_last_valid_position_id(position_ids, attention_mask)
        
        if position_ids.dim() == 3:
            control_delta_position_id = control_delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
            last_valid_pos_expanded = last_valid_pos.unsqueeze(-1)
        else:
            last_valid_pos_expanded = last_valid_pos.unsqueeze(-1)
        
        control_response_position_ids = last_valid_pos_expanded + control_delta_position_id
        control_position_ids = torch.cat([position_ids, control_response_position_ids], dim=-1)
        control_response_attention_mask = get_response_mask(response_id=control_responses, eos_token=eos_token_id, dtype=attention_mask.dtype)
        control_attention_mask = torch.cat((attention_mask, control_response_attention_mask), dim=-1)
        
        # Experimental stream - 使用原始 prompt（因为 hint 在 response 中）
        exp_prompt_ids = idx.clone()
        exp_prompt_attention_mask = attention_mask.clone()
        
        # 构建 exp_response_mask
        exp_response_mask = get_response_mask(response_id=exp_responses, eos_token=eos_token_id, dtype=attention_mask.dtype)
        
        # 11. 构建最终 batch
        batch = TensorDict(
            {
                "prompts": idx,
                "exp_prompts": exp_prompt_ids,
                "control_responses": control_responses,
                "exp_responses": exp_responses,
                "control_input_ids": control_seq,
                "exp_input_ids": exp_input_ids,
                "control_rollout_log_probs": control_log_probs,
                "exp_rollout_log_probs": None,  # 暂不计算，可以后续补充
                "control_attention_mask": control_attention_mask,
                "exp_attention_mask": exp_attention_mask,
                "control_position_ids": control_position_ids,
                "exp_position_ids": exp_position_ids,
                "control_response_mask": control_response_attention_mask,
                "exp_response_mask": exp_response_mask,
                "exp_loss_mask": exp_loss_mask,  # 关键新增：用于 loss 计算
            },
            batch_size=batch_size,
        )
        
        hints = np.array(all_hints, dtype=object)
        critiques = np.array(all_critiques, dtype=object)
        
        # Store intervention statistics for penalty calculation
        num_interventions = np.array([len(state['hints']) for state in sample_states], dtype=np.int32)
        hint_token_counts = np.array([
            sum(len(tokenizer.encode(h, add_special_tokens=False)) for h in state['hints'])
            for state in sample_states
        ], dtype=np.int32)
        
        non_tensor_batch = {
            "hints": hints,
            "critiques": critiques,
            "num_interventions": num_interventions,
            "hint_token_counts": hint_token_counts,
        }
        
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        
        timing_info["total"] = time.time() - start_time
        meta_info = prompts.meta_info.copy()
        meta_info["timing"] = timing_info
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
