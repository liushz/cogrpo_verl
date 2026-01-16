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
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import re
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Verifier prompts (matching format_training_data.py)
VERIFIER_SYSTEM_PROMPT = "You are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags."

VERIFIER_INTERVENE_PROMPT = """You are a **Socratic Reasoning Supervisor** and **Logic Auditor** for a large language model.
Your goal is to monitor the student's (model's) reasoning process step-by-step.

**Your Workflow:**
1.  **Analyze**: Read the provided `Question` and the student's `Current Reasoning Trace`.
2.  **Diagnose (System 2 Thought)**:
    - Generate a ```<think> ... </think>``` block.
    - Inside, perform a **Shadow Verification**:
        - If the step involves calculation, re-calculate it independently.
        - If the step involves logic, verify the rule application.
        - If you detect a pattern of error (e.g., hallucination, logic gap), diagnose it.
3.  **Act**:
    - If the reasoning is sound: Output `<GO>`.
    - If there is a critical error or high-risk pattern: Output `<WAIT>` followed by a brief, fuzzy guidance (imperative mood, do not leak answer).

**Output Format:**
```
<think>
(Your shadow verification and diagnosis)
</think>

<GO> or <WAIT> [Guidance]
```

**Question & Student Response:**
"""

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


class InterventionPolicy:
    """
    控制 Verifier 干预策略，避免过度干预或资源浪费。
    """
    def __init__(self, max_interventions: int = 3, confidence_threshold: float = 0.7):
        self.max_interventions = max_interventions
        self.confidence_threshold = confidence_threshold
    
    def should_intervene(self, state: Dict[str, Any], decision: Dict[str, Any]) -> bool:
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
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _extract_vllm_outputs(outputs, pad_token_id: int, max_length: int, device):
    """
    从 vLLM RequestOutput 对象列表中提取 token_ids 和 logprobs
    
    Args:
        outputs: List[RequestOutput] from vllm generate()
        pad_token_id: token id for padding
        max_length: maximum sequence length
        device: target device for tensors
        
    Returns:
        tuple: (response_tensor, log_probs_tensor)
            - response_tensor: (batch_size, max_length) 
            - log_probs_tensor: (batch_size, max_length)
    """
    response = []
    rollout_log_probs = []
    
    for output in outputs:
        for sample_id in range(len(output.outputs)):
            response_ids = output.outputs[sample_id].token_ids
            response.append(response_ids)
            curr_log_prob = []
            for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                curr_log_prob.append(logprob[response_ids[i]].logprob)
            rollout_log_probs.append(curr_log_prob)
    
    response_tensor = pad_2d_list_to_length(response, pad_token_id, max_length=max_length).to(device)
    log_probs_tensor = pad_2d_list_to_length(rollout_log_probs, -1, max_length=max_length).to(device)
    log_probs_tensor = log_probs_tensor.to(torch.float32)
    
    return response_tensor, log_probs_tensor


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
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
        self.tokenizer = tokenizer  # Store tokenizer for dual_stream_rollout
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # Extract verifier_lora_name and verifier_lora_path from kwargs if present
        verifier_lora_name = kwargs.pop("verifier_lora_name", None)
        verifier_lora_path = kwargs.pop("verifier_lora_path", None)
        
        # #region agent log
        # NOTE: Disabled ad-hoc debug logging to `.cursor/debug.log` (not needed in normal runs).
        # #endregion
        
        # KV Cache optimization: enable prefix caching for by_step mode
        # This allows vLLM to reuse attention cache across steps, reducing compute
        self.enable_kv_cache_optimization = kwargs.pop("enable_kv_cache_optimization", True)
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        # CRITICAL FIX: When enable_lora=True, vLLM may use a different tokenizer initialization
        # that causes garbled output. We ensure vLLM uses the base model's tokenizer by:
        # 1. Passing the base model's tokenizer revision if needed
        # 2. Ensuring the tokenizer is loaded from the base model path, not from any LoRA checkpoint
        if lora_kwargs.get("enable_lora", False):
            # When LoRA is enabled, explicitly set the tokenizer initialization to use base model
            engine_kwargs["tokenizer_mode"] = "auto"  # Ensure auto mode, not from LoRA

        # Optional verifier offload hint: map verifier.fsdp_config.* to safer vLLM settings.
        verifier_fsdp_config = config.get("verifier", {}).get("fsdp_config", {})
        self.verifier_offload_enabled = bool(verifier_fsdp_config.get("param_offload")) or bool(verifier_fsdp_config.get("optimizer_offload"))
        if self.verifier_offload_enabled:
            # If not provided, default to a conservative swap size (GB).
            swap_space_override = verifier_fsdp_config.get("swap_space", 64)
            current_swap_space = engine_kwargs.get("swap_space")
            if current_swap_space is None or current_swap_space < swap_space_override:
                engine_kwargs["swap_space"] = swap_space_override

            # Optionally tighten gpu_memory_utilization; fall back to 0.7 if not specified.
            gpu_mem_util = verifier_fsdp_config.get("gpu_memory_utilization", None)
            if gpu_mem_util is not None:
                config.gpu_memory_utilization = gpu_mem_util
            else:
                config.gpu_memory_utilization = min(getattr(config, "gpu_memory_utilization", 0.8), 0.7)

            logger.info(
                f"[VerifierOffload] Enabled verifier offload hints: swap_space={engine_kwargs.get('swap_space')}, "
                f"gpu_memory_utilization={getattr(config, 'gpu_memory_utilization', None)}"
            )

        # Ensure stop_token_ids is a plain list to satisfy vLLM SamplingParams type checks
        if "stop_token_ids" in kwargs and kwargs["stop_token_ids"] is not None and not isinstance(kwargs["stop_token_ids"], list):
            kwargs["stop_token_ids"] = list(kwargs["stop_token_ids"])

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )
        
        # #region agent log
        # NOTE: Disabled ad-hoc debug logging to `.cursor/debug.log` (not needed in normal runs).
        # #endregion

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        # NOTE: detokenize must be True when using stop strings
        has_stop_params = config.get("stop", None) is not None or config.get("stop_token_ids", None) is not None
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = has_stop_params  # True if stop strings used, False otherwise

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        # Hydra may hand us an OmegaConf ListConfig; vLLM expects a native list
        if "stop_token_ids" in kwargs and kwargs["stop_token_ids"] is not None and not isinstance(kwargs["stop_token_ids"], list):
            kwargs["stop_token_ids"] = list(kwargs["stop_token_ids"])

        # print(f"kwargs: {kwargs}")  # debug-only
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        # Verifier LoRA configuration for Co-GRPO
        # Use verifier_lora_name from kwargs if provided, otherwise from config, otherwise default
        if verifier_lora_name is not None:
            self.verifier_lora_name = verifier_lora_name
        else:
            self.verifier_lora_name = self.config.get("verifier_lora_name", "verifier_lora")
        self.verifier_lora_path = verifier_lora_path if verifier_lora_path else self.config.get("verifier", {}).get("lora_path", None)
        self.verifier_lora_int_id = None
        
        # Try to load Verifier LoRA if path is provided and LoRA is enabled
        # This initializes lora_manager if it's not already initialized
        if self.verifier_lora_path and hasattr(self, "lora_kwargs") and self.lora_kwargs.get("enable_lora", False):
            try:
                # Wake up engine to load LoRA
                self.inference_engine.wake_up()
                # Try to add LoRA adapter at initialization time
                # This will initialize lora_manager if it's not already initialized
                if hasattr(self.inference_engine, "add_lora"):
                    # vLLM supports runtime LoRA loading via add_lora
                    try:
                        self.inference_engine.add_lora(
                            adapter_name=self.verifier_lora_name,
                            adapter_path=self.verifier_lora_path
                        )
                        logger.info(f"Successfully loaded Verifier LoRA from {self.verifier_lora_path}")
                        # #region agent log
                        # NOTE: Disabled ad-hoc debug logging to `.cursor/debug.log` (not needed in normal runs).
                        # #endregion
                        # Put engine back to sleep
                        self.inference_engine.sleep(level=1)
                    except Exception as e:
                        logger.warning(f"Failed to load Verifier LoRA via add_lora: {e}")
                        # #region agent log
                        # NOTE: Disabled ad-hoc debug logging to `.cursor/debug.log` (not needed in normal runs).
                        # #endregion
                        self.inference_engine.sleep(level=1)
            except Exception as e:
                logger.warning(f"Error trying to load Verifier LoRA: {e}")
                # #region agent log
                # NOTE: Disabled ad-hoc debug logging to `.cursor/debug.log` (not needed in normal runs).
                # #endregion

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            # Check if stop sequences are being set - if so, detokenize must be True
            has_stop_sequences = kwargs.get('stop', None) is not None
            if has_stop_sequences:
                kwargs['detokenize'] = True

            for key, value in kwargs.items():
                # Always try to set the attribute, even if it doesn't exist yet
                try:
                    old_value = getattr(self.sampling_params, key, None)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
                except Exception as e:
                    logger.warning(f"Failed to set sampling param {key}={value}: {e}")
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            if value is not None:
                setattr(self.sampling_params, key, value)
            else:
                # Attribute didn't exist before, try to delete it
                try:
                    delattr(self.sampling_params, key)
                except:
                    pass


    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

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
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path=self.verifier_lora_path)] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)
                    # curr_input_log_prob = []
                    # for i, logprob in enumerate(output.prompt_logprobs):
                    #     if logprob is None:
                    #         assert i == 0
                    #         continue
                    #     curr_input_log_prob.append(logprob[output.prompt_token_ids[i]].logprob)
                    # input_log_probs.append(curr_input_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            # input_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=idx.size(1), left=True).to(idx.device).to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                # "input_log_probs": input_log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

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
        try:
            # #region agent log
            # NOTE: Disabled ad-hoc debug logging to `.cursor/debug.log` (not needed in normal runs).
            # #endregion
            
            # Strategy: Check if LoRA is enabled via lora_kwargs, then use fallback strategy
            # This avoids calling list_loras() which requires lora_manager to be initialized
            # In vLLM, lora_manager may not be initialized even if enable_lora=True
            # unless an actual LoRA adapter is loaded
            
            # Check if LoRA is enabled
            lora_enabled = False
            max_loras = 0
            if hasattr(self, "lora_kwargs"):
                lora_enabled = self.lora_kwargs.get("enable_lora", False)
                max_loras = self.lora_kwargs.get("max_loras", 0)
            
            # #region agent log
            # NOTE: Disabled ad-hoc debug logging to `.cursor/debug.log` (not needed in normal runs).
            # #endregion
            
            # If LoRA is enabled and max_loras > 1, use LoRA ID 1 for Verifier
            # This is a safe fallback that avoids accessing lora_manager
            if lora_enabled and max_loras > 1:
                self.verifier_lora_int_id = 1
                logger.info(f"Using LoRA ID 1 for Verifier (enable_lora={lora_enabled}, max_loras={max_loras}, verifier_lora_name={self.verifier_lora_name})")
                # #region agent log
                # NOTE: Disabled ad-hoc debug logging to `.cursor/debug.log` (not needed in normal runs).
                # #endregion
            else:
                self.verifier_lora_int_id = None
                logger.warning(f"LoRA not enabled or max_loras <= 1 (enable_lora={lora_enabled}, max_loras={max_loras}). Verifier will use base model.")
                # #region agent log
                # NOTE: Disabled ad-hoc debug logging to `.cursor/debug.log` (not needed in normal runs).
                # #endregion
        except Exception as e:
            logger.warning(f"Error in _register_verifier_lora: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            # #region agent log
            # NOTE: Disabled ad-hoc debug logging to `.cursor/debug.log` (not needed in normal runs).
            # #endregion
            # Fallback: If max_loras > 1, use LoRA ID 1
            if hasattr(self, "lora_kwargs") and self.lora_kwargs.get("max_loras", 0) > 1:
                self.verifier_lora_int_id = 1
                logger.info(f"Fallback: Using LoRA ID 1 for Verifier (max_loras={self.lora_kwargs.get('max_loras')})")
            else:
                self.verifier_lora_int_id = None

    @GPUMemoryLogger(role="vllm dual stream rollout", logger=logger)
    @torch.no_grad()
    def dual_stream_rollout(
        self, 
        prompts: DataProto, 
        intervention_mode: str = "by_response",
        token_check_interval: int = 1024,
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
        timing_info = {}
        start_time = time.time()
        
        tokenizer = self.tokenizer  # Use stored tokenizer
        
        # Rebuild vllm cache engine (with version compatibility)
        if self.config.free_cache_engine:
            try:
                self.inference_engine.wake_up()
            except AttributeError:
                # Fallback for older vLLM versions
                try:
                    self.inference_engine.init_cache_engine()
                except:
                    pass

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
        # TEMP: Disable by_response and by_token to focus on debugging by_step
        # if intervention_mode == "by_response":
        #     return self._dual_stream_rollout_by_response(
        #         prompts, timing_info, start_time, tokenizer, idx, attention_mask,
        #         position_ids, eos_token_id, batch_size, **kwargs
        #     )
        # elif intervention_mode == "by_token":
        #     return self._dual_stream_rollout_by_token(
        #         prompts, timing_info, start_time, tokenizer, idx, attention_mask,
        #         position_ids, eos_token_id, batch_size, token_check_interval,
        #         entropy_threshold, use_entropy_filter, **kwargs
        #     )
        # elif intervention_mode == "by_step":
        if intervention_mode == "by_step":
            # CRITICAL FIX: Pass dual_stream_rollout's explicit parameters to _dual_stream_rollout_by_step via kwargs
            # Otherwise token_check_interval won't be passed because it's an explicit parameter of dual_stream_rollout,
            # not included in **kwargs automatically
            kwargs['token_check_interval'] = token_check_interval
            # Keep min_step_tokens in sync with token_check_interval unless overridden
            kwargs.setdefault('min_step_tokens', token_check_interval)
            kwargs['entropy_threshold'] = entropy_threshold
            kwargs['use_entropy_filter'] = use_entropy_filter
            return self._dual_stream_rollout_by_step(
                prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                position_ids, eos_token_id, batch_size, **kwargs
            )
        else:
            # Fallback: Force by_step mode for debugging
            logger.warning(f"intervention_mode={intervention_mode} not supported yet, forcing by_step mode")
            kwargs['token_check_interval'] = token_check_interval
            kwargs['entropy_threshold'] = entropy_threshold
            kwargs['use_entropy_filter'] = use_entropy_filter
            return self._dual_stream_rollout_by_step(
                prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                position_ids, eos_token_id, batch_size, **kwargs
            )
    
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

        # =================================================================
        # FIX START: Robust Batch Size Check for Co-GRPO
        # =================================================================

        # 1. Basic configuration
        if not do_sample:
            control_kwargs = {
                "best_of": 1, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                "temperature": 0, "n": 1,
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

        # 2. Co-GRPO: Force n=1 to avoid double repetition
        # In Co-GRPO, trainer already repeats gen_batch BEFORE calling dual_stream_rollout
        # So we MUST use n=1 here, otherwise vLLM will repeat again causing batch_size mismatch
        if do_sample and not is_validate and self.sampling_params.n > 1:
            target_n = self.sampling_params.n
            logger.info(f"Co-GRPO by_step: Forcing vLLM n=1 (trainer already repeated with n={target_n})")
            control_kwargs["n"] = 1

        # =================================================================
        # FIX END
        # =================================================================

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
        
        control_responses, control_log_probs = _extract_vllm_outputs(
            control_output, self.pad_token_id, self.config.response_length, idx.device
        )
        timing_info["control_gen"] = time.time() - control_start

        # Handle num_generation_per_prompt > 1 case
        # IMPORTANT: Check if batch was already repeated by trainer (for Co-GRPO)
        # In Co-GRPO, trainer repeats gen_batch before calling dual_stream_rollout
        # So we should NOT repeat again here to avoid double repetition
        if self.sampling_params.n > 1 and do_sample:
            n = self.sampling_params.n
            # Check if batch was already repeated by checking if batch_size is a multiple of n
            # If batch_size >= n and is divisible by n, it was likely pre-repeated by trainer
            if batch_size % n == 0 and batch_size >= n:
                # Batch was already repeated by trainer, don't repeat again
                logger.info(f"Co-GRPO: Batch already repeated by trainer (size={batch_size}, n={n}), skipping rollout-side repeat")
            else:
                # Normal case: repeat batch in rollout
                idx = _repeat_interleave(idx, n)
                attention_mask = _repeat_interleave(attention_mask, n)
                position_ids = _repeat_interleave(position_ids, n)
                # Repeat idx_list to match the expanded batch size
                idx_list = idx_list * n
                batch_size = batch_size * n
                logger.info(f"Co-GRPO: Repeated batch in rollout (new size={batch_size})")

        # ========== Experimental Stream: Policy -> Verifier -> Policy ==========
        # Step 1: Policy generates initial draft (CoT)
        draft_start = time.time()
        draft_kwargs = control_kwargs.copy()
        # IMPORTANT: After batch expansion, we need n=1 for experimental stream
        # to avoid generating too many responses
        if do_sample and not is_validate:
            draft_kwargs["n"] = 1
        with self.update_sampling_params(**draft_kwargs):
            draft_output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                lora_request=None,  # No LoRA for initial draft
                use_tqdm=False,
            )
        
        draft_responses, draft_log_probs = _extract_vllm_outputs(
            draft_output, self.pad_token_id, self.config.response_length, idx.device
        )
        
        if draft_responses.shape[1] < self.config.response_length:
            draft_responses = pad_sequence_to_length(draft_responses, self.config.response_length, self.pad_token_id)
            draft_log_probs = pad_sequence_to_length(draft_log_probs, self.config.response_length, self.pad_token_id)
        timing_info["draft_gen"] = time.time() - draft_start

        # Step 2: Decode draft responses and construct Verifier input
        # Format: System prompt + Task instruction + Question + Student Response
        draft_texts = []
        prompt_texts = []
        for i in range(batch_size):
            prompt_text = tokenizer.decode(idx[i][attention_mask[i].bool()], skip_special_tokens=True)
            draft_text = tokenizer.decode(draft_responses[i], skip_special_tokens=True)
            prompt_texts.append(prompt_text)
            draft_texts.append(draft_text)
        
        # Construct Verifier input with system prompt and task instruction
        # Format matches format_training_data.py: system -> user (intervene_prompt + question + student response)
        # CRITICAL FIX: Use separate system and user messages to match training data format
        verifier_chat_inputs = []
        for prompt, draft in zip(prompt_texts, draft_texts):
            # Construct user content: intervene_prompt + question + student response
            user_content = f"{VERIFIER_INTERVENE_PROMPT} Question: {prompt}\n\nStudent Response: {draft}"

            # Format as separate system and user messages (matches training data)
            messages = [
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            # Apply chat template to get final text
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            verifier_chat_inputs.append(chat_text)

        # Tokenize Verifier inputs using chat template
        verifier_encoded = tokenizer(
            verifier_chat_inputs,  # Already chat-formatted
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
                            lora_path=getattr(self, 'verifier_lora_path', None)  # Use actual Verifier LoRA path
                )
            ] * batch_size

        # CRITICAL: Verifier must use temperature=0 for deterministic tag output
        # Override sampling params to force greedy decoding for Verifier
        verifier_kwargs = control_kwargs.copy()
        verifier_kwargs["temperature"] = 0
        verifier_kwargs["top_p"] = 1.0
        verifier_kwargs["top_k"] = -1

        with self.update_sampling_params(**verifier_kwargs):
            verifier_output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=verifier_idx_list,
                lora_request=verifier_lora_requests,
                use_tqdm=False,
            )
        
        verifier_responses, verifier_log_probs = _extract_vllm_outputs(
            verifier_output, self.pad_token_id, self.config.response_length, idx.device
        )
        
        if verifier_responses.shape[1] < self.config.response_length:
            verifier_responses = pad_sequence_to_length(verifier_responses, self.config.response_length, self.pad_token_id)
            verifier_log_probs = pad_sequence_to_length(verifier_log_probs, self.config.response_length, self.pad_token_id)

        # Decode Verifier outputs and extract actual hints (remove think blocks, extract <GO> or <WAIT> content)
        verifier_texts = []  # Full output for logging
        hints = []  # Extracted hints for actual use
        for i in range(batch_size):
            verifier_text = tokenizer.decode(verifier_responses[i].tolist(), skip_special_tokens=True)
            verifier_texts.append(verifier_text)
            # Extract actual hint (removes think blocks, extracts <GO> or <WAIT> content)
            hint = self._extract_verifier_hint(verifier_text)
            hints.append(hint)
        timing_info["verifier_gen"] = time.time() - verifier_start
        
        # Step 4: Policy generates final answer with Verifier guidance
        # Use extracted hints (without think blocks)
        final_inputs = []
        for prompt, hint in zip(prompt_texts, hints):
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
        
        exp_responses, exp_log_probs = _extract_vllm_outputs(
            final_output, self.pad_token_id, self.config.response_length, idx.device
        )
        
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
        # Use extracted hints (without think blocks) for actual use
        hints_array = np.array(hints, dtype=object)
        critiques = np.array(verifier_texts, dtype=object)  # Keep full output for logging/critique
        
        # Store intervention statistics for penalty calculation
        # In by_response mode, each sample has 1 intervention (the verifier output)
        num_interventions = np.ones(batch_size, dtype=np.int32)
        # Calculate token counts based on extracted hints (not full verifier output)
        hint_token_counts = np.array([
            len(tokenizer.encode(hint, add_special_tokens=False)) for hint in hints
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
        # IMPORTANT: Preserve the original non_tensor_batch fields (reward_model, data_source, etc.)
        # These are required for reward computation in NaiveRewardManager
        non_tensor_batch = prompts.non_tensor_batch.copy()

        # If batch was repeated (n > 1), we need to repeat fields in non_tensor_batch as well
        # to match the new batch_size
        original_batch_size = len(prompts.batch["input_ids"])
        if batch_size > original_batch_size:
            repeat_factor = batch_size // original_batch_size
            # Repeat all fields in non_tensor_batch to match the expanded batch size
            for key, val in non_tensor_batch.items():
                if isinstance(val, np.ndarray) and val.shape[0] == original_batch_size:
                    non_tensor_batch[key] = _repeat_interleave(val, repeat_factor)

        non_tensor_batch.update({
            "hints": hints_array,  # Use extracted hints (without think blocks)
            "critiques": critiques,
            "draft_responses": np.array([tokenizer.decode(draft_responses[i], skip_special_tokens=True) for i in range(batch_size)], dtype=object),
            "num_interventions": num_interventions,
            "hint_token_counts": hint_token_counts,
        })

        # Free vllm cache engine (with version compatibility)
        if self.config.free_cache_engine:
            try:
                self.inference_engine.sleep(level=1)
            except AttributeError:
                # Fallback for older vLLM versions
                try:
                    self.inference_engine.free_cache_engine()
                except:
                    pass

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
        # [".\n", "?\n","</think>", "<｜end of thought｜>", "\n\n", ".\n\n"]
        for seq in ["</think>", "<｜end of thought｜>", "\n\n", ".\n\n"]:
            stop_sequences.add(seq)
        
        # 2. 显式添加 \n 的 Token ID (防止 tokenizer 差异)
        try:
            nl_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
            stop_token_ids.add(nl_id)
        except:
            pass
            
        if eos_token_id is not None:
            # 处理 eos_token_id 可能是 list 的情况
            if isinstance(eos_token_id, (list, tuple)):
                stop_token_ids.update(eos_token_id)
            else:
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

            # CRITICAL FIX: Use chat template to format inputs
            # Verifier was trained with chat format, must use apply_chat_template
            # CRITICAL: Use separate system and user messages to match training data format
            verifier_chat_inputs = []
            for verifier_prompt in verifier_inputs:
                messages = [
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": verifier_prompt}
                ]
                chat_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                verifier_chat_inputs.append(chat_text)

            # 批量 tokenize
            verifier_encoded = tokenizer(
                verifier_chat_inputs,  # Use chat-formatted inputs
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
                                lora_path=getattr(self, 'verifier_lora_path', None)  # Use actual Verifier LoRA path
                    )
                ] * len(verifier_inputs)

            # CRITICAL: Verifier must use temperature=0 for deterministic tag output
            verifier_kwargs = exp_kwargs.copy()
            verifier_kwargs["temperature"] = 0
            verifier_kwargs["top_p"] = 1.0
            verifier_kwargs["top_k"] = -1

            # 批量生成 Verifier 响应
            with self.update_sampling_params(**verifier_kwargs):
                verifier_output = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=verifier_idx_list,
                    lora_request=verifier_lora_requests,
                    use_tqdm=False,
                )
            
            verifier_responses, verifier_log_probs = _extract_vllm_outputs(
                verifier_output, self.pad_token_id, self.config.response_length, idx_device
            )
            
            # 解析所有 Verifier 输出
            decisions = []
            for i in range(len(verifier_inputs)):
                verifier_response = verifier_responses[i]
                verifier_text = tokenizer.decode(verifier_response.tolist(), skip_special_tokens=True)
                decision = self._parse_verifier_decision(verifier_text)
                decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Verifier inference failed: {e}")
            # 返回默认决策（Pass）
            return [{'action': 'Pass', 'hint': None, 'critique': f'[Error: {str(e)}]'} for _ in verifier_inputs]
    
    def _extract_verifier_hint(self, verifier_text: str) -> str:
        """
        Extract actual hint from Verifier output, removing think blocks and extracting <GO> or <WAIT> content.
        
        Args:
            verifier_text: Full Verifier output, may contain think blocks (```...```)
        
        Returns:
            hint: Extracted hint content (only <GO> or <WAIT> followed content, without think blocks)
        """
        if not verifier_text:
            return ""
        
        # Step 1: Remove think blocks (```\n...\n``` or ```...```)
        # Match code blocks with various formats
        text_without_think = re.sub(
            r'```\s*\n?.*?\n?\s*```',
            '',
            verifier_text,
            flags=re.DOTALL
        )
        
        # Also remove <think> blocks if present
        text_without_think = re.sub(
            r'<think>.*?</think>',
            '',
            text_without_think,
            flags=re.DOTALL
        )
        
        # Step 2: Extract <GO> or <WAIT> followed content
        # Pattern: <GO> or <WAIT> followed by optional whitespace and content
        go_match = re.search(r'<GO>\s*(.*?)(?:\n|$)', text_without_think, re.IGNORECASE | re.DOTALL)
        wait_match = re.search(r'<WAIT>\s*(.*?)(?:\n|$)', text_without_think, re.IGNORECASE | re.DOTALL)
        
        if wait_match:
            hint = f"<WAIT> {wait_match.group(1).strip()}"
        elif go_match:
            hint = "<GO>"
        else:
            # Fallback: try to find <GO> or <WAIT> without strict format
            if "<WAIT>" in text_without_think.upper():
                # Extract everything after <WAIT>
                parts = re.split(r'<WAIT>', text_without_think, flags=re.IGNORECASE)
                if len(parts) > 1:
                    hint = f"<WAIT> {parts[-1].strip()}"
                else:
                    hint = ""
            elif "<GO>" in text_without_think.upper():
                hint = "<GO>"
            else:
                hint = ""
        
        return hint.strip()
    
    def _parse_verifier_decision(self, verifier_text: str) -> dict:
        """
        Parse Verifier output to extract decision (Pass/Intervene) and hint.
        
        New format: Extract <GO> or <WAIT> content after removing think blocks.
        
        Returns:
            dict with keys: "action" (str), "hint" (str), "critique" (str)
        """
        verifier_text = verifier_text.strip()
        
        # Extract actual hint (removes think blocks, extracts <GO> or <WAIT> content)
        hint = self._extract_verifier_hint(verifier_text)
        
        # Determine action based on hint
        if hint.upper().startswith("<WAIT>"):
            action = "Intervene"
            # Extract content after <WAIT>
            hint_content = hint.replace("<WAIT>", "").strip()
            hint = hint_content  # CRITICAL: Use just the content, NOT "<WAIT> {content}"
            # The <WAIT> tag confuses the Actor and causes it to repeat empty tokens!
        elif hint.upper().startswith("<GO>"):
            action = "Pass"
            hint = ""
        else:
            # Fallback: try old format parsing
            lower_text = verifier_text.lower()
            if "<wait>" in lower_text:
                hint_content = verifier_text.split("<WAIT>", 1)[-1].strip()
                action = "Intervene"
                hint = hint_content  # CRITICAL: Use just the content, NOT "<WAIT> {content}"
            elif "<go>" in lower_text:
                action = "Pass"
                hint = "<GO>"
            else:
                # Default: Pass if no clear signal
                action = "Pass"
                hint = ""
        
        return {
            "action": action,
            "hint": hint,
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
            "Falling back to by_step mode. "
            "Full implementation requires incremental generation support."
        )

        # Use by_step as fallback
        return self._dual_stream_rollout_by_step(
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

        # =================================================================
        # FIX START: Robust Batch Size Check for Co-GRPO
        # =================================================================

        # 1. Basic configuration
        if not do_sample:
            control_kwargs = {
                "best_of": 1, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                "temperature": 0, "n": 1,
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

        # 2. Co-GRPO: Force n=1 to avoid double repetition
        # In Co-GRPO, trainer already repeats gen_batch BEFORE calling dual_stream_rollout
        # So we MUST use n=1 here, otherwise vLLM will repeat again causing batch_size mismatch
        if do_sample and not is_validate and self.sampling_params.n > 1:
            target_n = self.sampling_params.n
            logger.info(f"Co-GRPO by_step: Forcing vLLM n=1 (trainer already repeated with n={target_n})")
            control_kwargs["n"] = 1

        # =================================================================
        # FIX END
        # =================================================================

        # Store the actual n used for control generation (for batch expansion)
        control_n = control_kwargs.get("n", self.sampling_params.n if do_sample and not is_validate else 1)

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

        control_responses, control_log_probs = _extract_vllm_outputs(
            control_output, self.pad_token_id, self.config.response_length, idx.device
        )
        timing_info["control_gen"] = time.time() - control_start

        # Handle num_generation_per_prompt > 1 case
        # IMPORTANT: Check if batch was already repeated by trainer (for Co-GRPO)
        # In Co-GRPO, trainer repeats gen_batch before calling dual_stream_rollout
        # So we should NOT repeat again here to avoid double repetition
        # IMPORTANT: Use control_n (the actual n used) not self.sampling_params.n
        if control_n > 1 and do_sample:
            n = control_n
            # Check if batch was already repeated by checking if batch_size is a multiple of n
            # If batch_size >= n and is divisible by n, it was likely pre-repeated by trainer
            if batch_size % n == 0 and batch_size >= n:
                # Batch was already repeated by trainer, don't repeat again
                pass
            else:
                # Normal case: repeat batch in rollout
                idx = _repeat_interleave(idx, n)
                attention_mask = _repeat_interleave(attention_mask, n)
                position_ids = _repeat_interleave(position_ids, n)
                # Repeat idx_list to match the expanded batch size
                idx_list = idx_list * n
                batch_size = batch_size * n

        # ========== Experimental Stream: 批量并行增量打断生成 ==========
        # 1. 获取停止配置
        stop_config = self._get_step_boundary_stop_config(tokenizer, eos_token_id)
        if not stop_config['stop_sequences'] and not stop_config['stop_token_ids']:
            logger.warning("No stop sequences found, using full generation without step boundaries")
            # Fall back to simple full generation (continue without step interruption)
            pass  # Continue with full generation

        # Pre-encode stop sequences for local detection (we ignore stop at the sampler level)
        stop_seq_token_ids = []
        for seq in stop_config['stop_sequences']:
            if not seq:
                continue
            ids = tokenizer.encode(seq, add_special_tokens=False)
            if ids:
                stop_seq_token_ids.append(ids)
        stop_token_ids = stop_config['stop_token_ids'] or []

        # 2. 初始化 InterventionPolicy
        max_interventions = kwargs.get('max_interventions', 3)
        confidence_threshold = kwargs.get('confidence_threshold', 0.7)
        intervention_policy = InterventionPolicy(
            max_interventions=max_interventions,
            confidence_threshold=confidence_threshold
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
            # IMPORTANT: When n > 1, we've already expanded idx/attention_mask/position_ids
            # so we need to set n=1 for experimental stream to avoid over-generating
            exp_kwargs = {"n": 1}
        
        # Defensive: ensure budget values are ints even if overridden as strings
        # NOTE: Must be defined before sample_states init (used for prompt length checks).
        response_budget = int(self.config.response_length)
        prompt_budget = int(self.config.prompt_length)

        # 4. 初始化所有样本的状态（批量管理）
        sample_states = []
        for b in range(batch_size):
            prompt_tokens = idx[b][attention_mask[b].bool()].tolist()
            # Defensive: strip trailing EOS to avoid immediate stop
            if eos_token_id is not None and prompt_tokens and prompt_tokens[-1] == eos_token_id:
                logger.warning(f"[PROMPT TRIM EOS] Sample {b}: stripping trailing EOS {eos_token_id}")
                prompt_tokens = prompt_tokens[:-1]
            if len(prompt_tokens) > prompt_budget:
                logger.warning(f"[PROMPT OVER BUDGET] Sample {b}: prompt_len={len(prompt_tokens)} > prompt_budget={prompt_budget}")
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
                'tokens_since_boundary': 0,  # Model tokens since last accepted boundary
                'last_finish_reason': None,  # track final finish reason for diagnostics/dump
                'context_exhausted': False,   # flag if we hit context budget
                'first_step_tokens_len': None,  # record first step token length for debugging
            })
        
        # 5. 监控指标
        metrics = {
            'total_steps': 0,
            'total_interventions': 0,
            'stop_sequence_hits': 0,
            'errors': 0,
        }
        
        exp_start = time.time()
        # Dynamic max_steps calculation based on response_budget and token_check_interval
        # This ensures we can generate up to response_budget tokens (not limited by hardcoded max_steps)
        token_check_interval = int(kwargs.get('token_check_interval', 1024))
        min_step_tokens = int(kwargs.get('min_step_tokens', token_check_interval))  # FIX: Extract from kwargs
        default_max_steps = (response_budget // token_check_interval) + 1  # +1 to handle remainder
        max_steps = int(kwargs.get('max_steps', default_max_steps))

        # Context budget: prompt + response + hint headroom
        # Reserve space for hints so that hint insertion doesn't cause context_exhausted
        # Note: response_budget is for model-generated tokens only; hints are extra
        max_interventions = int(kwargs.get('max_interventions', 5))
        estimated_hint_tokens = 256  # Average hint is 256 tokens
        hint_headroom = max_interventions * estimated_hint_tokens
        context_budget = prompt_budget + response_budget + hint_headroom

        logger.info(f"[BY_STEP] max_steps={max_steps} (response_budget={response_budget}, token_check_interval={token_check_interval}, "
                    f"min_step_tokens={min_step_tokens}, context_budget={context_budget}, hint_headroom={hint_headroom})")

        # 6. 批量增量生成循环
        for global_step in range(max_steps):
            metrics['total_steps'] += 1

            # 6.1 收集所有活跃样本（未完成的样本）
            active_indices = [b for b in range(batch_size) if not sample_states[b]['is_complete']]

            if not active_indices:
                break

            # 6.2 批量构建当前输入（并做逐样本 budget 过滤）
            batch_inputs = []
            batch_indices = []
            current_response_lens = []
            gen_remaining_list = []
            context_remaining_list = []

            for b in active_indices:
                state = sample_states[b]

                # Input = Prompt + Response (Hint included)
                current_input = state['prompt_tokens'] + state['response_tokens']
                context_remaining = context_budget - len(current_input)

                # Only count model-generated tokens against response budget; hints do not consume budget.
                gen_len = sum(1 for m in state['loss_masks'] if m == 1)
                gen_remaining = response_budget - gen_len

                if context_remaining <= 0:
                    logger.error(
                        f"[BUDGET EXHAUSTED] Sample {b}: context_len={len(current_input)} >= budget={context_budget}; marking complete"
                    )
                    state['is_complete'] = True
                    state['context_exhausted'] = True
                    state['last_finish_reason'] = 'context_exhausted'
                    continue

                if gen_remaining <= 0:
                    # Generated-token budget exhausted; stop generating but keep existing response/hints.
                    logger.info(
                        f"[GEN BUDGET EXHAUSTED] Sample {b}: gen_len={gen_len} >= response_budget={response_budget}; marking complete"
                    )
                    state['is_complete'] = True
                    state['last_finish_reason'] = 'gen_budget_exhausted'
                    continue

                batch_inputs.append(current_input)
                batch_indices.append(b)
                current_response_lens.append(len(state['response_tokens']))
                gen_remaining_list.append(gen_remaining)
                context_remaining_list.append(context_remaining)
            
            if not batch_inputs:
                break
            
            # 6.3 计算统一的 max_tokens（FIX: 使用 max 而不是 min）
            #
            # 问题：之前使用 min(gen_remaining_list) 导致 batch 中任何一个样本接近完成时，
            #      所有其他样本也被限制，只能生成很少的 tokens
            #
            # 解决：使用 max(gen_remaining_list)，让预算多的样本能生成更多
            #       然后通过后处理截断，防止超出单个样本的 budget
            #
            if not gen_remaining_list or not context_remaining_list:
                break

            # 使用最大的剩余 budget（而不是最小的）
            # 这样预算多的样本不会被预算少的样本拖累
            # FIX: gen_remaining 和 context_remaining 都用 max()
            active_gen_remaining = [r for r in gen_remaining_list if r > 0]
            if not active_gen_remaining:
                break
            active_context_remaining = [r for r in context_remaining_list if r > 0]
            if not active_context_remaining:
                break

            max_gen_remaining = max(active_gen_remaining)
            max_context_remaining = max(active_context_remaining)  # FIX: min → max
            raw_max_remaining = min(max_gen_remaining, max_context_remaining)
            max_remaining = min(token_check_interval, raw_max_remaining)

            # Debug log: 显示剩余 budget 分布
            if len(gen_remaining_list) > 1:
                logger.info(
                    f"[BY_STEP] Step {global_step}: gen_remaining=[min={min(gen_remaining_list)}, max={max_gen_remaining}], "
                    f"context_remaining=[min={min(context_remaining_list)}, max={max_context_remaining}], "
                    f"max_remaining={max_remaining}"
                )

            if max_remaining <= 0:
                logger.error(f"[BY_STEP] Step {global_step}: max_remaining<=0; breaking.")
                break

            # 6.4 批量生成到下一个 boundary
            batch_inputs_processed = []
            for inp in batch_inputs:
                batch_inputs_processed.append(
                    _pre_process_inputs(self.pad_token_id, torch.tensor(inp, device=idx.device))
                )

            try:
                sampling_override = dict(
                    stop=[],  # handle stop locally
                    stop_token_ids=[],
                    max_tokens=max_remaining,
                    include_stop_str_in_output=True,
                    **exp_kwargs,
                )
                # FIX: Don't ignore EOS to prevent infinite loops
                # Previously set ignore_eos=True which caused models to continue generating
                # after producing EOS, leading to repetitive output
                # if hasattr(self.sampling_params, "ignore_eos"):
                #     sampling_override["ignore_eos"] = True

                with self.update_sampling_params(**sampling_override):
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
                for b in batch_indices:
                    sample_states[b]['is_complete'] = True
                    sample_states[b]['error'] = str(e)
                continue

            # 6.5 提取新生成的 tokens（批量处理）
            step_responses = []
            step_log_probs = []
            step_finish_reasons = []
            for output in step_output:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    step_responses.append(response_ids)

                    # Extract logprobs
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    step_log_probs.append(curr_log_prob)

                    # Extract finish_reason
                    finish_reason = output.outputs[sample_id].finish_reason
                    step_finish_reasons.append(finish_reason)

            # 6.6 更新每个样本的状态并检测停止
            if len(step_responses) != len(batch_indices):
                logger.error(
                    f"[MISMATCH] vLLM output count != input count! "
                    f"len(step_responses)={len(step_responses)}, len(batch_indices)={len(batch_indices)}, "
                    f"len(batch_inputs)={len(batch_inputs)}, global_step={global_step}"
                )
                metrics['errors'] += len(batch_indices)
                for b in batch_indices:
                    sample_states[b]['is_complete'] = True
                    sample_states[b]['error'] = 'vllm_output_count_mismatch'
                    sample_states[b]['last_finish_reason'] = 'vllm_output_count_mismatch'
                continue
            logger.debug(f"[MATCH] vLLM output count matches input count: {len(step_responses)}")

            # 6.6 更新每个样本的状态并检测停止
            verifier_inputs = []
            verifier_input_map = []  # (batch_idx, step_idx)

            for i, b in enumerate(batch_indices):
                state = sample_states[b]
                step_tokens = list(step_responses[i])  # FIX: 确保是 list，vLLM 可能返回 tuple
                finish_reason = step_finish_reasons[i]  # 获取结束原因
                state['last_finish_reason'] = finish_reason  # Track last finish reason
                if state.get('first_step_tokens_len') is None:
                    state['first_step_tokens_len'] = len(step_tokens)

                # 获取该样本的剩余 budget
                gen_len = sum(1 for m in state['loss_masks'] if m == 1)
                sample_gen_remaining = response_budget - gen_len

                # 获取该样本的剩余 context budget（FIX: 也需要检查，防止超出 context budget）
                current_input_len = len(state['prompt_tokens']) + len(state['response_tokens'])
                sample_context_remaining = context_budget - current_input_len

                # 获取实际发送给 vLLM 的输入（去 padding 后），用于前缀检测
                sent_input_ids = batch_inputs_processed[i]

                # ========== 前缀检测：防止 vLLM 回显输入 ==========
                if len(step_tokens) >= len(sent_input_ids) and step_tokens[:len(sent_input_ids)] == sent_input_ids:
                    logger.warning(f"[ECHO DETECTED] Step {global_step}, Sample {b}: vLLM echoed input; stripping prefix len={len(sent_input_ids)}.")
                    new_tokens = step_tokens[len(sent_input_ids):]
                else:
                    new_tokens = step_tokens

                # ========== 后处理截断：防止超出该样本的 budget ==========
                # FIX: 由于使用 max() 而不是 min()，预算少的样本可能被超额生成
                # 需要截断到该样本的实际剩余 budget
                # 同时检查 gen_budget 和 context_budget，取最小值
                effective_budget = min(sample_gen_remaining, sample_context_remaining)

                # 边界检查：如果 effective_budget <= 0，说明该样本已经超出预算
                if effective_budget <= 0:
                    logger.warning(
                        f"[BUDGET OVERFLOW] Step {global_step}, Sample {b}: effective_budget={effective_budget} <= 0 "
                        f"(gen={sample_gen_remaining}, context={sample_context_remaining}), discarding new_tokens."
                    )
                    state['is_complete'] = True
                    state['last_finish_reason'] = 'budget_overflow'
                    continue

                truncated = False
                if len(new_tokens) > effective_budget:
                    logger.info(
                        f"[TRUNCATE] Step {global_step}, Sample {b}: Generated {len(new_tokens)} > effective_budget {effective_budget} "
                        f"(gen={sample_gen_remaining}, context={sample_context_remaining}), truncating."
                    )
                    new_tokens = new_tokens[:effective_budget]
                    truncated = True
                    # 注意：不在这里设置 is_complete，等添加 tokens 后再设置

                # ========== 空响应诊断 ==========
                if len(new_tokens) == 0:
                    logger.warning(
                        f"[EMPTY RESPONSE] Step {global_step}, Sample {b}: finish_reason='{finish_reason}', "
                        f"input_len={len(sent_input_ids)}, step_tokens_len={len(step_tokens)}, "
                        f"sample_gen_remaining={sample_gen_remaining}, sample_context_remaining={sample_context_remaining}, "
                        f"max_remaining={max_remaining}, prompt_tail={sent_input_ids[-5:]}, raw_output_snippet={step_tokens[:10]}..."
                    )
                    if finish_reason == 'length':
                        logger.error(
                            f"[CRITICAL] Step {global_step}, Sample {b}: length stop with 0 new tokens. "
                            f"state_response_len={len(state['response_tokens'])}, budget={response_budget}, "
                            f"total_context={current_input_len}"
                        )

                # 调试日志 - 增强版
                logger.debug(f"Step {global_step}, Sample {b}: step_tokens_len={len(step_tokens)}, new_tokens_len={len(new_tokens)}, "
                            f"finish_reason={finish_reason}, current_input_len={current_input_len}, "
                            f"sample_gen_remaining={sample_gen_remaining}, sample_context_remaining={sample_context_remaining}")

                # CRITICAL DIAGNOSTIC: Check for PAD tokens in new_tokens
                if self.pad_token_id in new_tokens:
                    pad_count = new_tokens.count(self.pad_token_id)
                    logger.warning(f"[PAD BUG] Step {global_step}, Sample {b}: Found {pad_count} PAD tokens in new_tokens! "
                                  f"This indicates padding bug not fully fixed.")
                else:
                    logger.debug(f"[PAD FIX] Step {global_step}, Sample {b}: No PAD tokens in new_tokens ✓")

                # Note: Echo handling is done above using the exact input ids sent to vLLM.

                # 检查是否生成了新内容
                if len(new_tokens) == 0:
                    # Enhanced diagnostic for empty output - CRITICAL for debugging
                    logger.warning(f"[EMPTY OUTPUT] Step {global_step}, Sample {b}: No new tokens generated! "
                                   f"finish_reason={finish_reason}, current_input_len={current_input_len}, "
                                   f"max_remaining={max_remaining}, step_tokens_raw_len={len(step_tokens)}")

                    # Show vLLM output details for debugging
                    if len(step_tokens) > 0:
                        # Show first few tokens to diagnose echo/padding
                        first_tokens = step_tokens[:10] if len(step_tokens) >= 10 else step_tokens
                        logger.warning(f"[EMPTY OUTPUT] step_tokens first 10: {first_tokens}")
                        # Check if it matches prompt (echo case not caught above)
                        if len(step_tokens) <= len(state['prompt_tokens']):
                            prefix_match = step_tokens == state['prompt_tokens'][:len(step_tokens)]
                            logger.warning(f"[EMPTY OUTPUT] Matches prompt prefix? {prefix_match}")
                    else:
                        # vLLM returned completely empty - this is the critical case
                        logger.error(f"[CRITICAL] Sample {b}: vLLM returned COMPLETELY EMPTY output! "
                                    f"finish_reason={finish_reason}, max_remaining={max_remaining}, "
                                    f"batch_idx={i}, global_step={global_step}")

                    state['is_complete'] = True
                    continue

                # 关键：追加到 response_tokens（不是 prompt_tokens）
                state['response_tokens'].extend(new_tokens)
                state['loss_masks'].extend([1] * len(new_tokens))  # Model Gen = 1
                state['step_count'] += 1
                state['tokens_since_boundary'] += len(new_tokens)

                # ========== 处理截断后的完成状态 ==========
                # FIX: 在添加 tokens 后再标记完成，确保截断的 tokens 不会丢失
                if truncated:
                    state['is_complete'] = True
                    if effective_budget == sample_gen_remaining:
                        state['last_finish_reason'] = 'gen_budget_exhausted'
                    else:
                        state['last_finish_reason'] = 'context_budget_exhausted'
                    continue  # 跳过 Verifier 调用

                # 检查是否应该结束
                if eos_token_id is not None and eos_token_id in new_tokens:
                    state['is_complete'] = True
                    continue
                # Hints do not consume the response budget; only model-generated tokens count.
                if sum(1 for m in state['loss_masks'] if m == 1) >= response_budget:
                    state['is_complete'] = True
                    continue

                # Local stop detection with minimum token gap
                hit_stop = False
                if state['tokens_since_boundary'] >= min_step_tokens:
                    # Check stop token ids in latest chunk
                    if stop_token_ids and any(tok in stop_token_ids for tok in new_tokens):
                        hit_stop = True
                    else:
                        # Check stop sequences as suffix match on full response
                        for seq_ids in stop_seq_token_ids:
                            if len(state['response_tokens']) >= len(seq_ids) and state['response_tokens'][-len(seq_ids):] == seq_ids:
                                hit_stop = True
                                break

                if hit_stop:
                    metrics['stop_sequence_hits'] += 1
                    state['tokens_since_boundary'] = 0  # reset window after accepting a boundary

                # CRITICAL FIX: ALWAYS call Verifier at each step boundary (unless max_interventions reached)
                # This ensures Verifier is called regularly, not just when stop sequence is hit
                # Without this fix, Verifier was almost never called → hints empty → EXP_STREAM detection failed
                # FIX: Also skip if sample is already complete (e.g., after truncation)
                if not state.get('is_complete', False) and len(state['hints']) < intervention_policy.max_interventions:
                    # Prepare Verifier input
                    current_reasoning = tokenizer.decode(state['response_tokens'], skip_special_tokens=True)

                    # CRITICAL FIX: Use training data format for Verifier input
                    # Format matches format_training_data.py: system -> user (intervene_prompt + question + student response)
                    # by_step mode: current_reasoning is the partial response so far
                    user_content = f"{VERIFIER_INTERVENE_PROMPT} Question: {state['prompt_text']}\n\nStudent Response: {current_reasoning}"
                    verifier_input = user_content  # Will be formatted with system message in _run_verifier_inference
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
        logger.info(f"[BY_STEP] metrics: avg_steps={timing_info['exp_metrics']['avg_steps']:.1f}, "
                   f"interventions={metrics['total_interventions']}, errors={metrics['errors']}")

        # 8. 数据后处理：构建完整的 DataProto（关键修正）
        exp_input_ids_list = []
        exp_attention_mask_list = []
        exp_position_ids_list = []
        exp_loss_mask_list = []  # 新增字段
        exp_responses_list = []
        exp_last_valid_pos_list = []
        all_hints = []
        all_critiques = []
        
        for b in range(batch_size):
            state = sample_states[b]
            # -----------------------------------------------------------------
            # CRITICAL FIX: enforce a stable two-segment layout for EXP stream
            # input_ids      = [prompt_segment(left-padded to prompt_budget)] + [response_segment(right-padded to response_budget)]
            # attention_mask = [0..0,1..1] + [1..1,0..0]
            # loss_mask      = [0..0] + [loss_masks_for_response_segment (0=hint,1=policy)]
            #
            # This matches the assumption in reward_manager/ray_trainer that the response segment
            # starts exactly at index prompt_budget. Without this, response tokens "leak" into the
            # prompt segment, causing valid_response_length to be 0/tiny (fake empty/truncated outputs).
            # -----------------------------------------------------------------
            total_len = prompt_budget + response_budget

            # 8.1 Prompt segment (left-pad)
            prompt_tokens = list(state.get("prompt_tokens") or [])
            if len(prompt_tokens) > prompt_budget:
                # Keep the tail for left-padded semantics.
                logger.warning(
                    f"[PROMPT TRUNCATION] Sample {b}: prompt_len={len(prompt_tokens)} > prompt_budget={prompt_budget}; keeping tail"
                )
                prompt_tokens = prompt_tokens[-prompt_budget:]

            prompt_ids = (
                torch.tensor(prompt_tokens, dtype=torch.long, device=idx.device)
                if prompt_tokens
                else torch.tensor([], dtype=torch.long, device=idx.device)
            )
            prompt_att = (
                torch.ones((len(prompt_tokens),), dtype=torch.long, device=idx.device)
                if prompt_tokens
                else torch.tensor([], dtype=torch.long, device=idx.device)
            )
            if prompt_ids.numel() == 0:
                prompt_seg_ids = torch.full((prompt_budget,), self.pad_token_id, dtype=torch.long, device=idx.device)
                prompt_seg_att = torch.zeros((prompt_budget,), dtype=torch.long, device=idx.device)
            else:
                prompt_seg_ids = pad_sequence_to_length(
                    prompt_ids.unsqueeze(0), prompt_budget, self.pad_token_id, left_pad=True
                ).squeeze(0)
                prompt_seg_att = pad_sequence_to_length(prompt_att.unsqueeze(0), prompt_budget, 0, left_pad=True).squeeze(0)

            # 8.2 Response segment (right-pad); ensure ids and loss_masks stay aligned.
            response_tokens = list(state.get("response_tokens") or [])
            response_loss_masks = list(state.get("loss_masks") or [])
            if len(response_tokens) != len(response_loss_masks):
                min_len = min(len(response_tokens), len(response_loss_masks))
                logger.error(
                    f"[MASK MISMATCH] Sample {b}: response_tokens={len(response_tokens)} != loss_masks={len(response_loss_masks)}; "
                    f"clamping to {min_len}"
                )
                response_tokens = response_tokens[:min_len]
                response_loss_masks = response_loss_masks[:min_len]

            # The returned response tensor is fixed-length (response_budget). If we exceed it (e.g. due to hints),
            # keep the tail to preserve the final answer.
            if len(response_tokens) > response_budget:
                logger.warning(
                    f"[RESPONSE TRUNCATION] Sample {b}: response_len={len(response_tokens)} > response_budget={response_budget}; keeping tail"
                )
                response_tokens = response_tokens[-response_budget:]
                response_loss_masks = response_loss_masks[-response_budget:]

            resp_ids = (
                torch.tensor(response_tokens, dtype=torch.long, device=idx.device)
                if response_tokens
                else torch.tensor([], dtype=torch.long, device=idx.device)
            )
            resp_att = (
                torch.ones((len(response_tokens),), dtype=torch.long, device=idx.device)
                if response_tokens
                else torch.tensor([], dtype=torch.long, device=idx.device)
            )
            resp_loss = (
                torch.tensor(response_loss_masks, dtype=torch.long, device=idx.device)
                if response_loss_masks
                else torch.tensor([], dtype=torch.long, device=idx.device)
            )

            if resp_ids.numel() == 0:
                resp_seg_ids = torch.full((response_budget,), self.pad_token_id, dtype=torch.long, device=idx.device)
                resp_seg_att = torch.zeros((response_budget,), dtype=torch.long, device=idx.device)
                resp_seg_loss = torch.zeros((response_budget,), dtype=torch.long, device=idx.device)
            else:
                resp_seg_ids = pad_sequence_to_length(resp_ids.unsqueeze(0), response_budget, self.pad_token_id).squeeze(0)
                resp_seg_att = pad_sequence_to_length(resp_att.unsqueeze(0), response_budget, 0).squeeze(0)
                resp_seg_loss = pad_sequence_to_length(resp_loss.unsqueeze(0), response_budget, 0).squeeze(0)

            # 8.3 Assemble full sequence (fixed boundary at prompt_budget)
            full_ids = torch.cat([prompt_seg_ids, resp_seg_ids], dim=0)
            att_mask = torch.cat([prompt_seg_att, resp_seg_att], dim=0)
            loss_mask = torch.cat(
                [torch.zeros((prompt_budget,), dtype=torch.long, device=idx.device), resp_seg_loss],
                dim=0,
            )

            # 8.4 Position IDs (consistent with chat_scheduler: (mask.cumsum-1)*mask)
            pos_ids = (att_mask.cumsum(dim=0) - 1) * att_mask
            pos_ids = pos_ids.to(torch.long)

            # Track last valid position in the response segment (hints included but capped by response_budget).
            exp_last_valid_pos_list.append(int(min(len(response_tokens), response_budget)))

            exp_input_ids_list.append(full_ids)
            exp_attention_mask_list.append(att_mask)
            exp_loss_mask_list.append(loss_mask)
            exp_position_ids_list.append(pos_ids)
            exp_responses_list.append(resp_seg_ids)
            
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
        control_last_valid_pos = torch.full((batch_size,), control_response_length, dtype=torch.long, device=idx.device)
        exp_last_valid_pos = torch.tensor(exp_last_valid_pos_list, dtype=torch.long, device=idx.device)

        # 11. 构建 TensorDict batch
        from tensordict import TensorDict
        batch = TensorDict({
            'control_input_ids': control_seq,
            'control_attention_mask': torch.cat([attention_mask, torch.ones_like(control_responses)], dim=-1),
            'control_position_ids': torch.cat([position_ids, control_delta_position_id], dim=-1),
            'control_log_probs': control_log_probs,
            'control_last_valid_pos': control_last_valid_pos,
            'control_responses': control_responses,  # Add responses separately for trainer
            'exp_input_ids': exp_input_ids,
            'exp_attention_mask': exp_attention_mask,
            'exp_position_ids': exp_position_ids,
            'exp_loss_mask': exp_loss_mask,  # 关键新增
            'exp_responses': exp_responses,
            'exp_last_valid_pos': exp_last_valid_pos,
        }, batch_size=batch_size)

        # 12. 构建 non_tensor_batch
        # IMPORTANT: Preserve the original non_tensor_batch fields (reward_model, data_source, etc.)
        # These are required for reward computation in NaiveRewardManager
        non_tensor_batch = prompts.non_tensor_batch.copy()

        # If batch was repeated (n > 1), we need to repeat fields in non_tensor_batch as well
        # to match the new batch_size
        original_batch_size = len(prompts.batch["input_ids"])
        if batch_size > original_batch_size:
            repeat_factor = batch_size // original_batch_size
            # Repeat all fields in non_tensor_batch to match the expanded batch size
            for key, val in non_tensor_batch.items():
                if isinstance(val, np.ndarray) and val.shape[0] == original_batch_size:
                    non_tensor_batch[key] = _repeat_interleave(val, repeat_factor)

        non_tensor_batch.update({
            'control_prompts': np.array([tokenizer.decode(idx[i].tolist(), skip_special_tokens=False) for i in range(batch_size)], dtype=object),
            'control_responses': np.array([tokenizer.decode(control_responses[i].tolist(), skip_special_tokens=False) for i in range(batch_size)], dtype=object),
            'exp_prompts': np.array([state['prompt_text'] for state in sample_states], dtype=object),
            'exp_responses': np.array([tokenizer.decode(exp_responses[i].tolist(), skip_special_tokens=False) for i in range(batch_size)], dtype=object),
            'hints': np.array(all_hints, dtype=object),
            'critiques': np.array(all_critiques, dtype=object),
            'num_interventions': np.array([len(state['hints']) for state in sample_states], dtype=np.int32),
            'hint_token_counts': np.array([
                sum(len(tokenizer.encode(h, add_special_tokens=False)) for h in state['hints'])
                if state['hints'] else 0
                for state in sample_states
            ], dtype=np.int32),
            # Extra diagnostics for dump/offline analysis
            'prompt_len': np.array([len(state['prompt_tokens']) for state in sample_states], dtype=np.int32),
            'response_len': np.array([len(state['response_tokens']) for state in sample_states], dtype=np.int32),
            'gen_len': np.array([sum(1 for m in state['loss_masks'] if m == 1) for state in sample_states], dtype=np.int32),
            'hint_len': np.array([sum(1 for m in state['loss_masks'] if m == 0) for state in sample_states], dtype=np.int32),
            'last_finish_reason': np.array([state.get('last_finish_reason') for state in sample_states], dtype=object),
            'context_exhausted': np.array([state.get('context_exhausted', False) for state in sample_states], dtype=np.bool_),
            'first_step_tokens_len': np.array([state.get('first_step_tokens_len') for state in sample_states], dtype=object),
        })

        # Ensure all vector-like fields in non_tensor_batch align with batch_size
        for key, val in list(non_tensor_batch.items()):
            if isinstance(val, (str, bytes, dict)) or val is None:
                continue
            try:
                length = len(val)
            except Exception:
                continue
            if length == 0:
                continue
            if length != batch_size:
                repeat_factor = math.ceil(batch_size / length)
                repeated = (list(val) * repeat_factor)[:batch_size]
                non_tensor_batch[key] = np.array(repeated, dtype=object)
            else:
                # normalize to numpy array for consistency
                non_tensor_batch[key] = np.array(val, dtype=object)

        # 13. 释放 KV Cache
        if hasattr(self.inference_engine, 'sleep_mode'):
            try:
                self.inference_engine.sleep(level=1)
            except AttributeError:
                try:
                    self.inference_engine.free_cache_engine()
                except:
                    pass

        # 14. 添加 timing 信息到 meta_info
        timing_info["total"] = time.time() - start_time
        meta_info = prompts.meta_info.copy()
        meta_info["timing"] = timing_info

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    def _run_verifier_inference(
        self, 
        verifier_inputs: List[str], 
        tokenizer, 
        idx_device: torch.device, 
        exp_kwargs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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

            # CRITICAL FIX: Use chat template to format inputs
            # Verifier was trained with chat format, must use apply_chat_template
            # CRITICAL: Use separate system and user messages to match training data format
            verifier_chat_inputs = []
            for verifier_prompt in verifier_inputs:
                messages = [
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": verifier_prompt}
                ]
                chat_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                verifier_chat_inputs.append(chat_text)

            # 批量 tokenize
            verifier_encoded = tokenizer(
                verifier_chat_inputs,  # Use chat-formatted inputs
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
                                lora_path=getattr(self, 'verifier_lora_path', None)  # Use actual Verifier LoRA path
                    )
                ] * len(verifier_inputs)

            # CRITICAL: Verifier must use temperature=0 for deterministic tag output
            verifier_kwargs = exp_kwargs.copy()
            verifier_kwargs["temperature"] = 0
            verifier_kwargs["top_p"] = 1.0
            verifier_kwargs["top_k"] = -1

            # 批量生成 Verifier 响应
            with self.update_sampling_params(**verifier_kwargs):
                verifier_output = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=verifier_idx_list,
                    lora_request=verifier_lora_requests,
                    use_tqdm=False,
                )
            
            verifier_responses, verifier_log_probs = _extract_vllm_outputs(
                verifier_output, self.pad_token_id, self.config.response_length, idx_device
            )
            
            # 解析所有 Verifier 输出
            decisions = []
            for i in range(len(verifier_inputs)):
                verifier_response = verifier_responses[i]
                verifier_text = tokenizer.decode(verifier_response.tolist(), skip_special_tokens=True)
                decision = self._parse_verifier_decision(verifier_text)
                decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Verifier inference failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            # 返回默认决策（Pass）
            return [{'action': 'Pass', 'hint': None, 'critique': f'[Error: {str(e)}]'} for _ in verifier_inputs]

    @GPUMemoryLogger(role="vllm dual stream rollout spmd", logger=logger)
    @torch.no_grad()
    def dual_stream_rollout(
        self, 
        prompts: DataProto, 
        intervention_mode: str = "by_response",
        token_check_interval: int = 2048,
        entropy_threshold: float = 0.5,
        use_entropy_filter: bool = True,
        **kwargs
    ) -> DataProto:
        """
        Perform dual-stream rollout for Co-GRPO (SPMD version adapted for vLLM 0.10+):
        - Control Stream: Policy generates without Verifier guidance
        - Experimental Stream: Policy generates with Verifier intervention based on mode
        
        Args:
            prompts: DataProto containing prompts
            intervention_mode: Verifier intervention mode. Options:
                - "by_response": Intervene after complete response (default)
                - "by_step": Intervene at step boundaries (line breaks, think tokens, etc.)
            token_check_interval: For by_token mode, check every N tokens (default: 5)
            entropy_threshold: For by_token mode, entropy threshold for filtering (default: 0.5)
            use_entropy_filter: For by_token mode, whether to use entropy filtering (default: True)
            
        Returns:
            DataProto with control_responses, exp_responses, hints, critiques, and associated metadata
        """
        timing_info = {}
        start_time = time.time()
        
        tokenizer = self.tokenizer  # Use stored tokenizer
        
        # Wake up vllm model (new API)
        if self.config.free_cache_engine:
            try:
                self.inference_engine.wake_up()
            except AttributeError:
                # Fallback for older API
                logger.warning("wake_up() not available, trying init_cache_engine()")
                try:
                    self.inference_engine.init_cache_engine()
                except:
                    pass

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)
        
        # Handle empty batch
        if batch_size == 0:
            empty_batch = TensorDict({}, batch_size=0)
            timing_info["total"] = time.time() - start_time
            meta_info = prompts.meta_info.copy()
            meta_info["timing"] = timing_info
            return DataProto(batch=empty_batch, non_tensor_batch={}, meta_info=meta_info)
        
        # Route to appropriate intervention mode
        # TEMP: Disable by_response and by_token to focus on debugging by_step
        # if intervention_mode == "by_response":
        #     return self._dual_stream_rollout_by_response(
        #         prompts, timing_info, start_time, tokenizer, idx, attention_mask,
        #         position_ids, eos_token_id, batch_size, **kwargs
        #     )
        # elif intervention_mode == "by_token":
        #     return self._dual_stream_rollout_by_token(
        #         prompts, timing_info, start_time, tokenizer, idx, attention_mask,
        #         position_ids, eos_token_id, batch_size, token_check_interval,
        #         entropy_threshold, use_entropy_filter, **kwargs
        #     )
        # elif intervention_mode == "by_step":
        if intervention_mode == "by_step":
            # CRITICAL FIX: Pass dual_stream_rollout's explicit parameters to _dual_stream_rollout_by_step via kwargs
            # Otherwise token_check_interval won't be passed because it's an explicit parameter of dual_stream_rollout,
            # not included in **kwargs automatically
            kwargs['token_check_interval'] = token_check_interval
            kwargs['entropy_threshold'] = entropy_threshold
            kwargs['use_entropy_filter'] = use_entropy_filter
            return self._dual_stream_rollout_by_step(
                prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                position_ids, eos_token_id, batch_size, **kwargs
            )
        else:
            # Fallback: Force by_step mode for debugging
            logger.warning(f"intervention_mode={intervention_mode} not supported yet, forcing by_step mode")
            kwargs['token_check_interval'] = token_check_interval
            kwargs['entropy_threshold'] = entropy_threshold
            kwargs['use_entropy_filter'] = use_entropy_filter
            return self._dual_stream_rollout_by_step(
                prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                position_ids, eos_token_id, batch_size, **kwargs
            )
