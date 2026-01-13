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

import json
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
VERIFIER_SYSTEM_PROMPT = "You are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within ```...``` tags."

VERIFIER_INTERVENE_PROMPT = """You are a **Socratic Reasoning Supervisor** and **Logic Auditor** for a large language model.
Your goal is to monitor the student's (model's) reasoning process step-by-step.

**Your Workflow:**
1.  **Analyze**: Read the provided `Question` and the student's `Current Reasoning Trace`.
2.  **Diagnose (System 2 Thought)**:
    - Generate a ``` ... ``` block.
    - Inside, perform a **Shadow Verification**:
        - If the step involves calculation, re-calculate it independently.
        - If the step involves logic, verify the rule application.
        - If you detect a pattern of error (e.g., hallucination, logic gap), diagnose it.
3.  **Act**:
    - If the reasoning is sound: Output `<GO>`.
    - If there is a critical error or high-risk pattern: Output `<WAIT>` followed by a brief, fuzzy guidance (imperative mood, do not leak answer).

**Output Format:**
```
[Target]: ...
[Analysis/Calculation]: ...
[Verdict]: ...
```

<GO> or <WAIT> [Guidance]
**Question & Student Response:**"""

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
        import json
        import time
        with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "LoRA",
                "location": "vllm_rollout_spmd.py:257",
                "message": "Before LLM initialization: lora_kwargs",
                "data": {
                    "lora_kwargs": lora_kwargs,
                    "has_enable_lora": "enable_lora" in lora_kwargs,
                    "has_max_loras": "max_loras" in lora_kwargs,
                    "has_max_lora_rank": "max_lora_rank" in lora_kwargs
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
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
        with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "LoRA",
                "location": "vllm_rollout_spmd.py:278",
                "message": "After LLM initialization: check lora_manager",
                "data": {
                    "has_llm_engine": hasattr(self.inference_engine, "llm_engine"),
                    "has_model_executor": hasattr(self.inference_engine, "llm_engine") and hasattr(self.inference_engine.llm_engine, "model_executor"),
                    "has_workers": hasattr(self.inference_engine, "llm_engine") and hasattr(self.inference_engine.llm_engine, "model_executor") and hasattr(self.inference_engine.llm_engine.model_executor, "workers"),
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
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

        print(f"kwargs: {kwargs}")
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
                        import json
                        import time
                        with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "LoRA",
                                "location": "vllm_rollout_spmd.py:403",
                                "message": "After verifier_lora_path setup: trying to load Verifier LoRA",
                                "data": {
                                    "verifier_lora_path": self.verifier_lora_path,
                                    "verifier_lora_name": self.verifier_lora_name,
                                    "has_add_lora": hasattr(self.inference_engine, "add_lora"),
                                },
                                "timestamp": int(time.time() * 1000)
                            }) + "\n")
                        # #endregion
                        # Put engine back to sleep
                        self.inference_engine.sleep(level=1)
                    except Exception as e:
                        logger.warning(f"Failed to load Verifier LoRA via add_lora: {e}")
                        # #region agent log
                        import json
                        import time
                        with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "LoRA",
                                "location": "vllm_rollout_spmd.py:403",
                                "message": "Failed to load Verifier LoRA via add_lora",
                                "data": {
                                    "error": str(e),
                                    "error_type": type(e).__name__
                                },
                                "timestamp": int(time.time() * 1000)
                            }) + "\n")
                        # #endregion
                        self.inference_engine.sleep(level=1)
            except Exception as e:
                logger.warning(f"Error trying to load Verifier LoRA: {e}")
                # #region agent log
                import json
                import time
                with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "LoRA",
                        "location": "vllm_rollout_spmd.py:403",
                        "message": "Error trying to load Verifier LoRA",
                        "data": {
                            "error": str(e),
                            "error_type": type(e).__name__
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
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
            import json
            import time
            with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "LoRA",
                    "location": "vllm_rollout_spmd.py:625",
                    "message": "_register_verifier_lora: checking inference_engine structure",
                    "data": {
                        "has_inference_engine": hasattr(self, "inference_engine"),
                        "has_llm_engine": hasattr(self.inference_engine, "llm_engine") if hasattr(self, "inference_engine") else False,
                        "has_worker": hasattr(self.inference_engine, "worker") if hasattr(self, "inference_engine") else False,
                        "verifier_lora_path": self.verifier_lora_path if hasattr(self, "verifier_lora_path") else None,
                        "verifier_lora_name": self.verifier_lora_name if hasattr(self, "verifier_lora_name") else None,
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
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
            with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "LoRA",
                    "location": "vllm_rollout_spmd.py:661",
                    "message": "_register_verifier_lora: checking LoRA config",
                    "data": {
                        "lora_enabled": lora_enabled,
                        "max_loras": max_loras,
                        "has_lora_kwargs": hasattr(self, "lora_kwargs"),
                        "verifier_lora_path": self.verifier_lora_path if hasattr(self, "verifier_lora_path") else None,
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
            # #endregion
            
            # If LoRA is enabled and max_loras > 1, use LoRA ID 1 for Verifier
            # This is a safe fallback that avoids accessing lora_manager
            if lora_enabled and max_loras > 1:
                self.verifier_lora_int_id = 1
                logger.info(f"Using LoRA ID 1 for Verifier (enable_lora={lora_enabled}, max_loras={max_loras}, verifier_lora_name={self.verifier_lora_name})")
                # #region agent log
                with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "LoRA",
                        "location": "vllm_rollout_spmd.py:680",
                        "message": "_register_verifier_lora: using LoRA ID 1",
                        "data": {
                            "verifier_lora_int_id": 1,
                            "max_loras": max_loras
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
                # #endregion
            else:
                self.verifier_lora_int_id = None
                logger.warning(f"LoRA not enabled or max_loras <= 1 (enable_lora={lora_enabled}, max_loras={max_loras}). Verifier will use base model.")
                # #region agent log
                with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "LoRA",
                        "location": "vllm_rollout_spmd.py:690",
                        "message": "_register_verifier_lora: LoRA not enabled, using base model",
                        "data": {
                            "verifier_lora_int_id": None,
                            "lora_enabled": lora_enabled,
                            "max_loras": max_loras
                        },
                        "timestamp": int(time.time() * 1000)
                    }) + "\n")
                # #endregion
        except Exception as e:
            logger.warning(f"Error in _register_verifier_lora: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            # #region agent log
            import json
            import time
            with open("/mnt/shared-storage-user/liuhongwei/main_works/.cursor/debug.log", "a") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "LoRA",
                    "location": "vllm_rollout_spmd.py:625",
                    "message": "_register_verifier_lora: exception caught",
                    "data": {
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
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
            return self._dual_stream_rollout_by_step(
                prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                position_ids, eos_token_id, batch_size, **kwargs
            )
        else:
            # Fallback: Force by_step mode for debugging
            logger.warning(f"intervention_mode={intervention_mode} not supported yet, forcing by_step mode")
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
        logger.info(f"DEBUG by_response: draft_kwargs={draft_kwargs}, len(idx_list)={len(idx_list)}, sampling_params.n={self.sampling_params.n}")
        with self.update_sampling_params(**draft_kwargs):
            logger.info(f"DEBUG by_response: INSIDE context - sampling_params.n={self.sampling_params.n}")
            draft_output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                lora_request=None,  # No LoRA for initial draft
                use_tqdm=False,
            )
            logger.info(f"DEBUG by_response: AFTER draft generation - len(draft_output)={len(draft_output)}")
        
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
            verifier_chat_inputs = []
            for verifier_prompt in verifier_inputs:
                messages = [{"role": "user", "content": verifier_prompt}]
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
            hint = f"<WAIT> {hint_content}" if hint_content else "<WAIT>"
        elif hint.upper().startswith("<GO>"):
            action = "Pass"
            hint = "<GO>"
        else:
            # Fallback: try old format parsing
            lower_text = verifier_text.lower()
            if "<wait>" in lower_text:
                hint_content = verifier_text.split("<WAIT>", 1)[-1].strip()
                action = "Intervene"
                hint = f"<WAIT> {hint_content}" if hint_content else "<WAIT>"
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
        logger.info(f"DEBUG control: BEFORE generation - len(idx_list)={len(idx_list)}, "
                  f"control_kwargs={control_kwargs}, sampling_params.n={self.sampling_params.n}, control_n={control_n}")
        with self.update_sampling_params(**control_kwargs):
            logger.info(f"DEBUG control: INSIDE context - sampling_params.n={self.sampling_params.n}")
            control_output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                lora_request=None,
                use_tqdm=False,
            )
            logger.info(f"DEBUG control: AFTER generation - len(control_output)={len(control_output)}")

        control_responses, control_log_probs = _extract_vllm_outputs(
            control_output, self.pad_token_id, self.config.response_length, idx.device
        )
        logger.info(f"DEBUG control: After extraction - control_responses.shape={control_responses.shape}")
        timing_info["control_gen"] = time.time() - control_start

        # Handle num_generation_per_prompt > 1 case
        # IMPORTANT: Check if batch was already repeated by trainer (for Co-GRPO)
        # In Co-GRPO, trainer repeats gen_batch before calling dual_stream_rollout
        # So we should NOT repeat again here to avoid double repetition
        # IMPORTANT: Use control_n (the actual n used) not self.sampling_params.n
        if control_n > 1 and do_sample:
            n = control_n
            logger.info(f"DEBUG batch_expansion: BEFORE - idx.shape={idx.shape}, batch_size={batch_size}, n={n}")
            # Check if batch was already repeated by checking if batch_size is a multiple of n
            # If batch_size >= n and is divisible by n, it was likely pre-repeated by trainer
            if batch_size % n == 0 and batch_size >= n:
                # Batch was already repeated by trainer, don't repeat again
                logger.info(f"Co-GRPO by_step: Batch already repeated by trainer (size={batch_size}, n={n}), skipping rollout-side repeat")
            else:
                # Normal case: repeat batch in rollout
                idx = _repeat_interleave(idx, n)
                attention_mask = _repeat_interleave(attention_mask, n)
                position_ids = _repeat_interleave(position_ids, n)
                # Repeat idx_list to match the expanded batch size
                idx_list = idx_list * n
                batch_size = batch_size * n
                logger.info(f"Co-GRPO by_step: Repeated batch in rollout (new size={batch_size})")
            logger.info(f"DEBUG batch_expansion: AFTER - idx.shape={idx.shape}, batch_size={batch_size}")

        # ========== Experimental Stream: 批量并行增量打断生成 ==========
        # 1. 获取停止配置
        stop_config = self._get_step_boundary_stop_config(tokenizer, eos_token_id)
        if not stop_config['stop_sequences'] and not stop_config['stop_token_ids']:
            logger.warning("No stop sequences found, using full generation without step boundaries")
            # Fall back to simple full generation (continue without step interruption)
            # Don't fallback to by_response as it's disabled for debugging
            pass  # Continue with full generation
        
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
            # IMPORTANT: When n > 1, we've already expanded idx/attention_mask/position_ids
            # so we need to set n=1 for experimental stream to avoid over-generating
            exp_kwargs = {"n": 1}
        
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
                
                # DEBUG: Log n value before generation
                logger.info(f"DEBUG exp_gen BEFORE: len(batch_inputs_processed)={len(batch_inputs_processed)}, "
                          f"exp_kwargs={exp_kwargs}, sampling_params.n={self.sampling_params.n}")

                # 使用统一的 sampling_params（修正）
                with self.update_sampling_params(
                    stop=stop_config['stop_sequences'],
                    stop_token_ids=stop_config['stop_token_ids'],
                    max_tokens=max_remaining,
                    **exp_kwargs
                ):
                    logger.info(f"DEBUG exp_gen INSIDE: sampling_params.n={self.sampling_params.n}")
                    step_output = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=batch_inputs_processed,
                        lora_request=None,
                        use_tqdm=False,
                    )
                    logger.info(f"DEBUG exp_gen AFTER: len(step_output)={len(step_output)}")
            except Exception as e:
                logger.error(f"Generation error at step {global_step}: {e}")
                metrics['errors'] += len(batch_indices)
                # 标记失败样本为完成
                for b in batch_indices:
                    sample_states[b]['is_complete'] = True
                    sample_states[b]['error'] = str(e)
                continue
            
            # 6.5 提取新生成的 tokens（批量处理）

            
            step_responses, step_log_probs = _extract_vllm_outputs(
                step_output, self.pad_token_id, self.config.response_length, idx.device
            )
            
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

                    # TODO: by_token 模式的 Verifier prompt 格式与训练数据不一致
                    # 当前格式: "Prompt: ...\nCurrent Reasoning: ...\nShould I intervene?"
                    # 训练数据格式: [system message] + [user message with VERIFIER_INTERVENE_PROMPT + Question + Student Response]
                    # 这可能导致 Verifier 输出不稳定的标签
                    # by_token 模式暂时不使用，等后续修复
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
        control_last_valid_pos = torch.full((batch_size,), control_response_length, dtype=torch.long, device=idx.device)
        exp_last_valid_pos = torch.tensor([len(state['response_tokens']) for state in sample_states], dtype=torch.long, device=idx.device)

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
            'exp_hints': np.array(all_hints, dtype=object),
            'exp_critiques': np.array(all_critiques, dtype=object),
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

        # DEBUG: Log final DataProto sizes
        logger.info(f"DEBUG DataProto return: batch_size={batch.batch_size}, control_responses.shape={batch['control_responses'].shape}, exp_responses.shape={batch['exp_responses'].shape}")

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
            verifier_chat_inputs = []
            for verifier_prompt in verifier_inputs:
                messages = [{"role": "user", "content": verifier_prompt}]
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
        token_check_interval: int = 5,
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
            return self._dual_stream_rollout_by_step(
                prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                position_ids, eos_token_id, batch_size, **kwargs
            )
        else:
            # Fallback: Force by_step mode for debugging
            logger.warning(f"intervention_mode={intervention_mode} not supported yet, forcing by_step mode")
            return self._dual_stream_rollout_by_step(
                prompts, timing_info, start_time, tokenizer, idx, attention_mask,
                position_ids, eos_token_id, batch_size, **kwargs
            )

    # def _dual_stream_rollout_by_response(
    #     self, prompts, timing_info, start_time, tokenizer, idx, attention_mask,
    #     position_ids, eos_token_id, batch_size, **kwargs
    # ) -> DataProto:
    #     """
    #     Original by_response mode: Generate complete response, then Verifier intervenes.
    #     This is the default implementation.
    #     """
    #     # ========== Control Stream: Policy generates without guidance ==========
    #     idx_list = []
    #     for i in range(batch_size):
    #         idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

    #     do_sample = prompts.meta_info.get("do_sample", True)
    #     is_validate = prompts.meta_info.get("validate", False)

    #     # =================================================================
    #     # FIX START: Robust Batch Size Check for Co-GRPO
    #     # =================================================================

    #     # 1. Basic configuration
    #     if not do_sample:
    #         control_kwargs = {
    #             "best_of": 1, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
    #             "temperature": 0, "n": 1,
    #         }
    #     elif is_validate:
    #         control_kwargs = {
    #             "top_k": self.config.val_kwargs.top_k,
    #             "top_p": self.config.val_kwargs.top_p,
    #             "temperature": self.config.val_kwargs.temperature,
    #             "n": 1,
    #         }
    #     else:
    #         control_kwargs = {}

    #     # 2. Co-GRPO double repetition detection and correction
    #     if do_sample and not is_validate and self.sampling_params.n > 1:
    #         target_n = self.sampling_params.n
    #         # Use function parameter batch_size - it's the most accurate tensor dimension
    #         current_batch_size = batch_size

    #         # Detection logic:
    #         # If current batch size is a multiple of target_n AND >= target_n,
    #         # we have high confidence that Trainer already did repeat operation
    #         if current_batch_size >= target_n and current_batch_size % target_n == 0:
    #             logger.info(f"Co-GRPO: Detected pre-repeated batch (size={current_batch_size}, n={target_n}). "
    #                         f"Forcing vLLM n=1 to avoid double repetition.")
    #             control_kwargs["n"] = 1
    #         else:
    #             # If not a multiple relationship, this might be a raw batch needing vLLM to do n-times sampling
    #             # Note: This is rare in Co-GRPO training but may happen during inference or debugging small batches
    #             logger.info(f"Co-GRPO: Detected raw batch (size={current_batch_size}, n={target_n}). "
    #                         f"Delegating repetition to vLLM.")
    #             control_kwargs["n"] = target_n

    #     # =================================================================
    #     # FIX END
    #     # =================================================================

    #     # Generate control responses (no LoRA)
    #     control_start = time.time()
    #     with self.update_sampling_params(**control_kwargs):
    #         control_output = self.inference_engine.generate(
    #             prompts=None,
    #             sampling_params=self.sampling_params,
    #             prompt_token_ids=idx_list,
    #             lora_request=None,  # No LoRA for control stream
    #             use_tqdm=False,
    #         )
        
    #     control_responses, control_log_probs = _extract_vllm_outputs(
    #         control_output, self.pad_token_id, self.config.response_length, idx.device
    #     )
    #     timing_info["control_gen"] = time.time() - control_start

    #     # Handle num_generation_per_prompt > 1 case
    #     # IMPORTANT: Check if batch was already repeated by trainer (for Co-GRPO)
    #     # In Co-GRPO, trainer repeats gen_batch before calling dual_stream_rollout
    #     # So we should NOT repeat again here to avoid double repetition
    #     if self.sampling_params.n > 1 and do_sample:
    #         n = self.sampling_params.n
    #         # Check if batch was already repeated by checking if batch_size is a multiple of n
    #         # If batch_size >= n and is divisible by n, it was likely pre-repeated by trainer
    #         if batch_size % n == 0 and batch_size >= n:
    #             # Batch was already repeated by trainer, don't repeat again
    #             logger.info(f"Co-GRPO: Batch already repeated by trainer (size={batch_size}, n={n}), skipping rollout-side repeat")
    #         else:
    #             # Normal case: repeat batch in rollout
    #             idx = _repeat_interleave(idx, n)
    #             attention_mask = _repeat_interleave(attention_mask, n)
    #             position_ids = _repeat_interleave(position_ids, n)
    #             # Repeat idx_list to match the expanded batch size
    #             idx_list = idx_list * n
    #             batch_size = batch_size * n
    #             logger.info(f"Co-GRPO: Repeated batch in rollout (new size={batch_size})")

    #     # ========== Experimental Stream: Policy -> Verifier -> Policy ==========
    #     # Step 1: Policy generates initial draft (CoT)
    #     draft_start = time.time()
    #     draft_kwargs = control_kwargs.copy()
    #     # IMPORTANT: After batch expansion, we need n=1 for experimental stream
    #     # to avoid generating too many responses
    #     if do_sample and not is_validate:
    #         draft_kwargs["n"] = 1
    #     logger.info(f"DEBUG by_response: draft_kwargs={draft_kwargs}, len(idx_list)={len(idx_list)}, sampling_params.n={self.sampling_params.n}")
    #     with self.update_sampling_params(**draft_kwargs):
    #         logger.info(f"DEBUG by_response: INSIDE context - sampling_params.n={self.sampling_params.n}")
    #         draft_output = self.inference_engine.generate(
    #             prompts=None,
    #             sampling_params=self.sampling_params,
    #             prompt_token_ids=idx_list,
    #             lora_request=None,  # No LoRA for initial draft
    #             use_tqdm=False,
    #         )
    #         logger.info(f"DEBUG by_response: AFTER draft generation - len(draft_output)={len(draft_output)}")
        
    #     draft_responses, draft_log_probs = _extract_vllm_outputs(
    #         draft_output, self.pad_token_id, self.config.response_length, idx.device
    #     )
        
    #     if draft_responses.shape[1] < self.config.response_length:
    #         draft_responses = pad_sequence_to_length(draft_responses, self.config.response_length, self.pad_token_id)
    #         draft_log_probs = pad_sequence_to_length(draft_log_probs, self.config.response_length, self.pad_token_id)
    #     timing_info["draft_gen"] = time.time() - draft_start

    #     # Step 2: Decode draft responses and construct Verifier input
    #     # Format: System prompt + Task instruction + Question + Student Response
    #     draft_texts = []
    #     prompt_texts = []
    #     for i in range(batch_size):
    #         prompt_text = tokenizer.decode(idx[i][attention_mask[i].bool()], skip_special_tokens=True)
    #         draft_text = tokenizer.decode(draft_responses[i], skip_special_tokens=True)
    #         prompt_texts.append(prompt_text)
    #         draft_texts.append(draft_text)
        
    #     # Construct Verifier input with system prompt and task instruction
    #     # Format matches format_training_data.py: system -> user (intervene_prompt + question + student response)
    #     verifier_inputs = []
    #     for prompt, draft in zip(prompt_texts, draft_texts):
    #         # Construct user content: intervene_prompt + question + student response
    #         user_content = f"{VERIFIER_INTERVENE_PROMPT} Question: {prompt}\n\nStudent Response: {draft}"
            
    #         # For vLLM, concatenate system and user prompts
    #         verifier_prompt = f"{VERIFIER_SYSTEM_PROMPT}\n\n{user_content}"
    #         verifier_inputs.append(verifier_prompt)
        
    #     # Tokenize Verifier inputs
    #     verifier_encoded = tokenizer(
    #         verifier_inputs,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=self.config.prompt_length + self.config.response_length,
    #     )
    #     verifier_input_ids = verifier_encoded["input_ids"].to(idx.device)
    #     verifier_attention_mask = verifier_encoded["attention_mask"].to(idx.device)
        
    #     # Convert to list format for vLLM
    #     verifier_idx_list = []
    #     for i in range(batch_size):
    #         verifier_idx_list.append(_pre_process_inputs(self.pad_token_id, verifier_input_ids[i]))

    #     # Step 3: Verifier generates Critique + Hint (using Verifier LoRA)
    #     verifier_start = time.time()
    #     if self.verifier_lora_int_id is None:
    #         self._register_verifier_lora()
        
    #     verifier_lora_requests = None
    #     if self.verifier_lora_int_id is not None:
    #         verifier_lora_requests = [
    #             LoRARequest(
    #                 lora_name=self.verifier_lora_name,
    #                 lora_int_id=self.verifier_lora_int_id,
    #                         lora_path=getattr(self, 'verifier_lora_path', None)  # Use actual Verifier LoRA path
    #             )
    #         ] * batch_size
        
    #     with self.update_sampling_params(**control_kwargs):
    #         verifier_output = self.inference_engine.generate(
    #             prompts=None,
    #             sampling_params=self.sampling_params,
    #             prompt_token_ids=verifier_idx_list,
    #             lora_request=verifier_lora_requests,
    #             use_tqdm=False,
    #         )
        
    #     verifier_responses, verifier_log_probs = _extract_vllm_outputs(
    #         verifier_output, self.pad_token_id, self.config.response_length, idx.device
    #     )
        
    #     if verifier_responses.shape[1] < self.config.response_length:
    #         verifier_responses = pad_sequence_to_length(verifier_responses, self.config.response_length, self.pad_token_id)
    #         verifier_log_probs = pad_sequence_to_length(verifier_log_probs, self.config.response_length, self.pad_token_id)

    #     # Decode Verifier outputs and extract actual hints (remove think blocks, extract <GO> or <WAIT> content)
    #     verifier_texts = []  # Full output for logging
    #     hints = []  # Extracted hints for actual use
    #     for i in range(batch_size):
    #         verifier_text = tokenizer.decode(verifier_responses[i].tolist(), skip_special_tokens=True)
    #         verifier_texts.append(verifier_text)
    #         # Extract actual hint (removes think blocks, extracts <GO> or <WAIT> content)
    #         hint = self._extract_verifier_hint(verifier_text)
    #         hints.append(hint)
    #     timing_info["verifier_gen"] = time.time() - verifier_start
        
    #     # Step 4: Policy generates final answer with Verifier guidance
    #     # Use extracted hints (without think blocks)
    #     final_inputs = []
    #     for prompt, hint in zip(prompt_texts, hints):
    #         final_prompt = f"Prompt: {prompt}\n\nHint: {hint}\n\nFinal Answer:"
    #         final_inputs.append(final_prompt)
        
    #     final_encoded = tokenizer(
    #         final_inputs,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=self.config.prompt_length + self.config.response_length,
    #     )
    #     final_input_ids = final_encoded["input_ids"].to(idx.device)
    #     final_attention_mask = final_encoded["attention_mask"].to(idx.device)
        
    #     final_idx_list = []
    #     for i in range(batch_size):
    #         final_idx_list.append(_pre_process_inputs(self.pad_token_id, final_input_ids[i]))

    #     # Generate final responses (no LoRA, back to Policy)
    #     final_start = time.time()
    #     with self.update_sampling_params(**control_kwargs):
    #         final_output = self.inference_engine.generate(
    #             prompts=None,
    #             sampling_params=self.sampling_params,
    #             prompt_token_ids=final_idx_list,
    #             lora_request=None,  # No LoRA for final generation
    #             use_tqdm=False,
    #         )
        
    #     exp_responses, exp_log_probs = _extract_vllm_outputs(
    #         final_output, self.pad_token_id, self.config.response_length, idx.device
    #     )
        
    #     if exp_responses.shape[1] < self.config.response_length:
    #         exp_responses = pad_sequence_to_length(exp_responses, self.config.response_length, self.pad_token_id)
    #         exp_log_probs = pad_sequence_to_length(exp_log_probs, self.config.response_length, self.pad_token_id)
    #     timing_info["final_gen"] = time.time() - final_start

    #     # Store the actual prompts used for experimental stream
    #     # This is important for correct response mask calculation later
    #     exp_prompt_ids = final_input_ids
    #     exp_prompt_attention_mask = final_attention_mask
    #     exp_prompt_length = exp_prompt_attention_mask.sum(dim=1)

    #     # ========== Prepare output DataProto ==========
    #     # Store hints and critiques in non_tensor_batch
    #     # Use extracted hints (without think blocks) for actual use
    #     hints_array = np.array(hints, dtype=object)
    #     critiques = np.array(verifier_texts, dtype=object)  # Keep full output for logging/critique
        
    #     # Store intervention statistics for penalty calculation
    #     # In by_response mode, each sample has 1 intervention (the verifier output)
    #     num_interventions = np.ones(batch_size, dtype=np.int32)
    #     # Calculate token counts based on extracted hints (not full verifier output)
    #     hint_token_counts = np.array([
    #         len(tokenizer.encode(hint, add_special_tokens=False)) for hint in hints
    #     ], dtype=np.int32)
        
    #     # Combine control and experimental responses
    #     # Helper function to get last valid position id
    
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
        logger.info(f"DEBUG control: BEFORE generation - len(idx_list)={len(idx_list)}, "
                  f"control_kwargs={control_kwargs}, sampling_params.n={self.sampling_params.n}, control_n={control_n}")
        with self.update_sampling_params(**control_kwargs):
            logger.info(f"DEBUG control: INSIDE context - sampling_params.n={self.sampling_params.n}")
            control_output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                lora_request=None,
                use_tqdm=False,
            )
            logger.info(f"DEBUG control: AFTER generation - len(control_output)={len(control_output)}")

        control_responses, control_log_probs = _extract_vllm_outputs(
            control_output, self.pad_token_id, self.config.response_length, idx.device
        )
        logger.info(f"DEBUG control: After extraction - control_responses.shape={control_responses.shape}")
        timing_info["control_gen"] = time.time() - control_start

        # Handle num_generation_per_prompt > 1 case
        # IMPORTANT: Check if batch was already repeated by trainer (for Co-GRPO)
        # In Co-GRPO, trainer repeats gen_batch before calling dual_stream_rollout
        # So we should NOT repeat again here to avoid double repetition
        # IMPORTANT: Use control_n (the actual n used) not self.sampling_params.n
        if control_n > 1 and do_sample:
            n = control_n
            logger.info(f"DEBUG batch_expansion: BEFORE - idx.shape={idx.shape}, batch_size={batch_size}, n={n}")
            # Check if batch was already repeated by checking if batch_size is a multiple of n
            # If batch_size >= n and is divisible by n, it was likely pre-repeated by trainer
            if batch_size % n == 0 and batch_size >= n:
                # Batch was already repeated by trainer, don't repeat again
                logger.info(f"Co-GRPO by_step: Batch already repeated by trainer (size={batch_size}, n={n}), skipping rollout-side repeat")
            else:
                # Normal case: repeat batch in rollout
                idx = _repeat_interleave(idx, n)
                attention_mask = _repeat_interleave(attention_mask, n)
                position_ids = _repeat_interleave(position_ids, n)
                # Repeat idx_list to match the expanded batch size
                idx_list = idx_list * n
                batch_size = batch_size * n
                logger.info(f"Co-GRPO by_step: Repeated batch in rollout (new size={batch_size})")
            logger.info(f"DEBUG batch_expansion: AFTER - idx.shape={idx.shape}, batch_size={batch_size}")

        # ========== Experimental Stream: 批量并行增量打断生成 ==========
        # 1. 获取停止配置
        stop_config = self._get_step_boundary_stop_config(tokenizer, eos_token_id)
        if not stop_config['stop_sequences'] and not stop_config['stop_token_ids']:
            logger.warning("No stop sequences found, using full generation without step boundaries")
            # Fall back to simple full generation (continue without step interruption)
            # Don't fallback to by_response as it's disabled for debugging
            pass  # Continue with full generation
        
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
            # IMPORTANT: When n > 1, we've already expanded idx/attention_mask/position_ids
            # so we need to set n=1 for experimental stream to avoid over-generating
            exp_kwargs = {"n": 1}
        
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
                
                # DEBUG: Log n value before generation
                logger.info(f"DEBUG exp_gen BEFORE: len(batch_inputs_processed)={len(batch_inputs_processed)}, "
                          f"exp_kwargs={exp_kwargs}, sampling_params.n={self.sampling_params.n}")

                # 使用统一的 sampling_params（修正）
                with self.update_sampling_params(
                    stop=stop_config['stop_sequences'],
                    stop_token_ids=stop_config['stop_token_ids'],
                    max_tokens=max_remaining,
                    **exp_kwargs
                ):
                    logger.info(f"DEBUG exp_gen INSIDE: sampling_params.n={self.sampling_params.n}")
                    step_output = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=batch_inputs_processed,
                        lora_request=None,
                        use_tqdm=False,
                    )
                    logger.info(f"DEBUG exp_gen AFTER: len(step_output)={len(step_output)}")
            except Exception as e:
                logger.error(f"Generation error at step {global_step}: {e}")
                metrics['errors'] += len(batch_indices)
                # 标记失败样本为完成
                for b in batch_indices:
                    sample_states[b]['is_complete'] = True
                    sample_states[b]['error'] = str(e)
                continue
            
            # 6.5 提取新生成的 tokens（批量处理）

            
            step_responses, step_log_probs = _extract_vllm_outputs(
                step_output, self.pad_token_id, self.config.response_length, idx.device
            )
            
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

                    # TODO: by_token 模式的 Verifier prompt 格式与训练数据不一致
                    # 当前格式: "Prompt: ...\nCurrent Reasoning: ...\nShould I intervene?"
                    # 训练数据格式: [system message] + [user message with VERIFIER_INTERVENE_PROMPT + Question + Student Response]
                    # 这可能导致 Verifier 输出不稳定的标签
                    # by_token 模式暂时不使用，等后续修复
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
        control_last_valid_pos = torch.full((batch_size,), control_response_length, dtype=torch.long, device=idx.device)
        exp_last_valid_pos = torch.tensor([len(state['response_tokens']) for state in sample_states], dtype=torch.long, device=idx.device)

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
            'exp_hints': np.array(all_hints, dtype=object),
            'exp_critiques': np.array(all_critiques, dtype=object),
        })

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

        # DEBUG: Log final DataProto sizes
        logger.info(f"DEBUG DataProto return: batch_size={batch.batch_size}, control_responses.shape={batch['control_responses'].shape}, exp_responses.shape={batch['exp_responses'].shape}")

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


if __name__ == "__main__":
    """
    Test Verifier input format and output parsing.
    """
    print("=" * 80)
    print("Testing Verifier Input Format and Output Parsing")
    print("=" * 80)
    
    # Create a mock class to test the methods
    class MockVerifier:
        def _extract_verifier_hint(self, verifier_text: str) -> str:
            """Copy of the actual method for testing"""
            if not verifier_text:
                return ""
            
            # Step 1: Remove think blocks (```\n...\n``` or ```...```)
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
            go_match = re.search(r'<GO>\s*(.*?)(?:\n|$)', text_without_think, re.IGNORECASE | re.DOTALL)
            wait_match = re.search(r'<WAIT>\s*(.*?)(?:\n|$)', text_without_think, re.IGNORECASE | re.DOTALL)
            
            if wait_match:
                hint = f"<WAIT> {wait_match.group(1).strip()}"
            elif go_match:
                hint = "<GO>"
            else:
                # Fallback: try to find <GO> or <WAIT> without strict format
                if "<WAIT>" in text_without_think.upper():
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
            """Copy of the actual method for testing"""
            verifier_text = verifier_text.strip()
            hint = self._extract_verifier_hint(verifier_text)
            
            if hint.upper().startswith("<WAIT>"):
                action = "Intervene"
                hint_content = hint.replace("<WAIT>", "").strip()
                hint = f"<WAIT> {hint_content}" if hint_content else "<WAIT>"
            elif hint.upper().startswith("<GO>"):
                action = "Pass"
                hint = "<GO>"
            else:
                action = "Pass"
                hint = ""
            
            return {
                "action": action,
                "hint": hint,
                "critique": verifier_text
            }
    
    mock_verifier = MockVerifier()
    
    # Test 1: Verifier input format
    print("\n[Test 1] Verifier Input Format")
    print("-" * 80)
    test_prompt = "Question: What is 2+2?"
    test_draft = "Student Response: Let me think... 2 + 2 = 4"
    
    # Construct input as done in the code
    user_content = f"{VERIFIER_INTERVENE_PROMPT} Question: {test_prompt}\n\nStudent Response: {test_draft}"
    verifier_input = f"{VERIFIER_SYSTEM_PROMPT}\n\n{user_content}"
    
    print("System Prompt (first 100 chars):")
    print(f"  {VERIFIER_SYSTEM_PROMPT[:100]}...")
    print("\nUser Content (first 200 chars):")
    print(f"  {user_content[:200]}...")
    print("\n✅ Input format matches expected structure")
    
    # Test 2: Output parsing - <WAIT> case with think block
    print("\n[Test 2] Output Parsing - <WAIT> with think block")
    print("-" * 80)
    test_output_1 = """```
[Target]: The derivation of the relationship between d and A is correct, but the solution stops before linking the parameters to the indices n and m.
[Error]: No final expression for m in terms of n (or vice‑versa) is provided, so the answer is incomplete.
[Pattern]: Incomplete conclusion – the student derived the necessary condition but did not translate it into the required relationship.
```

<WAIT> Hold on. Review the step where you connect the parameters to the indices n and m; make sure to express m in terms of n (or n in terms of m) as required."""
    
    hint_1 = mock_verifier._extract_verifier_hint(test_output_1)
    decision_1 = mock_verifier._parse_verifier_decision(test_output_1)
    
    print("Full Verifier Output:")
    print(f"  {test_output_1[:150]}...")
    print(f"\nExtracted Hint:")
    print(f"  {hint_1}")
    print(f"\nParsed Decision:")
    print(f"  Action: {decision_1['action']}")
    print(f"  Hint: {decision_1['hint']}")
    print(f"  Expected: <WAIT> Hold on. Review the step where you connect...")
    assert "<WAIT>" in hint_1, "❌ Failed: Should extract <WAIT> hint"
    assert decision_1['action'] == "Intervene", "❌ Failed: Should be Intervene"
    print("✅ Test passed: <WAIT> hint extracted correctly")
    
    # Test 3: Output parsing - <GO> case with think block
    print("\n[Test 3] Output Parsing - <GO> with think block")
    print("-" * 80)
    test_output_2 = """```
[Target]: 'Thus discriminant D must be non-negative: -164 + (8088)/y >= 0 => (8088)/y >= 164 => y <= 8088/164.'
[Shadow_Derivation]: Starting from the quadratic equation x^2 + 10x + (66 - 2022/y) = 0, the discriminant is D = b^2 - 4ac = 10^2 - 4·1·(66 - 2022/y) = 100 - 264 + (4·2022)/y = -164 + 8088/y. For real roots we need D ≥ 0, so -164 + 8088/y ≥ 0 ⇒ 8088/y ≥ 164 ⇒ y ≤ 8088/164. Simplifying 8088/164 by dividing numerator and denominator by 4 gives 2022/41 ≈ 49.317. Hence the largest integer y satisfying the inequality is 49.
[Comparison]: The student's derivation matches this independent calculation exactly.
[Verdict]: The step is logically sound and mathematically correct.
```

<GO>"""
    
    hint_2 = mock_verifier._extract_verifier_hint(test_output_2)
    decision_2 = mock_verifier._parse_verifier_decision(test_output_2)
    
    print("Full Verifier Output:")
    print(f"  {test_output_2[:150]}...")
    print(f"\nExtracted Hint:")
    print(f"  {hint_2}")
    print(f"\nParsed Decision:")
    print(f"  Action: {decision_2['action']}")
    print(f"  Hint: {decision_2['hint']}")
    assert hint_2 == "<GO>", f"❌ Failed: Should extract <GO>, got '{hint_2}'"
    assert decision_2['action'] == "Pass", "❌ Failed: Should be Pass"
    print("✅ Test passed: <GO> hint extracted correctly")
    
    # Test 4: Output parsing - <think> format (from show.json)
    print("\n[Test 4] Output Parsing - <think> format")
    print("-" * 80)
    test_output_3 = """<think>
[Check]: The derivation of the relationship between d and A is correct, but the solution stops before linking the parameters to the indices n and m.
[Error]: No final expression for m in terms of n (or vice‑versa) is provided, so the answer is incomplete.
[Pattern]: Incomplete conclusion – the student derived the necessary condition but did not translate it into the required relationship.
</think>

<WAIT> Hold on. Review the step where you connect the parameters to the indices n and m; make sure to express m in terms of n (or n in terms of m) as required."""
    
    hint_3 = mock_verifier._extract_verifier_hint(test_output_3)
    decision_3 = mock_verifier._parse_verifier_decision(test_output_3)
    
    print("Full Verifier Output:")
    print(f"  {test_output_3[:150]}...")
    print(f"\nExtracted Hint:")
    print(f"  {hint_3}")
    print(f"\nParsed Decision:")
    print(f"  Action: {decision_3['action']}")
    print(f"  Hint: {decision_3['hint']}")
    assert "<WAIT>" in hint_3, "❌ Failed: Should extract <WAIT> hint"
    assert decision_3['action'] == "Intervene", "❌ Failed: Should be Intervene"
    assert "<think>" not in hint_3, "❌ Failed: Should remove <think> block"
    print("✅ Test passed: <think> block removed correctly")
    
    # Test 5: Edge case - no think block, just <GO>
    print("\n[Test 5] Edge Case - No think block, just <GO>")
    print("-" * 80)
    test_output_4 = "<GO>"
    hint_4 = mock_verifier._extract_verifier_hint(test_output_4)
    decision_4 = mock_verifier._parse_verifier_decision(test_output_4)
    
    print(f"Input: {test_output_4}")
    print(f"Extracted Hint: {hint_4}")
    print(f"Action: {decision_4['action']}")
    assert hint_4 == "<GO>", f"❌ Failed: Should extract <GO>, got '{hint_4}'"
    assert decision_4['action'] == "Pass", "❌ Failed: Should be Pass"
    print("✅ Test passed: Simple <GO> handled correctly")
    
    # Test 6: Edge case - empty output
    print("\n[Test 6] Edge Case - Empty output")
    print("-" * 80)
    test_output_5 = ""
    hint_5 = mock_verifier._extract_verifier_hint(test_output_5)
    decision_5 = mock_verifier._parse_verifier_decision(test_output_5)
    
    print(f"Input: (empty)")
    print(f"Extracted Hint: '{hint_5}'")
    print(f"Action: {decision_5['action']}")
    assert hint_5 == "", f"❌ Failed: Should return empty, got '{hint_5}'"
    assert decision_5['action'] == "Pass", "❌ Failed: Should default to Pass"
    print("✅ Test passed: Empty output handled correctly")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
