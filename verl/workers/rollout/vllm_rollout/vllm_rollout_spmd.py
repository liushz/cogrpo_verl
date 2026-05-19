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
import math
import os
import re
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

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
from verl.utils.torch_functional import (
    get_response_mask,
    pad_2d_list_to_length,
    pad_sequence_to_length,
)
from verl.workers.rollout.base import BaseRollout

# Shared (online-canonical) Verifier hint injection helpers.
from .verifier_hint_injection import (
    VERIFIER_INTERVENE_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    build_verifier_user_prompt,
    compute_tail_rollback_keep_len as _compute_tail_rollback_keep_len,
    find_last_eos_index as _find_last_eos_index,
    find_last_marker_span as _find_last_marker_span,
    find_last_subsequence_index as _find_last_subsequence_index,
    find_last_think_close_span as _find_last_think_close_span,
    find_last_trim_boundary as _find_last_trim_boundary,
    find_prethink_rollback_keep_len as _find_prethink_rollback_keep_len,
    find_think_close_pos as _find_think_close_pos,
    hint_prefix_for_tail as _hint_prefix_for_tail,
    insert_hint_tokens as _insert_hint_tokens,
    is_post_think_finalized as _is_post_think_finalized,
    last_hint_end as _last_hint_end,
    rollback_prethink_once as _rollback_prethink_once,
    rollback_state_to_keep_len as _rollback_state_to_keep_len,
    rollback_tail_for_hint as _rollback_tail_for_hint,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _normalize_eos_token_ids(eos_token_id):
    """Return a set[int] of eos token ids (handles int/list/tuple)."""
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        out = set()
        for t in eos_token_id:
            try:
                out.add(int(t))
            except Exception:
                continue
        return out
    try:
        return {int(eos_token_id)}
    except Exception:
        return set()


def _find_first_eos_index(tokens, eos_token_ids: set) -> Optional[int]:
    if not tokens or not eos_token_ids:
        return None
    for i, t in enumerate(tokens):
        if t in eos_token_ids:
            return i
    return None


def _hash_token_sequence(tokens: List[int], tail_k: int = 512) -> int:
    """Deterministic rolling hash for token ids (used for bucketing rollout states)."""
    if not tokens:
        return 0
    try:
        tail_k = int(tail_k)
    except Exception:
        tail_k = 512
    if tail_k > 0 and len(tokens) > tail_k:
        tokens = tokens[-tail_k:]

    # 64-bit rolling hash (deterministic across processes).
    h = 1469598103934665603  # FNV offset basis
    for t in tokens:
        try:
            v = int(t)
        except Exception:
            v = 0
        h ^= v & 0xFFFFFFFFFFFFFFFF
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(h)


_CF_CONTROL_BATCH_DROP_NONTENSOR_KEYS = (
    # Long decoded strings (can be 32k tokens) - huge in Ray object store.
    "exp_prompts",
    "exp_responses",
    "control_prompts",
    "control_responses",
    "hints",
    "critiques",
    # Diagnostics only.
    "num_interventions",
    "hint_token_counts",
    "prompt_len",
    "response_len",
    "gen_len",
    "hint_len",
    "last_finish_reason",
    "context_exhausted",
    "first_step_tokens_len",
)


def _build_cf_reward_tail_batch(
    cf_control_batch: DataProto,
    pad_token_id: int,
    tail_tokens: int,
) -> DataProto:
    """Build a compact CF batch for reward evaluation.

    CF-branch returns an auxiliary rollout batch (`cf_control_batch`) whose only
    consumer is the trainer-side reward computation. Shipping the full 32k
    tensors (and decoded strings) through Ray can trigger severe object store
    spilling. Here we keep only:
      - `prompts` (prompt segment, left-padded, fixed length)
      - `responses` (last `tail_tokens` valid response tokens, right-padded)
      - `attention_mask` (prompt+response)

    Everything else is dropped to minimize Ray transfer.
    """
    if cf_control_batch is None or len(cf_control_batch) == 0:
        return cf_control_batch

    try:
        tail_tokens = int(tail_tokens)
    except Exception:
        return cf_control_batch
    if tail_tokens <= 0:
        return cf_control_batch

    if cf_control_batch.batch is None:
        return cf_control_batch

    batch_keys = set(cf_control_batch.batch.keys())
    required = {"exp_input_ids", "exp_attention_mask", "exp_responses"}
    if not required.issubset(batch_keys):
        return cf_control_batch

    exp_input_ids = cf_control_batch.batch["exp_input_ids"]
    exp_attention_mask = cf_control_batch.batch["exp_attention_mask"]
    exp_responses = cf_control_batch.batch["exp_responses"]

    if exp_responses.dim() != 2 or exp_input_ids.dim() != 2 or exp_attention_mask.dim() != 2:
        return cf_control_batch

    response_window_len = int(exp_responses.size(1))
    if response_window_len <= 0:
        return cf_control_batch

    tail_len = min(int(tail_tokens), response_window_len)
    total_len = int(exp_input_ids.size(1))
    prompt_len = total_len - response_window_len
    if prompt_len <= 0:
        return cf_control_batch

    # Prefer the rollout-provided response valid lengths (includes hints, excludes PAD).
    valid_response_len = None
    if "exp_last_valid_pos" in batch_keys:
        valid_response_len = cf_control_batch.batch["exp_last_valid_pos"]
    if valid_response_len is None:
        try:
            valid_response_len = exp_attention_mask[:, prompt_len:].sum(dim=1)
        except Exception:
            valid_response_len = None
    if valid_response_len is None:
        return cf_control_batch

    valid_response_len = valid_response_len.to(torch.long).clamp(
        min=0, max=response_window_len
    )
    start = (valid_response_len - tail_len).clamp(min=0)

    device = exp_responses.device
    arange = torch.arange(tail_len, device=device, dtype=torch.long)
    gather_idx = start.unsqueeze(1) + arange.unsqueeze(0)
    # exp_responses is right-padded, so indices beyond valid_response_len point to PAD.
    responses_tail = exp_responses.gather(1, gather_idx)

    tail_counts = torch.minimum(
        valid_response_len, torch.tensor(tail_len, device=device, dtype=torch.long)
    )
    tail_att = (arange.unsqueeze(0) < tail_counts.unsqueeze(1)).to(
        exp_attention_mask.dtype
    )

    prompts_ids = exp_input_ids[:, :prompt_len]
    prompts_att = exp_attention_mask[:, :prompt_len]
    attention_mask = torch.cat([prompts_att, tail_att], dim=1)

    # Move compact tensors to CPU to avoid CUDA IPC / cross-node GPU serialization overhead.
    prompts_ids = prompts_ids.to("cpu")
    responses_tail = responses_tail.to("cpu")
    attention_mask = attention_mask.to("cpu")

    # Extra safety: ensure padded positions are PAD tokens.
    if tail_len > 0:
        pad_mask = attention_mask[:, prompt_len:] == 0
        if pad_mask.any():
            responses_tail[pad_mask] = int(pad_token_id)

    compact_batch = TensorDict(
        {
            "prompts": prompts_ids,
            "responses": responses_tail,
            "attention_mask": attention_mask,
        },
        batch_size=prompts_ids.size(0),
    )

    non_tensor_batch = dict(cf_control_batch.non_tensor_batch or {})
    for key in _CF_CONTROL_BATCH_DROP_NONTENSOR_KEYS:
        non_tensor_batch.pop(key, None)

    meta_info = dict(cf_control_batch.meta_info or {})
    meta_info["cf_reward_tail_len"] = int(tail_len)
    meta_info["cf_reward_original_response_len"] = int(response_window_len)

    return DataProto(batch=compact_batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


def _extract_verifier_question_from_prompt(prompt_text: str) -> str:
    """Extract a cleaner user question from decoded chat prompt text for Verifier input."""
    if not prompt_text:
        return ""

    text = str(prompt_text).replace("\r\n", "\n")
    candidate = text.strip()

    def _strip_answer_format_fragments(raw: str) -> str:
        if not raw:
            return ""
        s = str(raw)
        patterns = [
            # EN directives with boxed format.
            r"(?i)(?:^|[\s,;:.，；：。])remember\s+to[^\n]*?\\boxed\{\}[^\n]*",
            r"(?i)(?:^|[\s,;:.，；：。])(?:please\s+)?(?:put|write|format|provide)[^\n]*?final\s+answer[^\n]*",
            r"(?i)###\s*final\s*answer[^\n]*",
            # CN directives with boxed format.
            r"(?:^|[\s,;:.，；：。])请将[^\n]*?(?:最终答案|答案)[^\n]*?\\boxed\{\}[^\n]*",
            r"(?:^|[\s,;:.，；：。])输出格式[^\n]*",
        ]
        for pat in patterns:
            s = re.sub(pat, " ", s)
        s = re.sub(r"\s+", " ", s).strip(" `\"'.,:;!?，。；：")
        return s

    # Prefer the last user turn when the prompt is chat-rendered like:
    #   system\n...\nuser\n...\nassistant\n
    role_matches = list(re.finditer(r"(?:^|\n)\s*user\s*\n", text, flags=re.IGNORECASE))
    if role_matches:
        start = role_matches[-1].end()
        tail = text[start:]
        end_match = re.search(r"\n\s*assistant\s*(?:\n|$)", tail, flags=re.IGNORECASE)
        if end_match:
            tail = tail[: end_match.start()]
        candidate = tail.strip()

    # Drop obvious answer-format directives from the question body to reduce mode confusion.
    # Keep this conservative: only strip lines that explicitly mention final-answer formatting.
    cleaned_lines = []
    removed_any = False
    for raw_line in candidate.split("\n"):
        line = raw_line.strip()
        if not line:
            cleaned_lines.append(raw_line)
            continue

        lower_line = line.lower()
        mentions_boxed = "\\boxed" in line
        stripped_line = _strip_answer_format_fragments(raw_line)
        if stripped_line != raw_line.strip():
            removed_any = True
        if not stripped_line:
            continue

        looks_like_answer_format = (
            "final answer" in lower_line
            or "remember to" in lower_line
            or "format" in lower_line
            or "output format" in lower_line
            or "### final answer" in lower_line
            or ("答案" in line and "最终" in line)
        )
        if mentions_boxed and looks_like_answer_format:
            removed_any = True
            continue
        if "### final answer" in lower_line and "boxed" in lower_line:
            removed_any = True
            continue

        cleaned_lines.append(stripped_line)

    cleaned = "\n".join(cleaned_lines).strip()
    if cleaned:
        return cleaned
    if removed_any:
        coarse = _strip_answer_format_fragments(candidate)
        if coarse:
            return coarse
    if candidate:
        return candidate
    return text.strip()


# Hard filters to prevent verifier "spam hints" / mode collapse.
BANNED_PHRASES = [
    "provide the final numeric answer",
    "finish the solution",
    "answer the question",
    "provide the final answer",
    "final numeric answer",
    "output the decision line",
    "plus one short guidance hint",
    "remember to format",
    "output format",
    "thus we output",
]

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


class InterventionPolicy:
    """
    控制 Verifier 干预策略，避免过度干预或资源浪费。
    """

    def __init__(self, max_interventions: int = 3, confidence_threshold: float = 0.0):
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
        import difflib

        def _normalize_hint(text: str) -> str:
            if not text:
                return ""
            s = text.strip().lower()
            s = re.sub(r"\s+", " ", s)
            return s

        def _strip_banned_phrases(text: str) -> str:
            if not text:
                return ""
            cleaned = str(text)
            for phrase in BANNED_PHRASES:
                cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" `\"'.,:;!?-\t\n\r")
            return cleaned

        # 1. 达到最大干预次数
        if len(state["hints"]) >= self.max_interventions:
            return False

        # 2. 决策不是 Intervene
        if decision.get("action") != "Intervene":
            return False

        # 3. 没有 hint
        if not decision.get("hint"):
            return False

        # 4. Optional WAIT-confidence gate (disabled when threshold <= 0).
        try:
            threshold = float(self.confidence_threshold)
        except Exception:
            threshold = 0.0
        if threshold > 0:
            wait_conf = decision.get("wait_confidence", None)
            if wait_conf is None:
                return False
            try:
                if float(wait_conf) < threshold:
                    return False
            except Exception:
                return False

        # === Hard filters: sanitize blacklist + deduplication ===
        raw_hint = str(decision.get("hint", ""))
        hint_norm = _normalize_hint(raw_hint)
        if not hint_norm:
            return False

        # Do not drop the whole intervention immediately when banned phrases appear.
        # First sanitize hint text; only drop when nothing meaningful remains.
        if any(phrase in hint_norm for phrase in BANNED_PHRASES):
            sanitized_hint = _strip_banned_phrases(raw_hint)
            sanitized_norm = _normalize_hint(sanitized_hint)
            if len(sanitized_norm) < 4:
                return False
            decision["hint"] = sanitized_hint
            hint_norm = sanitized_norm

        prev_hints = state.get("hints", []) or []
        for prev in prev_hints:
            prev_norm = _normalize_hint(str(prev))
            if not prev_norm:
                continue
            if hint_norm in prev_norm or prev_norm in hint_norm:
                return False
            if difflib.SequenceMatcher(a=hint_norm, b=prev_norm).ratio() >= 0.9:
                return False

        return True


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][
        0
    ]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(
    value: Union[torch.Tensor, np.ndarray], repeats: int
) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, (int, np.integer)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off", ""):
            return False
    return bool(val)


def _best_effort_reset_prefix_cache(engine: Any) -> bool:
    """Try to reset vLLM prefix cache across version/object layouts.

    Some vLLM versions require an explicit prefix-cache reset before calling
    sleep/offload/free-cache-engine, otherwise long-running by_step rollouts can
    accumulate KV blocks and lead to memory growth / instability.
    """
    if engine is None:
        return False

    candidates = [engine]
    for attr in ("llm_engine", "engine"):
        obj = getattr(engine, attr, None)
        if obj is not None:
            candidates.append(obj)

    for obj in candidates:
        fn = getattr(obj, "reset_prefix_cache", None)
        if not callable(fn):
            continue
        try:
            ret = fn()
            # Async API: nothing we can do safely in this synchronous code path.
            if hasattr(ret, "__await__"):
                logger.warning(
                    "[vLLM] reset_prefix_cache() appears async; skipping await in sync rollout."
                )
            return True
        except Exception as e:
            logger.warning(f"[vLLM] reset_prefix_cache() failed: {e}")
            return False

    return False


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

    response_tensor = pad_2d_list_to_length(
        response, pad_token_id, max_length=max_length
    ).to(device)
    log_probs_tensor = pad_2d_list_to_length(
        rollout_log_probs, -1, max_length=max_length
    ).to(device)
    log_probs_tensor = log_probs_tensor.to(torch.float32)

    return response_tensor, log_probs_tensor


def _compute_wait_confidence(
    response_tokens: torch.Tensor,
    log_probs: torch.Tensor,
    pad_token_id: int,
    tail_tokens: int = 64,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate confidence for a `<WAIT>` decision from verifier token logprobs.

    Returns:
        tuple: (confidence, avg_logprob)
    """
    try:
        if response_tokens is None or log_probs is None:
            return None, None
        if response_tokens.numel() == 0 or log_probs.numel() == 0:
            return None, None

        if pad_token_id is None:
            valid_mask = log_probs != -1
        else:
            valid_mask = response_tokens != int(pad_token_id)
            if not torch.any(valid_mask):
                valid_mask = log_probs != -1

        valid_log_probs = log_probs[valid_mask]
        if valid_log_probs.numel() == 0:
            return None, None

        finite_mask = torch.isfinite(valid_log_probs)
        if not torch.any(finite_mask):
            return None, None
        valid_log_probs = valid_log_probs[finite_mask]

        # Focus on the tail where the decision line lives; otherwise the long
        # `<think>` block dominates and makes the signal too smooth.
        try:
            tail_tokens = int(tail_tokens)
        except Exception:
            tail_tokens = 64
        if tail_tokens > 0 and valid_log_probs.numel() > tail_tokens:
            valid_log_probs = valid_log_probs[-tail_tokens:]

        avg_logprob = float(valid_log_probs.mean().item())
        avg_logprob = float(min(0.0, max(-20.0, avg_logprob)))
        confidence = float(math.exp(avg_logprob))
        return confidence, avg_logprob
    except Exception:
        return None, None


class vLLMRollout(BaseRollout):
    def __init__(
        self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.model_path = model_path
        self.config = config
        self.tokenizer = tokenizer  # Store tokenizer for dual_stream_rollout
        assert not (not config.enforce_eager and config.free_cache_engine), (
            "disable CUDA graph (enforce_eager = False) if free cache engine"
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
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
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size,
                    num_tp_per_train_tp=num_tp_per_train_tp,
                )
            else:
                vllm_ps.initialize_model_parallel(
                    tensor_model_parallel_size=tensor_parallel_size
                )

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = (
                    model_hf_config.llm_config.max_position_embeddings
                )
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = (
                    model_hf_config.text_config.max_position_embeddings
                )
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert (
                max_position_embeddings >= config.prompt_length + config.response_length
            ), "model context length should be greater than total sequence length"

        max_model_len = int(
            config.max_model_len or config.prompt_length + config.response_length
        )

        # Align tokenizer's declared max length with vLLM's max_model_len.
        #
        # Many Qwen/Qwen3 tokenizers ship with `model_max_length=32768` even when the
        # underlying model supports a larger context window (e.g. 40960). In by_step
        # rollouts we repeatedly re-encode a growing prefix; leaving model_max_length
        # smaller produces noisy warnings and can trigger unintended truncation in
        # downstream tokenization code.
        try:
            if hasattr(tokenizer, "model_max_length"):
                tokenizer.model_max_length = max_model_len
        except Exception:
            pass

        if (
            max_num_batched_tokens < max_model_len
            and self.config.enable_chunked_prefill
        ):
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = (
            "dummy" if config.load_format.startswith("dummy") else config.load_format
        )

        lora_kwargs = kwargs.pop("lora_kwargs", {}) or {}
        verifier_config = kwargs.pop("verifier_config", None)
        # Sanitize LoRA kwargs: vLLM expects integer limits (e.g., max_loras). Some configs may
        # accidentally pass `None`, which can later crash inside add_lora/remove_lora with
        # TypeErrors like `'>' not supported between instances of 'NoneType' and 'int'`.
        lora_kwargs = {k: v for k, v in dict(lora_kwargs).items() if v is not None}
        if lora_kwargs.get("enable_lora", False) and "max_loras" not in lora_kwargs:
            # Default to 1 so verifier LoRA hot-reload has a slot.
            lora_kwargs["max_loras"] = 1
        self.lora_kwargs = lora_kwargs

        # Verifier config may live outside rollout sub-config (Hydra top-level `verifier.*`).
        # Pass it explicitly from the trainer/worker so verifier inference uses the intended
        # max_new_tokens / temperature / prompt budgets.
        self.verifier_config = None
        if verifier_config is not None:
            try:
                if isinstance(verifier_config, DictConfig):
                    self.verifier_config = OmegaConf.to_container(
                        verifier_config, resolve=True
                    )
                else:
                    self.verifier_config = dict(verifier_config)
            except Exception:
                self.verifier_config = verifier_config
        # Extract verifier_lora_name and verifier_lora_path from kwargs if present
        verifier_lora_name = kwargs.pop("verifier_lora_name", None)
        verifier_lora_path = kwargs.pop("verifier_lora_path", None)

        # KV Cache optimization: whether to enable vLLM prefix caching.
        # NOTE: In by_step mode we call vLLM generate() repeatedly. Prefix caching can
        # improve throughput but may also cause prefix-cache reset failures / block leaks
        # in some vLLM versions. Make it configurable via config (preferred) or kwargs.
        config_enable_kv_cache_optimization = config.get(
            "enable_kv_cache_optimization", None
        )
        if config_enable_kv_cache_optimization is None:
            config_enable_kv_cache_optimization = kwargs.pop(
                "enable_kv_cache_optimization", True
            )
        self.enable_kv_cache_optimization = _as_bool(
            config_enable_kv_cache_optimization
        )
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {
            key: val for key, val in engine_kwargs.items() if val is not None
        }
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
        self.verifier_offload_enabled = bool(
            verifier_fsdp_config.get("param_offload")
        ) or bool(verifier_fsdp_config.get("optimizer_offload"))
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
                config.gpu_memory_utilization = min(
                    getattr(config, "gpu_memory_utilization", 0.8), 0.7
                )

            logger.info(
                f"[VerifierOffload] Enabled verifier offload hints: swap_space={engine_kwargs.get('swap_space')}, "
                f"gpu_memory_utilization={getattr(config, 'gpu_memory_utilization', None)}"
            )

        # Ensure stop_token_ids is a plain list to satisfy vLLM SamplingParams type checks
        if (
            "stop_token_ids" in kwargs
            and kwargs["stop_token_ids"] is not None
            and not isinstance(kwargs["stop_token_ids"], list)
        ):
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
            enable_prefix_caching=self.enable_kv_cache_optimization,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # vLLM maintains its own tokenizer instance; keep its model_max_length consistent
        # as well to avoid "Token indices sequence length ..." warnings when running
        # near the context limit.
        try:
            vllm_tokenizer = None
            if hasattr(self.inference_engine, "get_tokenizer"):
                vllm_tokenizer = self.inference_engine.get_tokenizer()
            if vllm_tokenizer is None and hasattr(self.inference_engine, "tokenizer"):
                vllm_tokenizer = getattr(self.inference_engine, "tokenizer", None)
            if vllm_tokenizer is None and hasattr(self.inference_engine, "llm_engine"):
                vllm_tokenizer = getattr(self.inference_engine.llm_engine, "tokenizer", None)
            if vllm_tokenizer is not None and hasattr(vllm_tokenizer, "model_max_length"):
                vllm_tokenizer.model_max_length = max_model_len
        except Exception:
            pass

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        # NOTE: detokenize must be True when using stop strings
        has_stop_params = (
            config.get("stop", None) is not None
            or config.get("stop_token_ids", None) is not None
        )
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = (
                has_stop_params  # True if stop strings used, False otherwise
            )

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        # Hydra may hand us an OmegaConf ListConfig; vLLM expects a native list
        if (
            "stop_token_ids" in kwargs
            and kwargs["stop_token_ids"] is not None
            and not isinstance(kwargs["stop_token_ids"], list)
        ):
            kwargs["stop_token_ids"] = list(kwargs["stop_token_ids"])

        # print(f"kwargs: {kwargs}")  # debug-only
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        # Verifier LoRA configuration for Co-GRPO
        # Use verifier_lora_name from kwargs if provided, otherwise from config, otherwise default
        if verifier_lora_name is not None:
            self.verifier_lora_name = verifier_lora_name
        else:
            self.verifier_lora_name = self.config.get(
                "verifier_lora_name", "verifier_lora"
            )
        self.verifier_lora_path = (
            verifier_lora_path
            if verifier_lora_path
            else self.config.get("verifier", {}).get("lora_path", None)
        )
        self.verifier_lora_int_id = None

        self._load_verifier_lora_if_needed()

    def _get_lora_engine(self):
        lora_engine = self.inference_engine
        if lora_engine is None:
            return None
        if not hasattr(lora_engine, "add_lora") and hasattr(lora_engine, "llm_engine"):
            if hasattr(lora_engine.llm_engine, "add_lora"):
                lora_engine = lora_engine.llm_engine
        return lora_engine

    def _load_verifier_lora_if_needed(self):
        if not hasattr(self, "_verifier_lora_load_logged"):
            self._verifier_lora_load_logged = False

        def _log_once(status: str):
            if self._verifier_lora_load_logged:
                return
            self._verifier_lora_load_logged = True
            try:
                lora_kwargs = getattr(self, "lora_kwargs", {}) or {}
                logger.warning(
                    "[VerifierLoRA] %s (base_model=%s, lora_path=%s, lora_int_id=%s, enable_lora=%s, max_loras=%s)",
                    status,
                    getattr(self, "model_path", None),
                    getattr(self, "verifier_lora_path", None),
                    getattr(self, "verifier_lora_int_id", None),
                    lora_kwargs.get("enable_lora", False),
                    lora_kwargs.get("max_loras", None),
                )
            except Exception:
                pass

        if not self.verifier_lora_path:
            # No verifier LoRA configured; use base model for verifier.
            _log_once("SKIP: verifier_lora_path is empty (Verifier will use base model)")
            return
        if not hasattr(self, "lora_kwargs") or not self.lora_kwargs.get(
            "enable_lora", False
        ):
            _log_once("SKIP: lora_kwargs.enable_lora=False (Verifier will use base model)")
            return
        if self.inference_engine is None:
            _log_once("SKIP: inference_engine is None (Verifier will use base model)")
            return
        try:
            if hasattr(self.inference_engine, "wake_up"):
                self.inference_engine.wake_up()
            lora_engine = self._get_lora_engine()
            if lora_engine is None or not hasattr(lora_engine, "add_lora"):
                _log_once("SKIP: vLLM engine has no add_lora() (Verifier will use base model)")
                return
            try:
                import inspect

                sig = inspect.signature(lora_engine.add_lora)
                if "lora_request" in sig.parameters:
                    if self.verifier_lora_int_id is None:
                        self._register_verifier_lora()
                    if self.verifier_lora_int_id is None:
                        logger.warning(
                            "Verifier LoRA int id is None after registration; "
                            "skipping add_lora (LoRA likely disabled or max_loras misconfigured)."
                        )
                        _log_once(
                            "FAIL: verifier_lora_int_id is None after registration (use base model)"
                        )
                        return
                    lora_req = LoRARequest(
                        lora_name=self.verifier_lora_name,
                        lora_int_id=int(self.verifier_lora_int_id),
                        lora_path=self.verifier_lora_path,
                    )
                    success = lora_engine.add_lora(lora_request=lora_req)
                    if success is False:
                        logger.warning(
                            "Failed to load Verifier LoRA: add_lora returned False"
                        )
                        _log_once("FAIL: vLLM add_lora returned False (use base model)")
                    else:
                        _log_once("OK: Verifier LoRA loaded")
                else:
                    lora_ret = lora_engine.add_lora(
                        adapter_name=self.verifier_lora_name,
                        adapter_path=self.verifier_lora_path,
                    )
                    if isinstance(lora_ret, int):
                        self.verifier_lora_int_id = lora_ret
                    else:
                        self._register_verifier_lora()
                    _log_once("OK: Verifier LoRA loaded")

            except Exception as e:
                logger.warning(f"Failed to load Verifier LoRA via add_lora: {e}")
                _log_once(f"FAIL: exception during add_lora ({type(e).__name__})")
                try:
                    self._register_verifier_lora()
                except Exception as reg_e:
                    logger.warning(
                        f"Failed to register Verifier LoRA in fallback: {reg_e}"
                    )
        except Exception as e:
            logger.warning(f"Error trying to load Verifier LoRA: {e}")
            _log_once(f"FAIL: exception trying to load LoRA ({type(e).__name__})")
        finally:
            if hasattr(self.inference_engine, "sleep"):
                try:
                    if getattr(self, "enable_kv_cache_optimization", False):
                        _best_effort_reset_prefix_cache(self.inference_engine)
                    self.inference_engine.sleep(level=1)
                except Exception:
                    pass

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            # Check if stop sequences are being set - if so, detokenize must be True
            has_stop_sequences = kwargs.get("stop", None) is not None
            if has_stop_sequences:
                kwargs["detokenize"] = True

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
        # Rebuild vLLM cache engine / wake up executor.
        # NOTE: vLLM APIs changed across versions:
        # - newer versions expose `wake_up()` / `sleep()`
        # - older versions used `init_cache_engine()` / `free_cache_engine()`
        if self.config.free_cache_engine:
            try:
                self.inference_engine.wake_up()
            except AttributeError:
                try:
                    self.inference_engine.init_cache_engine()
                except Exception:
                    pass

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ],
                dtype=object,
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # -----------------------------------------------------------------
        # Stability: avoid vLLM internal `n>1` path for long-context rollouts.
        #
        # We have observed sporadic "CUDA illegal memory access" crashes inside
        # vLLM sampler when using large `n` (e.g. 16) together with long
        # generations (32k). Co-GRPO's by_step rollout already forces `n=1`
        # and repeats prompts on the driver; apply the same idea here by
        # repeating prompts locally and setting vLLM `n=1` for this call.
        #
        # Default behavior:
        # - Only triggers for training rollouts (do_sample=True, validate=False)
        # - Only triggers when `effective_n > 1`
        # - Only triggers for long context (response_length >= 16384) unless
        #   explicitly enabled via VERL_VLLM_FORCE_EXTERNAL_N=1.
        #
        # Override:
        # - Set VERL_VLLM_USE_INTERNAL_N=1 to keep vLLM internal n-path.
        # -----------------------------------------------------------------
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        effective_n = int(kwargs.get("n", getattr(self.sampling_params, "n", 1) or 1))
        if effective_n < 1:
            effective_n = 1

        force_external_n = False
        if do_sample and (not is_validate) and effective_n > 1:
            if _as_bool(os.environ.get("VERL_VLLM_USE_INTERNAL_N", "0")):
                force_external_n = False
            else:
                auto = self.config.response_length >= 16384
                force_external_n = auto or _as_bool(
                    os.environ.get("VERL_VLLM_FORCE_EXTERNAL_N", "0")
                )

        if force_external_n:
            if not hasattr(self, "_force_external_n_logged"):
                self._force_external_n_logged = True
                logger.warning(
                    f"[vLLM] Forcing external prompt repetition (n={effective_n} -> 1) "
                    f"for stability (response_length={self.config.response_length}). "
                    f"Set VERL_VLLM_USE_INTERNAL_N=1 to disable."
                )

            orig_batch_size = batch_size
            repeat_n = effective_n
            # Repeat tensor prompts (for building the returned DataProto).
            idx = _repeat_interleave(idx, repeat_n)
            attention_mask = _repeat_interleave(attention_mask, repeat_n)
            position_ids = _repeat_interleave(position_ids, repeat_n)
            batch_size = batch_size * repeat_n

            # Repeat non-tensor metadata to align with expanded batch.
            for key, val in list(non_tensor_batch.items()):
                if isinstance(val, np.ndarray) and val.shape[0] == orig_batch_size:
                    non_tensor_batch[key] = _repeat_interleave(val, repeat_n)

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
            ):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                    }
                )
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids}
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

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

        # IMPORTANT: Don't apply any loaded LoRA by default for normal rollout generation.
        # In Co-GRPO, Verifier LoRA is applied explicitly only during Verifier generation calls.
        lora_requests = None

        # users can customize different sampling_params at different run
        if force_external_n and do_sample and (not is_validate):
            # vLLM `n` is now encoded in the repeated prompts.
            kwargs = dict(kwargs)
            kwargs["n"] = 1
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

            response = pad_2d_list_to_length(
                response, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(
                rollout_log_probs, -1, max_length=self.config.response_length
            ).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            # input_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=idx.size(1), left=True).to(idx.device).to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                repeat_n = int(self.sampling_params.n)
                orig_batch_size = batch_size
                idx = _repeat_interleave(idx, repeat_n)
                attention_mask = _repeat_interleave(attention_mask, repeat_n)
                position_ids = _repeat_interleave(position_ids, repeat_n)
                batch_size = batch_size * repeat_n
                # Repeat per-sample metadata in non_tensor_batch to match expanded batch size.
                for key, val in list(non_tensor_batch.items()):
                    if isinstance(val, np.ndarray) and val.shape[0] == orig_batch_size:
                        non_tensor_batch[key] = _repeat_interleave(val, repeat_n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device
        )
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, 3, -1
            )

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
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

        # Free vLLM KV cache / offload executor when configured.
        if self.config.free_cache_engine:
            try:
                if getattr(self, "enable_kv_cache_optimization", False):
                    _best_effort_reset_prefix_cache(self.inference_engine)
                if hasattr(self.inference_engine, "sleep"):
                    self.inference_engine.sleep(level=1)
                elif hasattr(self.inference_engine, "free_cache_engine"):
                    self.inference_engine.free_cache_engine()
            except Exception:
                pass

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
            if self.verifier_lora_int_id is not None:
                return
            if not hasattr(self, "_verifier_register_debug_once"):
                self._verifier_register_debug_once = False

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
                if max_loras is None:
                    max_loras = 0

            # If LoRA is enabled, use a stable LoRA ID for Verifier requests.
            # NOTE: vLLM's LoRARequest requires an int id; we use 1 by convention.
            if lora_enabled and max_loras >= 1:
                self.verifier_lora_int_id = 1
                logger.info(
                    f"Using LoRA ID 1 for Verifier (enable_lora={lora_enabled}, max_loras={max_loras}, verifier_lora_name={self.verifier_lora_name})"
                )
                if not self._verifier_register_debug_once:
                    if _as_bool(os.environ.get("VERL_VERIFIER_DEBUG_LOG", False)):
                        logger.warning(
                            "[VerifierDebug] register_lora: enable_lora=%s max_loras=%s verifier_lora_path=%s assigned_lora_int_id=%s",
                            lora_enabled,
                            max_loras,
                            getattr(self, "verifier_lora_path", None),
                            self.verifier_lora_int_id,
                        )
                    self._verifier_register_debug_once = True

            else:
                self.verifier_lora_int_id = None
                logger.warning(
                    f"LoRA not enabled or max_loras < 1 (enable_lora={lora_enabled}, max_loras={max_loras}). Verifier will use base model."
                )
                if not self._verifier_register_debug_once:
                    if _as_bool(os.environ.get("VERL_VERIFIER_DEBUG_LOG", False)):
                        logger.warning(
                            "[VerifierDebug] register_lora: enable_lora=%s max_loras=%s verifier_lora_path=%s assigned_lora_int_id=%s",
                            lora_enabled,
                            max_loras,
                            getattr(self, "verifier_lora_path", None),
                            self.verifier_lora_int_id,
                        )
                    self._verifier_register_debug_once = True

        except Exception as e:
            logger.warning(f"Error in _register_verifier_lora: {e}")
            import traceback

            logger.warning(traceback.format_exc())

            # Fallback: If max_loras > 1, use LoRA ID 1
            if (
                hasattr(self, "lora_kwargs")
                and self.lora_kwargs.get("max_loras", 0) > 1
            ):
                self.verifier_lora_int_id = 1
                logger.info(
                    f"Fallback: Using LoRA ID 1 for Verifier (max_loras={self.lora_kwargs.get('max_loras')})"
                )
            else:
                self.verifier_lora_int_id = None

    def reload_verifier_lora(self, verifier_lora_path: str):
        """Hot-reload verifier LoRA in the vLLM engine (best-effort)."""
        if not verifier_lora_path:
            logger.warning("reload_verifier_lora called with empty path; skipping")
            return False

        self.verifier_lora_path = verifier_lora_path
        if self.verifier_lora_int_id is None:
            # Ensure we have a valid int id before calling vLLM add_lora/remove_lora.
            self._register_verifier_lora()
        if self.verifier_lora_int_id is None:
            logger.warning(
                "Verifier LoRA int id is None (LoRA likely disabled); cannot reload via vLLM add_lora."
            )
            return False
        prev_lora_int_id = int(self.verifier_lora_int_id)
        lora_engine = self._get_lora_engine()
        if lora_engine is None:
            logger.warning("vLLM engine not ready; cannot reload verifier LoRA")
            return False
        try:
            if hasattr(self.inference_engine, "wake_up"):
                self.inference_engine.wake_up()

            # Try to remove old LoRA first
            if hasattr(lora_engine, "remove_lora"):
                try:
                    # Check if it uses LoRARequest or simple name
                    if hasattr(lora_engine.remove_lora, "__code__"):
                        import inspect

                        sig = inspect.signature(lora_engine.remove_lora)
                        if "lora_id" in sig.parameters:
                            # AsyncLLMEngine uses lora_id (int)
                            lora_engine.remove_lora(lora_id=int(prev_lora_int_id))
                        else:
                            # Fallback: try name-based removal
                            lora_engine.remove_lora(lora_name=self.verifier_lora_name)
                except Exception as e:
                    logger.debug(f"Failed to remove old LoRA (may not exist): {e}")

            # Try to add new LoRA
            if hasattr(lora_engine, "add_lora"):
                try:
                    # Check if add_lora expects LoRARequest (AsyncLLMEngine) or kwargs (older API)
                    import inspect

                    sig = inspect.signature(lora_engine.add_lora)

                    if "lora_request" in sig.parameters:
                        # AsyncLLMEngine API: add_lora(lora_request: LoRARequest) -> bool
                        from vllm.lora.request import LoRARequest

                        lora_req = LoRARequest(
                            lora_name=self.verifier_lora_name,
                            lora_int_id=prev_lora_int_id,
                            lora_path=verifier_lora_path,
                        )
                        success = lora_engine.add_lora(lora_request=lora_req)
                        if success:
                            self.verifier_lora_int_id = prev_lora_int_id
                            logger.info(
                                f"Successfully reloaded Verifier LoRA via AsyncLLMEngine API: {verifier_lora_path}"
                            )
                        else:
                            logger.error(
                                f"Failed to reload Verifier LoRA: add_lora returned False"
                            )
                            return False
                    else:
                        # Older/unknown API: try kwargs
                        lora_ret = lora_engine.add_lora(
                            adapter_name=self.verifier_lora_name,
                            adapter_path=verifier_lora_path,
                        )
                        if isinstance(lora_ret, int):
                            self.verifier_lora_int_id = lora_ret
                        else:
                            self.verifier_lora_int_id = prev_lora_int_id
                        logger.info(
                            f"Successfully reloaded Verifier LoRA via legacy API: {verifier_lora_path}"
                        )

                    self._register_verifier_lora()

                    # Verify LoRA is actually loaded
                    if hasattr(lora_engine, "list_loras"):
                        try:
                            loaded_loras = lora_engine.list_loras()
                            if hasattr(loaded_loras, "__iter__"):
                                if all(isinstance(l, int) for l in loaded_loras):
                                    if self.verifier_lora_int_id not in loaded_loras:
                                        logger.warning(
                                            f"Verifier LoRA ID {self.verifier_lora_int_id} not found in loaded LoRAs: {loaded_loras}"
                                        )
                                    else:
                                        logger.info(
                                            f"Verified Verifier LoRA ID {self.verifier_lora_int_id} is loaded"
                                        )
                                else:
                                    lora_names = [
                                        l.lora_name
                                        if hasattr(l, "lora_name")
                                        else str(l)
                                        for l in loaded_loras
                                    ]
                                    if self.verifier_lora_name not in lora_names:
                                        logger.warning(
                                            f"Verifier LoRA {self.verifier_lora_name} not found in loaded LoRAs: {lora_names}"
                                        )
                                    else:
                                        logger.info(
                                            f"Verified Verifier LoRA {self.verifier_lora_name} is loaded with ID={self.verifier_lora_int_id}"
                                        )
                        except Exception as e:
                            logger.debug(f"Could not verify LoRA loading: {e}")

                    return True

                except Exception as e:
                    logger.error(f"Failed to reload Verifier LoRA via add_lora: {e}")
                    return False
            else:
                logger.warning(
                    "vLLM engine does not support add_lora; cannot reload verifier LoRA"
                )
                return False
        finally:
            if hasattr(self.inference_engine, "sleep"):
                try:
                    if getattr(self, "enable_kv_cache_optimization", False):
                        _best_effort_reset_prefix_cache(self.inference_engine)
                    self.inference_engine.sleep(level=1)
                except Exception:
                    pass

    @GPUMemoryLogger(role="vllm dual stream rollout", logger=logger)
    @torch.no_grad()
    def dual_stream_rollout(
        self,
        prompts: DataProto,
        intervention_mode: str = "by_step",
        stream_mode: str = "both",
        token_check_interval: int = 1024,
        **kwargs,
    ) -> DataProto:
        """
        Perform dual-stream rollout for Co-GRPO:
        - Control Stream: Policy generates without Verifier guidance
        - Experimental Stream: Policy generates with Verifier intervention based on mode

        Args:
            prompts: DataProto containing prompts
            intervention_mode: active mode is "by_step"; others fallback to by_step.
            token_check_interval: by_step trigger interval in generated tokens.

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
        eos_token_ids = _normalize_eos_token_ids(eos_token_id)
        batch_size = idx.size(0)

        # Handle empty batch
        if batch_size == 0:
            from tensordict import TensorDict

            empty_batch = TensorDict({}, batch_size=0)
            timing_info["total"] = time.time() - start_time
            meta_info = prompts.meta_info.copy()
            meta_info["timing"] = timing_info
            return DataProto(
                batch=empty_batch, non_tensor_batch={}, meta_info=meta_info
            )

        # Route to by_step rollout (the only supported path in this file).
        if intervention_mode == "by_step":
            # CRITICAL FIX: Pass dual_stream_rollout's explicit parameters to _dual_stream_rollout_by_step via kwargs
            # Otherwise token_check_interval won't be passed because it's an explicit parameter of dual_stream_rollout,
            # not included in **kwargs automatically
            kwargs["token_check_interval"] = token_check_interval
            # Keep min_step_tokens in sync with token_check_interval unless overridden
            kwargs.setdefault("min_step_tokens", token_check_interval)
            kwargs["stream_mode"] = stream_mode
            return self._dual_stream_rollout_by_step(
                prompts,
                timing_info,
                start_time,
                tokenizer,
                idx,
                attention_mask,
                position_ids,
                eos_token_id,
                batch_size,
                **kwargs,
            )
        else:
            # Fallback: Force by_step mode for debugging
            logger.warning(
                f"intervention_mode={intervention_mode} not supported yet, forcing by_step mode"
            )
            kwargs["token_check_interval"] = token_check_interval
            kwargs["stream_mode"] = stream_mode
            return self._dual_stream_rollout_by_step(
                prompts,
                timing_info,
                start_time,
                tokenizer,
                idx,
                attention_mask,
                position_ids,
                eos_token_id,
                batch_size,
                **kwargs,
            )

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

        # 1. Prefer coarser boundaries (used for local trimming / optional stop sequences).
        # NOTE: Some tokenizers fuse the closing marker with newlines (e.g. Qwen2: "</think>\n\n"),
        # so we include the common newline variants explicitly.
        for seq in [
            "</think>",
            "</think>\n",
            "</think>\n\n",
            "<｜end of thought｜>",
            "<|end of thought|>",
            "<|end_of_thought|>",
            "\n\n",
            ".\n\n",
        ]:
            stop_sequences.add(seq)

        if eos_token_id is not None:
            # 处理 eos_token_id 可能是 list 的情况
            if isinstance(eos_token_id, (list, tuple)):
                stop_token_ids.update(eos_token_id)
            else:
                stop_token_ids.add(eos_token_id)

        # Also stop when the model starts a new chat message.
        try:
            im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[-1]
            stop_token_ids.add(int(im_start_id))
        except Exception:
            pass

        # Treat pad/endoftext as a stop token as well.
        try:
            if getattr(self, "pad_token_id", None) is not None:
                stop_token_ids.add(int(self.pad_token_id))
        except Exception:
            pass

        return {
            "stop_token_ids": list(stop_token_ids),
            "stop_sequences": list(stop_sequences),
        }

    def _run_verifier_inference(
        self,
        verifier_inputs,
        tokenizer,
        idx_device,
        exp_kwargs,
        return_trajectories: bool = False,
    ):
        """
        批量调用 Verifier 进行推理。

        Args:
            verifier_inputs: List[str] - Verifier 输入文本列表
            tokenizer: Tokenizer instance
            idx_device: Device for tensors
            exp_kwargs: Sampling parameters

        Returns:
            If return_trajectories is False:
                List[dict]: 每个输入的决策结果，包含 'action', 'hint', 'critique'
            If return_trajectories is True:
                Tuple[List[dict], Dict[str, Any]]: (decisions, trajectories)
        """
        if not verifier_inputs:
            return ([], {}) if return_trajectories else []

        try:
            if not hasattr(self, "_verifier_debug_once"):
                self._verifier_debug_once = False
            if not hasattr(self, "_verifier_malformed_warn_count"):
                self._verifier_malformed_warn_count = 0

            verifier_cfg = None
            if getattr(self, "verifier_config", None) is not None:
                verifier_cfg = self.verifier_config
            if verifier_cfg is None:
                verifier_cfg = self.config.get("verifier", {})
            verifier_max_new_tokens = int(verifier_cfg.get("max_new_tokens", 512))
            verifier_logprobs = int(verifier_cfg.get("logprobs", 1))
            verifier_lora_rank = int(verifier_cfg.get("lora_rank", 0))
            verifier_debug_log = _as_bool(
                verifier_cfg.get(
                    "debug_log",
                    os.environ.get("VERL_VERIFIER_DEBUG_LOG", False),
                )
            )

            # NOTE:
            # `self.config` here is rollout sub-config, which may not carry verifier.lora_rank.
            # Do not disable LoRA solely based on verifier_cfg.lora_rank.
            lora_enabled = bool(
                getattr(self, "lora_kwargs", {}).get("enable_lora", False)
            )
            verifier_lora_path = getattr(self, "verifier_lora_path", None)
            use_verifier_lora = lora_enabled and bool(verifier_lora_path)

            if use_verifier_lora and self.verifier_lora_int_id is None:
                self._register_verifier_lora()

            max_model_len = int(
                self.config.max_model_len
                or (self.config.prompt_length + self.config.response_length)
            )
            verifier_max_prompt_length = int(
                verifier_cfg.get(
                    "max_prompt_length", max_model_len - verifier_max_new_tokens
                )
            )
            # vLLM requires: len(prompt) + max_tokens <= max_model_len
            verifier_prompt_budget = max(
                1,
                min(
                    verifier_max_prompt_length, max_model_len - verifier_max_new_tokens
                ),
            )

            verifier_temperature = float(verifier_cfg.get("temperature", 1.0))
            if verifier_temperature <= 0:
                verifier_temperature = 1.0

            verifier_top_p = float(verifier_cfg.get("top_p", 1.0))
            verifier_top_k = int(verifier_cfg.get("top_k", -1))
            verifier_do_sample = bool(verifier_cfg.get("do_sample", False))
            if not verifier_do_sample:
                # Deterministic decode while keeping temperature valid for PPO logprob computation.
                verifier_temperature = 1.0
                verifier_top_p = 1.0
                verifier_top_k = 1

            # Build per-sample prompt_token_ids without batch padding (avoid OOM).
            verifier_idx_list = []
            verifier_prompt_token_ids = []
            first_sep_found = False
            first_student_resp_len = 0
            first_student_resp_preview = ""
            first_question_preview = ""
            for prompt_idx, verifier_prompt in enumerate(verifier_inputs):
                # Heuristic structured truncation: keep the latest part of "Student Response".
                prefix_text, sep, student_resp = verifier_prompt.partition(
                    "**Student Response (So Far):**"
                )
                if sep:
                    user_prefix = f"{prefix_text}**Student Response (So Far):**\n"
                else:
                    user_prefix = verifier_prompt
                    student_resp = ""

                if prompt_idx == 0:
                    first_sep_found = bool(sep)
                    first_student_resp_len = len(student_resp)
                    first_student_resp_preview = str(student_resp)[:220].replace(
                        "\n", "\\n"
                    )
                    try:
                        q_part = (
                            prefix_text.split("**Question:**", 1)[1]
                            if "**Question:**" in prefix_text
                            else prefix_text
                        )
                    except Exception:
                        q_part = prefix_text
                    first_question_preview = str(q_part)[:220].replace("\n", "\\n")

                base_messages = [
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prefix},
                ]
                try:
                    base_ids = tokenizer.apply_chat_template(
                        base_messages, tokenize=True, add_generation_prompt=True
                    )
                except Exception:
                    base_text = tokenizer.apply_chat_template(
                        base_messages, tokenize=False, add_generation_prompt=True
                    )
                    base_ids = tokenizer.encode(base_text, add_special_tokens=False)

                remaining = verifier_prompt_budget - len(base_ids)
                if remaining > 0 and student_resp:
                    tail_ids = tokenizer.encode(student_resp, add_special_tokens=False)
                    tail_ids = tail_ids[-remaining:]
                    truncated_student_resp = tokenizer.decode(
                        tail_ids, skip_special_tokens=True
                    )
                    user_full = user_prefix + truncated_student_resp
                else:
                    user_full = user_prefix

                messages = [
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_full},
                ]
                try:
                    prompt_ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                except Exception:
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

                # Final safety clamp (keep tail if still over budget).
                if len(prompt_ids) > verifier_prompt_budget:
                    prompt_ids = prompt_ids[-verifier_prompt_budget:]

                verifier_idx_list.append(prompt_ids)
                verifier_prompt_token_ids.append(prompt_ids)

            # 准备 LoRA requests
            verifier_lora_requests = None
            if use_verifier_lora and self.verifier_lora_int_id is not None:
                verifier_lora_requests = [
                    LoRARequest(
                        lora_name=self.verifier_lora_name,
                        lora_int_id=self.verifier_lora_int_id,
                        lora_path=verifier_lora_path,
                    )
                    for _ in range(len(verifier_inputs))
                ]

            if verifier_debug_log and (not self._verifier_debug_once):
                try:
                    prompt_preview = (
                        str(verifier_inputs[0])[:260].replace("\n", "\\n")
                        if verifier_inputs
                        else ""
                    )
                    logger.warning(
                        "[VerifierDebug] first_call: lora_rank=%s lora_enabled=%s use_verifier_lora=%s lora_int_id=%s has_lora_request=%s num_inputs=%s max_prompt_budget=%s max_new_tokens=%s lora_path=%s sep_found=%s student_resp_len=%s question_preview=%s student_resp_preview=%s prompt_preview=%s",
                        verifier_lora_rank,
                        lora_enabled,
                        use_verifier_lora,
                        self.verifier_lora_int_id,
                        bool(verifier_lora_requests),
                        len(verifier_inputs),
                        verifier_prompt_budget,
                        verifier_max_new_tokens,
                        verifier_lora_path,
                        first_sep_found,
                        first_student_resp_len,
                        first_question_preview,
                        first_student_resp_preview,
                        prompt_preview,
                    )
                except Exception:
                    pass

            verifier_kwargs = exp_kwargs.copy()
            verifier_kwargs["temperature"] = verifier_temperature
            verifier_kwargs["top_p"] = verifier_top_p
            verifier_kwargs["top_k"] = verifier_top_k
            verifier_kwargs["n"] = 1
            verifier_kwargs["max_tokens"] = verifier_max_new_tokens
            verifier_kwargs["logprobs"] = verifier_logprobs

            # Best-effort early stop for the common-pass case.
            # NOTE: Do NOT stop on `<WAIT>`; we need the hint text after it.
            verifier_kwargs["stop"] = ["\n<GO>"]
            verifier_kwargs["include_stop_str_in_output"] = True

            # 批量生成 Verifier 响应
            with self.update_sampling_params(**verifier_kwargs):
                verifier_output = self.inference_engine.generate(
                    prompts=None,
                    sampling_params=self.sampling_params,
                    prompt_token_ids=verifier_idx_list,
                    lora_request=verifier_lora_requests,
                    use_tqdm=False,
                )

            # Extra debug: finish_reason distribution helps confirm truncation vs stop.
            if verifier_debug_log and (not self._verifier_debug_once):
                try:
                    finish_reason_counts = {}
                    for out in verifier_output:
                        for seq_out in getattr(out, "outputs", []) or []:
                            fr = getattr(seq_out, "finish_reason", None)
                            fr = str(fr) if fr is not None else "None"
                            finish_reason_counts[fr] = (
                                finish_reason_counts.get(fr, 0) + 1
                            )
                    logger.warning(
                        "[VerifierDebug] finish_reason_counts=%s",
                        finish_reason_counts,
                    )
                except Exception:
                    pass

            verifier_responses, verifier_log_probs = _extract_vllm_outputs(
                verifier_output, self.pad_token_id, verifier_max_new_tokens, idx_device
            )

            # 解析所有 Verifier 输出
            decisions = []
            malformed_like = 0
            tag_go = 0
            tag_wait = 0
            parsed_go = 0
            parsed_wait = 0
            no_valid_decision = 0
            closed_think = 0
            no_decision_samples = []
            max_no_decision_samples = 3
            for i in range(len(verifier_inputs)):
                verifier_response = verifier_responses[i]
                verifier_text = tokenizer.decode(
                    verifier_response.tolist(), skip_special_tokens=True
                )
                low_text = verifier_text.lower()
                if "<go>" in low_text:
                    tag_go += 1
                if "<wait>" in low_text:
                    tag_wait += 1
                if (
                    "</think>" in low_text
                    or "<｜end of thought｜>" in low_text
                    or "<|end of thought|>" in low_text
                    or "<|end_of_thought|>" in low_text
                ):
                    closed_think += 1
                decision = self._parse_verifier_decision(verifier_text)
                decision_tag = decision.get("decision_tag", None)
                if decision_tag == "GO":
                    parsed_go += 1
                elif decision_tag == "WAIT":
                    parsed_wait += 1
                else:
                    no_valid_decision += 1
                if decision.get("action") == "Intervene":
                    wait_confidence, wait_avg_logprob = _compute_wait_confidence(
                        verifier_response,
                        verifier_log_probs[i],
                        self.pad_token_id,
                    )
                    if wait_confidence is not None:
                        decision["wait_confidence"] = wait_confidence
                        decision["wait_avg_logprob"] = wait_avg_logprob
                decisions.append(decision)

                if decision_tag not in ("GO", "WAIT"):
                    if ("final answer" in low_text) or ("\\boxed" in verifier_text):
                        malformed_like += 1
                    if (
                        verifier_debug_log
                        and len(no_decision_samples) < max_no_decision_samples
                    ):
                        head = str(verifier_text)[:180].replace("\n", "\\n")
                        tail = str(verifier_text)[-180:].replace("\n", "\\n")
                        no_decision_samples.append(f"head={head} ... tail={tail}")

                if verifier_debug_log and (not self._verifier_debug_once) and i == 0:
                    try:
                        logger.warning(
                            "[VerifierDebug] first_output: parsed_action=%s hint_len=%s text_preview=%s",
                            decision.get("action"),
                            len(decision.get("hint") or ""),
                            str(verifier_text)[:260].replace("\n", "\\n"),
                        )
                    except Exception:
                        pass

            if verifier_debug_log and (not self._verifier_debug_once):
                try:
                    logger.warning(
                        "[VerifierDebug] output_tag_stats: raw_go=%s raw_wait=%s parsed_go=%s parsed_wait=%s no_valid_decision=%s closed_think=%s",
                        tag_go,
                        tag_wait,
                        parsed_go,
                        parsed_wait,
                        no_valid_decision,
                        closed_think,
                    )
                except Exception:
                    pass

            if (
                len(verifier_inputs) > 0
                and no_valid_decision / len(verifier_inputs) >= 0.5
            ):
                if verifier_debug_log and (self._verifier_malformed_warn_count < 5):
                    logger.warning(
                        "[VerifierDebug] verifier_no_valid_decision=%s/%s malformed_final_like=%s/%s "
                        "(missing <GO>/<WAIT> decision tag; malformed_final_like is the subset that also looks like a final answer)",
                        no_valid_decision,
                        len(verifier_inputs),
                        malformed_like,
                        len(verifier_inputs),
                    )
                    try:
                        finish_reason_counts = {}
                        for out in verifier_output:
                            for seq_out in getattr(out, "outputs", []) or []:
                                fr = getattr(seq_out, "finish_reason", None)
                                fr = str(fr) if fr is not None else "None"
                                finish_reason_counts[fr] = (
                                    finish_reason_counts.get(fr, 0) + 1
                                )
                        logger.warning(
                            "[VerifierDebug] no_decision_finish_reasons=%s",
                            finish_reason_counts,
                        )
                    except Exception:
                        pass
                    if no_decision_samples:
                        logger.warning(
                            "[VerifierDebug] no_decision_samples=%s",
                            no_decision_samples,
                        )
                    self._verifier_malformed_warn_count += 1
                elif (not verifier_debug_log) and (self._verifier_malformed_warn_count < 1):
                    logger.warning(
                        "[Verifier] High no-decision rate: %s/%s (set VERL_VERIFIER_DEBUG_LOG=1 or verifier.debug_log=True for detailed samples).",
                        no_valid_decision,
                        len(verifier_inputs),
                    )
                    self._verifier_malformed_warn_count += 1

            self._verifier_debug_once = True

            if not return_trajectories:
                return decisions

            trajectories = {
                "prompt_token_ids": verifier_prompt_token_ids,
                "responses": verifier_responses,
                "old_log_probs": verifier_log_probs,
                "temperature": verifier_temperature,
                "max_new_tokens": verifier_max_new_tokens,
            }
            return decisions, trajectories

        except Exception as e:
            logger.warning(f"Verifier inference failed: {e}")
            # 返回默认决策（Pass）
            fallback = [
                {"action": "Pass", "hint": None, "critique": f"[Error: {str(e)}]"}
                for _ in verifier_inputs
            ]
            return (fallback, {}) if return_trajectories else fallback

    def _extract_verifier_hint(self, verifier_text: str) -> str:
        """
        Extract actual hint from Verifier output, removing think blocks and extracting <GO> or <WAIT> content.

        Parsing policy is intentionally strict to avoid leaking verifier meta-instructions into actor hints.
        """
        if not verifier_text:
            return ""

        raw_text = str(verifier_text)

        def _sanitize_hint_text(text: str) -> str:
            if not text:
                return ""
            normalized = re.sub(r"\s+", " ", text).strip().strip("`\"'")
            if not normalized:
                return ""

            lower_text = normalized.lower()
            blocked_markers = (
                "final answer",
                "output the decision line",
                "output format",
                "remember to format",
                "thus we output",
                "plus one short guidance hint",
                "<go>",
                "<wait>",
                "<think>",
                "</think>",
                "```",
            )
            if any(marker in lower_text for marker in blocked_markers):
                return ""
            return normalized

        # Remove fenced code blocks first.
        without_code = re.sub(r"```[\s\S]*?```", "", raw_text)

        # Prefer parsing decision from the tail after the last closing think tag.
        # NOTE: Some base models may emit alternative "end of thought" markers.
        decision_scope = without_code
        think_close_pattern = r"(?:</think\s*>|<｜end of thought｜>|<\|end of thought\|>|<\|end_of_thought\|>)"
        think_tail_parts = re.split(
            think_close_pattern,
            without_code,
            flags=re.IGNORECASE,
        )
        if len(think_tail_parts) > 1:
            decision_scope = think_tail_parts[-1]
        else:
            # Remove fully-closed think blocks; if we still have an unclosed <think>, drop from that tag onward.
            decision_scope = re.sub(
                rf"<think>.*?{think_close_pattern}",
                "",
                without_code,
                flags=re.IGNORECASE | re.DOTALL,
            )
            decision_scope = re.sub(
                r"<think>[\s\S]*$", "", decision_scope, flags=re.IGNORECASE
            )

        def _extract_from_lines(lines):
            tag_line_pattern = re.compile(r"^<(GO|WAIT)>\s*(.*)$", re.IGNORECASE)
            for idx, line in enumerate(lines):
                match = tag_line_pattern.match(line)
                if not match:
                    continue

                tag = match.group(1).upper()
                if tag == "GO":
                    return "<GO>"

                # WAIT: prefer same-line hint; if empty, allow one next non-tag line.
                hint_line = match.group(2).strip()
                if (
                    not hint_line
                    and idx + 1 < len(lines)
                    and not tag_line_pattern.match(lines[idx + 1])
                ):
                    hint_line = lines[idx + 1].strip()

                hint_line = _sanitize_hint_text(hint_line)
                if not hint_line:
                    return ""
                return f"<WAIT> {hint_line}"
            return ""

        lines = [
            line.strip()
            for line in decision_scope.replace("\r\n", "\n").split("\n")
            if line.strip()
        ]
        hint = _extract_from_lines(lines)
        if hint:
            return hint

        # If think-style output exists but no valid tail decision is found, treat it as malformed.
        # This avoids parsing quoted `<GO>/<WAIT>` tags from instruction text inside `<think>`.
        if re.search(r"</?think\b", without_code, flags=re.IGNORECASE):
            return ""

        # Legacy fallback (for outputs that do not use <think> blocks).
        fallback_lines = [
            line.strip()
            for line in without_code.replace("\r\n", "\n").split("\n")
            if line.strip()
        ]
        return _extract_from_lines(fallback_lines)

    def _truncate_verifier_hint_tokens(self, hint: str, tokenizer) -> str:
        if not hint:
            return ""
        verifier_cfg = None
        if getattr(self, "verifier_config", None) is not None:
            verifier_cfg = self.verifier_config
        if verifier_cfg is None:
            verifier_cfg = (
                self.config.get("verifier", {}) if hasattr(self.config, "get") else {}
            )
        try:
            max_hint_tokens = (
                verifier_cfg.get("max_hint_tokens", None)
                if hasattr(verifier_cfg, "get")
                else None
            )
        except Exception:
            max_hint_tokens = None
        if max_hint_tokens is None:
            return hint
        try:
            max_hint_tokens = int(max_hint_tokens)
        except Exception:
            return hint
        if max_hint_tokens <= 0:
            return hint
        try:
            hint_ids = tokenizer.encode(hint, add_special_tokens=False)
            if len(hint_ids) <= max_hint_tokens:
                return hint
            return tokenizer.decode(
                hint_ids[:max_hint_tokens], skip_special_tokens=True
            ).strip()
        except Exception:
            return hint

    def _parse_verifier_decision(self, verifier_text: str) -> dict:
        """
        Parse Verifier output to extract decision (Pass/Intervene) and hint.

        New format: Extract <GO> or <WAIT> content after removing think blocks.

        Returns:
            dict with keys: "action" (str), "hint" (str), "critique" (str)
        """
        verifier_text = verifier_text.strip()

        # Extract actual hint (removes think blocks, extracts <GO> or <WAIT> content)
        raw_decision = self._extract_verifier_hint(verifier_text)
        decision_tag = None
        if re.match(r"^\s*<WAIT>(?:\s|$)", raw_decision, flags=re.IGNORECASE):
            decision_tag = "WAIT"
        elif re.match(r"^\s*<GO>(?:\s|$)", raw_decision, flags=re.IGNORECASE):
            decision_tag = "GO"
        hint = raw_decision

        # Determine action based on hint
        if re.match(r"^\s*<WAIT>(?:\s|$)", hint, flags=re.IGNORECASE):
            action = "Intervene"
            # Extract content after <WAIT>
            hint_content = re.sub(
                r"^\s*<WAIT>\s*", "", hint, flags=re.IGNORECASE
            ).strip()
            hint = (
                hint_content  # CRITICAL: Use just the content, NOT "<WAIT> {content}"
            )
            # The <WAIT> tag confuses the Actor and causes it to repeat empty tokens!
            if not hint:
                action = "Pass"
        elif re.match(r"^\s*<GO>(?:\s|$)", hint, flags=re.IGNORECASE):
            action = "Pass"
            hint = ""
        else:
            action = "Pass"
            hint = ""

        return {
            "action": action,
            "hint": hint,
            "critique": verifier_text,  # Keep full output for logging
            "decision_tag": decision_tag,  # "GO" | "WAIT" | None
        }


    def _dual_stream_rollout_by_step(
        self,
        prompts,
        timing_info,
        start_time,
        tokenizer,
        idx,
        attention_mask,
        position_ids,
        eos_token_id,
        batch_size,
        **kwargs,
    ) -> DataProto:
        """
        by_step mode: 使用 stop_sequences 实现批量并行增量打断生成（修正版）。

        关键修正：
        1. 统一 sampling_params（计算 max_remaining_tokens）
        2. 引入 loss_mask 区分 Hint（0）和生成内容（1）
        3. Token ID + String Suffix 双重停止检测
        4. 完整的 DataProto 构建
        """
        stream_mode = kwargs.pop("stream_mode", "both")
        if stream_mode not in ("both", "control", "exp"):
            logger.warning(f"Unknown stream_mode={stream_mode}, falling back to 'both'")
            stream_mode = "both"

        # Optional: seed sample states from an existing partial response (used by cf-branch rollouts).
        initial_state_overrides = kwargs.pop("initial_state_overrides", None)
        collect_verifier_trajectories = _as_bool(
            kwargs.pop("collect_verifier_trajectories", True)
        )
        # NOTE: Decoding full 32k responses into strings and shipping them through Ray
        # can bloat Ray object store usage and trigger spilling. Keep decoded text
        # fields off by default; trainer-side dumps already decode when needed.
        build_decoded_fields = _as_bool(kwargs.pop("build_decoded_fields", False))

        # Counterfactual branching (cf-branch) config: build additional "no-hint" rollouts
        # for each applied intervention, to estimate marginal contribution Δ = R(with) - R(without).
        cf_branching = _as_bool(kwargs.pop("cf_branching", False))
        try:
            cf_branch_prob = float(kwargs.pop("cf_branch_prob", 1.0))
        except Exception:
            cf_branch_prob = 1.0
        try:
            cf_branch_max_events_per_sample = int(
                kwargs.pop("cf_branch_max_events_per_sample", 0)
            )
        except Exception:
            cf_branch_max_events_per_sample = 0
        try:
            cf_branch_state_hash_mod = int(kwargs.pop("cf_branch_state_hash_mod", 1024))
        except Exception:
            cf_branch_state_hash_mod = 1024
        try:
            cf_branch_k = int(kwargs.pop("cf_branch_k", 1))
        except Exception:
            cf_branch_k = 1
        if cf_branch_k < 1:
            cf_branch_k = 1
        try:
            cf_branch_reward_tail_tokens = int(
                kwargs.pop("cf_branch_reward_tail_tokens", 2048)
            )
        except Exception:
            cf_branch_reward_tail_tokens = 2048
        if cf_branch_reward_tail_tokens < 0:
            cf_branch_reward_tail_tokens = 0

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        control_responses = None
        control_log_probs = None
        control_n = 1
        control_prompt_lens = None
        control_response_lens = None
        control_finish_reasons = None

        # ========== Control Stream: Policy generates without guidance ==========
        if stream_mode in ("both", "control"):
            idx_list = []
            for i in range(batch_size):
                idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

            # =================================================================
            # FIX START: Robust Batch Size Check for Co-GRPO
            # =================================================================

            # 1. Basic configuration
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

            # 2. Co-GRPO: Force n=1 to avoid double repetition
            # In Co-GRPO, trainer already repeats gen_batch BEFORE calling dual_stream_rollout
            # So we MUST use n=1 here, otherwise vLLM will repeat again causing batch_size mismatch
            if do_sample and not is_validate and self.sampling_params.n > 1:
                target_n = self.sampling_params.n
                logger.info(
                    f"Co-GRPO by_step: Forcing vLLM n=1 (trainer already repeated with n={target_n})"
                )
                control_kwargs["n"] = 1

            # =================================================================
            # FIX END
            # =================================================================

            # Store the actual n used for control generation (for batch expansion)
            control_n = control_kwargs.get(
                "n", self.sampling_params.n if do_sample and not is_validate else 1
            )

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

            # Extract exactly 1 response per prompt. In long-context RL we have seen
            # vLLM occasionally return multiple outputs per prompt (stale n/best_of),
            # which can crash downstream tensor concatenation if we flatten them.
            control_response_ids_list = []
            control_logprob_list = []
            control_finish_reasons = []
            multi_output_prompts = 0
            empty_output_prompts = 0
            if len(control_output) != len(idx_list):
                logger.error(
                    f"[CONTROL][MISMATCH] vLLM request count != input count! "
                    f"len(control_output)={len(control_output)}, len(idx_list)={len(idx_list)}"
                )
            for out in control_output:
                outputs = getattr(out, "outputs", None) or []
                if len(outputs) == 0:
                    empty_output_prompts += 1
                    control_response_ids_list.append([])
                    control_logprob_list.append([])
                    control_finish_reasons.append("vllm_empty_output")
                    continue
                if len(outputs) != 1:
                    multi_output_prompts += 1
                out0 = outputs[0]
                response_ids = out0.token_ids
                control_response_ids_list.append(response_ids)
                curr_log_prob = []
                try:
                    for j, logprob in enumerate(out0.logprobs or []):
                        curr_log_prob.append(logprob[response_ids[j]].logprob)
                except Exception:
                    pass
                control_logprob_list.append(curr_log_prob)
                fr = getattr(out0, "finish_reason", None) or "unknown"
                if fr == "length":
                    fr = "gen_budget_exhausted"
                elif fr == "stop":
                    fr = "eos"
                control_finish_reasons.append(fr)

            if empty_output_prompts > 0 or multi_output_prompts > 0:
                logger.warning(
                    f"[CONTROL] vLLM returned empty/multi outputs "
                    f"(empty={empty_output_prompts}, multi={multi_output_prompts}); "
                    f"using first output per prompt."
                )

            expected_ctrl = int(batch_size)
            if len(control_response_ids_list) != expected_ctrl:
                logger.error(
                    f"[CONTROL][MISMATCH] control output count != batch_size "
                    f"(count={len(control_response_ids_list)}, batch_size={expected_ctrl}); "
                    f"trunc/pad to match."
                )
                if len(control_response_ids_list) < expected_ctrl:
                    pad_n = expected_ctrl - len(control_response_ids_list)
                    control_response_ids_list.extend([[]] * pad_n)
                    control_logprob_list.extend([[]] * pad_n)
                    control_finish_reasons.extend(["unknown"] * pad_n)
                else:
                    control_response_ids_list = control_response_ids_list[:expected_ctrl]
                    control_logprob_list = control_logprob_list[:expected_ctrl]
                    control_finish_reasons = control_finish_reasons[:expected_ctrl]

            control_responses = pad_2d_list_to_length(
                control_response_ids_list,
                self.pad_token_id,
                max_length=self.config.response_length,
            ).to(idx.device)
            control_log_probs = pad_2d_list_to_length(
                control_logprob_list, -1, max_length=self.config.response_length
            ).to(idx.device)
            control_log_probs = control_log_probs.to(torch.float32)
            timing_info["control_gen"] = time.time() - control_start

            # Control-side rollout diagnostics (for exp/control truncation comparison).
            # `control_finish_reasons` is collected above (1 per prompt).

            try:
                # attention_mask here is the *prompt* attention mask.
                prompt_lens = (
                    attention_mask.to(torch.long)
                    .sum(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
                n_ctrl = int(control_responses.size(0))
                if int(prompt_lens.shape[0]) != n_ctrl and int(prompt_lens.shape[0]) > 0:
                    rep = int(math.ceil(n_ctrl / int(prompt_lens.shape[0])))
                    prompt_lens = np.tile(prompt_lens, rep)[:n_ctrl]
                control_prompt_lens = prompt_lens
            except Exception:
                control_prompt_lens = None

            try:
                resp_lens = (
                    (control_responses != int(self.pad_token_id))
                    .to(torch.long)
                    .sum(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
                control_response_lens = resp_lens
            except Exception:
                control_response_lens = None

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
        # Normalize eos token ids (some tokenizers expose list/tuple).
        eos_token_ids = _normalize_eos_token_ids(eos_token_id)
        stop_token_ids_set = set(
            int(x) for x in (stop_config.get("stop_token_ids") or [])
        )
        if not stop_config["stop_sequences"] and not stop_config["stop_token_ids"]:
            logger.warning(
                "No stop sequences found, using full generation without step boundaries"
            )
            # Fall back to simple full generation (continue without step interruption)
            pass  # Continue with full generation

        # Pre-encode stop sequences for local detection (we ignore stop at the sampler level)
        stop_seq_token_ids = []
        for seq in stop_config["stop_sequences"]:
            if not seq:
                continue
            ids = tokenizer.encode(seq, add_special_tokens=False)
            if ids:
                stop_seq_token_ids.append(ids)
        stop_token_ids = stop_config["stop_token_ids"] or []

        # 2. 初始化 InterventionPolicy
        max_interventions = kwargs.get("max_interventions", 3)
        confidence_threshold = kwargs.get("confidence_threshold", 0.0)
        intervention_policy = InterventionPolicy(
            max_interventions=max_interventions,
            confidence_threshold=confidence_threshold,
        )
        try:
            confidence_threshold_value = float(intervention_policy.confidence_threshold)
        except Exception:
            confidence_threshold_value = 0.0

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
            if eos_token_ids and prompt_tokens and prompt_tokens[-1] in eos_token_ids:
                logger.warning(
                    f"[PROMPT TRIM EOS] Sample {b}: stripping trailing EOS {eos_token_id}"
                )
                prompt_tokens = prompt_tokens[:-1]
            if len(prompt_tokens) > prompt_budget:
                logger.warning(
                    f"[PROMPT OVER BUDGET] Sample {b}: prompt_len={len(prompt_tokens)} > prompt_budget={prompt_budget}"
                )
            prompt_text_raw = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            verifier_question = _extract_verifier_question_from_prompt(prompt_text_raw)

            sample_states.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "prompt_len": int(len(prompt_tokens)),
                    "prompt_text": prompt_text_raw,
                    "verifier_question": verifier_question,
                    # Cache full input tokens (prompt + response) to avoid rebuilding
                    # `prompt_tokens + response_tokens` on every by_step loop.
                    "input_tokens": list(prompt_tokens),
                    "response_tokens": [],  # 关键：hint 追加到这里
                    "loss_masks": [],  # 关键新增：1 for Model Gen, 0 for Hint
                    # Fast counters to avoid O(seq_len) scans of loss_masks in hot loops.
                    # Keep consistent with loss_masks (1=policy, 0=hint).
                    "gen_len": 0,
                    "hint_len": 0,
                    # Token count of the *raw* hint strings (for penalty/analysis); updated per intervention.
                    # This matches the previous semantics of summing tokenizer.encode(hint) at the end,
                    # but avoids re-tokenizing all hints for every sample.
                    "raw_hint_token_count": 0,
                    "hints": [],
                    "critiques": [],
                    "step_count": 0,
                    "is_complete": False,
                    "error": None,
                    "tokens_since_boundary": 0,  # Model tokens since last accepted boundary
                    "last_finish_reason": None,  # track final finish reason for diagnostics/dump
                    "context_exhausted": False,  # flag if we hit context budget
                    "first_step_tokens_len": None,  # record first step token length for debugging
                    "_prethink_rollback_used": False,  # allow at most one rollback-to-</think> per sample
                    "_hint_skipped_late_stage": False,
                    "_cf_branch_event_count": 0,  # number of cf events already created for this sample
                }
            )

        if initial_state_overrides is not None:
            try:
                if len(initial_state_overrides) != batch_size:
                    logger.warning(
                        f"[BY_STEP] initial_state_overrides size mismatch: "
                        f"len={len(initial_state_overrides)} vs batch_size={batch_size}; ignoring overrides."
                    )
                else:
                    for b in range(batch_size):
                        override = initial_state_overrides[b]
                        if not isinstance(override, dict):
                            continue
                        for k, v in override.items():
                            if k in ("response_tokens", "loss_masks", "hints", "critiques"):
                                sample_states[b][k] = list(v) if v is not None else []
                            else:
                                sample_states[b][k] = v
                        # Keep response_tokens/loss_masks aligned.
                        rt = list(sample_states[b].get("response_tokens") or [])
                        lm = list(sample_states[b].get("loss_masks") or [])
                        if rt and not lm:
                            lm = [1] * len(rt)
                        if len(rt) != len(lm):
                            keep = min(len(rt), len(lm))
                            rt = rt[:keep]
                            lm = lm[:keep]
                        sample_states[b]["response_tokens"] = rt
                        sample_states[b]["loss_masks"] = lm
                        # Keep fast counters and cached full tokens consistent with overrides.
                        try:
                            sample_states[b]["gen_len"] = int(
                                sum(1 for m in lm if int(m) == 1)
                            )
                            sample_states[b]["hint_len"] = int(
                                sum(1 for m in lm if int(m) == 0)
                            )
                        except Exception:
                            sample_states[b]["gen_len"] = int(
                                sample_states[b].get("gen_len") or 0
                            )
                            sample_states[b]["hint_len"] = int(
                                sample_states[b].get("hint_len") or 0
                            )
                        try:
                            prompt_tokens = list(
                                sample_states[b].get("prompt_tokens") or []
                            )
                            sample_states[b]["prompt_len"] = int(len(prompt_tokens))
                            sample_states[b]["input_tokens"] = prompt_tokens + rt
                        except Exception:
                            pass
                        try:
                            if "raw_hint_token_count" not in override:
                                hints_val = sample_states[b].get("hints") or []
                                if isinstance(hints_val, str):
                                    hints_val = [hints_val]
                                sample_states[b]["raw_hint_token_count"] = int(
                                    sum(
                                        len(
                                            tokenizer.encode(
                                                h, add_special_tokens=False
                                            )
                                        )
                                        for h in hints_val
                                    )
                                    if hints_val
                                    else 0
                                )
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"[BY_STEP] Failed to apply initial_state_overrides: {e}")

        # 5. 监控指标
        metrics = {
            "total_steps": 0,
            "total_interventions": 0,
            "stop_sequence_hits": 0,
            "hint_skipped_late_stage": 0,
            "verifier_skipped_no_insert_anchor": 0,
            "errors": 0,
            "verifier_outputs": 0,
            "verifier_no_valid_decision": 0,
            "verifier_no_valid_decision_final_like": 0,
            "verifier_wait_total": 0,
            "verifier_wait_conf_count": 0,
            "verifier_wait_conf_sum": 0.0,
            "verifier_wait_conf_min": 1.0,
            "verifier_wait_conf_max": 0.0,
            "verifier_wait_conf_missing": 0,
            "verifier_wait_conf_invalid": 0,
            "verifier_wait_blocked_low_conf": 0,
            "verifier_request_candidates": 0,
            "verifier_skipped_low_budget": 0,
            "token_check_interval_effective_sum": 0.0,
            "token_check_interval_effective_count": 0,
        }

        # Verifier PPO trajectories (only intervention steps are used for training).
        prompt_uids = None
        prompt_sample_uids = None
        if getattr(prompts, "non_tensor_batch", None) is not None:
            if "uid" in prompts.non_tensor_batch:
                prompt_uids = prompts.non_tensor_batch["uid"]
            if "sample_uid" in prompts.non_tensor_batch:
                prompt_sample_uids = prompts.non_tensor_batch["sample_uid"]
        verifier_train_prompt_token_ids = []
        verifier_train_response_ids = []
        verifier_train_old_log_probs = []
        verifier_train_parent_uids = []
        verifier_train_parent_sample_uids = []
        verifier_train_step_indices = []
        verifier_train_event_uids = []
        verifier_train_prefix_lens = []
        verifier_train_wait_confidence = []
        verifier_train_wait_avg_logprob = []
        verifier_train_hint_token_counts = []
        verifier_train_prethink_anchor = []
        verifier_train_state_hash_bucket = []

        cf_events = []  # [{event_uid, parent_idx, initial_state_overrides, ...}]

        exp_start = time.time()
        # Dynamic max_steps calculation based on response_budget and token_check_interval
        # This ensures we can generate up to response_budget tokens (not limited by hardcoded max_steps)
        token_check_interval = int(kwargs.get("token_check_interval", 1024))
        min_step_tokens = int(
            kwargs.get("min_step_tokens", token_check_interval)
        )  # FIX: Extract from kwargs
        verifier_skip_budget_tokens = int(kwargs.get("verifier_skip_budget_tokens", 0))
        token_check_late_start_tokens = int(
            kwargs.get("token_check_late_start_tokens", 0)
        )
        token_check_interval_late = int(kwargs.get("token_check_interval_late", 0))
        if verifier_skip_budget_tokens < 0:
            verifier_skip_budget_tokens = 0
        if token_check_late_start_tokens < 0:
            token_check_late_start_tokens = 0
        if token_check_interval_late < 0:
            token_check_interval_late = 0

        # Context budget: prompt + response(gen budget) + hint headroom
        # Reserve space for hints so that hint insertion doesn't cause context_exhausted.
        # Note: response_budget is for model-generated tokens only; hints are extra.
        max_interventions = int(kwargs.get("max_interventions", 5))
        if cf_branch_max_events_per_sample <= 0:
            cf_branch_max_events_per_sample = max_interventions
        cf_branch_prob = max(0.0, min(1.0, float(cf_branch_prob)))
        pending_hint_tail_decode_tokens = int(
            kwargs.get("pending_hint_tail_decode_tokens", 128)
        )
        hint_rollback_window_tokens = int(kwargs.get("hint_rollback_window_tokens", 512))

        conservative_step_tokens = max(1, token_check_interval)
        default_max_steps = (
            response_budget + conservative_step_tokens - 1
        ) // conservative_step_tokens
        # Extra slack for boundary-seeking / hint-commit rounds near the end.
        default_max_steps += max_interventions * 4 + 8
        max_steps = int(kwargs.get("max_steps", default_max_steps))
        verifier_cfg = None
        if getattr(self, "verifier_config", None) is not None:
            verifier_cfg = self.verifier_config
        if verifier_cfg is None:
            verifier_cfg = self.config.get("verifier", {})
        # Prefer using the configured hard cap for hint tokens as the headroom estimate.
        # This avoids under-reserving when hints are long (which can trigger context_exhausted / vLLM max_len errors).
        estimated_hint_tokens = int(verifier_cfg.get("max_hint_tokens", 256))
        hint_headroom = max_interventions * estimated_hint_tokens
        # Total response window for EXP/CONTROL sequences stored in DataProto.
        # This keeps policy-generated tokens capped at response_budget (32k), while allowing hints to extend past it.
        response_total_budget = response_budget + hint_headroom
        context_budget = prompt_budget + response_budget + hint_headroom

        logger.info(
            f"[BY_STEP] max_steps={max_steps} (response_budget={response_budget}, token_check_interval={token_check_interval}, "
            f"min_step_tokens={min_step_tokens}, conservative_step_tokens={conservative_step_tokens}, "
            f"context_budget={context_budget}, hint_headroom={hint_headroom}, "
            f"response_total_budget={response_total_budget})"
        )

        # 6. 批量增量生成循环
        for global_step in range(max_steps):
            metrics["total_steps"] += 1

            # 6.1 收集所有活跃样本（未完成的样本）
            active_indices = [
                b for b in range(batch_size) if not sample_states[b]["is_complete"]
            ]

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
                current_input = state.get("input_tokens")
                if current_input is None:
                    current_input = state["prompt_tokens"] + state["response_tokens"]
                context_remaining = context_budget - len(current_input)

                # Only count model-generated tokens against response budget; hints do not consume budget.
                gen_len = state.get("gen_len")
                if gen_len is None:
                    gen_len = sum(1 for m in state["loss_masks"] if m == 1)
                gen_remaining = response_budget - gen_len

                if context_remaining <= 0:
                    logger.error(
                        f"[BUDGET EXHAUSTED] Sample {b}: context_len={len(current_input)} >= budget={context_budget}; marking complete"
                    )
                    state["is_complete"] = True
                    state["context_exhausted"] = True
                    state["last_finish_reason"] = "context_exhausted"
                    continue

                if gen_remaining <= 0:
                    # Generated-token budget exhausted; stop generating but keep existing response/hints.
                    logger.info(
                        f"[GEN BUDGET EXHAUSTED] Sample {b}: gen_len={gen_len} >= response_budget={response_budget}; marking complete"
                    )
                    state["is_complete"] = True
                    state["last_finish_reason"] = "gen_budget_exhausted"
                    continue

                batch_inputs.append(current_input)
                batch_indices.append(b)
                current_response_lens.append(len(state["response_tokens"]))
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

            # Fixed-step generation (e.g., 2k tokens). We'll trim locally to a natural boundary.
            max_remaining = min(token_check_interval, raw_max_remaining)

            # Debug log: 显示剩余 budget 分布
            if len(gen_remaining_list) > 1:
                logger.info(
                    f"[BY_STEP] Step {global_step}: gen_remaining=[min={min(gen_remaining_list)}, max={max_gen_remaining}], "
                    f"context_remaining=[min={min(context_remaining_list)}, max={max_context_remaining}], "
                    f"max_remaining={max_remaining}"
                )

            if max_remaining <= 0:
                logger.error(
                    f"[BY_STEP] Step {global_step}: max_remaining<=0; breaking."
                )
                break

            # 6.4 批量生成到下一个 boundary
            # NOTE: `batch_inputs` are already unpadded Python `List[int]` built from
            # `prompt_tokens + response_tokens`. Avoid the expensive list->tensor->list
            # roundtrip (and GPU transfers) here.
            batch_inputs_processed = batch_inputs

            try:
                sampling_override = dict(
                    stop=[],
                    stop_token_ids=stop_token_ids or [],
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
                metrics["errors"] += len(batch_indices)
                for b in batch_indices:
                    sample_states[b]["is_complete"] = True
                    sample_states[b]["error"] = str(e)
                continue

            # 6.5 提取新生成的 tokens（批量处理）
            step_responses = []
            step_log_probs = []
            step_finish_reasons = []
            if len(step_output) != len(batch_indices):
                logger.error(
                    f"[MISMATCH] vLLM request count != input count! "
                    f"len(step_output)={len(step_output)}, len(batch_indices)={len(batch_indices)}, "
                    f"len(batch_inputs)={len(batch_inputs)}, global_step={global_step}"
                )
                metrics["errors"] += len(batch_indices)
                for b in batch_indices:
                    sample_states[b]["is_complete"] = True
                    sample_states[b]["error"] = "vllm_request_count_mismatch"
                    sample_states[b]["last_finish_reason"] = (
                        "vllm_request_count_mismatch"
                    )
                continue

            multi_output_prompts = 0
            empty_output_prompts = 0
            for i, output in enumerate(step_output):
                b = batch_indices[i]
                outputs = getattr(output, "outputs", None) or []
                if len(outputs) == 0:
                    empty_output_prompts += 1
                    metrics["errors"] += 1
                    sample_states[b]["is_complete"] = True
                    sample_states[b]["error"] = "vllm_empty_output"
                    sample_states[b]["last_finish_reason"] = "vllm_empty_output"
                    step_responses.append([])
                    step_log_probs.append([])
                    step_finish_reasons.append("vllm_empty_output")
                    continue

                # vLLM should return exactly 1 output per prompt when sampling_params.n=1.
                # Some versions may still return >1 outputs (e.g., stale n/best_of). To
                # keep the training loop stable, we pick the first output and ignore
                # the rest (log once per step).
                if len(outputs) != 1:
                    multi_output_prompts += 1
                out0 = outputs[0]

                response_ids = out0.token_ids
                step_responses.append(response_ids)

                # Extract logprobs (best-effort; can be missing/None depending on config).
                curr_log_prob = []
                try:
                    for j, logprob in enumerate(out0.logprobs or []):
                        curr_log_prob.append(logprob[response_ids[j]].logprob)
                except Exception:
                    pass
                step_log_probs.append(curr_log_prob)

                finish_reason = getattr(out0, "finish_reason", None) or "unknown"
                step_finish_reasons.append(finish_reason)

            if empty_output_prompts > 0 or multi_output_prompts > 0:
                logger.warning(
                    f"[BY_STEP] Step {global_step}: vLLM returned empty/multi outputs "
                    f"(empty={empty_output_prompts}, multi={multi_output_prompts}); "
                    f"using first output per prompt."
                )

            # 6.6 更新每个样本的状态并检测停止
            if len(step_responses) != len(batch_indices):
                logger.error(
                    f"[MISMATCH] vLLM output count != input count! "
                    f"len(step_responses)={len(step_responses)}, len(batch_indices)={len(batch_indices)}, "
                    f"len(batch_inputs)={len(batch_inputs)}, global_step={global_step}"
                )
                metrics["errors"] += len(batch_indices)
                for b in batch_indices:
                    sample_states[b]["is_complete"] = True
                    sample_states[b]["error"] = "vllm_output_count_mismatch"
                    sample_states[b]["last_finish_reason"] = (
                        "vllm_output_count_mismatch"
                    )
                continue
            logger.debug(
                f"[MATCH] vLLM output count matches input count: {len(step_responses)}"
            )

            # 6.6 更新每个样本的状态并检测停止
            verifier_inputs = []
            verifier_input_map = []  # (batch_idx, step_idx, hint_anchor_keep, use_prethink_anchor)

            for i, b in enumerate(batch_indices):
                state = sample_states[b]
                state["_intervened_this_step"] = False
                step_tokens = list(
                    step_responses[i]
                )  # FIX: 确保是 list，vLLM 可能返回 tuple
                finish_reason = step_finish_reasons[i]  # 获取结束原因
                state["last_finish_reason"] = finish_reason  # Track last finish reason
                if state.get("first_step_tokens_len") is None:
                    state["first_step_tokens_len"] = len(step_tokens)

                # 获取该样本的剩余 budget
                gen_len = state.get("gen_len")
                if gen_len is None:
                    gen_len = sum(1 for m in state["loss_masks"] if m == 1)
                try:
                    gen_len = int(gen_len)
                except Exception:
                    gen_len = int(sum(1 for m in state["loss_masks"] if m == 1))
                sample_gen_remaining = response_budget - gen_len

                # 获取该样本的剩余 context budget（FIX: 也需要检查，防止超出 context budget）
                input_tokens = state.get("input_tokens")
                if input_tokens is None:
                    input_tokens = state["prompt_tokens"] + state["response_tokens"]
                current_input_len = len(input_tokens)
                sample_context_remaining = context_budget - current_input_len

                # 获取实际发送给 vLLM 的输入（去 padding 后），用于前缀检测
                sent_input_ids = batch_inputs_processed[i]

                # ========== 前缀检测：防止 vLLM 回显输入 ==========
                if (
                    len(step_tokens) >= len(sent_input_ids)
                    and step_tokens[: len(sent_input_ids)] == sent_input_ids
                ):
                    logger.warning(
                        f"[ECHO DETECTED] Step {global_step}, Sample {b}: vLLM echoed input; stripping prefix len={len(sent_input_ids)}."
                    )
                    new_tokens = step_tokens[len(sent_input_ids) :]
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
                    state["is_complete"] = True
                    state["last_finish_reason"] = "budget_overflow"
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
                    if finish_reason == "length":
                        logger.error(
                            f"[CRITICAL] Step {global_step}, Sample {b}: length stop with 0 new tokens. "
                            f"state_response_len={len(state['response_tokens'])}, budget={response_budget}, "
                            f"total_context={current_input_len}"
                        )

                # 调试日志 - 增强版
                logger.debug(
                    f"Step {global_step}, Sample {b}: step_tokens_len={len(step_tokens)}, new_tokens_len={len(new_tokens)}, "
                    f"finish_reason={finish_reason}, current_input_len={current_input_len}, "
                    f"sample_gen_remaining={sample_gen_remaining}, sample_context_remaining={sample_context_remaining}"
                )

                saw_stop_token = False
                if stop_token_ids_set and new_tokens:
                    stop_rel = None
                    stop_tok = None
                    for j, t in enumerate(new_tokens):
                        if int(t) in stop_token_ids_set:
                            stop_rel = j
                            stop_tok = int(t)
                            break
                    if stop_rel is not None:
                        saw_stop_token = True
                        # Do not keep special stop tokens in the stored response.
                        new_tokens = new_tokens[: int(stop_rel)]
                        logger.info(
                            f"[STOP TOKEN] Step {global_step}, Sample {b}: saw stop_token_id={stop_tok}, truncating output and completing."
                        )

                # Note: Echo handling is done above using the exact input ids sent to vLLM.

                # If we hit the fixed per-step budget (finish_reason==length), roll back to a natural
                # boundary within this chunk to avoid keeping an unfinished trailing fragment.
                if finish_reason == "length" and new_tokens:
                    cut = _find_last_trim_boundary(new_tokens, tokenizer)
                    if cut is not None:
                        new_tokens = new_tokens[: int(cut)]

                # 检查是否生成了新内容
                if len(new_tokens) == 0:
                    if finish_reason == "stop" or saw_stop_token:
                        state["is_complete"] = True
                        state["last_finish_reason"] = "eos"
                        continue
                    # Enhanced diagnostic for empty output - CRITICAL for debugging
                    logger.warning(
                        f"[EMPTY OUTPUT] Step {global_step}, Sample {b}: No new tokens generated! "
                        f"finish_reason={finish_reason}, current_input_len={current_input_len}, "
                        f"max_remaining={max_remaining}, step_tokens_raw_len={len(step_tokens)}"
                    )

                    # Show vLLM output details for debugging
                    if len(step_tokens) > 0:
                        # Show first few tokens to diagnose echo/padding
                        first_tokens = (
                            step_tokens[:10] if len(step_tokens) >= 10 else step_tokens
                        )
                        logger.warning(
                            f"[EMPTY OUTPUT] step_tokens first 10: {first_tokens}"
                        )
                        # Check if it matches prompt (echo case not caught above)
                        if len(step_tokens) <= len(state["prompt_tokens"]):
                            prefix_match = (
                                step_tokens
                                == state["prompt_tokens"][: len(step_tokens)]
                            )
                            logger.warning(
                                f"[EMPTY OUTPUT] Matches prompt prefix? {prefix_match}"
                            )
                    else:
                        # vLLM returned completely empty - this is the critical case
                        logger.error(
                            f"[CRITICAL] Sample {b}: vLLM returned COMPLETELY EMPTY output! "
                            f"finish_reason={finish_reason}, max_remaining={max_remaining}, "
                            f"batch_idx={i}, global_step={global_step}"
                        )

                    state["is_complete"] = True
                    continue

                # 关键：追加到 response_tokens（不是 prompt_tokens）
                state["response_tokens"].extend(new_tokens)
                state["loss_masks"].extend([1] * len(new_tokens))  # Model Gen = 1
                try:
                    state["gen_len"] = int(state.get("gen_len") or 0) + int(
                        len(new_tokens)
                    )
                except Exception:
                    pass
                try:
                    if isinstance(state.get("input_tokens"), list):
                        state["input_tokens"].extend(new_tokens)
                except Exception:
                    pass
                state["step_count"] += 1
                state["tokens_since_boundary"] += len(new_tokens)

                # 记录本步是否在本轮后应结束；先不 continue，给短输出保留一次 Verifier 机会。
                pending_complete_reason = None
                if truncated:
                    if effective_budget == sample_gen_remaining:
                        pending_complete_reason = "gen_budget_exhausted"
                    else:
                        pending_complete_reason = "context_budget_exhausted"
                elif _find_first_eos_index(new_tokens, eos_token_ids) is not None:
                    pending_complete_reason = "eos"
                elif finish_reason == "stop" or saw_stop_token:
                    pending_complete_reason = "eos"
                # NOTE: In fixed-chunk mode we do not pass stop_sequences to vLLM. Therefore
                # finish_reason == "stop" is only expected from stop_token_ids (EOS/pad/<|im_start|>).
                # Hints do not consume the response budget; only model-generated tokens count.
                elif int(state.get("gen_len") or 0) >= response_budget:
                    pending_complete_reason = "gen_budget_exhausted"

                # Boundary detection for step/end markers.
                boundary_hit = False
                if stop_token_ids and any(tok in stop_token_ids for tok in new_tokens):
                    boundary_hit = True
                else:
                    for seq_ids in stop_seq_token_ids:
                        if (
                            len(state["response_tokens"]) >= len(seq_ids)
                            and state["response_tokens"][-len(seq_ids) :] == seq_ids
                        ):
                            boundary_hit = True
                            break

                # Verifier trigger policy (strict):
                # 1) periodic trigger every min_step_tokens policy tokens;
                # 2) for short outputs near completion, allow one final trigger.
                post_step_gen_len = int(state.get("gen_len") or 0)
                effective_token_check_interval = int(token_check_interval)
                if (
                    token_check_interval_late > 0
                    and token_check_late_start_tokens > 0
                    and post_step_gen_len >= token_check_late_start_tokens
                ):
                    effective_token_check_interval = int(token_check_interval_late)
                periodic_threshold = max(
                    int(min_step_tokens), int(effective_token_check_interval)
                )
                periodic_hit = state["tokens_since_boundary"] >= periodic_threshold
                final_boundary_hit = pending_complete_reason is not None
                should_trigger_verifier = periodic_hit or final_boundary_hit

                # Only request Verifier when trigger condition is met and no pending hint is waiting.
                should_request_verifier = (
                    (not state.get("is_complete", False))
                    and (len(state["hints"]) < intervention_policy.max_interventions)
                    and should_trigger_verifier
                )

                if should_request_verifier:
                    metrics["verifier_request_candidates"] += 1
                    metrics["token_check_interval_effective_sum"] += float(
                        effective_token_check_interval
                    )
                    metrics["token_check_interval_effective_count"] += 1
                    remaining_after_step = max(0, response_budget - post_step_gen_len)
                    if (
                        verifier_skip_budget_tokens > 0
                        and remaining_after_step < verifier_skip_budget_tokens
                    ):
                        metrics["verifier_skipped_low_budget"] += 1
                        state["tokens_since_boundary"] = 0
                        state["_pending_complete_reason"] = pending_complete_reason
                        continue
                    if boundary_hit:
                        metrics["stop_sequence_hits"] += 1
                    # Reset periodic window only when a verifier request is actually issued.
                    state["tokens_since_boundary"] = 0

                    # Prepare Verifier input from the same anchor where hint will be inserted.
                    hint_anchor_keep = len(state["response_tokens"])
                    tail_keep = _compute_tail_rollback_keep_len(
                        state["response_tokens"],
                        tokenizer,
                        hint_rollback_window_tokens,
                    )
                    if tail_keep is not None:
                        hint_anchor_keep = int(tail_keep)

                    # Never roll back past any already-inserted hints; otherwise we would
                    # silently delete prior interventions and break the mapping between
                    # (hints/critiques/event_uid) and the final response.
                    last_hint_keep = 0
                    try:
                        last_hint_keep = _last_hint_end(state.get("loss_masks") or [])
                        if last_hint_keep > int(hint_anchor_keep):
                            hint_anchor_keep = int(last_hint_keep)
                    except Exception:
                        last_hint_keep = 0

                    # If we just hit EOS/stop-token, drop it from the anchor so a hint can
                    # keep generation running (otherwise vLLM may immediately stop / return
                    # empty output on the next step).
                    if pending_complete_reason == "eos":
                        try:
                            if (
                                int(hint_anchor_keep) > 0
                                and int(
                                    state["response_tokens"][int(hint_anchor_keep) - 1]
                                )
                                in stop_token_ids_set
                            ):
                                hint_anchor_keep = max(
                                    int(last_hint_keep), int(hint_anchor_keep) - 1
                                )
                        except Exception:
                            pass

                    use_prethink_anchor = False
                    anchor_tokens = state["response_tokens"][:hint_anchor_keep]
                    # If we've already produced substantive post-</think> content, we should NEVER
                    # insert hints after it. Instead, best-effort roll back to before the last
                    # think-close marker once per sample.
                    if not state.get("_prethink_rollback_used", False):
                        if (
                            _find_think_close_pos(anchor_tokens, tokenizer) is None
                            and _is_post_think_finalized(
                                anchor_tokens,
                                tokenizer,
                                decode_tail_tokens=pending_hint_tail_decode_tokens,
                            )
                        ):
                            prethink_keep = _find_prethink_rollback_keep_len(
                                anchor_tokens, tokenizer
                            )
                            if (
                                prethink_keep is not None
                                and int(prethink_keep) < int(hint_anchor_keep)
                            ):
                                hint_anchor_keep = int(prethink_keep)
                                use_prethink_anchor = True
                                # After rolling back to prethink, re-apply tail rollback within
                                # the new anchor so the Verifier context matches insertion context.
                                tail_keep = _compute_tail_rollback_keep_len(
                                    state["response_tokens"][:hint_anchor_keep],
                                    tokenizer,
                                    hint_rollback_window_tokens,
                                )
                                if tail_keep is not None:
                                    hint_anchor_keep = int(tail_keep)
                                # Re-clamp after the second tail-rollback: the new boundary search
                                # may land *inside* an existing hint span (loss_mask==0) and would
                                # silently truncate already-inserted hints.
                                try:
                                    last_hint_keep = _last_hint_end(
                                        state.get("loss_masks") or []
                                    )
                                    if last_hint_keep > int(hint_anchor_keep):
                                        hint_anchor_keep = int(last_hint_keep)
                                except Exception:
                                    pass

                    # Strong coupling: only request Verifier when a same-round insertion anchor exists.
                    anchor_tokens = state["response_tokens"][:hint_anchor_keep]
                    has_safe_think_anchor = (
                        _find_think_close_pos(anchor_tokens, tokenizer) is not None
                    )
                    is_post_think_finalized = _is_post_think_finalized(
                        anchor_tokens,
                        tokenizer,
                        decode_tail_tokens=pending_hint_tail_decode_tokens,
                    )
                    anchor_insertable = has_safe_think_anchor or (
                        not is_post_think_finalized
                    )
                    if not anchor_insertable:
                        metrics["verifier_skipped_no_insert_anchor"] += 1
                        logger.info(
                            f"[VERIFIER SKIP NO ANCHOR] sample={b}, step={state.get('step_count', -1)}: "
                            "no safe insertion anchor (post-final and no think-close)."
                        )
                        # Still keep completion semantics consistent for this step.
                        state["_pending_complete_reason"] = pending_complete_reason
                        continue

                    verifier_state = {
                        "response_tokens": list(state["response_tokens"][:hint_anchor_keep]),
                        "loss_masks": list(state["loss_masks"][:hint_anchor_keep]),
                    }
                    current_reasoning = tokenizer.decode(
                        verifier_state["response_tokens"], skip_special_tokens=True
                    )

                    # CRITICAL FIX: Use training data format for Verifier input
                    # Format matches format_training_data.py: system -> user (intervene_prompt + question + student response)
                    # by_step mode: current_reasoning is the partial response so far
                    verifier_question = state.get(
                        "verifier_question", state.get("prompt_text", "")
                    )
                    user_content = build_verifier_user_prompt(
                        verifier_question, current_reasoning
                    )
                    verifier_input = user_content  # Will be formatted with system message in _run_verifier_inference
                    verifier_inputs.append(verifier_input)
                    verifier_input_map.append(
                        (
                            b,
                            state["step_count"],
                            int(hint_anchor_keep),
                            bool(use_prethink_anchor),
                        )
                    )

                # Verifier 输入准备完后再执行完成标记，确保短输出有机会被介入一次。
                state["_pending_complete_reason"] = pending_complete_reason

            # 6.7 批量调用 Verifier
            if verifier_inputs:
                decisions, verifier_traj = self._run_verifier_inference(
                    verifier_inputs,
                    tokenizer,
                    idx.device,
                    exp_kwargs,
                    return_trajectories=True,
                )

                # Track verifier output format quality. "Valid decision" means the output
                # contains a parsed <GO>/<WAIT> decision tag (in the proper decision scope).
                try:
                    metrics["verifier_outputs"] += int(len(decisions))
                    for d in decisions:
                        if d.get("decision_tag") in ("GO", "WAIT"):
                            continue
                        metrics["verifier_no_valid_decision"] += 1
                        crit = str(d.get("critique") or "")
                        low = crit.lower()
                        if ("final answer" in low) or ("\\boxed" in crit):
                            metrics["verifier_no_valid_decision_final_like"] += 1
                except Exception:
                    pass

                for i, (b, step_idx, hint_anchor_keep, use_prethink_anchor) in enumerate(
                    verifier_input_map
                ):
                    state = sample_states[b]
                    decision = decisions[i]
                    if decision.get("hint"):
                        decision["hint"] = self._truncate_verifier_hint_tokens(
                            decision["hint"], tokenizer
                        )

                    # 记录 critique
                    critique = decision.get("critique", decision.get("hint", ""))
                    state["critiques"].append(critique)

                    if decision.get("action") == "Intervene":
                        metrics["verifier_wait_total"] += 1
                        wait_conf_raw = decision.get("wait_confidence", None)
                        if wait_conf_raw is None:
                            metrics["verifier_wait_conf_missing"] += 1
                        else:
                            try:
                                wait_conf = float(wait_conf_raw)
                                metrics["verifier_wait_conf_count"] += 1
                                metrics["verifier_wait_conf_sum"] += wait_conf
                                metrics["verifier_wait_conf_min"] = min(
                                    metrics["verifier_wait_conf_min"], wait_conf
                                )
                                metrics["verifier_wait_conf_max"] = max(
                                    metrics["verifier_wait_conf_max"], wait_conf
                                )
                                if (
                                    confidence_threshold_value > 0
                                    and wait_conf < confidence_threshold_value
                                ):
                                    metrics["verifier_wait_blocked_low_conf"] += 1
                            except Exception:
                                metrics["verifier_wait_conf_invalid"] += 1

                    # 使用 InterventionPolicy 判断是否应该干预
                    if intervention_policy.should_intervene(state, decision):
                        hint = decision["hint"]
                        hint_applied = False
                        parent_uid = (
                            str(prompt_uids[b]) if prompt_uids is not None else str(b)
                        )
                        parent_sample_uid = (
                            str(prompt_sample_uids[b])
                            if prompt_sample_uids is not None
                            else str(b)
                        )
                        hint_idx = len(state.get("hints", []) or [])
                        event_uid = f"{parent_sample_uid}:{hint_idx}"

                        # Restore to verifier anchor first, so generated hint context and
                        # insertion position are strictly aligned.
                        _rollback_state_to_keep_len(state, int(hint_anchor_keep))
                        if use_prethink_anchor:
                            state["_prethink_rollback_used"] = True

                        # If anchor is still post-final and not a safe think-close insertion,
                        # allow one best-effort rollback to prethink boundary.
                        if (
                            state.get("_pending_complete_reason") is not None
                            and (_find_think_close_pos(state["response_tokens"], tokenizer) is None)
                            and _is_post_think_finalized(
                                state["response_tokens"],
                                tokenizer,
                                decode_tail_tokens=pending_hint_tail_decode_tokens,
                            )
                        ):
                            _rollback_prethink_once(state, tokenizer)

                        # Snapshot the exact prefix state used for insertion. This is the counterfactual
                        # branch start state (no-hint continuation).
                        cf_prefix_response_tokens = list(state.get("response_tokens") or [])
                        cf_prefix_loss_masks = list(state.get("loss_masks") or [])
                        cf_prefix_hints = list(state.get("hints") or [])
                        cf_prefix_prethink_used = bool(state.get("_prethink_rollback_used", False))
                        prefix_len = len(cf_prefix_response_tokens)
                        state_hash_bucket = 0
                        if cf_branch_state_hash_mod > 0:
                            try:
                                state_hash_bucket = int(
                                    _hash_token_sequence(cf_prefix_response_tokens)
                                    % int(cf_branch_state_hash_mod)
                                )
                            except Exception:
                                state_hash_bucket = 0
                        try:
                            tail = tokenizer.decode(
                                state["response_tokens"][
                                    -int(pending_hint_tail_decode_tokens) :
                                ],
                                skip_special_tokens=True,
                            )
                        except Exception:
                            tail = ""
                        prefix = _hint_prefix_for_tail(tail)
                        hint_text = f"{prefix}{(hint or '').strip()}\n\n"
                        hint_tokens = tokenizer.encode(
                            hint_text, add_special_tokens=False
                        )
                        hint_token_count = int(len(hint_tokens) if hint_tokens else 0)
                        if hint_tokens:
                            inserted = _insert_hint_tokens(
                                state, hint_tokens, tokenizer, eos_token_ids
                            )
                            if not inserted and state.pop("_hint_skipped_late_stage", False):
                                metrics["hint_skipped_late_stage"] += 1
                                logger.info(
                                    f"[HINT SKIP LATE] sample={b}, step={step_idx}: skip hint insertion after final answer marker."
                                )
                            hint_applied = bool(inserted)

                        if hint_applied:
                            state["hints"].append(hint)
                            try:
                                state["raw_hint_token_count"] = int(
                                    state.get("raw_hint_token_count") or 0
                                ) + int(
                                    len(
                                        tokenizer.encode(
                                            hint, add_special_tokens=False
                                        )
                                    )
                                )
                            except Exception:
                                pass
                            metrics["total_interventions"] += 1
                            state["_intervened_this_step"] = True
                            state["tokens_since_boundary"] = 0

                            # If the sample was about to finish (EOS/stop), keep it running so the hint
                            # can affect the continuation.
                            state["_pending_complete_reason"] = None

                            # Record counterfactual branch request (no-hint continuation) for this event.
                            if cf_branching:
                                try:
                                    if (
                                        state.get("_cf_branch_event_count", 0)
                                        < cf_branch_max_events_per_sample
                                    ) and (np.random.rand() < cf_branch_prob):
                                        state["_cf_branch_event_count"] = int(
                                            state.get("_cf_branch_event_count", 0)
                                        ) + 1
                                        cf_events.append(
                                            {
                                                "event_uid": str(event_uid),
                                                "parent_idx": int(b),
                                                "parent_uid": str(parent_uid),
                                                "parent_sample_uid": str(parent_sample_uid),
                                                "step_idx": int(step_idx),
                                                "prefix_len": int(prefix_len),
                                                "wait_confidence": decision.get(
                                                    "wait_confidence", None
                                                ),
                                                "wait_avg_logprob": decision.get(
                                                    "wait_avg_logprob", None
                                                ),
                                                "hint_token_count": int(hint_token_count),
                                                "prethink_anchor": bool(use_prethink_anchor),
                                                "state_hash_bucket": int(state_hash_bucket),
                                                "initial_state": {
                                                    "response_tokens": cf_prefix_response_tokens,
                                                    "loss_masks": cf_prefix_loss_masks,
                                                    "hints": cf_prefix_hints,
                                                    "_prethink_rollback_used": bool(
                                                        cf_prefix_prethink_used
                                                    ),
                                                },
                                            }
                                        )
                                except Exception as e:
                                    logger.warning(
                                        f"[CF_BRANCH] Failed to record cf event (b={b}, step_idx={step_idx}): {e}"
                                    )

                        # Record verifier trajectories for training (prompt + full response + old logprobs).
                        try:
                            if verifier_traj and hint_applied and collect_verifier_trajectories:
                                prompt_token_ids = verifier_traj["prompt_token_ids"][i]
                                response_ids = (
                                    verifier_traj["responses"][i].detach().to("cpu")
                                )
                                old_log_probs = (
                                    verifier_traj["old_log_probs"][i]
                                    .detach()
                                    .to("cpu")
                                    .to(torch.float32)
                                )

                                verifier_train_prompt_token_ids.append(prompt_token_ids)
                                verifier_train_response_ids.append(response_ids)
                                verifier_train_old_log_probs.append(old_log_probs)
                                verifier_train_parent_uids.append(parent_uid)
                                verifier_train_parent_sample_uids.append(
                                    parent_sample_uid
                                )
                                verifier_train_step_indices.append(int(step_idx))
                                verifier_train_event_uids.append(str(event_uid))
                                verifier_train_prefix_lens.append(int(prefix_len))
                                verifier_train_wait_confidence.append(
                                    decision.get("wait_confidence", None)
                                )
                                verifier_train_wait_avg_logprob.append(
                                    decision.get("wait_avg_logprob", None)
                                )
                                verifier_train_hint_token_counts.append(
                                    int(hint_token_count)
                                )
                                verifier_train_prethink_anchor.append(
                                    bool(use_prethink_anchor)
                                )
                                verifier_train_state_hash_bucket.append(
                                    int(state_hash_bucket)
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to record verifier trajectory (b={b}, step_idx={step_idx}): {e}"
                            )

            # Finalize completion for this step.
            for b in batch_indices:
                st = sample_states[b]
                reason = st.pop("_pending_complete_reason", None)
                intervened = bool(st.pop("_intervened_this_step", False))
                if reason is not None and (not intervened):
                    st["is_complete"] = True
                    st["last_finish_reason"] = reason

        remaining_active = [
            b
            for b in range(batch_size)
            if not sample_states[b].get("is_complete", False)
        ]
        if remaining_active:
            logger.warning(
                f"[BY_STEP] Reached max_steps={max_steps} with active_samples={len(remaining_active)}/{batch_size}; "
                f"responses may be early-stopped (consider increasing max_steps or step-token caps)."
            )
            for b in remaining_active:
                if not sample_states[b].get("last_finish_reason"):
                    sample_states[b]["last_finish_reason"] = "max_steps_exhausted"

        timing_info["exp_gen"] = time.time() - exp_start

        # 7. 记录监控指标
        wait_total = metrics["verifier_wait_total"]
        wait_conf_count = metrics["verifier_wait_conf_count"]
        verifier_request_candidates = metrics["verifier_request_candidates"]
        verifier_skipped_low_budget = metrics["verifier_skipped_low_budget"]
        interval_effective_count = metrics["token_check_interval_effective_count"]
        timing_info["exp_metrics"] = {
            "avg_steps": sum(s["step_count"] for s in sample_states) / batch_size
            if batch_size > 0
            else 0,
            "intervention_rate": metrics["total_interventions"] / batch_size
            if batch_size > 0
            else 0,
            "intervention_count": metrics["total_interventions"],
            "cf_branching": int(bool(cf_branching)),
            "cf_event_count": int(len(cf_events)),
            "cf_event_per_intervention": float(len(cf_events))
            / float(max(1, metrics["total_interventions"])),
            "stop_sequence_hit_rate": metrics["stop_sequence_hits"]
            / metrics["total_steps"]
            if metrics["total_steps"] > 0
            else 0,
            "error_rate": metrics["errors"] / batch_size if batch_size > 0 else 0,
            "error_count": metrics["errors"],
            "hint_skipped_late_stage": metrics["hint_skipped_late_stage"],
            "verifier_skipped_no_insert_anchor": metrics[
                "verifier_skipped_no_insert_anchor"
            ],
            "confidence_threshold": confidence_threshold_value,
            "verifier_outputs": metrics["verifier_outputs"],
            "verifier_no_valid_decision": metrics["verifier_no_valid_decision"],
            "verifier_no_valid_decision_rate": metrics["verifier_no_valid_decision"]
            / metrics["verifier_outputs"]
            if metrics["verifier_outputs"] > 0
            else 0,
            "verifier_no_valid_decision_final_like": metrics[
                "verifier_no_valid_decision_final_like"
            ],
            "verifier_no_valid_decision_final_like_rate": metrics[
                "verifier_no_valid_decision_final_like"
            ]
            / metrics["verifier_outputs"]
            if metrics["verifier_outputs"] > 0
            else 0,
            "wait_total": wait_total,
            "wait_conf_coverage": wait_conf_count / wait_total if wait_total > 0 else 0,
            "wait_conf_mean": metrics["verifier_wait_conf_sum"] / wait_conf_count
            if wait_conf_count > 0
            else 0,
            "wait_conf_min": metrics["verifier_wait_conf_min"]
            if wait_conf_count > 0
            else 0,
            "wait_conf_max": metrics["verifier_wait_conf_max"]
            if wait_conf_count > 0
            else 0,
            "wait_conf_missing": metrics["verifier_wait_conf_missing"],
            "wait_conf_invalid": metrics["verifier_wait_conf_invalid"],
            "wait_blocked_low_conf": metrics["verifier_wait_blocked_low_conf"],
            "wait_blocked_low_conf_rate": metrics["verifier_wait_blocked_low_conf"]
            / wait_total
            if wait_total > 0
            else 0,
            "verifier_skipped_low_budget_count": verifier_skipped_low_budget,
            "verifier_skipped_low_budget_ratio": verifier_skipped_low_budget
            / verifier_request_candidates
            if verifier_request_candidates > 0
            else 0,
            "token_check_interval_effective_mean": metrics[
                "token_check_interval_effective_sum"
            ]
            / interval_effective_count
            if interval_effective_count > 0
            else float(token_check_interval),
        }
        logger.info(
            f"[BY_STEP] metrics: avg_steps={timing_info['exp_metrics']['avg_steps']:.1f}, "
            f"interventions={metrics['total_interventions']}, "
            f"wait_total={wait_total}, "
            f"wait_blocked_low_conf={metrics['verifier_wait_blocked_low_conf']}, "
            f"wait_conf_mean={timing_info['exp_metrics']['wait_conf_mean']:.4f}, "
            f"hint_skipped_late_stage={metrics['hint_skipped_late_stage']}, "
            f"verifier_skipped_low_budget={verifier_skipped_low_budget}/{verifier_request_candidates}, "
            f"verifier_skipped_no_insert_anchor={metrics['verifier_skipped_no_insert_anchor']}, "
            f"errors={metrics['errors']}"
        )

        # 8. 数据后处理：构建完整的 DataProto（关键修正）
        exp_input_ids_list = []
        exp_attention_mask_list = []
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
            total_len = prompt_budget + response_total_budget

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
                prompt_seg_ids = torch.full(
                    (prompt_budget,),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=idx.device,
                )
                prompt_seg_att = torch.zeros(
                    (prompt_budget,), dtype=torch.long, device=idx.device
                )
            else:
                prompt_seg_ids = pad_sequence_to_length(
                    prompt_ids.unsqueeze(0),
                    prompt_budget,
                    self.pad_token_id,
                    left_pad=True,
                ).squeeze(0)
                prompt_seg_att = pad_sequence_to_length(
                    prompt_att.unsqueeze(0), prompt_budget, 0, left_pad=True
                ).squeeze(0)

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

            # The stored response window is fixed-length (response_total_budget).
            # Policy-generated tokens are still capped by response_budget during rollout; hints can extend past it.
            # If we ever exceed response_total_budget (shouldn't happen because hints are capped), keep the tail.
            if len(response_tokens) > response_total_budget:
                logger.warning(
                    f"[RESPONSE TRUNCATION] Sample {b}: response_len={len(response_tokens)} > response_total_budget={response_total_budget}; keeping tail"
                )
                response_tokens = response_tokens[-response_total_budget:]
                response_loss_masks = response_loss_masks[-response_total_budget:]

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
                resp_seg_ids = torch.full(
                    (response_total_budget,),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=idx.device,
                )
                resp_seg_att = torch.zeros(
                    (response_total_budget,), dtype=torch.long, device=idx.device
                )
                resp_seg_loss = torch.zeros(
                    (response_total_budget,), dtype=torch.long, device=idx.device
                )
            else:
                resp_seg_ids = pad_sequence_to_length(
                    resp_ids.unsqueeze(0), response_total_budget, self.pad_token_id
                ).squeeze(0)
                resp_seg_att = pad_sequence_to_length(
                    resp_att.unsqueeze(0), response_total_budget, 0
                ).squeeze(0)
                resp_seg_loss = pad_sequence_to_length(
                    resp_loss.unsqueeze(0), response_total_budget, 0
                ).squeeze(0)

            # 8.3 Assemble full sequence (fixed boundary at prompt_budget)
            full_ids = torch.cat([prompt_seg_ids, resp_seg_ids], dim=0)
            att_mask_long = torch.cat([prompt_seg_att, resp_seg_att], dim=0)
            loss_mask_long = torch.cat(
                [
                    torch.zeros((prompt_budget,), dtype=torch.long, device=idx.device),
                    resp_seg_loss,
                ],
                dim=0,
            )

            # Store masks as int8 to reduce Ray object size (cast back to long when needed).
            att_mask = att_mask_long.to(torch.int8)
            loss_mask = loss_mask_long.to(torch.int8)

            # Track last valid position in the response segment (hints included but capped by response_budget).
            exp_last_valid_pos_list.append(
                int(min(len(response_tokens), response_total_budget))
            )

            exp_input_ids_list.append(full_ids)
            exp_attention_mask_list.append(att_mask)
            exp_loss_mask_list.append(loss_mask)
            exp_responses_list.append(resp_seg_ids)

            # Reconstruct hints directly from `loss_masks` (0-spans) to guarantee that
            # dumped hints always match the final token sequence (no "hint lost").
            try:
                reconstructed_hints = []
                in_span = False
                span_start = 0
                for j, m in enumerate(response_loss_masks):
                    if int(m) == 0:
                        if not in_span:
                            span_start = j
                            in_span = True
                    elif in_span:
                        hint_text = tokenizer.decode(
                            response_tokens[span_start:j], skip_special_tokens=True
                        ).strip()
                        if hint_text:
                            reconstructed_hints.append(hint_text)
                        in_span = False
                if in_span:
                    hint_text = tokenizer.decode(
                        response_tokens[span_start:], skip_special_tokens=True
                    ).strip()
                    if hint_text:
                        reconstructed_hints.append(hint_text)
                state["hints"] = reconstructed_hints
            except Exception:
                pass

            all_hints.append("\n".join(state["hints"]))
            all_critiques.append(
                "\n---\n".join(state["critiques"])
                if state["critiques"]
                else "No interventions"
            )

        # 9. Stack Tensors
        exp_input_ids = torch.stack(exp_input_ids_list)  # [batch_size, total_len]
        exp_attention_mask = torch.stack(exp_attention_mask_list)
        exp_loss_mask = torch.stack(exp_loss_mask_list)  # 关键新增
        exp_responses = torch.stack(exp_responses_list)  # [batch_size, response_length]

        # 10. 构建完整的 DataProto（参考现有实现）
        exp_last_valid_pos = torch.tensor(
            exp_last_valid_pos_list, dtype=torch.long, device=idx.device
        )
        from tensordict import TensorDict

        batch_dict = {
            "exp_input_ids": exp_input_ids,
            "exp_attention_mask": exp_attention_mask,
            "exp_loss_mask": exp_loss_mask,  # 关键新增
            "exp_responses": exp_responses,
            "exp_last_valid_pos": exp_last_valid_pos,
        }

        if stream_mode in ("both", "control"):
            # Control stream data
            # Pad to match exp response window so Co-GRPO can mix streams without shape mismatch.
            if control_responses is None:
                raise RuntimeError(
                    "stream_mode includes control but control_responses is None"
                )
            if control_log_probs is None:
                raise RuntimeError(
                    "stream_mode includes control but control_log_probs is None"
                )

            control_responses = pad_sequence_to_length(
                control_responses, response_total_budget, self.pad_token_id
            )
            # control_log_probs uses -1 padding in other helpers; keep 0 for masked tokens.
            control_log_probs = pad_sequence_to_length(
                control_log_probs, response_total_budget, 0
            )

            control_seq = torch.cat([idx, control_responses], dim=-1)
            control_att_long = torch.cat(
                [
                    attention_mask,
                    (control_responses != self.pad_token_id).to(attention_mask.dtype),
                ],
                dim=-1,
            )
            control_att = control_att_long.to(torch.int8)
            control_last_valid_pos = (
                control_att_long[:, -response_total_budget:].sum(dim=1)
            ).to(torch.long)

            batch_dict.update(
                {
                    "control_input_ids": control_seq,
                    "control_attention_mask": control_att,
                    "control_rollout_log_probs": control_log_probs,
                    "control_last_valid_pos": control_last_valid_pos,
                    "control_responses": control_responses,  # Add responses separately for trainer
                }
            )

        # Provide per-stream response masks to avoid trainer-side recomputation (and to correctly ignore PADs).
        # Combine: before-eos mask AND non-pad mask.
        from verl.utils.torch_functional import get_response_mask

        exp_response_mask = get_response_mask(
            exp_responses, eos_token=eos_token_id, dtype=exp_attention_mask.dtype
        )
        exp_response_mask = exp_response_mask * (exp_responses != self.pad_token_id).to(
            exp_attention_mask.dtype
        )
        batch_dict["exp_response_mask"] = exp_response_mask
        if stream_mode in ("both", "control"):
            control_response_mask = get_response_mask(
                control_responses, eos_token=eos_token_id, dtype=control_att.dtype
            )
            control_response_mask = control_response_mask * (
                control_responses != self.pad_token_id
            ).to(control_att.dtype)
            batch_dict["control_response_mask"] = control_response_mask

        # 11. 构建 TensorDict batch
        batch = TensorDict(batch_dict, batch_size=batch_size)

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

        # Minimal per-sample intervention metadata (always keep; used for reward/cost/dump).
        non_tensor_batch.update(
            {
                "hints": np.array(all_hints, dtype=object),
                "critiques": np.array(all_critiques, dtype=object),
                "num_interventions": np.array(
                    [len(state["hints"]) for state in sample_states],
                    dtype=np.int32,
                ),
                "hint_token_counts": np.array(
                    [
                        int(
                            state.get("raw_hint_token_count")
                            if state.get("raw_hint_token_count") is not None
                            else (
                                sum(
                                    len(tokenizer.encode(h, add_special_tokens=False))
                                    for h in state["hints"]
                                )
                                if state["hints"]
                                else 0
                            )
                        )
                        for state in sample_states
                    ],
                    dtype=np.int32,
                ),
                # Extra diagnostics for dump/offline analysis
                "prompt_len": np.array(
                    [len(state["prompt_tokens"]) for state in sample_states],
                    dtype=np.int32,
                ),
                "response_len": np.array(
                    [len(state["response_tokens"]) for state in sample_states],
                    dtype=np.int32,
                ),
                "gen_len": np.array(
                    [
                        int(
                            state.get("gen_len")
                            if state.get("gen_len") is not None
                            else sum(1 for m in state["loss_masks"] if m == 1)
                        )
                        for state in sample_states
                    ],
                    dtype=np.int32,
                ),
                "hint_len": np.array(
                    [
                        int(
                            state.get("hint_len")
                            if state.get("hint_len") is not None
                            else sum(1 for m in state["loss_masks"] if m == 0)
                        )
                        for state in sample_states
                    ],
                    dtype=np.int32,
                ),
                "last_finish_reason": np.array(
                    [state.get("last_finish_reason") for state in sample_states],
                    dtype=object,
                ),
                "context_exhausted": np.array(
                    [state.get("context_exhausted", False) for state in sample_states],
                    dtype=np.bool_,
                ),
                "first_step_tokens_len": np.array(
                    [state.get("first_step_tokens_len") for state in sample_states],
                    dtype=object,
                ),
            }
        )

        # Control stream diagnostics (used to compare truncation between exp vs control).
        # Note: exp-side metadata above comes from `sample_states`, which tracks the EXP stream.
        if (
            stream_mode in ("both", "control")
            and control_responses is not None
            and control_response_lens is not None
        ):
            try:
                n_ctrl = int(len(control_response_lens))
                prompt_lens = control_prompt_lens
                if prompt_lens is None or int(len(prompt_lens)) != n_ctrl:
                    prompt_lens = np.zeros((n_ctrl,), dtype=np.int32)

                finish_reasons = control_finish_reasons
                if finish_reasons is None or int(len(finish_reasons)) != n_ctrl:
                    finish_reasons = ["unknown"] * n_ctrl

                non_tensor_batch.update(
                    {
                        "control_prompt_len": np.asarray(prompt_lens, dtype=np.int32),
                        "control_response_len": np.asarray(
                            control_response_lens, dtype=np.int32
                        ),
                        "control_gen_len": np.asarray(
                            control_response_lens, dtype=np.int32
                        ),
                        "control_hint_len": np.zeros((n_ctrl,), dtype=np.int32),
                        "control_last_finish_reason": np.asarray(
                            finish_reasons, dtype=object
                        ),
                        "control_context_exhausted": np.zeros((n_ctrl,), dtype=np.bool_),
                        "control_first_step_tokens_len": np.asarray(
                            control_response_lens, dtype=np.int32
                        ),
                    }
                )
            except Exception:
                pass

        # Optional decoded prompt text for quick debugging (responses are decoded in trainer dumps).
        if build_decoded_fields:
            non_tensor_batch["exp_prompt_text"] = np.array(
                [state.get("prompt_text", "") for state in sample_states],
                dtype=object,
            )

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

        cf_control_batch = None
        if cf_branching and cf_events:
            try:
                t0 = time.time()
                # Expand each event K times to reduce baseline variance:
                #   R0(event) = mean_k R0_k
                expanded_events = cf_events
                if cf_branch_k > 1:
                    expanded_events = []
                    for e in cf_events:
                        expanded_events.extend([e] * int(cf_branch_k))

                cf_parent_indices = [int(e.get("parent_idx", 0)) for e in expanded_events]
                cf_prompts = prompts.select_idxs(cf_parent_indices)
                # Keep the original (repeated) batch index for joining reward metadata in trainer.
                # NOTE: Generation prompts passed into rollout do not carry reward metadata
                # (data_source/reward_model/extra_info). Trainer must re-attach them using this index.
                cf_prompts.non_tensor_batch["cf_parent_idx"] = np.array(
                    cf_parent_indices, dtype=np.int32
                )
                cf_prompts.non_tensor_batch["cf_event_uid"] = np.array(
                    [str(e.get("event_uid", "")) for e in expanded_events], dtype=object
                )
                cf_prompts.non_tensor_batch["cf_parent_uid"] = np.array(
                    [str(e.get("parent_uid", "")) for e in expanded_events], dtype=object
                )
                cf_prompts.non_tensor_batch["cf_parent_sample_uid"] = np.array(
                    [str(e.get("parent_sample_uid", "")) for e in expanded_events], dtype=object
                )
                cf_prompts.non_tensor_batch["cf_step_idx"] = np.array(
                    [int(e.get("step_idx", 0)) for e in expanded_events], dtype=np.int32
                )

                initial_overrides = [
                    dict(e.get("initial_state") or {}) for e in expanded_events
                ]

                # Counterfactual control rollouts: start from the exact insertion prefix,
                # do NOT apply the current hint, and continue with the same verifier policy.
                cf_control_batch = self.dual_stream_rollout(
                    cf_prompts,
                    intervention_mode="by_step",
                    stream_mode="exp",
                    token_check_interval=token_check_interval,
                    min_step_tokens=min_step_tokens,
                    max_interventions=max_interventions,
                    confidence_threshold=confidence_threshold_value,
                    pending_hint_tail_decode_tokens=pending_hint_tail_decode_tokens,
                    hint_rollback_window_tokens=hint_rollback_window_tokens,
                    initial_state_overrides=initial_overrides,
                    cf_branching=False,
                    collect_verifier_trajectories=False,
                    build_decoded_fields=False,
                )
                timing_info.setdefault("exp_metrics", {})[
                    "cf_control_gen_s"
                ] = float(time.time() - t0)
                if (
                    cf_control_batch is not None
                    and len(cf_control_batch) > 0
                    and cf_branch_reward_tail_tokens > 0
                ):
                    try:
                        t_shrink = time.time()
                        cf_control_batch = _build_cf_reward_tail_batch(
                            cf_control_batch,
                            pad_token_id=self.pad_token_id,
                            tail_tokens=cf_branch_reward_tail_tokens,
                        )
                        timing_info.setdefault("exp_metrics", {})[
                            "cf_control_shrink_s"
                        ] = float(time.time() - t_shrink)
                    except Exception as e:
                        logger.warning(
                            f"[CF_BRANCH] Failed to shrink cf_control_batch for reward: {e}"
                        )
            except Exception as e:
                logger.warning(f"[CF_BRANCH] Failed to generate cf_control_batch: {e}")
                cf_control_batch = None

        # 13. 释放 KV Cache / offload executor (vLLM version compatible).
        if hasattr(self.inference_engine, "sleep") or hasattr(
            self.inference_engine, "free_cache_engine"
        ):
            try:
                if getattr(self, "enable_kv_cache_optimization", False):
                    _best_effort_reset_prefix_cache(self.inference_engine)
                if hasattr(self.inference_engine, "sleep"):
                    self.inference_engine.sleep(level=1)
                else:
                    self.inference_engine.free_cache_engine()
            except Exception:
                pass

        # 14. 添加 timing 信息到 meta_info
        timing_info["total"] = time.time() - start_time
        meta_info = prompts.meta_info.copy()
        meta_info["timing"] = timing_info
        if cf_control_batch is not None and len(cf_control_batch) > 0:
            meta_info["cf_control_batch"] = cf_control_batch

        # Build verifier training batch (separate DataProto) to avoid mixing batch sizes.
        if verifier_train_prompt_token_ids:
            verifier_cfg = self.config.get("verifier", {})
            verifier_temperature = float(verifier_cfg.get("temperature", 1.0))
            if verifier_temperature <= 0:
                verifier_temperature = 1.0

            verifier_prompt_len = max(len(p) for p in verifier_train_prompt_token_ids)
            verifier_prompt_ids = pad_2d_list_to_length(
                verifier_train_prompt_token_ids,
                self.pad_token_id,
                max_length=verifier_prompt_len,
                left=True,
            ).to("cpu")
            verifier_prompt_att = (verifier_prompt_ids != self.pad_token_id).to(
                torch.long
            )

            verifier_responses = torch.stack(verifier_train_response_ids, dim=0).to(
                "cpu"
            )
            verifier_old_log_probs = (
                torch.stack(verifier_train_old_log_probs, dim=0)
                .to("cpu")
                .to(torch.float32)
            )
            verifier_response_att = (verifier_responses != self.pad_token_id).to(
                torch.long
            )

            verifier_input_ids = torch.cat(
                [verifier_prompt_ids, verifier_responses], dim=1
            )
            verifier_attention_mask = torch.cat(
                [verifier_prompt_att, verifier_response_att], dim=1
            )
            verifier_position_ids = (
                verifier_attention_mask.cumsum(dim=1) - 1
            ) * verifier_attention_mask

            verifier_batch = TensorDict(
                {
                    "input_ids": verifier_input_ids,
                    "attention_mask": verifier_attention_mask,
                    "position_ids": verifier_position_ids,
                    "responses": verifier_responses,
                    "old_log_probs": verifier_old_log_probs,
                },
                batch_size=verifier_input_ids.size(0),
            )

            verifier_non_tensor_batch = {
                "parent_uid": np.array(verifier_train_parent_uids, dtype=object),
                "parent_sample_uid": np.array(
                    verifier_train_parent_sample_uids, dtype=object
                ),
                "step_idx": np.array(verifier_train_step_indices, dtype=np.int32),
                "event_uid": np.array(verifier_train_event_uids, dtype=object),
                "prefix_len": np.array(verifier_train_prefix_lens, dtype=np.int32),
                "wait_confidence": np.array(
                    [
                        float(x) if x is not None else float("nan")
                        for x in verifier_train_wait_confidence
                    ],
                    dtype=np.float32,
                ),
                "wait_avg_logprob": np.array(
                    [
                        float(x) if x is not None else float("nan")
                        for x in verifier_train_wait_avg_logprob
                    ],
                    dtype=np.float32,
                ),
                "hint_token_count": np.array(
                    verifier_train_hint_token_counts, dtype=np.int32
                ),
                "prethink_anchor": np.array(
                    verifier_train_prethink_anchor, dtype=np.bool_
                ),
                "state_hash_bucket": np.array(
                    verifier_train_state_hash_bucket, dtype=np.int32
                ),
            }
            verifier_meta_info = {
                "temperature": verifier_temperature,
                "verifier_prompt_len": int(verifier_prompt_len),
                "verifier_response_len": int(verifier_responses.size(1)),
            }
            meta_info["verifier_batch"] = DataProto(
                batch=verifier_batch,
                non_tensor_batch=verifier_non_tensor_batch,
                meta_info=verifier_meta_info,
            )

        return DataProto(
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info
        )


class vLLMAsyncRollout(vLLMRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(
        self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs
    ):
        # Skip vLLMRollout.__init__ to avoid building a local LLM engine.
        BaseRollout.__init__(self)
        self.config = config
        self.tokenizer = tokenizer
        assert not (not config.enforce_eager and config.free_cache_engine), (
            "disable CUDA graph (enforce_eager = False) if free cache engine"
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size,
                    num_tp_per_train_tp=num_tp_per_train_tp,
                )
            else:
                vllm_ps.initialize_model_parallel(
                    tensor_model_parallel_size=tensor_parallel_size
                )

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = (
                    model_hf_config.llm_config.max_position_embeddings
                )
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = (
                    model_hf_config.text_config.max_position_embeddings
                )
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert (
                max_position_embeddings >= config.prompt_length + config.response_length
            ), "model context length should be greater than total sequence length"

        max_model_len = int(
            config.max_model_len or config.prompt_length + config.response_length
        )
        if (
            max_num_batched_tokens < max_model_len
            and self.config.enable_chunked_prefill
        ):
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, "
                "please increase max_num_batched_tokens or disable chunked prefill"
            )

        lora_kwargs = kwargs.pop("lora_kwargs", {}) or {}
        lora_kwargs = {k: v for k, v in dict(lora_kwargs).items() if v is not None}
        if lora_kwargs.get("enable_lora", False) and "max_loras" not in lora_kwargs:
            lora_kwargs["max_loras"] = 1
        self.lora_kwargs = lora_kwargs
        verifier_lora_name = kwargs.pop("verifier_lora_name", None)
        verifier_lora_path = kwargs.pop("verifier_lora_path", None)
        self.enable_kv_cache_optimization = _as_bool(
            kwargs.pop("enable_kv_cache_optimization", True)
        )

        sampling_kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
        )
        has_stop_params = (
            config.get("stop", None) is not None
            or config.get("stop_token_ids", None) is not None
        )
        if vllm_version != "0.3.1":
            sampling_kwargs["detokenize"] = has_stop_params
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                sampling_kwargs[k] = config.get(k)
        if (
            "stop_token_ids" in sampling_kwargs
            and sampling_kwargs["stop_token_ids"] is not None
            and not isinstance(sampling_kwargs["stop_token_ids"], list)
        ):
            sampling_kwargs["stop_token_ids"] = list(sampling_kwargs["stop_token_ids"])
        self.sampling_params = SamplingParams(**sampling_kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        if verifier_lora_name is not None:
            self.verifier_lora_name = verifier_lora_name
        else:
            self.verifier_lora_name = self.config.get(
                "verifier_lora_name", "verifier_lora"
            )
        self.verifier_lora_path = (
            verifier_lora_path
            if verifier_lora_path
            else self.config.get("verifier", {}).get("lora_path", None)
        )
        self.verifier_lora_int_id = None

        # Engine is deferred to be initialized in init_worker.
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

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
        self._load_verifier_lora_if_needed()

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
        exp_kwargs: Dict[str, Any],
        return_trajectories: bool = False,
    ) -> Union[List[Dict[str, Any]], tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        批量调用 Verifier 进行推理。

        Args:
            verifier_inputs: List[str] - Verifier 输入文本列表
            tokenizer: Tokenizer instance
            idx_device: Device for tensors
            exp_kwargs: Sampling parameters

        Returns:
            If return_trajectories is False:
                List[dict]: 每个输入的决策结果，包含 'action', 'hint', 'critique'
            If return_trajectories is True:
                Tuple[List[dict], Dict[str, Any]]: (decisions, trajectories)
        """
        if not verifier_inputs:
            return ([], {}) if return_trajectories else []

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
                    {"role": "user", "content": verifier_prompt},
                ]
                chat_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
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
                verifier_idx_list.append(
                    _pre_process_inputs(self.pad_token_id, verifier_input_ids[i])
                )

            # 准备 LoRA requests
            verifier_lora_requests = None
            if self.verifier_lora_int_id is not None:
                verifier_lora_requests = [
                    LoRARequest(
                        lora_name=self.verifier_lora_name,
                        lora_int_id=self.verifier_lora_int_id,
                        lora_path=getattr(self, "verifier_lora_path", None),
                    )
                    for _ in range(len(verifier_inputs))
                ]

            # Use verifier-specific sampling params.
            verifier_kwargs = exp_kwargs.copy()
            verifier_cfg = self.config.get("verifier", {})
            verifier_temperature = float(verifier_cfg.get("temperature", 1.0))
            if verifier_temperature <= 0:
                verifier_temperature = 1.0
            verifier_kwargs["temperature"] = verifier_temperature
            verifier_kwargs["top_p"] = 1.0
            # Deterministic decode by default (stable tags): sample from top-1.
            verifier_kwargs["top_k"] = 1

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
                verifier_output,
                self.pad_token_id,
                self.config.response_length,
                idx_device,
            )

            # 解析所有 Verifier 输出
            decisions = []
            for i in range(len(verifier_inputs)):
                verifier_response = verifier_responses[i]
                verifier_text = tokenizer.decode(
                    verifier_response.tolist(), skip_special_tokens=True
                )
                decision = self._parse_verifier_decision(verifier_text)
                if decision.get("action") == "Intervene":
                    wait_confidence, wait_avg_logprob = _compute_wait_confidence(
                        verifier_response,
                        verifier_log_probs[i],
                        self.pad_token_id,
                    )
                    if wait_confidence is not None:
                        decision["wait_confidence"] = wait_confidence
                        decision["wait_avg_logprob"] = wait_avg_logprob
                decisions.append(decision)

            if not return_trajectories:
                return decisions

            trajectories = {
                "prompt_token_ids": verifier_idx_list,
                "responses": verifier_responses,
                "old_log_probs": verifier_log_probs,
                "temperature": verifier_temperature,
                "max_new_tokens": int(
                    getattr(
                        self.sampling_params, "max_tokens", self.config.response_length
                    )
                ),
            }
            return decisions, trajectories

        except Exception as e:
            logger.warning(f"Verifier inference failed: {e}")
            import traceback

            logger.warning(traceback.format_exc())
            # 返回默认决策（Pass）
            fallback = [
                {"action": "Pass", "hint": None, "critique": f"[Error: {str(e)}]"}
                for _ in verifier_inputs
            ]
            return (fallback, {}) if return_trajectories else fallback

    @GPUMemoryLogger(role="vllm dual stream rollout spmd", logger=logger)
    @torch.no_grad()
    def dual_stream_rollout(
        self,
        prompts: DataProto,
        intervention_mode: str = "by_step",
        stream_mode: str = "both",
        token_check_interval: int = 2048,
        **kwargs,
    ) -> DataProto:
        return super().dual_stream_rollout(
            prompts=prompts,
            intervention_mode=intervention_mode,
            stream_mode=stream_mode,
            token_check_interval=token_check_interval,
            **kwargs,
        )
