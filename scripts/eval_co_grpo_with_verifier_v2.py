#!/usr/bin/env python3
"""
Eval Co-GRPO with Verifier LoRA interventions.

Goals:
- Match training-side prompt format and truncation ("**Student Response (So Far):**").
- Support vLLM backend (fast, consistent with CoGRPO rollout).
- Keep a HF backend fallback for environments without vLLM.

NOTE: The original script may be executed by remote jobs on shared FS (ETXTBSY).
      This v2 script avoids in-place edits of the original.
"""

import argparse
import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


# Verifier prompts (must match training: verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py)
VERIFIER_SYSTEM_PROMPT = (
    "You are an expert reasoner with extensive experience in all areas. You approach problems through systematic "
    "thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, "
    "making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags."
)

VERIFIER_INTERVENE_PROMPT = """You are monitoring a student's reasoning trace.
Given the `Question` and the `Student Response` so far, do the workflow below.

**Phase 1: Shadow Verification (inside <think>)**
Write a careful, non-redundant shadow verification. You MUST:
1) **Re-calculate**: actually redo the math for the current step (not just “seems right”).
2) **Check logic**: verify conditions/edge cases for the theorem/rule being used.
3) **Check progress / efficiency** (crucial): ask “Did this step create new information or remove uncertainty?”
   - If the student is repeating already-established facts → mark as **Stagnation**.
   - If a final answer is already effectively finished but the student keeps expanding → mark as **Late-stage Verbosity**.
4) **Avoid loops**: do not repeat the same check multiple times. If you already verified something, move to the next missing check.

**Phase 2: Decision**
- If the step is correct AND it makes real progress → output `<GO>`.
- Otherwise (error / missing justification / stagnation / late-stage verbosity) → output `<WAIT>` plus ONE short guidance hint.

**Guidance Hint Rules (strict)**
- The hint MUST be **first-person inner voice** (e.g., “Wait, I should … / Hold on, I should …”).
- The hint MUST be **in the same language** as the student response.
- The hint MUST be **actionable and minimal** (1–2 sentences): tell what to check/redo next, not a full solution.
- The hint MUST NOT reveal the final answer (no `\\boxed{}`, no “Final Answer”, no “答案：”).
- The hint MUST NOT contain any extra tags or special markers:
  - DO NOT include `<GO>`, `<WAIT>`, `<think>`, `</think>` inside the hint body.
  - DO NOT use Markdown code blocks (```).
- If unsure which detail is wrong, ask for a *targeted check* (e.g., “Wait, I should verify the sign/limits/domain here.”).

**Output Format (strict)**
- Start directly with `<think>` (no preface).
- Then output EXACTLY one decision line:
  - `<GO>`
  - OR `<WAIT> ` + your hint (single line)

Format:
<think>
[shadow verification...]
</think>
<GO>
OR
<think>
[shadow verification...]
</think>
<WAIT> [first-person guidance hint]

**Question:**
{question}

**Student Response (So Far):**
{student_response}
"""


def build_verifier_user_prompt(question: str, student_response: str) -> str:
    # Keep placeholders safe if question/response contains "{...}".
    template = VERIFIER_INTERVENE_PROMPT.replace("{question}", "__VERIFIER_QUESTION__").replace(
        "{student_response}", "__VERIFIER_RESPONSE__"
    )
    return template.replace("__VERIFIER_QUESTION__", question).replace("__VERIFIER_RESPONSE__", student_response)


BANNED_PHRASES = [
    "provide the final numeric answer",
    "finish the solution",
    "answer the question",
    "provide the final answer",
    "final numeric answer",
]


@dataclass
class Decision:
    action: str  # "Pass" or "Intervene"
    hint: str
    critique: str


_ONLINE_HINT_INJECTION_MODULE: Any = None


def _load_online_hint_injection_module() -> Any:
    """
    Load the online-canonical hint injection helpers (natural insertion, rollback guards).

    Preferred: import from an installed `verl` package.
    Fallback: dynamic import from a repo checkout path (set via $REPRO_ROOT or $ONLINE_HINT_INJECTION_PY).
    """
    # 1) Try normal import first (works if repro is installed / in PYTHONPATH).
    try:
        from verl.workers.rollout.vllm_rollout import verifier_hint_injection as inj  # type: ignore

        return inj
    except Exception:
        pass

    # 2) Dynamic import from file path.
    candidates: List[Path] = []
    p_env = (os.environ.get("ONLINE_HINT_INJECTION_PY") or "").strip()
    if p_env:
        candidates.append(Path(p_env))
    repro_root = (os.environ.get("REPRO_ROOT") or "").strip()
    if repro_root:
        candidates.append(
            Path(repro_root)
            / "verl"
            / "workers"
            / "rollout"
            / "vllm_rollout"
            / "verifier_hint_injection.py"
        )
    # Default: repo-relative.
    try:
        repo_root = Path(__file__).resolve().parents[1]
        candidates.append(
            repo_root
            / "verl"
            / "workers"
            / "rollout"
            / "vllm_rollout"
            / "verifier_hint_injection.py"
        )
    except Exception:
        pass

    for p in candidates:
        try:
            if not p or not p.exists():
                continue
            spec = importlib.util.spec_from_file_location("_online_hint_injection", str(p))
            if spec is None or spec.loader is None:
                continue
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)  # type: ignore[attr-defined]
            return m
        except Exception:
            continue

    raise FileNotFoundError(
        "Failed to load online hint injection helpers. Set $REPRO_ROOT to the repro checkout root "
        "or $ONLINE_HINT_INJECTION_PY to verifier_hint_injection.py."
    )


def _get_online_hint_injection_module() -> Any:
    global _ONLINE_HINT_INJECTION_MODULE
    if _ONLINE_HINT_INJECTION_MODULE is None:
        _ONLINE_HINT_INJECTION_MODULE = _load_online_hint_injection_module()
    return _ONLINE_HINT_INJECTION_MODULE


def _normalize_eos_token_ids(eos_token_id) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        out: set[int] = set()
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


def _build_step_boundary_stop_config(tokenizer, extra_stop_token_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    stop_token_ids = set(int(x) for x in (extra_stop_token_ids or []) if x is not None)
    stop_sequences = {
        "</think>",
        "</think>\n",
        "</think>\n\n",
        "<｜end of thought｜>",
        "<|end of thought|>",
        "<|end_of_thought|>",
        "\n\n",
        ".\n\n",
    }
    eos_ids = set(int(x) for x in stop_token_ids)

    try:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            stop_token_ids.add(int(tokenizer.pad_token_id))
    except Exception:
        pass

    try:
        im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        if im_start_ids:
            stop_token_ids.add(int(im_start_ids[-1]))
    except Exception:
        pass

    stop_seq_token_ids: List[List[int]] = []
    for seq in stop_sequences:
        try:
            ids = tokenizer.encode(seq, add_special_tokens=False)
            if ids:
                stop_seq_token_ids.append([int(x) for x in ids])
        except Exception:
            continue

    return {
        "stop_token_ids": sorted(stop_token_ids),
        "stop_sequences": sorted(stop_sequences),
        "stop_seq_token_ids": stop_seq_token_ids,
        "eos_token_ids": eos_ids,
    }


def _build_control_stop_token_ids(tokenizer, extra_stop_token_ids: Optional[List[int]] = None) -> List[int]:
    stop_token_ids = set(int(x) for x in (extra_stop_token_ids or []) if x is not None)
    try:
        im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        if im_start_ids:
            stop_token_ids.add(int(im_start_ids[-1]))
    except Exception:
        pass
    return sorted(stop_token_ids)


def _response_hits_local_stop(
    response_ids: List[int],
    new_tokens: List[int],
    stop_token_ids: set[int],
    stop_seq_token_ids: List[List[int]],
) -> bool:
    if stop_token_ids and any(int(t) in stop_token_ids for t in (new_tokens or [])):
        return True
    if response_ids and stop_seq_token_ids:
        for seq_ids in stop_seq_token_ids:
            if seq_ids and len(response_ids) >= len(seq_ids) and response_ids[-len(seq_ids):] == seq_ids:
                return True
    return False


# Prefer online-canonical verifier prompts from the shared hint injection module.
try:
    _inj_prompts0 = _get_online_hint_injection_module()
    VERIFIER_SYSTEM_PROMPT = str(getattr(_inj_prompts0, "VERIFIER_SYSTEM_PROMPT", VERIFIER_SYSTEM_PROMPT))
    VERIFIER_INTERVENE_PROMPT = str(getattr(_inj_prompts0, "VERIFIER_INTERVENE_PROMPT", VERIFIER_INTERVENE_PROMPT))
except Exception:
    pass


def _messages_to_prompt_ids(tokenizer, messages: List[Dict[str, str]]) -> List[int]:
    """
    Match rl_dataset.py encoding:
    - apply_chat_template(..., tokenize=False) to get raw string
    - encode with add_special_tokens=False
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer has no chat template support (apply_chat_template missing).")
    raw = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(raw, add_special_tokens=False)


def _extract_verifier_hint(verifier_text: str) -> str:
    import re

    if not verifier_text:
        return ""
    text = verifier_text
    text = re.sub(r"```\s*\n?.*?\n?\s*```", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # IMPORTANT: do not truncate at first newline; <WAIT> hints are commonly multi-line checklists.
    if "<WAIT>" in text.upper():
        m = re.search(r"<WAIT>\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        payload = (m.group(1) if m else "").strip()
        return f"<WAIT> {payload}".strip()
    if "<GO>" in text.upper():
        return "<GO>"
    return ""


def _parse_verifier_decision(verifier_text: str) -> Decision:
    verifier_text = (verifier_text or "").strip()
    hint = _extract_verifier_hint(verifier_text)

    if hint.upper().startswith("<WAIT>"):
        action = "Intervene"
        hint = hint.replace("<WAIT>", "").strip()  # CRITICAL: do not keep the tag.
    elif hint.upper().startswith("<GO>"):
        action = "Pass"
        hint = ""
    else:
        lower = verifier_text.lower()
        if "<wait>" in lower:
            action = "Intervene"
            hint = verifier_text.split("<WAIT>", 1)[-1].strip()
        else:
            action = "Pass"
            hint = ""

    return Decision(action=action, hint=hint, critique=verifier_text)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _json_safe(obj: Any) -> Any:
    """
    Best-effort conversion to JSON-serializable objects.
    Keeps eval robust when input rows contain non-JSON primitives (e.g. numpy scalars).
    """
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _iter_prompt_items_from_file(path: Path, prompt_key: str) -> List[Tuple[Any, Dict[str, Any]]]:
    """
    Return a list of (prompt_obj, origin_info) pairs.

    - parquet/jsonl/json: prompt_key can be either:
      * str (single-turn user prompt)
      * list[dict] (OpenAI messages, same as CoGRPO RL dataset)
    - txt: one prompt string per line
    """
    suf = path.suffix.lower()
    if suf == ".parquet":
        # Training uses parquet with prompt_key as OpenAI messages (list[dict]).
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError("Reading .parquet requires pandas/pyarrow in the environment.") from e

        df = pd.read_parquet(path)
        if prompt_key not in df.columns:
            raise KeyError(f"Missing column '{prompt_key}' in {path}")
        items: List[Tuple[Any, Dict[str, Any]]] = []
        # Avoid df.to_dict("records") to reduce peak memory on large eval sets.
        for row_idx in range(len(df)):
            try:
                row = df.iloc[row_idx].to_dict()
            except Exception:
                row = {prompt_key: df[prompt_key].iloc[row_idx]}

            v = row.get(prompt_key)
            # Some pipelines store JSON string; accept both.
            if isinstance(v, str):
                vv = v.strip()
                if vv.startswith("[") and vv.endswith("]"):
                    try:
                        v = json.loads(vv)
                    except Exception:
                        v = v
            items.append((v, _json_safe(row)))
        return items

    if suf in (".jsonl", ".json"):
        rows = _read_jsonl(path) if suf == ".jsonl" else json.load(path.open("r", encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("data", [])
        items: List[Tuple[Any, Dict[str, Any]]] = []
        for r in rows:
            if not isinstance(r, dict) or prompt_key not in r:
                continue
            items.append((r[prompt_key], _json_safe(r)))
        return items

    # txt: one prompt per line
    items_txt: List[Tuple[Any, Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                items_txt.append((line, {"raw": line}))
    return items_txt


def _iter_prompts_from_file(path: Path, prompt_key: str) -> List[str]:
    # Back-compat helper: return prompts only.
    return [p for p, _ in _iter_prompt_items_from_file(path, prompt_key)]  # type: ignore[return-value]


def _detect_base_model_dir(run_dir: Path) -> Path:
    marker = run_dir / "latest_checkpointed_iteration.txt"
    step = None
    if marker.exists():
        try:
            step = int(marker.read_text(encoding="utf-8").strip())
        except Exception:
            step = None

    candidates: List[Path] = []
    if step is not None:
        candidates.extend(
            [
                run_dir / f"global_step_{step}" / "actor" / "huggingface",
                run_dir / f"global_step_{step}" / "actor",
            ]
        )
    for p in sorted(run_dir.glob("global_step_*")):
        candidates.append(p / "actor" / "huggingface")
        candidates.append(p / "actor")

    for p in candidates:
        if p.is_dir() and (p / "config.json").exists():
            return p
    raise FileNotFoundError(f"Could not find base model dir under: {run_dir}")


def _detect_verifier_lora_dir(run_dir: Path) -> Path:
    p = run_dir / "verifier_lora_latest"
    if p.is_dir() and (p / "adapter_config.json").exists():
        return p
    for c in sorted(run_dir.glob("verifier_lora*")):
        if c.is_dir() and (c / "adapter_config.json").exists():
            return c
    raise FileNotFoundError(f"Could not find verifier LoRA dir under: {run_dir}")


def _load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _load_model_with_verifier_lora(model_dir: str, verifier_lora_dir: str, dtype: torch.dtype, device_map: Optional[str]):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    model.eval()
    model = PeftModel.from_pretrained(model, verifier_lora_dir, is_trainable=False)
    model.eval()
    return model


def _load_hf_model_no_lora(model_dir: str, dtype: torch.dtype, device_map: Optional[str]):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    model.eval()
    return model


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _chat_prompt_ids(tokenizer, user_prompt: str, system_prompt: Optional[str]) -> List[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return _messages_to_prompt_ids(tokenizer, messages)
    return tokenizer.encode(user_prompt, add_special_tokens=False)


def _prompt_obj_to_token_ids(tokenizer, prompt_obj: Any, system_prompt: Optional[str]) -> List[int]:
    # Training RL dataset uses prompt_key as OpenAI messages (list[dict]).
    if isinstance(prompt_obj, list) and hasattr(tokenizer, "apply_chat_template"):
        # IMPORTANT: match verl/utils/dataset/rl_dataset.py
        raw = tokenizer.apply_chat_template(prompt_obj, add_generation_prompt=True, tokenize=False)
        return tokenizer.encode(raw, add_special_tokens=False)
    if not isinstance(prompt_obj, str):
        # Best-effort: serialize unknown objects.
        prompt_obj = json.dumps(prompt_obj, ensure_ascii=False)
    return _chat_prompt_ids(tokenizer, prompt_obj, system_prompt=system_prompt)


def _prompt_obj_to_prompt_text(tokenizer, prompt_obj: Any, system_prompt: Optional[str]) -> str:
    """
    Convert prompt object to a raw prompt string suitable for text-based engines (e.g., lmdeploy pipeline).
    Tries to match training-side chat template formatting.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        if not isinstance(prompt_obj, str):
            prompt_obj = json.dumps(prompt_obj, ensure_ascii=False)
        if system_prompt:
            return f"{system_prompt}\n\n{prompt_obj}"
        return str(prompt_obj)

    if isinstance(prompt_obj, list):
        messages = list(prompt_obj)
        if system_prompt and not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages
        raw = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": str(prompt_obj)})
        raw = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # LMDeploy pipeline tokenizes prompts with add_special_tokens=True; remove explicit bos prefix
    # to avoid double-bos effects (matches OpenCompass workaround).
    try:
        bos = getattr(tokenizer, "bos_token", None)
        if bos and isinstance(raw, str) and raw.startswith(bos):
            raw = raw[len(bos) :]
    except Exception:
        pass
    return str(raw)


def _build_verifier_prompt_ids(
    tokenizer,
    *,
    question: str,
    current_reasoning: str,
    verifier_max_prompt_length: int,
    verifier_max_new_tokens: int,
    max_model_len: int,
) -> List[int]:
    user_content = build_verifier_user_prompt(question, current_reasoning)

    verifier_max_prompt_length = int(verifier_max_prompt_length)
    verifier_max_new_tokens = int(verifier_max_new_tokens)
    max_model_len = int(max_model_len)
    verifier_prompt_budget = max(1, min(verifier_max_prompt_length, max_model_len - verifier_max_new_tokens))

    prefix_text, sep, student_resp = user_content.partition("**Student Response (So Far):**")
    if sep:
        user_prefix = f"{prefix_text}**Student Response (So Far):**\n"
    else:
        user_prefix = user_content
        student_resp = ""

    base_messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prefix},
    ]
    try:
        base_ids = tokenizer.apply_chat_template(base_messages, tokenize=True, add_generation_prompt=True)
    except Exception:
        base_text = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=True)
        base_ids = tokenizer.encode(base_text, add_special_tokens=False)

    remaining = verifier_prompt_budget - len(base_ids)
    if remaining > 0 and student_resp:
        tail_ids = tokenizer.encode(student_resp, add_special_tokens=False)
        tail_ids = tail_ids[-remaining:]
        truncated_student_resp = tokenizer.decode(tail_ids, skip_special_tokens=True)
        user_full = user_prefix + truncated_student_resp
    else:
        user_full = user_prefix

    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_full},
    ]
    try:
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    if len(prompt_ids) > verifier_prompt_budget:
        prompt_ids = prompt_ids[-verifier_prompt_budget:]
    return prompt_ids


def _normalize_hint(text: str) -> str:
    import re

    if not text:
        return ""
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _hint_allowed(hint: str, previous_hints: List[str]) -> bool:
    import difflib

    hint_norm = _normalize_hint(hint)
    if not hint_norm:
        return False

    for phrase in BANNED_PHRASES:
        if phrase in hint_norm:
            return False

    for prev in previous_hints or []:
        prev_norm = _normalize_hint(prev)
        if not prev_norm:
            continue
        if hint_norm in prev_norm or prev_norm in hint_norm:
            return False
        if difflib.SequenceMatcher(a=hint_norm, b=prev_norm).ratio() >= 0.9:
            return False

    return True


def _format_hint_text(*, hint: str, response_ids: List[int], tokenizer) -> str:
    inj = _get_online_hint_injection_module()
    try:
        tail = tokenizer.decode(response_ids[-128:], skip_special_tokens=True) if response_ids else ""
    except Exception:
        tail = ""
    return inj.format_hint_text(hint=(hint or ""), tail_text=tail)


def _infer_lora_rank(lora_dir: Path, default_rank: int = 64) -> int:
    cfg = lora_dir / "adapter_config.json"
    if not cfg.exists():
        return int(default_rank)
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return int(data.get("r", default_rank))
    except Exception:
        return int(default_rank)


def _init_vllm_engine(
    *,
    model_dir: Path,
    tokenizer_dir: Optional[Path],
    verifier_lora_dir: Optional[Path],
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool = False,
    max_num_batched_tokens: int = 0,
    enable_chunked_prefill: Optional[bool] = None,
):
    from vllm import LLM

    kwargs: Dict[str, Any] = {
        "model": str(model_dir),
        "tokenizer": str(tokenizer_dir if tokenizer_dir is not None else model_dir),
        "trust_remote_code": True,
        "tensor_parallel_size": int(tensor_parallel_size),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "max_model_len": int(max_model_len),
        "enforce_eager": bool(enforce_eager),
    }
    if int(max_num_batched_tokens) > 0:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if enable_chunked_prefill is not None:
        kwargs["enable_chunked_prefill"] = bool(enable_chunked_prefill)
    if verifier_lora_dir is not None:
        lora_rank = _infer_lora_rank(verifier_lora_dir, default_rank=64)
        kwargs.update(
            {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": int(lora_rank),
            }
        )
    llm = LLM(**kwargs)
    return llm


def _vllm_generate_tokens(
    llm,
    *,
    prompt_token_ids: List[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stop_token_ids: List[int],
    stop_sequences: Optional[List[str]] = None,
    seed: Optional[int],
    lora_request: Optional[object] = None,
) -> Tuple[List[int], str, float]:
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams

    t0 = time.time()
    params = SamplingParams(
        n=1,
        temperature=float(temperature),
        top_p=float(top_p) if top_p is not None else 1.0,
        top_k=int(top_k) if top_k is not None and int(top_k) > 0 else 0,
        max_tokens=int(max_tokens),
        stop_token_ids=[int(x) for x in (stop_token_ids or [])],
        stop=list(stop_sequences or []),
        seed=int(seed) if seed is not None else None,
    )
    out = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=list(prompt_token_ids))],
        sampling_params=params,
        use_tqdm=False,
        lora_request=[lora_request] if lora_request is not None else None,
    )
    t1 = time.time()
    gen = out[0].outputs[0]
    return list(gen.token_ids), str(gen.text or ""), float(t1 - t0)


def _vllm_generate_tokens_with_logprobs(
    llm,
    *,
    prompt_token_ids: List[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stop_token_ids: List[int],
    stop_sequences: Optional[List[str]] = None,
    seed: Optional[int],
    logprobs: int,
    lora_request: Optional[object] = None,
) -> Tuple[List[int], str, List[float], float]:
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams

    t0 = time.time()
    params = SamplingParams(
        n=1,
        temperature=float(temperature),
        top_p=float(top_p) if top_p is not None else 1.0,
        top_k=int(top_k) if top_k is not None and int(top_k) > 0 else 0,
        max_tokens=int(max_tokens),
        stop_token_ids=[int(x) for x in (stop_token_ids or [])],
        stop=list(stop_sequences or []),
        seed=int(seed) if seed is not None else None,
        logprobs=int(logprobs) if int(logprobs) > 0 else 0,
    )
    out = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=list(prompt_token_ids))],
        sampling_params=params,
        use_tqdm=False,
        lora_request=[lora_request] if lora_request is not None else None,
    )
    t1 = time.time()
    gen = out[0].outputs[0]
    token_ids = list(gen.token_ids)
    text = str(gen.text or "")
    token_logprobs = _extract_generated_token_logprobs(gen)
    return token_ids, text, token_logprobs, float(t1 - t0)


def _vllm_generate_batch(
    llm,
    *,
    prompt_token_ids_list: List[List[int]],
    max_tokens: Any,
    temperature: float,
    top_p: float,
    top_k: int,
    stop_token_ids: List[int],
    stop_sequences: Optional[List[str]] = None,
    seed: Optional[int],
    lora_requests: Optional[List[object]] = None,
) -> Tuple[List[List[int]], List[str], float]:
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams

    if not prompt_token_ids_list:
        return [], [], 0.0

    t0 = time.time()
    stop_ids = [int(x) for x in (stop_token_ids or [])]
    seed_i = int(seed) if seed is not None else None

    if isinstance(max_tokens, (list, tuple)):
        params = [
            SamplingParams(
                n=1,
                temperature=float(temperature),
                top_p=float(top_p) if top_p is not None else 1.0,
                top_k=int(top_k) if top_k is not None and int(top_k) > 0 else 0,
                max_tokens=int(mt),
                stop_token_ids=stop_ids,
                stop=list(stop_sequences or []),
                seed=seed_i,
            )
            for mt in max_tokens
        ]
    else:
        params = SamplingParams(
            n=1,
            temperature=float(temperature),
            top_p=float(top_p) if top_p is not None else 1.0,
            top_k=int(top_k) if top_k is not None and int(top_k) > 0 else 0,
            max_tokens=int(max_tokens),
            stop_token_ids=stop_ids,
            stop=list(stop_sequences or []),
            seed=seed_i,
        )
    out = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=list(ids)) for ids in prompt_token_ids_list],
        sampling_params=params,
        use_tqdm=False,
        lora_request=lora_requests,
    )
    t1 = time.time()

    token_ids_list: List[List[int]] = []
    texts: List[str] = []
    for req in out:
        gen = req.outputs[0]
        token_ids_list.append(list(gen.token_ids))
        texts.append(str(gen.text or ""))
    return token_ids_list, texts, float(t1 - t0)


def _vllm_generate_batch_with_logprobs(
    llm,
    *,
    prompt_token_ids_list: List[List[int]],
    max_tokens: Any,
    temperature: float,
    top_p: float,
    top_k: int,
    stop_token_ids: List[int],
    stop_sequences: Optional[List[str]] = None,
    seed: Optional[int],
    logprobs: int,
    lora_requests: Optional[List[object]] = None,
) -> Tuple[List[List[int]], List[str], List[List[float]], float]:
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams

    if not prompt_token_ids_list:
        return [], [], [], 0.0

    t0 = time.time()
    stop_ids = [int(x) for x in (stop_token_ids or [])]
    seed_i = int(seed) if seed is not None else None
    lp = int(logprobs) if int(logprobs) > 0 else 0

    if isinstance(max_tokens, (list, tuple)):
        params = [
            SamplingParams(
                n=1,
                temperature=float(temperature),
                top_p=float(top_p) if top_p is not None else 1.0,
                top_k=int(top_k) if top_k is not None and int(top_k) > 0 else 0,
                max_tokens=int(mt),
                stop_token_ids=stop_ids,
                stop=list(stop_sequences or []),
                seed=seed_i,
                logprobs=lp,
            )
            for mt in max_tokens
        ]
    else:
        params = SamplingParams(
            n=1,
            temperature=float(temperature),
            top_p=float(top_p) if top_p is not None else 1.0,
            top_k=int(top_k) if top_k is not None and int(top_k) > 0 else 0,
            max_tokens=int(max_tokens),
            stop_token_ids=stop_ids,
            stop=list(stop_sequences or []),
            seed=seed_i,
            logprobs=lp,
        )
    out = llm.generate(
        prompts=[TokensPrompt(prompt_token_ids=list(ids)) for ids in prompt_token_ids_list],
        sampling_params=params,
        use_tqdm=False,
        lora_request=lora_requests,
    )
    t1 = time.time()

    token_ids_list: List[List[int]] = []
    texts: List[str] = []
    token_logprobs_list: List[List[float]] = []
    for req in out:
        gen = req.outputs[0]
        token_ids_list.append(list(gen.token_ids))
        texts.append(str(gen.text or ""))
        token_logprobs_list.append(_extract_generated_token_logprobs(gen))
    return token_ids_list, texts, token_logprobs_list, float(t1 - t0)


def _extract_generated_token_logprobs(gen: Any) -> List[float]:
    """
    Extract per-token logprobs for the generated token_ids.

    vLLM returns `gen.logprobs` as a list where each entry is a dict mapping
    token_id -> Logprob (or float-like). We only need the logprob of the
    actually generated token at each step.
    """
    out: List[float] = []
    logprobs = getattr(gen, "logprobs", None)
    token_ids = list(getattr(gen, "token_ids", []) or [])
    if not logprobs or not token_ids:
        return out

    for tok, lp_dict in zip(token_ids, logprobs):
        val = None
        try:
            if isinstance(lp_dict, dict):
                item = lp_dict.get(tok, None)
                if item is None:
                    item = lp_dict.get(str(tok), None)
                if item is None and lp_dict:
                    item = next(iter(lp_dict.values()))
                if item is not None:
                    if hasattr(item, "logprob"):
                        val = float(getattr(item, "logprob"))
                    else:
                        val = float(item)
            elif lp_dict is not None:
                val = float(lp_dict)
        except Exception:
            val = None

        out.append(float(val) if val is not None else float("nan"))
    return out


def _compute_wait_confidence_from_logprobs(
    token_logprobs: List[float], *, tail_tokens: int = 64
) -> Tuple[Optional[float], Optional[float]]:
    """
    Match online `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py::_compute_wait_confidence`
    as closely as possible:
      confidence = exp(mean(last N token logprobs))
    """
    if not token_logprobs:
        return None, None
    vals: List[float] = []
    for x in token_logprobs:
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            vals.append(xf)
    if not vals:
        return None, None
    try:
        tail_tokens = int(tail_tokens)
    except Exception:
        tail_tokens = 64
    if tail_tokens > 0 and len(vals) > tail_tokens:
        vals = vals[-tail_tokens:]

    avg_logprob = float(sum(vals) / max(1, len(vals)))
    avg_logprob = float(min(0.0, max(-20.0, avg_logprob)))
    return float(math.exp(avg_logprob)), avg_logprob


def _generate_control_vllm(
    *,
    llm,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stop_token_ids: List[int],
    seed: Optional[int],
) -> Tuple[str, List[int], float]:
    gen_ids, gen_text, dt = _vllm_generate_tokens(
        llm,
        prompt_token_ids=prompt_ids,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_token_ids=stop_token_ids,
        stop_sequences=None,
        seed=seed,
        lora_request=None,
    )
    return gen_text, gen_ids, dt


def _generate_with_verifier_vllm(
    *,
    llm,
    tokenizer,
    verifier_lora_dir: Optional[Path],
    question: str,
    prompt_ids: List[int],
    max_model_len: int,
    max_model_tokens: int,
    token_check_interval: int,
    min_step_tokens: int,
    max_interventions: int,
    verifier_max_prompt_length: int,
    verifier_max_new_tokens: int,
    verifier_max_hint_tokens: int,
    confidence_threshold: float,
    wait_conf_tail_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stop_token_ids: List[int],
    seed: Optional[int],
) -> Dict[str, Any]:
    verifier_lora_req = None
    if verifier_lora_dir is not None:
        from vllm.lora.request import LoRARequest

        verifier_lora_req = LoRARequest(
            lora_name="verifier_lora",
            lora_int_id=1,
            lora_path=str(verifier_lora_dir),
        )

    inj = _get_online_hint_injection_module()
    stop_cfg = _build_step_boundary_stop_config(tokenizer, stop_token_ids)
    eos_token_ids = set(int(x) for x in (stop_cfg.get("eos_token_ids") or []))
    terminal_stop_set = set(int(x) for x in (stop_cfg.get("stop_token_ids") or []))
    boundary_stop_set = set(int(x) for x in (stop_cfg.get("stop_token_ids") or []))
    stop_seq_token_ids = list(stop_cfg.get("stop_seq_token_ids") or [])

    response_ids: List[int] = []
    interventions: List[Dict[str, Any]] = []
    previous_hints: List[str] = []
    tokens_since_boundary = 0
    model_gen_tokens = 0

    t0 = time.time()
    while True:
        if model_gen_tokens >= int(max_model_tokens):
            break

        context_ids = list(prompt_ids) + list(response_ids)
        step_tokens = min(int(token_check_interval), int(max_model_tokens) - model_gen_tokens)
        step_tokens = max(1, int(step_tokens))

        if len(context_ids) + step_tokens > int(max_model_len):
            break

        new_tokens, _, _ = _vllm_generate_tokens(
            llm,
            prompt_token_ids=context_ids,
            max_tokens=step_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # Match training (vllm_rollout_spmd.py): do NOT pass stop_token_ids into vLLM;
            # stop tokens are handled locally for boundary detection.
            stop_token_ids=[],
            stop_sequences=None,
            seed=seed,
            lora_request=None,
        )
        if not new_tokens:
            break

        response_ids.extend(new_tokens)
        model_gen_tokens += len(new_tokens)
        tokens_since_boundary += len(new_tokens)

        if terminal_stop_set and any(int(t) in terminal_stop_set for t in (new_tokens or [])):
            break

        if len(interventions) >= int(max_interventions):
            continue

        # Stop boundary (training uses this for stop detection; verifier call happens every step regardless).
        if tokens_since_boundary >= int(min_step_tokens) and _response_hits_local_stop(
            response_ids, new_tokens, boundary_stop_set, stop_seq_token_ids
        ):
            tokens_since_boundary = 0

        current_reasoning = tokenizer.decode(response_ids, skip_special_tokens=True)
        verifier_prompt_ids = _build_verifier_prompt_ids(
            tokenizer,
            question=question,
            current_reasoning=current_reasoning,
            verifier_max_prompt_length=verifier_max_prompt_length,
            verifier_max_new_tokens=verifier_max_new_tokens,
            max_model_len=max_model_len,
        )

        verifier_new_tokens, verifier_text, verifier_logprobs, _ = _vllm_generate_tokens_with_logprobs(
            llm,
            prompt_token_ids=verifier_prompt_ids,
            max_tokens=int(verifier_max_new_tokens),
            # Match training: deterministic decode via top-1, keep temperature valid.
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            stop_token_ids=[],
            stop_sequences=None,
            seed=None,
            logprobs=1,
            lora_request=verifier_lora_req,
        )
        if not verifier_text and verifier_new_tokens:
            verifier_text = tokenizer.decode(verifier_new_tokens, skip_special_tokens=True)

        decision = _parse_verifier_decision(verifier_text)
        wait_confidence, wait_avg_logprob = (None, None)
        if decision.action == "Intervene":
            wait_confidence, wait_avg_logprob = _compute_wait_confidence_from_logprobs(
                verifier_logprobs, tail_tokens=int(wait_conf_tail_tokens)
            )
            if float(confidence_threshold) > 0:
                if wait_confidence is None or float(wait_confidence) < float(confidence_threshold):
                    continue
        if decision.hint and verifier_max_hint_tokens > 0:
            hint_ids = tokenizer.encode(decision.hint, add_special_tokens=False)
            if len(hint_ids) > verifier_max_hint_tokens:
                decision = Decision(
                    action=decision.action,
                    hint=tokenizer.decode(hint_ids[:verifier_max_hint_tokens], skip_special_tokens=True).strip(),
                    critique=decision.critique,
                )

        if decision.action != "Intervene" or not decision.hint:
            continue
        if not _hint_allowed(decision.hint, previous_hints):
            continue

        hint_text = _format_hint_text(hint=decision.hint, response_ids=response_ids, tokenizer=tokenizer)
        hint_ids = tokenizer.encode(hint_text, add_special_tokens=False)
        hint_applied = False
        if hint_ids:
            state = {"response_tokens": response_ids, "loss_masks": [1] * len(response_ids)}
            hint_applied = bool(inj.insert_hint_tokens(state, hint_ids, tokenizer, eos_token_ids))
            if hint_applied:
                previous_hints.append(decision.hint)
                tokens_since_boundary = 0  # align with online behavior
        if not hint_applied:
            continue

        interventions.append(
            {
                "hint": decision.hint,
                "hint_text": hint_text,
                "verifier_action": decision.action,
                "verifier_critique": decision.critique,
                "wait_confidence": wait_confidence,
                "wait_avg_logprob": wait_avg_logprob,
                "response_tokens_after": len(response_ids),
            }
        )

    t1 = time.time()
    return {
        "response_text": tokenizer.decode(response_ids, skip_special_tokens=True),
        "response_token_ids": response_ids,
        "interventions": interventions,
        "model_gen_tokens": int(model_gen_tokens),
        "gen_s": float(t1 - t0),
    }


@dataclass
class _ExpState:
    idx: int
    prompt_obj: Any
    prompt_ids: List[int]
    prompt_text: str
    response_ids: List[int]
    loss_masks: List[int]
    interventions: List[Dict[str, Any]]
    previous_hints: List[str]
    tokens_since_boundary: int
    model_gen_tokens: int
    is_complete: bool
    pending_complete_reason: Optional[str] = None
    prethink_rollback_used: bool = False
    error: Optional[str] = None


def _rollback_exp_state_to_keep_len(state: _ExpState, keep_len: int) -> bool:
    inj = _get_online_hint_injection_module()
    raw_state = {
        "response_tokens": list(state.response_ids),
        "loss_masks": list(state.loss_masks),
        "gen_len": int(state.model_gen_tokens),
    }
    ok = bool(inj.rollback_state_to_keep_len(raw_state, int(keep_len)))
    if not ok:
        return False
    state.response_ids = list(raw_state.get("response_tokens") or [])
    state.loss_masks = list(raw_state.get("loss_masks") or [])
    try:
        state.model_gen_tokens = int(raw_state.get("gen_len") or 0)
    except Exception:
        pass
    return True


def _compute_exp_anchor_keep(
    state: _ExpState,
    tokenizer,
    stop_token_ids_set: set[int],
    *,
    hint_rollback_window_tokens: int = 512,
    pending_hint_tail_decode_tokens: int = 128,
) -> Tuple[int, bool, bool]:
    inj = _get_online_hint_injection_module()

    hint_anchor_keep = len(state.response_ids)
    try:
        tail_keep = inj.compute_tail_rollback_keep_len(
            state.response_ids,
            tokenizer,
            int(hint_rollback_window_tokens),
        )
        if tail_keep is not None:
            hint_anchor_keep = int(tail_keep)
    except Exception:
        pass

    last_hint_keep = 0
    try:
        last_hint_keep = int(inj.last_hint_end(state.loss_masks))
        if last_hint_keep > int(hint_anchor_keep):
            hint_anchor_keep = int(last_hint_keep)
    except Exception:
        last_hint_keep = 0

    if state.pending_complete_reason == "eos":
        try:
            if (
                int(hint_anchor_keep) > 0
                and int(state.response_ids[int(hint_anchor_keep) - 1]) in stop_token_ids_set
            ):
                hint_anchor_keep = max(int(last_hint_keep), int(hint_anchor_keep) - 1)
        except Exception:
            pass

    use_prethink_anchor = False
    anchor_tokens = state.response_ids[: int(hint_anchor_keep)]
    try:
        has_safe_think_anchor = inj.find_think_close_pos(anchor_tokens, tokenizer) is not None
        is_post_think_finalized = bool(
            inj.is_post_think_finalized(
                anchor_tokens,
                tokenizer,
                decode_tail_tokens=int(pending_hint_tail_decode_tokens),
            )
        )
    except Exception:
        has_safe_think_anchor = False
        is_post_think_finalized = False

    if (not state.prethink_rollback_used) and (not has_safe_think_anchor) and is_post_think_finalized:
        try:
            prethink_keep = inj.find_prethink_rollback_keep_len(anchor_tokens, tokenizer)
        except Exception:
            prethink_keep = None
        if prethink_keep is not None and int(prethink_keep) < int(hint_anchor_keep):
            hint_anchor_keep = int(prethink_keep)
            use_prethink_anchor = True
            try:
                tail_keep = inj.compute_tail_rollback_keep_len(
                    state.response_ids[: int(hint_anchor_keep)],
                    tokenizer,
                    int(hint_rollback_window_tokens),
                )
                if tail_keep is not None:
                    hint_anchor_keep = int(tail_keep)
            except Exception:
                pass
            try:
                last_hint_keep = int(inj.last_hint_end(state.loss_masks))
                if last_hint_keep > int(hint_anchor_keep):
                    hint_anchor_keep = int(last_hint_keep)
            except Exception:
                pass
            anchor_tokens = state.response_ids[: int(hint_anchor_keep)]
            try:
                has_safe_think_anchor = inj.find_think_close_pos(anchor_tokens, tokenizer) is not None
                is_post_think_finalized = bool(
                    inj.is_post_think_finalized(
                        anchor_tokens,
                        tokenizer,
                        decode_tail_tokens=int(pending_hint_tail_decode_tokens),
                    )
                )
            except Exception:
                has_safe_think_anchor = False
                is_post_think_finalized = False

    anchor_insertable = bool(has_safe_think_anchor or (not is_post_think_finalized))
    return int(hint_anchor_keep), bool(use_prethink_anchor), bool(anchor_insertable)


def _generate_with_verifier_vllm_batch(
    *,
    llm,
    tokenizer,
    verifier_lora_dir: Optional[Path],
    states: List[_ExpState],
    max_model_len: int,
    max_model_tokens: int,
    token_check_interval: int,
    min_step_tokens: int,
    max_interventions: int,
    token_check_interval_late: int,
    token_check_late_start_tokens: int,
    verifier_skip_budget_tokens: int,
    verifier_max_prompt_length: int,
    verifier_max_new_tokens: int,
    verifier_max_hint_tokens: int,
    confidence_threshold: float,
    wait_conf_tail_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    stop_token_ids: List[int],
    hint_rollback_window_tokens: int,
    pending_hint_tail_decode_tokens: int,
    seed: Optional[int],
) -> None:
    """
    Batched by_step rollout with verifier interventions, aligned with training logic:
    - actor chunks: no stop_token_ids passed into vLLM; stop tokens handled locally.
    - verifier called once per step chunk (per sample) unless sample complete or max_interventions reached.
    """
    verifier_lora_req = None
    if verifier_lora_dir is not None:
        from vllm.lora.request import LoRARequest

        verifier_lora_req = LoRARequest(
            lora_name="verifier_lora",
            lora_int_id=1,
            lora_path=str(verifier_lora_dir),
        )

    inj = _get_online_hint_injection_module()
    stop_cfg = _build_step_boundary_stop_config(tokenizer, stop_token_ids)
    eos_token_ids = set(int(x) for x in (stop_cfg.get("eos_token_ids") or []))
    terminal_stop_set = set(int(x) for x in (stop_cfg.get("stop_token_ids") or []))
    boundary_stop_set = set(int(x) for x in (stop_cfg.get("stop_token_ids") or []))
    stop_seq_token_ids = list(stop_cfg.get("stop_seq_token_ids") or [])
    stop_token_ids_set = set(int(x) for x in (stop_token_ids or []))

    t0 = time.time()
    while True:
        active: List[_ExpState] = []
        for s in states:
            if s.is_complete:
                continue
            if s.model_gen_tokens >= int(max_model_tokens):
                s.is_complete = True
                continue
            if len(s.prompt_ids) + len(s.response_ids) >= int(max_model_len):
                s.is_complete = True
                continue
            active.append(s)
        if not active:
            break

        # Per-sample max_tokens (matches training after per-sample truncation, but avoids vLLM length errors).
        max_tokens_list: List[int] = []
        for s in active:
            gen_remaining = int(max_model_tokens) - int(s.model_gen_tokens)
            ctx_remaining = int(max_model_len) - (len(s.prompt_ids) + len(s.response_ids))
            mt = min(int(token_check_interval), int(gen_remaining), int(ctx_remaining))
            max_tokens_list.append(max(1, int(mt)))

        actor_prompts = [s.prompt_ids + s.response_ids for s in active]
        try:
            new_token_lists, _, _ = _vllm_generate_batch(
                llm,
                prompt_token_ids_list=actor_prompts,
                max_tokens=max_tokens_list,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_token_ids=[],  # local stop detection only
                stop_sequences=None,
                seed=seed,
                lora_requests=None,
            )
        except Exception as e:
            for s in active:
                s.is_complete = True
                s.error = f"actor_generate_error: {e}"
            break

        # Update states with model-generated tokens.
        for s, new_tokens in zip(active, new_token_lists):
            if not new_tokens:
                s.is_complete = True
                continue
            s.response_ids.extend(new_tokens)
            s.loss_masks.extend([1] * len(new_tokens))
            s.model_gen_tokens += len(new_tokens)
            s.tokens_since_boundary += len(new_tokens)
            if terminal_stop_set and any(int(t) in terminal_stop_set for t in (new_tokens or [])):
                s.pending_complete_reason = "eos"
                continue
            if s.model_gen_tokens >= int(max_model_tokens):
                s.is_complete = True
                continue

            # Local stop detection is only used for boundary semantics; unlike the
            # old offline version, do not reset the periodic window here.
            _ = _response_hits_local_stop(s.response_ids, new_tokens, boundary_stop_set, stop_seq_token_ids)

        # Prepare verifier calls (batched) for samples still eligible.
        verifier_batch: List[_ExpState] = []
        verifier_prompt_ids_list: List[List[int]] = []
        verifier_anchor_keep_list: List[int] = []
        verifier_use_prethink_anchor_list: List[bool] = []
        for s in states:
            if s.is_complete:
                continue
            effective_token_check_interval = int(token_check_interval)
            if (
                int(token_check_interval_late) > 0
                and int(token_check_late_start_tokens) > 0
                and int(s.model_gen_tokens) >= int(token_check_late_start_tokens)
            ):
                effective_token_check_interval = int(token_check_interval_late)
            periodic_threshold = max(int(min_step_tokens), int(effective_token_check_interval))
            periodic_hit = int(s.tokens_since_boundary) >= int(periodic_threshold)
            final_boundary_hit = s.pending_complete_reason is not None
            should_request_verifier = (
                final_boundary_hit or periodic_hit
            ) and (len(s.interventions) < int(max_interventions))
            if not should_request_verifier:
                continue
            try:
                remaining_after_step = max(0, int(max_model_tokens) - int(s.model_gen_tokens))
                if (
                    int(verifier_skip_budget_tokens) > 0
                    and int(remaining_after_step) < int(verifier_skip_budget_tokens)
                ):
                    s.tokens_since_boundary = 0
                    continue
                s.tokens_since_boundary = 0
                hint_anchor_keep, use_prethink_anchor, anchor_insertable = _compute_exp_anchor_keep(
                    s,
                    tokenizer,
                    stop_token_ids_set,
                    hint_rollback_window_tokens=hint_rollback_window_tokens,
                    pending_hint_tail_decode_tokens=pending_hint_tail_decode_tokens,
                )
                if not anchor_insertable:
                    continue
                current_reasoning = tokenizer.decode(
                    s.response_ids[: int(hint_anchor_keep)], skip_special_tokens=True
                )
                vp_ids = _build_verifier_prompt_ids(
                    tokenizer,
                    question=s.prompt_text,
                    current_reasoning=current_reasoning,
                    verifier_max_prompt_length=verifier_max_prompt_length,
                    verifier_max_new_tokens=verifier_max_new_tokens,
                    max_model_len=max_model_len,
                )
                verifier_batch.append(s)
                verifier_prompt_ids_list.append(vp_ids)
                verifier_anchor_keep_list.append(int(hint_anchor_keep))
                verifier_use_prethink_anchor_list.append(bool(use_prethink_anchor))
            except Exception as e:
                # If verifier prompt build fails, just skip verifier for this step.
                s.error = f"verifier_prompt_build_error: {e}"

        if not verifier_batch:
            continue

        verifier_lora_requests = None
        if verifier_lora_req is not None:
            verifier_lora_requests = [verifier_lora_req] * len(verifier_batch)

        try:
            v_token_lists, v_texts, v_logprobs_lists, _ = _vllm_generate_batch_with_logprobs(
                llm,
                prompt_token_ids_list=verifier_prompt_ids_list,
                max_tokens=int(verifier_max_new_tokens),
                temperature=1.0,
                top_p=1.0,
                top_k=1,
                stop_token_ids=[],
                stop_sequences=None,
                seed=None,
                logprobs=1,
                lora_requests=verifier_lora_requests,
            )
        except Exception as e:
            for s in verifier_batch:
                s.error = f"verifier_generate_error: {e}"
            continue

        intervened_this_round: set[int] = set()
        for s, v_tokens, v_text, v_logprobs, hint_anchor_keep, use_prethink_anchor in zip(
            verifier_batch,
            v_token_lists,
            v_texts,
            v_logprobs_lists,
            verifier_anchor_keep_list,
            verifier_use_prethink_anchor_list,
        ):
            if not v_text and v_tokens:
                v_text = tokenizer.decode(v_tokens, skip_special_tokens=True)
            decision = _parse_verifier_decision(v_text)
            wait_confidence, wait_avg_logprob = (None, None)
            if decision.action == "Intervene":
                wait_confidence, wait_avg_logprob = _compute_wait_confidence_from_logprobs(
                    v_logprobs, tail_tokens=int(wait_conf_tail_tokens)
                )
                if float(confidence_threshold) > 0:
                    if wait_confidence is None or float(wait_confidence) < float(confidence_threshold):
                        continue
            if decision.hint and verifier_max_hint_tokens > 0:
                hint_ids = tokenizer.encode(decision.hint, add_special_tokens=False)
                if len(hint_ids) > int(verifier_max_hint_tokens):
                    decision = Decision(
                        action=decision.action,
                        hint=tokenizer.decode(hint_ids[: int(verifier_max_hint_tokens)], skip_special_tokens=True).strip(),
                        critique=decision.critique,
                    )

            if decision.action != "Intervene" or not decision.hint:
                continue
            if not _hint_allowed(decision.hint, s.previous_hints):
                continue

            if int(hint_anchor_keep) < len(s.response_ids):
                _rollback_exp_state_to_keep_len(s, int(hint_anchor_keep))
            if use_prethink_anchor:
                s.prethink_rollback_used = True

            hint_text = _format_hint_text(hint=decision.hint, response_ids=s.response_ids, tokenizer=tokenizer)
            hint_ids = tokenizer.encode(hint_text, add_special_tokens=False)
            hint_applied = False
            if hint_ids:
                state = {"response_tokens": s.response_ids, "loss_masks": s.loss_masks}
                hint_applied = bool(inj.insert_hint_tokens(state, hint_ids, tokenizer, eos_token_ids))
                if hint_applied:
                    s.response_ids = list(state.get("response_tokens") or [])
                    s.loss_masks = list(state.get("loss_masks") or [])
                    s.previous_hints.append(decision.hint)
                    s.tokens_since_boundary = 0
                    s.pending_complete_reason = None
                    intervened_this_round.add(int(s.idx))
            if not hint_applied:
                continue

            s.interventions.append(
                {
                    "hint": decision.hint,
                    "hint_text": hint_text,
                    "verifier_action": decision.action,
                    "verifier_critique": decision.critique,
                    "wait_confidence": wait_confidence,
                    "wait_avg_logprob": wait_avg_logprob,
                    "response_tokens_after": len(s.response_ids),
                }
            )

        for s in states:
            if s.is_complete:
                continue
            if s.pending_complete_reason is not None and int(s.idx) not in intervened_this_round:
                s.is_complete = True

    # Attach timing info per-sample (roughly equal across the batch).
    dt = float(time.time() - t0)
    for s in states:
        # Only set if not already set elsewhere.
        if not hasattr(s, "_gen_s"):
            setattr(s, "_gen_s", dt)


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval Co-GRPO verifier interventions (training-consistent).")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--verifier-lora", default=None)
    ap.add_argument(
        "--tokenizer-path",
        default="",
        help="Optional tokenizer dir. If set, use it for prompt encoding and vLLM tokenizer init.",
    )

    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--prompt", action="append", default=[])

    ap.add_argument("--out-jsonl", default="eval_verifier_results.jsonl")

    ap.add_argument("--backend", default="vllm", choices=["hf", "vllm", "lmdeploy"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device-map", default="auto", help='HF device_map (ignored for vLLM). Use "none" to disable.')

    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--max-response-tokens", type=int, default=32768)
    ap.add_argument("--token-check-interval", type=int, default=4096)
    ap.add_argument("--min-step-tokens", type=int, default=4096)
    ap.add_argument("--max-interventions", type=int, default=5)
    ap.add_argument("--token-check-interval-late", type=int, default=0)
    ap.add_argument("--token-check-late-start-tokens", type=int, default=0)
    ap.add_argument("--verifier-skip-budget-tokens", type=int, default=0)
    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Only apply <WAIT> hints when wait_confidence>=threshold. "
        "wait_confidence is computed from verifier token logprobs (exp(mean tail logprob)); 0 disables the gate.",
    )
    ap.add_argument(
        "--wait-conf-tail-tokens",
        type=int,
        default=64,
        help="Tail tokens used to estimate wait_confidence (match online default=64).",
    )

    ap.add_argument("--verifier-max-prompt-length", type=int, default=16384)
    ap.add_argument("--verifier-max-new-tokens", type=int, default=2048)
    ap.add_argument("--verifier-max-hint-tokens", type=int, default=512)
    ap.add_argument("--hint-rollback-window-tokens", type=int, default=512)
    ap.add_argument("--pending-hint-tail-decode-tokens", type=int, default=128)

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-token-id", action="append", default=[])

    ap.add_argument("--vllm-tp", type=int, default=1)
    ap.add_argument("--vllm-gpu-mem-util", type=float, default=0.8)
    ap.add_argument("--vllm-max-model-len", type=int, default=0)
    ap.add_argument("--vllm-enforce-eager", action="store_true")
    ap.add_argument("--vllm-max-num-batched-tokens", type=int, default=0)
    ap.add_argument(
        "--vllm-enable-chunked-prefill",
        dest="vllm_enable_chunked_prefill",
        action="store_true",
        help="Force enable chunked prefill for vLLM.",
    )
    ap.add_argument(
        "--vllm-disable-chunked-prefill",
        dest="vllm_enable_chunked_prefill",
        action="store_false",
        help="Force disable chunked prefill for vLLM.",
    )
    ap.set_defaults(vllm_enable_chunked_prefill=None)
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size for vLLM (control + exp).")

    ap.add_argument("--lmdeploy-backend", default="pytorch", choices=["pytorch", "turbomind"])
    ap.add_argument("--lmdeploy-session-len", type=int, default=0, help="LMDeploy engine session_len (0=use max_model_len).")
    ap.add_argument("--lmdeploy-max-batch-size", type=int, default=128, help="LMDeploy engine max_batch_size (best-effort).")
    ap.add_argument("--lmdeploy-log-level", default="WARNING", help="LMDeploy pipeline log_level.")

    ap.add_argument(
        "--mode",
        default="both",
        choices=["control", "exp", "both"],
        help="control=no verifier, exp=with verifier, both=default",
    )
    ap.add_argument(
        "--use-verifier-system-prompt",
        action="store_true",
        help="Include training-style system prompt for base actor generation (aligns with meta_template begin prompt).",
    )
    ap.add_argument(
        "--no-use-verifier-system-prompt",
        dest="use_verifier_system_prompt",
        action="store_false",
        help="Disable system prompt for base actor generation.",
    )
    ap.set_defaults(use_verifier_system_prompt=False)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if args.base_model:
        base_model_dir = Path(args.base_model).resolve()
    else:
        if run_dir is None:
            raise SystemExit("Must provide --base-model or --run-dir")
        base_model_dir = _detect_base_model_dir(run_dir)

    run_control = args.mode in ["control", "both"]
    run_exp = args.mode in ["exp", "both"]

    verifier_lora_dir: Optional[Path] = None
    if args.verifier_lora:
        verifier_lora_dir = Path(args.verifier_lora).resolve()
    elif run_exp and run_dir is not None:
        # Best-effort auto-detect (kept for backwards compatibility with old run-dir based eval).
        try:
            verifier_lora_dir = _detect_verifier_lora_dir(run_dir)
        except Exception:
            verifier_lora_dir = None

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device_map = None if str(args.device_map).lower() in ("none", "null", "") else args.device_map

    tokenizer_dir: Optional[Path] = None
    if str(args.tokenizer_path or "").strip():
        tokenizer_dir = Path(args.tokenizer_path).resolve()
        if not tokenizer_dir.exists():
            raise SystemExit(f"tokenizer_path not found: {tokenizer_dir}")

    tokenizer = _load_tokenizer(str(tokenizer_dir if tokenizer_dir is not None else base_model_dir))
    # Load training chat_template if present.
    if verifier_lora_dir is not None:
        tmpl = verifier_lora_dir / "chat_template.jinja"
        if tmpl.exists():
            try:
                tokenizer.chat_template = tmpl.read_text(encoding="utf-8")
            except Exception:
                pass

    max_model_len = int(args.vllm_max_model_len) if int(args.vllm_max_model_len) > 0 else int(args.max_prompt_tokens + args.max_response_tokens)

    model = None
    llm = None
    lmdeploy_pipe = None
    if args.backend == "hf":
        if verifier_lora_dir is not None:
            model = _load_model_with_verifier_lora(str(base_model_dir), str(verifier_lora_dir), dtype=dtype, device_map=device_map)
        else:
            model = _load_hf_model_no_lora(str(base_model_dir), dtype=dtype, device_map=device_map)
    elif args.backend == "lmdeploy":
        try:
            from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, pipeline  # type: ignore
        except Exception as e:
            raise SystemExit(f"lmdeploy backend selected but lmdeploy is not available: {e}")
        session_len = int(args.lmdeploy_session_len) if int(args.lmdeploy_session_len) > 0 else int(max_model_len)
        max_batch_size = int(args.lmdeploy_max_batch_size) if int(args.lmdeploy_max_batch_size) > 0 else 128
        engine_kwargs = {"session_len": int(session_len), "max_batch_size": int(max_batch_size)}
        if str(args.lmdeploy_backend) == "turbomind":
            filtered = {k: v for k, v in engine_kwargs.items() if hasattr(TurbomindEngineConfig, k)}
            backend_config = TurbomindEngineConfig(**filtered)
        else:
            filtered = {k: v for k, v in engine_kwargs.items() if hasattr(PytorchEngineConfig, k)}
            backend_config = PytorchEngineConfig(**filtered)
        lmdeploy_pipe = pipeline(
            str(base_model_dir),
            backend_config=backend_config,
            log_level=str(args.lmdeploy_log_level),
            max_log_len=int(os.environ.get("LMDEPLOY_MAX_LOG_LEN", "10")),
        )
        if run_exp:
            raise SystemExit("backend=lmdeploy currently supports only --mode control (verifier LoRA interventions require vLLM).")
    else:
        llm = _init_vllm_engine(
            model_dir=base_model_dir,
            tokenizer_dir=tokenizer_dir,
            verifier_lora_dir=verifier_lora_dir,
            tensor_parallel_size=args.vllm_tp,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
            max_model_len=max_model_len,
            enforce_eager=bool(args.vllm_enforce_eager),
            max_num_batched_tokens=int(args.vllm_max_num_batched_tokens),
            enable_chunked_prefill=args.vllm_enable_chunked_prefill,
        )

    prompt_items: List[Tuple[Any, Dict[str, Any]]] = []
    if args.prompts_file:
        prompt_items.extend(_iter_prompt_items_from_file(Path(args.prompts_file), args.prompt_key))
    # CLI --prompt is treated as a plain user string.
    for p in [p for p in (args.prompt or []) if p]:  # type: ignore[arg-type]
        prompt_items.append((p, {"prompt": p, "source": "cli"}))
    if not prompt_items:
        raise SystemExit("No prompts provided. Use --prompt or --prompts-file.")

    stop_token_ids: List[int] = []
    if args.stop_token_id:
        for x in args.stop_token_id:
            try:
                stop_token_ids.append(int(x))
            except Exception:
                pass
    else:
        stop_token_ids = [151645]

    system_prompt = VERIFIER_SYSTEM_PROMPT if args.use_verifier_system_prompt else None

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        batch_size = max(1, int(args.batch_size))
        seed = int(args.seed) if int(args.seed) != 0 else None

        total = len(prompt_items)
        batch_starts = range(0, total, batch_size)
        if args.progress and tqdm is not None:
            batch_starts = tqdm(batch_starts, total=(total + batch_size - 1) // batch_size, desc="eval")

        for start in batch_starts:
            batch_items = [
                (start + j, prompt_items[start + j][0], prompt_items[start + j][1])
                for j in range(0, min(batch_size, total - start))
            ]

            prompt_ids_list: List[List[int]] = []
            prompt_text_list: List[str] = []
            for _, prompt, _origin in batch_items:
                p_ids = _prompt_obj_to_token_ids(tokenizer, prompt, system_prompt=system_prompt)
                if args.max_prompt_tokens and int(args.max_prompt_tokens) > 0 and len(p_ids) > int(args.max_prompt_tokens):
                    p_ids = p_ids[-int(args.max_prompt_tokens) :]
                prompt_ids_list.append(p_ids)
                prompt_text_list.append(tokenizer.decode(p_ids, skip_special_tokens=True))

            recs: List[Dict[str, Any]] = []
            for (i, prompt, origin_info), p_ids, p_text in zip(batch_items, prompt_ids_list, prompt_text_list):
                recs.append(
                    {
                        "idx": i,
                        "prompt": prompt,
                        "prompt_text": p_text,
                        "origin_info": origin_info,
                        "base_model_dir": str(base_model_dir),
                        "verifier_lora_dir": str(verifier_lora_dir) if verifier_lora_dir is not None else "",
                        "backend": args.backend,
                        "mode": args.mode,
                        "system_prompt": "VERIFIER_SYSTEM_PROMPT" if args.use_verifier_system_prompt else None,
                        "max_model_len": int(max_model_len),
                    }
                )

            if run_control:
                if args.backend == "vllm":
                    assert llm is not None
                    control_stop_token_ids = _build_control_stop_token_ids(tokenizer, stop_token_ids)
                    token_lists, texts, dt = _vllm_generate_batch(
                        llm,
                        prompt_token_ids_list=prompt_ids_list,
                        max_tokens=int(args.max_response_tokens),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=int(args.top_k),
                        stop_token_ids=control_stop_token_ids,
                        stop_sequences=None,
                        seed=seed,
                        lora_requests=None,
                    )
                    for rec, ids, txt in zip(recs, token_lists, texts):
                        rec["control"] = {"response_text": txt, "response_tokens": len(ids), "gen_s": float(dt)}
                else:
                    # HF backend: keep per-sample for simplicity.
                    assert model is not None
                    device = _get_model_device(model)
                    for rec, p_ids in zip(recs, prompt_ids_list):
                        input_ids = torch.tensor([p_ids], dtype=torch.long, device=device)
                        with model.disable_adapter():
                            t0 = time.time()
                            out = model.generate(
                                input_ids=input_ids,
                                max_new_tokens=int(args.max_response_tokens),
                                do_sample=True,
                                temperature=float(args.temperature),
                                top_p=float(args.top_p),
                                top_k=int(args.top_k) if int(args.top_k) > 0 else None,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                use_cache=True,
                            )
                            dt = time.time() - t0
                        ids = out[0, input_ids.size(1) :].tolist()
                        txt = tokenizer.decode(ids, skip_special_tokens=True)
                        rec["control"] = {"response_text": txt, "response_tokens": len(ids), "gen_s": float(dt)}

            if run_exp:
                if args.backend != "vllm":
                    raise SystemExit("HF exp mode is not supported in v2 (too slow). Use --backend vllm.")
                assert llm is not None

                states: List[_ExpState] = []
                for rec, p_ids, p_text in zip(recs, prompt_ids_list, prompt_text_list):
                    states.append(
                        _ExpState(
                            idx=int(rec["idx"]),
                            prompt_obj=rec["prompt"],
                            prompt_ids=list(p_ids),
                            prompt_text=str(p_text),
                            response_ids=[],
                            loss_masks=[],
                            interventions=[],
                            previous_hints=[],
                            tokens_since_boundary=0,
                            model_gen_tokens=0,
                            is_complete=False,
                        )
                    )

                _generate_with_verifier_vllm_batch(
                    llm=llm,
                    tokenizer=tokenizer,
                    verifier_lora_dir=verifier_lora_dir,
                    states=states,
                    max_model_len=int(max_model_len),
                    max_model_tokens=int(args.max_response_tokens),
                    token_check_interval=int(args.token_check_interval),
                    min_step_tokens=int(args.min_step_tokens),
                    max_interventions=int(args.max_interventions),
                    token_check_interval_late=int(args.token_check_interval_late),
                    token_check_late_start_tokens=int(args.token_check_late_start_tokens),
                    verifier_skip_budget_tokens=int(args.verifier_skip_budget_tokens),
                    verifier_max_prompt_length=int(args.verifier_max_prompt_length),
                    verifier_max_new_tokens=int(args.verifier_max_new_tokens),
                    verifier_max_hint_tokens=int(args.verifier_max_hint_tokens),
                    confidence_threshold=float(args.confidence_threshold),
                    wait_conf_tail_tokens=int(args.wait_conf_tail_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    stop_token_ids=stop_token_ids,
                    hint_rollback_window_tokens=int(args.hint_rollback_window_tokens),
                    pending_hint_tail_decode_tokens=int(args.pending_hint_tail_decode_tokens),
                    seed=seed,
                )

                for rec, s in zip(recs, states):
                    rec["exp"] = {
                        "response_text": tokenizer.decode(s.response_ids, skip_special_tokens=True),
                        "response_tokens_total": len(s.response_ids),
                        "response_tokens_model_gen": int(s.model_gen_tokens),
                        "interventions": s.interventions,
                        "num_interventions": len(s.interventions),
                        "gen_s": float(getattr(s, "_gen_s", 0.0)),
                        "error": s.error,
                    }

            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
