#!/usr/bin/env python3
"""
LMDeploy backend eval for Co-GRPO with Verifier LoRA interventions (64k-friendly).

This script is meant to align with older OpenCompass(LMDeploy) eval configs:
- session_len / max_seq_len up to 65536
- chat_template formatting

It generates:
- control: base actor output (no verifier)
- exp: step-wise generation with verifier interventions (hint injection), using a Verifier LoRA.

Notes / tradeoffs:
- Actor generation uses LMDeploy pipeline (stateless): each step re-sends full prompt + response-so-far.
- Verifier generation uses Transformers + PEFT on the same GPU (a second model copy). Keep batch_size small.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


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


def _extract_verifier_hint(verifier_text: str) -> str:
    import re

    if not verifier_text:
        return ""
    text = verifier_text
    text = re.sub(r"```\s*\n?.*?\n?\s*```", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

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
        hint = hint.replace("<WAIT>", "").strip()
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


def _format_hint_text(*, hint: str, response_text: str) -> str:
    hint = (hint or "").strip()
    if not hint:
        return ""

    if response_text.endswith("\n\n"):
        prefix = ""
    elif response_text.endswith("\n"):
        prefix = "\n"
    else:
        prefix = "\n\n"
    return f"{prefix}{hint}\n\n"


def _load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _get_potential_stop_words(tokenizer, model_dir: str) -> List[str]:
    """
    Best-effort: match OpenCompass' lmdeploy wrapper behavior by harvesting EOS stop strings.
    This helps prevent runaway generations when the model emits special EOS markers.
    """
    stop_words: List[str] = []
    try:
        from transformers import GenerationConfig  # type: ignore

        gen_cfg = GenerationConfig.from_pretrained(model_dir)
        eos_token_id = getattr(gen_cfg, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            try:
                stop_words.append(tokenizer.decode([int(eos_token_id)]))
            except Exception:
                pass
        elif isinstance(eos_token_id, list):
            for tid in eos_token_id:
                try:
                    s = tokenizer.decode([int(tid)])
                except Exception:
                    continue
                if s.startswith(" "):
                    s = s.strip()
                if s:
                    stop_words.append(s)
    except Exception:
        pass

    try:
        eos_tok = getattr(tokenizer, "eos_token", None)
        if eos_tok:
            stop_words.append(str(eos_tok))
    except Exception:
        pass

    # Remove empty/duplicates while preserving order.
    seen = set()
    uniq: List[str] = []
    for s in stop_words:
        s = str(s)
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _apply_stop_words(text: str, stop_words: List[str]) -> Tuple[str, bool]:
    if not text or not stop_words:
        return text, False
    cut = None
    for s in stop_words:
        if not s:
            continue
        j = text.find(s)
        if j >= 0:
            if cut is None or j < cut:
                cut = j
    if cut is None:
        return text, False
    return text[:cut], True


def _too_repetitive_suffix(text: str) -> bool:
    """
    Very cheap degeneration guard:
    - If the last paragraph repeats the same line many times, stop early.
    This is intended to catch pathological loops like 'Thus maybe ...' repeated hundreds of times.
    """
    if not text:
        return False
    tail = text[-8192:]
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if len(lines) < 20:
        return False
    window = lines[-40:]

    # If the suffix collapses into a tiny set of repeated lines, stop.
    # This catches both "AAAAA..." and alternating patterns like "A B A B ...".
    from collections import Counter

    counts = Counter(window)
    (most_common_line, most_common_n) = counts.most_common(1)[0]
    if len(most_common_line) >= 20 and most_common_n >= 10:
        return True

    # Backward-compatible check: last line repeats heavily.
    last = window[-1]
    if len(last) >= 20:
        same = sum(1 for ln in window if ln == last)
        if same >= 10:
            return True

    # Another cheap heuristic: very low unique ratio in suffix.
    uniq = len(counts)
    if uniq <= 3 and len(window) >= 24:
        long_lines = sum(1 for ln in window if len(ln) >= 20)
        if long_lines >= 12:
            return True

    return False


def _decode_stop_token_ids(tokenizer, token_ids: List[int]) -> List[str]:
    out: List[str] = []
    for tid in token_ids or []:
        try:
            s = tokenizer.decode([int(tid)])
        except Exception:
            continue
        if not s:
            continue
        if s.startswith(" "):
            s = s.strip()
        if s:
            out.append(s)
    # de-dup preserving order
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _http_json(method: str, url: str, payload: Optional[dict], timeout_s: int) -> dict:
    body = None
    headers = {"content-type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=body, headers=headers, method=method.upper())
    try:
        with urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        try:
            msg = e.read().decode("utf-8", errors="ignore")
        except Exception:
            msg = str(e)
        raise RuntimeError(f"HTTP {e.code} calling {url}: {msg}") from e
    except URLError as e:
        raise RuntimeError(f"URL error calling {url}: {e}") from e


def _normalize_api_base(api_base: str) -> str:
    api_base = (api_base or "").strip().rstrip("/")
    if not api_base:
        raise ValueError("Empty actor api_base.")
    if not api_base.endswith("/v1"):
        api_base = api_base + "/v1"
    return api_base


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def _openai_pick_model_id(api_base: str, timeout_s: int) -> str:
    api_base = _normalize_api_base(api_base)
    resp = _http_json("GET", f"{api_base}/models", None, timeout_s)
    data = resp.get("data")
    if isinstance(data, list) and data:
        mid = data[0].get("id")
        if isinstance(mid, str) and mid:
            return mid
    models = resp.get("models")
    if isinstance(models, list) and models:
        mid = models[0].get("id")
        if isinstance(mid, str) and mid:
            return mid
    raise RuntimeError(f"Failed to pick model id from {api_base}/models: keys={list(resp.keys())}")


def _wait_openai_ready(api_base: str, timeout_s: int) -> str:
    api_base = _normalize_api_base(api_base)
    deadline = time.time() + float(timeout_s)
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            return _openai_pick_model_id(api_base, timeout_s=30)
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"Timeout waiting for server ready: {api_base} (last_err={last_err})")


def _openai_completions(
    *,
    api_base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]],
    timeout_s: int,
) -> str:
    api_base = _normalize_api_base(api_base)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": 1,
    }
    if stop:
        payload["stop"] = list(stop)
    resp = _http_json("POST", f"{api_base}/completions", payload, timeout_s)
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Bad /completions response: keys={list(resp.keys())}")
    c0 = choices[0]
    if isinstance(c0, dict):
        txt = c0.get("text", "")
        if txt is None:
            txt = ""
        return str(txt)
    return str(c0)


def _openai_chat_completions(
    *,
    api_base: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]],
    timeout_s: int,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """
    OpenAI-compatible /v1/chat/completions.
    If response contains `reasoning_content`, we emulate OpenCompass behavior by
    concatenating it with '</think>' and `content`.
    """
    api_base = _normalize_api_base(api_base)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": 1,
        "stop": stop,
    }
    if extra_body:
        payload.update(extra_body)
    resp = _http_json("POST", f"{api_base}/chat/completions", payload, timeout_s)
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Bad /chat/completions response: keys={list(resp.keys())}")
    msg = (choices[0] or {}).get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise RuntimeError("Bad /chat/completions response: missing choices[0].message")
    content = msg.get("content", "") or ""
    reasoning_content = msg.get("reasoning_content", "") or ""
    if reasoning_content:
        if content:
            return str(reasoning_content) + "</think>" + str(content)
        return str(reasoning_content)
    return str(content)


def _prompt_obj_to_raw_prompt(tokenizer, prompt_obj: Any, system_prompt: Optional[str]) -> str:
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

    # OpenCompass workaround: remove explicit bos prefix to avoid double-bos when engine tokenizes with add_special_tokens=True.
    try:
        bos = getattr(tokenizer, "bos_token", None)
        if bos and isinstance(raw, str) and raw.startswith(bos):
            raw = raw[len(bos) :]
    except Exception:
        pass
    return str(raw)


def _prompt_obj_to_chat_messages(prompt_obj: Any, system_prompt: Optional[str]) -> List[Dict[str, Any]]:
    """
    Convert prompt object (string or list[dict]) into OpenAI chat messages.
    This is used by --actor-transport=openai (api_server) to align with OpenCompass.
    """
    if isinstance(prompt_obj, list):
        messages: List[Dict[str, Any]] = []
        for m in prompt_obj:
            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content")
                if isinstance(role, str) and content is not None:
                    messages.append({"role": role, "content": str(content)})
            elif isinstance(m, str):
                messages.append({"role": "user", "content": m})
        if system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": str(system_prompt)}] + messages
        return messages

    # Scalar prompt: wrap as user message.
    if not isinstance(prompt_obj, str):
        try:
            prompt_obj = json.dumps(prompt_obj, ensure_ascii=False)
        except Exception:
            prompt_obj = str(prompt_obj)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": str(prompt_obj)})
    return messages


def _tail_truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[-max_tokens:]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _iter_prompt_items_from_file(path: Path, prompt_key: str) -> List[Tuple[Any, Dict[str, Any]]]:
    rows: List[Tuple[Any, Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                rows.append((obj, {"raw": obj}))
                continue
            prompt_obj = obj.get(prompt_key)
            if prompt_obj is None:
                # fall back to common key
                prompt_obj = obj.get("question")
            rows.append((prompt_obj, obj))
    return rows


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    verifier_prompt_budget = max(1, min(int(verifier_max_prompt_length), int(max_model_len) - int(verifier_max_new_tokens)))

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


def _load_verifier_model(base_model_dir: str, verifier_lora_dir: str, dtype: torch.dtype, device_map: Optional[str]):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    model.eval()
    model = PeftModel.from_pretrained(model, verifier_lora_dir, is_trainable=False)
    model.eval()
    return model


def _verifier_generate(
    *,
    model,
    tokenizer,
    question: str,
    current_reasoning: str,
    verifier_max_prompt_length: int,
    verifier_max_new_tokens: int,
    max_model_len: int,
) -> str:
    device = _get_model_device(model)
    prompt_ids = _build_verifier_prompt_ids(
        tokenizer,
        question=question,
        current_reasoning=current_reasoning,
        verifier_max_prompt_length=verifier_max_prompt_length,
        verifier_max_new_tokens=verifier_max_new_tokens,
        max_model_len=max_model_len,
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        t0 = time.time()
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=int(verifier_max_new_tokens),
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        _ = time.time() - t0
    gen_ids = out[0, input_ids.size(1) :].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval Co-GRPO with verifier interventions (actor=LMDeploy, 64k).")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--verifier-lora", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--prompt-key", default="question")
    ap.add_argument("--out-jsonl", required=True)

    ap.add_argument("--mode", default="both", choices=["control", "exp", "both"])
    ap.add_argument("--use-verifier-system-prompt", action="store_true")

    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device-map", default="auto")

    ap.add_argument(
        "--actor-transport",
        default="pipeline",
        choices=["pipeline", "openai"],
        help="Actor transport: lmdeploy pipeline (default) or OpenAI-compatible /v1/chat/completions.",
    )
    ap.add_argument(
        "--actor-api-base",
        default="",
        help="OpenAI-compatible API base for actor (e.g. http://127.0.0.1:23333/v1). Required when --actor-transport=openai.",
    )
    ap.add_argument("--actor-api-timeout", type=int, default=600, help="HTTP timeout seconds for OpenAI API calls.")
    ap.add_argument(
        "--start-actor-api-server",
        action="store_true",
        help="Start an in-job `lmdeploy serve api_server ...` and use it as actor (OpenAI transport).",
    )
    ap.add_argument("--actor-api-port", type=int, default=0, help="Port for in-job api_server (0=auto pick free port).")
    ap.add_argument("--actor-api-tp", type=int, default=1)
    ap.add_argument("--actor-api-worker-num", type=int, default=1)
    ap.add_argument(
        "--actor-api-extra-cli",
        default="",
        help="Extra CLI appended to lmdeploy serve api_server (e.g. \"--backend pytorch --session-len 65536 --max-batch-size 128\").",
    )
    ap.add_argument(
        "--actor-disable-thinking",
        action="store_true",
        help="In OpenAI actor mode, do not send chat_template_kwargs.enable_thinking=True.",
    )

    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--max-response-tokens", type=int, default=65536)
    ap.add_argument("--token-check-interval", type=int, default=4096)
    ap.add_argument("--min-step-tokens", type=int, default=4096)
    ap.add_argument("--max-interventions", type=int, default=5)

    ap.add_argument("--verifier-max-prompt-length", type=int, default=16384)
    ap.add_argument("--verifier-max-new-tokens", type=int, default=2048)
    ap.add_argument("--verifier-max-hint-tokens", type=int, default=512)

    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--stop-token-id", action="append", default=[])
    ap.add_argument("--stop-words", action="append", default=[], help="Additional stop words for LMDeploy actor.")
    ap.add_argument("--stop-at-think-end", action="store_true", help="Stop actor generation when '</think>' appears.")
    ap.add_argument("--degeneration-guard", action="store_true", help="Stop early on obvious repetition loops.")

    ap.add_argument("--lmdeploy-backend", default="pytorch", choices=["pytorch", "turbomind"])
    ap.add_argument("--lmdeploy-session-len", type=int, default=65536)
    ap.add_argument("--lmdeploy-max-batch-size", type=int, default=128)
    ap.add_argument("--lmdeploy-log-level", default="WARNING")

    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    base_model_dir = str(Path(args.base_model).resolve())
    verifier_lora_dir = str(Path(args.verifier_lora).resolve())
    prompts_file = Path(args.prompts_file).resolve()
    out_path = Path(args.out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device_map = None if str(args.device_map).lower() in ("none", "null", "") else str(args.device_map)

    tokenizer = _load_tokenizer(base_model_dir)
    # If verifier adapter carries a training chat_template, use it for prompt formatting.
    tmpl = Path(verifier_lora_dir) / "chat_template.jinja"
    if tmpl.exists():
        try:
            tokenizer.chat_template = tmpl.read_text(encoding="utf-8")
        except Exception:
            pass

    system_prompt = VERIFIER_SYSTEM_PROMPT if args.use_verifier_system_prompt else None

    stop_token_ids: List[int] = []
    if args.stop_token_id:
        for x in args.stop_token_id:
            try:
                stop_token_ids.append(int(x))
            except Exception:
                pass
    else:
        stop_token_ids = [151645]

    # Actor stop words (applied both in pipeline and OpenAI /completions mode).
    base_stop_words = _get_potential_stop_words(tokenizer, base_model_dir)
    base_stop_words.extend(_decode_stop_token_ids(tokenizer, stop_token_ids))
    user_stop_words = [str(s) for s in (args.stop_words or []) if str(s)]
    if args.stop_at_think_end:
        user_stop_words.append("</think>")
    stop_words = list(dict.fromkeys(base_stop_words + user_stop_words))

    actor_transport = str(args.actor_transport)
    actor_api_base = str(args.actor_api_base or "").strip()
    actor_api_model = ""
    actor_proc: Optional[subprocess.Popen] = None

    actor_pipe = None
    GenerationConfig = None
    supports_stop_words = False

    if args.start_actor_api_server:
        extra_cli = str(args.actor_api_extra_cli or "").strip()
        if not extra_cli:
            extra_cli = (
                f"--backend {args.lmdeploy_backend} "
                f"--session-len {int(args.lmdeploy_session_len)} "
                f"--max-batch-size {int(args.lmdeploy_max_batch_size)}"
            )

        # IMPORTANT: do NOT use the auto-eval-pipeline llm_start.py directly here because it
        # hard-codes ports (23333+i) and can conflict under `--host-network=true` when multiple
        # 1-GPU jobs land on the same node. Instead, we mimic its behavior but choose a free port.
        port = int(args.actor_api_port)
        if port <= 0:
            port = _pick_free_local_port()

        env = os.environ.copy()
        env.setdefault("LMDEPLOY_SKIP_WARMUP", "1")
        # Match llm_start.py offline envs.
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_EVALUATE_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")

        cmd = [
            "lmdeploy",
            "serve",
            "api_server",
            base_model_dir,
            "--tp",
            str(int(args.actor_api_tp)),
            "--server-port",
            str(int(port)),
        ]
        cmd.extend(shlex.split(extra_cli))
        actor_proc = subprocess.Popen(cmd, env=env)

        actor_transport = "openai"
        actor_api_base = f"http://127.0.0.1:{int(port)}/v1"
        actor_api_model = _wait_openai_ready(actor_api_base, timeout_s=int(args.actor_api_timeout))
    elif actor_transport == "openai":
        if not actor_api_base:
            raise SystemExit("--actor-transport=openai requires --actor-api-base (or --start-actor-api-server).")
        actor_api_model = _wait_openai_ready(actor_api_base, timeout_s=int(args.actor_api_timeout))

    if actor_transport == "pipeline":
        # LMDeploy actor pipeline.
        try:
            from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, GenerationConfig as _GenCfg, pipeline  # type: ignore
        except Exception as e:
            raise SystemExit(f"Missing lmdeploy (backend=lmdeploy required): {e}")

        GenerationConfig = _GenCfg
        engine_kwargs = {
            "session_len": int(args.lmdeploy_session_len),
            "max_batch_size": int(args.lmdeploy_max_batch_size),
        }
        if args.lmdeploy_backend == "turbomind":
            filtered = {k: v for k, v in engine_kwargs.items() if hasattr(TurbomindEngineConfig, k)}
            backend_config = TurbomindEngineConfig(**filtered)
        else:
            filtered = {k: v for k, v in engine_kwargs.items() if hasattr(PytorchEngineConfig, k)}
            backend_config = PytorchEngineConfig(**filtered)
        actor_pipe = pipeline(
            base_model_dir,
            backend_config=backend_config,
            log_level=str(args.lmdeploy_log_level),
        )
        if stop_words:
            try:
                _GenCfg(stop_words=stop_words)  # type: ignore
                supports_stop_words = True
            except TypeError:
                supports_stop_words = False

    actor_extra_body: Optional[Dict[str, Any]] = None
    if actor_transport == "openai" and not args.actor_disable_thinking:
        # Match OpenCompass' OpenAISDK usage for InternS1.
        actor_extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

    # Verifier model (Transformers + PEFT).
    verifier_model = _load_verifier_model(base_model_dir, verifier_lora_dir, dtype=dtype, device_map=device_map)

    prompt_items = _iter_prompt_items_from_file(prompts_file, args.prompt_key)
    if not prompt_items:
        raise SystemExit(f"No prompts found in: {prompts_file}")

    run_control = args.mode in ("control", "both")
    run_exp = args.mode in ("exp", "both")

    idxs = list(range(len(prompt_items)))
    if args.progress and tqdm is not None:
        idxs = tqdm(idxs, total=len(idxs), desc="eval", unit="rec")  # type: ignore[assignment]

    with out_path.open("w", encoding="utf-8") as w:
        for i in idxs:
            prompt_obj, origin_info = prompt_items[i]
            question = None
            if isinstance(origin_info, dict):
                question = origin_info.get(args.prompt_key) or origin_info.get("question")
            if question is None:
                question = prompt_obj if isinstance(prompt_obj, str) else ""

            base_prompt_raw = _prompt_obj_to_raw_prompt(tokenizer, prompt_obj, system_prompt=system_prompt)
            base_prompt_raw = _tail_truncate_to_tokens(tokenizer, base_prompt_raw, int(args.max_prompt_tokens))
            actor_base_messages = _prompt_obj_to_chat_messages(prompt_obj, system_prompt=system_prompt)

            rec: Dict[str, Any] = {
                "idx": int(i),
                "prompt": prompt_obj,
                "prompt_text": base_prompt_raw,
                "origin_info": origin_info,
                "base_model_dir": base_model_dir,
                "verifier_lora_dir": verifier_lora_dir,
                "backend": "lmdeploy",
                "mode": args.mode,
                "system_prompt": "VERIFIER_SYSTEM_PROMPT" if args.use_verifier_system_prompt else None,
                "max_model_len": int(args.lmdeploy_session_len),
            }

            if run_control:
                # Chunked generation with degeneration guard (mirrors EXP loop but without verifier calls).
                response_text = ""
                model_gen_tokens = 0
                error: Optional[str] = None

                t0 = time.time()
                while True:
                    if model_gen_tokens >= int(args.max_response_tokens):
                        break

                    # Context length check (best-effort, based on tokenizer).
                    try:
                        ctx_len = len(tokenizer.encode(base_prompt_raw + response_text, add_special_tokens=False))
                    except Exception:
                        ctx_len = 0
                    if ctx_len >= int(args.lmdeploy_session_len):
                        break

                    gen_remaining = int(args.max_response_tokens) - int(model_gen_tokens)
                    step_tokens = min(int(args.token_check_interval), int(gen_remaining))
                    # Also respect session length.
                    if ctx_len > 0:
                        ctx_remaining = int(args.lmdeploy_session_len) - int(ctx_len)
                        step_tokens = min(int(step_tokens), int(ctx_remaining))
                    step_tokens = max(1, int(step_tokens))

                    try:
                        if actor_transport == "openai":
                            msgs = actor_base_messages
                            if response_text:
                                msgs = msgs + [{"role": "assistant", "content": response_text}]
                            new_text = _openai_chat_completions(
                                api_base=actor_api_base,
                                model=actor_api_model,
                                messages=msgs,
                                max_tokens=int(step_tokens),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p) if args.top_p is not None else 1.0,
                                stop=stop_words,
                                timeout_s=int(args.actor_api_timeout),
                                extra_body=actor_extra_body,
                            )
                        else:
                            assert actor_pipe is not None and GenerationConfig is not None
                            gen_kwargs = dict(
                                max_new_tokens=int(step_tokens),
                                min_new_tokens=1,
                                temperature=float(args.temperature),
                                top_p=float(args.top_p) if args.top_p is not None else 1.0,
                                top_k=int(args.top_k) if int(args.top_k) > 0 else 0,
                            )
                            if supports_stop_words and stop_words:
                                gen_kwargs["stop_words"] = stop_words
                            gen_cfg = GenerationConfig(**gen_kwargs)
                            out = actor_pipe([base_prompt_raw + response_text], gen_config=gen_cfg, do_preprocess=False)[0]
                            new_text = str(getattr(out, "text", "") or "")
                    except Exception as e:
                        error = f"actor_generate_error: {e}"
                        break
                    new_text, hit_stop = _apply_stop_words(new_text, stop_words)
                    if not new_text:
                        break
                    response_text += new_text
                    model_gen_tokens += len(tokenizer.encode(new_text, add_special_tokens=False))
                    if hit_stop:
                        break
                    if args.degeneration_guard and _too_repetitive_suffix(response_text):
                        break

                dt = float(time.time() - t0)
                # Token counts after any post-cut.
                tok_total = len(tokenizer.encode(response_text, add_special_tokens=False)) if response_text else 0
                rec["control"] = {
                    "response_text": response_text,
                    "response_tokens_model_gen": int(model_gen_tokens),
                    "response_tokens_total": int(tok_total),
                    "gen_s": dt,
                    "error": error,
                }

            if run_exp:
                response_text = ""
                interventions: List[Dict[str, Any]] = []
                previous_hints: List[str] = []
                model_gen_tokens = 0
                error: Optional[str] = None

                t0 = time.time()
                while True:
                    if model_gen_tokens >= int(args.max_response_tokens):
                        break
                    if len(interventions) >= int(args.max_interventions):
                        # No more verifier calls; still allow generation to finish.
                        pass

                    # Context length check (best-effort, based on tokenizer).
                    try:
                        ctx_len = len(tokenizer.encode(base_prompt_raw + response_text, add_special_tokens=False))
                    except Exception:
                        ctx_len = 0
                    if ctx_len >= int(args.lmdeploy_session_len):
                        break

                    gen_remaining = int(args.max_response_tokens) - int(model_gen_tokens)
                    step_tokens = min(int(args.token_check_interval), int(gen_remaining))
                    # Also respect session length.
                    if ctx_len > 0:
                        ctx_remaining = int(args.lmdeploy_session_len) - int(ctx_len)
                        step_tokens = min(int(step_tokens), int(ctx_remaining))
                    step_tokens = max(1, int(step_tokens))

                    try:
                        if actor_transport == "openai":
                            msgs = actor_base_messages
                            if response_text:
                                msgs = msgs + [{"role": "assistant", "content": response_text}]
                            new_text = _openai_chat_completions(
                                api_base=actor_api_base,
                                model=actor_api_model,
                                messages=msgs,
                                max_tokens=int(step_tokens),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p) if args.top_p is not None else 1.0,
                                stop=stop_words,
                                timeout_s=int(args.actor_api_timeout),
                                extra_body=actor_extra_body,
                            )
                        else:
                            assert actor_pipe is not None and GenerationConfig is not None
                            gen_kwargs = dict(
                                max_new_tokens=int(step_tokens),
                                min_new_tokens=1,
                                temperature=float(args.temperature),
                                top_p=float(args.top_p) if args.top_p is not None else 1.0,
                                top_k=int(args.top_k) if int(args.top_k) > 0 else 0,
                            )
                            if supports_stop_words and stop_words:
                                gen_kwargs["stop_words"] = stop_words
                            gen_cfg = GenerationConfig(**gen_kwargs)
                            out = actor_pipe([base_prompt_raw + response_text], gen_config=gen_cfg, do_preprocess=False)[0]
                            new_text = str(getattr(out, "text", "") or "")
                    except Exception as e:
                        error = f"actor_generate_error: {e}"
                        break
                    new_text, hit_stop = _apply_stop_words(new_text, stop_words)
                    if not new_text:
                        break
                    response_text += new_text
                    # Use tokenizer-based token length for consistency with any post-cut by stop_words.
                    model_gen_tokens += len(tokenizer.encode(new_text, add_special_tokens=False))
                    if hit_stop:
                        break
                    if args.degeneration_guard and _too_repetitive_suffix(response_text):
                        break

                    if len(interventions) >= int(args.max_interventions):
                        continue

                    # Always call verifier after each chunk (training-consistent).
                    try:
                        verifier_text = _verifier_generate(
                            model=verifier_model,
                            tokenizer=tokenizer,
                            question=str(question),
                            current_reasoning=response_text,
                            verifier_max_prompt_length=int(args.verifier_max_prompt_length),
                            verifier_max_new_tokens=int(args.verifier_max_new_tokens),
                            max_model_len=int(args.verifier_max_prompt_length) + int(args.verifier_max_new_tokens),
                        )
                    except Exception as e:
                        error = f"verifier_generate_error: {e}"
                        break
                    decision = _parse_verifier_decision(verifier_text)

                    hint = (decision.hint or "").strip()
                    if not hint or decision.action != "Intervene":
                        continue
                    # Truncate hint tokens.
                    if int(args.verifier_max_hint_tokens) > 0:
                        hint_ids = tokenizer.encode(hint, add_special_tokens=False)
                        if len(hint_ids) > int(args.verifier_max_hint_tokens):
                            hint = tokenizer.decode(hint_ids[: int(args.verifier_max_hint_tokens)], skip_special_tokens=True).strip()
                    if not _hint_allowed(hint, previous_hints):
                        continue
                    hint_text = _format_hint_text(hint=hint, response_text=response_text)
                    if hint_text:
                        response_text += hint_text
                        previous_hints.append(hint)
                    interventions.append(
                        {
                            "hint": hint,
                            "hint_text": hint_text,
                            "verifier_action": decision.action,
                            "verifier_critique": verifier_text,
                            "response_tokens_after": len(tokenizer.encode(response_text, add_special_tokens=False)),
                        }
                    )

                dt = float(time.time() - t0)
                rec["exp"] = {
                    "response_text": response_text,
                    "response_tokens_total": len(tokenizer.encode(response_text, add_special_tokens=False)),
                    "response_tokens_model_gen": int(model_gen_tokens),
                    "interventions": interventions,
                    "num_interventions": len(interventions),
                    "gen_s": dt,
                    "error": error,
                }

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.flush()

    if actor_proc is not None:
        try:
            actor_proc.terminate()
        except Exception:
            pass
        try:
            actor_proc.wait(timeout=10)
        except Exception:
            try:
                actor_proc.kill()
            except Exception:
                pass

    print(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
