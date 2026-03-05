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
import hashlib
import importlib.util
import json
import math
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


# Prefer online-canonical verifier prompts from the shared hint injection module.
try:
    _inj_prompts0 = _load_online_hint_injection_module()
    VERIFIER_SYSTEM_PROMPT = str(getattr(_inj_prompts0, "VERIFIER_SYSTEM_PROMPT", VERIFIER_SYSTEM_PROMPT))
    VERIFIER_INTERVENE_PROMPT = str(getattr(_inj_prompts0, "VERIFIER_INTERVENE_PROMPT", VERIFIER_INTERVENE_PROMPT))
except Exception:
    pass


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

    # Fallback for single-line repetition (no '\n'): split by punctuation and
    # detect repeated "sentences" near the tail.
    try:
        import re
        from collections import Counter

        parts = [p.strip() for p in re.split(r"[\\.!?]+", tail) if p.strip()]
        if len(parts) >= 40:
            win = parts[-80:]
            cnt = Counter(win)
            most, n = cnt.most_common(1)[0]
            if len(most) >= 20 and n >= 10:
                return True
            last = win[-1]
            if len(last) >= 20:
                same = sum(1 for p in win if p == last)
                if same >= 10:
                    return True

        # Token-level n-gram repetition (robust to minor punctuation/whitespace diffs).
        words = [w for w in re.split(r"\s+", tail.lower()) if w]
        if len(words) >= 200:
            seq = words[-500:]
            ngram_n = 4
            grams = [
                (seq[i], seq[i + 1], seq[i + 2], seq[i + 3])
                for i in range(0, max(0, len(seq) - ngram_n + 1))
            ]
            if grams:
                uniq_ratio = float(len(set(grams)) / max(1, len(grams)))
                # Empirically, pathological loops collapse uniq_ratio << 0.2.
                if uniq_ratio < 0.15 and len(seq) >= 300:
                    return True
    except Exception:
        pass

    return False


def _has_complete_boxed_answer(text: str) -> bool:
    """
    Heuristic early-stop for math eval: if '\\boxed{...}' appears with a closing brace,
    consider the sample "finished" to avoid tail degeneration filling the remaining budget.
    """
    if not text:
        return False
    j = text.rfind("\\boxed{")
    if j < 0:
        return False
    k = text.find("}", j + len("\\boxed{"))
    if k < 0:
        return False
    # Avoid triggering on extremely early/accidental occurrences (shouldn't happen in prompts).
    return (k - j) <= 512


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
    api_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("ACTOR_API_KEY") or "").strip()
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
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


def _openai_pick_model_id(api_base: str, timeout_s: int, preferred_model: str = "") -> str:
    api_base = _normalize_api_base(api_base)
    resp = _http_json("GET", f"{api_base}/models", None, timeout_s)
    preferred_model = str(preferred_model or "").strip()
    if preferred_model:
        data = resp.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and str(item.get("id") or "").strip() == preferred_model:
                    return preferred_model
        models = resp.get("models")
        if isinstance(models, list):
            for item in models:
                if isinstance(item, dict) and str(item.get("id") or "").strip() == preferred_model:
                    return preferred_model
        raise RuntimeError(
            f"Preferred model not found on {api_base}/models: preferred={preferred_model}"
        )
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


def _wait_openai_ready(api_base: str, timeout_s: int, preferred_model: str = "") -> str:
    api_base = _normalize_api_base(api_base)
    deadline = time.time() + float(timeout_s)
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            return _openai_pick_model_id(api_base, timeout_s=30, preferred_model=preferred_model)
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
    top_k: Optional[int],
    seed: Optional[int],
    stop: Optional[List[str]],
    timeout_s: int,
) -> Tuple[str, str]:
    api_base = _normalize_api_base(api_base)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": 1,
    }
    if top_k is not None:
        payload["top_k"] = int(top_k)
    if seed is not None:
        payload["seed"] = int(seed)
    if stop:
        payload["stop"] = list(stop)
    resp = _http_json("POST", f"{api_base}/completions", payload, timeout_s)
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Bad /completions response: keys={list(resp.keys())}")
    c0 = choices[0]
    finish_reason = ""
    if isinstance(c0, dict):
        finish_reason = str(c0.get("finish_reason") or "")
    if isinstance(c0, dict):
        txt = c0.get("text", "")
        if txt is None:
            txt = ""
        return str(txt), finish_reason
    return str(c0), finish_reason


def _openai_chat_completions(
    *,
    api_base: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    seed: Optional[int],
    stop: Optional[List[str]],
    timeout_s: int,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
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
    if top_k is not None:
        payload["top_k"] = int(top_k)
    if seed is not None:
        payload["seed"] = int(seed)
    if extra_body:
        payload.update(extra_body)
    resp = _http_json("POST", f"{api_base}/chat/completions", payload, timeout_s)
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Bad /chat/completions response: keys={list(resp.keys())}")
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    finish_reason = str(c0.get("finish_reason") or "")
    msg = (c0 or {}).get("message") if isinstance(c0, dict) else None
    if not isinstance(msg, dict):
        raise RuntimeError("Bad /chat/completions response: missing choices[0].message")
    content = msg.get("content", "") or ""
    reasoning_content = msg.get("reasoning_content", "") or ""
    if reasoning_content:
        if content:
            return str(reasoning_content) + "</think>" + str(content), finish_reason
        return str(reasoning_content), finish_reason
    return str(content), finish_reason


def _is_404_on_completions(err: Exception) -> bool:
    msg = str(err)
    return "HTTP 404" in msg and "/completions" in msg


def _openai_generate_chunk_auto(
    *,
    api_mode_state: Dict[str, str],
    api_base: str,
    model: str,
    base_prompt_raw: str,
    base_messages: List[Dict[str, Any]],
    response_so_far: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    seed: Optional[int],
    stop: Optional[List[str]],
    timeout_s: int,
    extra_body: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    mode = str(api_mode_state.get("mode", "auto") or "auto").lower()
    if mode not in ("auto", "completions", "chat"):
        mode = "auto"

    first_err: Optional[Exception] = None
    if mode in ("auto", "completions"):
        try:
            text, finish_reason = _openai_completions(
                api_base=api_base,
                model=model,
                prompt=base_prompt_raw + response_so_far,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=top_k,
                seed=seed,
                stop=stop,
                timeout_s=int(timeout_s),
            )
            api_mode_state["mode"] = "completions"
            return text, finish_reason
        except Exception as e:
            first_err = e
            if mode == "completions" or not _is_404_on_completions(e):
                raise
            api_mode_state["mode"] = "chat"
            print(
                "[warn] /v1/completions unavailable (404); falling back to /v1/chat/completions.",
                flush=True,
            )

    messages = list(base_messages)
    if response_so_far:
        messages.append({"role": "assistant", "content": response_so_far})
    try:
        text, finish_reason = _openai_chat_completions(
            api_base=api_base,
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=top_k,
            seed=seed,
            stop=stop,
            timeout_s=int(timeout_s),
            extra_body=extra_body,
        )
        api_mode_state["mode"] = "chat"
        return text, finish_reason
    except Exception as e:
        if first_err is None:
            raise
        raise RuntimeError(f"{first_err}; chat_fallback_failed: {e}") from e


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


ONLINE_MATH_BOXED_REMINDER = "\nRemember to put your final answer within \\boxed{}."


def _append_user_prompt_suffix(prompt_obj: Any, suffix: str) -> Any:
    """
    Append a suffix to the last user message (or scalar prompt).

    This is used to align local eval prompts with online OpenCompass templates, e.g.:
      {question}\\nRemember to put your final answer within \\boxed{}.
    """
    suffix = str(suffix or "")
    if not suffix:
        return prompt_obj

    # Scalar prompt (common for math jsonl: {"question": "..."}).
    if isinstance(prompt_obj, str):
        # Avoid duplicating common online reminder.
        if "Remember to put your final answer within" in prompt_obj:
            return prompt_obj
        return prompt_obj.rstrip() + suffix

    # Chat messages: append to the last user message.
    if isinstance(prompt_obj, list):
        messages: List[Any] = []
        last_user_idx: Optional[int] = None
        for m in prompt_obj:
            if isinstance(m, dict):
                mm = dict(m)
                role = mm.get("role")
                if isinstance(role, str) and role.lower() in ("user", "human"):
                    last_user_idx = len(messages)
                messages.append(mm)
            else:
                messages.append(m)

        if last_user_idx is not None and isinstance(messages[last_user_idx], dict):
            content = messages[last_user_idx].get("content", "") or ""
            content = str(content)
            if "Remember to put your final answer within" in content:
                return messages
            messages[last_user_idx]["content"] = content.rstrip() + suffix
            return messages

        # Fallback: append a new user message (avoid leading newline).
        messages.append({"role": "user", "content": suffix.lstrip("\n")})
        return messages

    return prompt_obj


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


def _load_verifier_model(
    verifier_model_dir: str,
    verifier_lora_dir: Optional[str],
    dtype: torch.dtype,
    device_map: Optional[str],
):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        verifier_model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    model.eval()

    lora_dir = str(verifier_lora_dir or "").strip()
    if lora_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_dir, is_trainable=False)
        model.eval()
        print(f"[verifier] loaded LoRA: {lora_dir}", flush=True)
    else:
        print(f"[verifier] using full model (no LoRA): {verifier_model_dir}", flush=True)

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
    compute_wait_confidence: bool,
    wait_conf_tail_tokens: int,
) -> Tuple[str, Optional[float], Optional[float]]:
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
    if not compute_wait_confidence:
        with torch.no_grad():
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
        gen_ids = out[0, input_ids.size(1) :].tolist()
        return tokenizer.decode(gen_ids, skip_special_tokens=True), None, None

    # IMPORTANT: do NOT use `output_scores=True` here; it stores full vocab logits
    # for every generated token and can easily blow up memory when max_new_tokens is large.
    # Instead, run a greedy cached decode loop and only keep logprobs for the tail window.
    from collections import deque

    eos_id = tokenizer.eos_token_id
    try:
        tail = int(wait_conf_tail_tokens)
    except Exception:
        tail = 64
    if tail <= 0:
        tail = 64
    logprob_tail = deque(maxlen=tail)
    generated: List[int] = []

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        past = getattr(outputs, "past_key_values", None)
        next_logits = outputs.logits[:, -1, :]

        for _ in range(int(verifier_max_new_tokens)):
            token_id = int(torch.argmax(next_logits, dim=-1).item())

            try:
                logits_vec = next_logits[0].to(torch.float32)
                lp = float((logits_vec[token_id] - torch.logsumexp(logits_vec, dim=-1)).item())
                if math.isfinite(lp):
                    logprob_tail.append(lp)
            except Exception:
                pass

            generated.append(token_id)

            if eos_id is not None and int(token_id) == int(eos_id):
                break

            token_in = torch.tensor([[token_id]], dtype=torch.long, device=device)
            outputs = model(input_ids=token_in, use_cache=True, past_key_values=past)
            past = getattr(outputs, "past_key_values", None)
            next_logits = outputs.logits[:, -1, :]

    text = tokenizer.decode(generated, skip_special_tokens=True)
    if not logprob_tail:
        return text, None, None

    avg = float(sum(logprob_tail) / max(1, len(logprob_tail)))
    avg = float(min(0.0, max(-20.0, avg)))
    return text, float(math.exp(avg)), avg


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval Co-GRPO with verifier interventions (actor=LMDeploy, 64k).")
    ap.add_argument("--base-model", required=True)
    ap.add_argument(
        "--verifier-model",
        default="",
        help="Optional: full HF verifier model dir. If set, overrides --base-model for verifier loading.",
    )
    ap.add_argument(
        "--verifier-lora",
        default="",
        help="Optional: PEFT adapter dir for verifier. If omitted, verifier runs as a full model (no adapter).",
    )
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--prompt-key", default="question")
    ap.add_argument("--out-jsonl", required=True)

    ap.add_argument("--mode", default="both", choices=["control", "exp", "both"])
    ap.add_argument("--use-verifier-system-prompt", action="store_true")
    ap.add_argument(
        "--online-math-prompt",
        action="store_true",
        help="Align actor user prompt with OpenCompass math template by appending: "
        "'Remember to put your final answer within \\\\boxed{}.'",
    )

    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device-map", default="auto")

    ap.add_argument(
        "--actor-transport",
        default="pipeline",
        choices=["pipeline", "openai"],
        help="Actor transport: lmdeploy pipeline (default) or OpenAI-compatible API (auto uses /v1/completions, falls back to /v1/chat/completions on 404).",
    )
    ap.add_argument(
        "--actor-api-base",
        default="",
        help="OpenAI-compatible API base for actor (e.g. http://127.0.0.1:23333/v1). Required when --actor-transport=openai.",
    )
    ap.add_argument(
        "--actor-api-model",
        default="",
        help="OpenAI model id for actor API calls. When set, enforce this model id from /v1/models.",
    )
    ap.add_argument(
        "--actor-api-mode",
        default="auto",
        choices=["auto", "completions", "chat"],
        help="OpenAI API mode for actor calls: auto (default), completions, or chat.",
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
    ap.add_argument(
        "--hint-rollback-window-tokens",
        type=int,
        default=512,
        help="Align with online by_step: trim a trailing fragment to a natural boundary for hint anchor.",
    )
    ap.add_argument(
        "--pending-hint-tail-decode-tokens",
        type=int,
        default=128,
        help="Align with online by_step: decode this many tail tokens for post-</think> guard + hint prefix.",
    )

    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    # Align with LMDeploy OpenAI server default (top_k=40) when using temperature sampling.
    ap.add_argument("--top-k", type=int, default=40)
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
    ap.add_argument(
        "--max-sample-seconds",
        type=int,
        default=0,
        help="Hard wall-clock budget per sample (0 disables). Useful to avoid 64k tail loops hanging a shard.",
    )
    args = ap.parse_args()

    run_control = args.mode in ("control", "both")
    run_exp = args.mode in ("exp", "both")

    base_model_dir = str(Path(args.base_model).resolve())
    verifier_model_dir = str(Path(args.verifier_model).resolve()) if str(args.verifier_model or "").strip() else ""
    verifier_lora_dir = str(Path(args.verifier_lora).resolve()) if str(args.verifier_lora or "").strip() else ""
    if run_exp:
        if not verifier_lora_dir and not verifier_model_dir:
            raise SystemExit(
                "Need at least one of: --verifier-model (full model) or --verifier-lora (adapter) "
                "when --mode includes exp."
            )
        if not verifier_model_dir:
            verifier_model_dir = base_model_dir
    prompts_file = Path(args.prompts_file).resolve()
    out_path = Path(args.out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device_map = None if str(args.device_map).lower() in ("none", "null", "") else str(args.device_map)

    tokenizer = _load_tokenizer(base_model_dir)
    verifier_tokenizer = None
    if run_exp:
        verifier_tokenizer = tokenizer if verifier_model_dir == base_model_dir else _load_tokenizer(verifier_model_dir)

    # If verifier adapter carries a training chat_template, use it for verifier prompt formatting.
    if run_exp and verifier_lora_dir and verifier_tokenizer is not None:
        tmpl = Path(verifier_lora_dir) / "chat_template.jinja"
        if tmpl.exists():
            try:
                verifier_tokenizer.chat_template = tmpl.read_text(encoding="utf-8")
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

    # Online-canonical hint injection/rollback helpers (single source of truth).
    inj = None
    hint_rollback_window_tokens = 512
    pending_hint_tail_decode_tokens = 128
    if run_exp:
        inj = _load_online_hint_injection_module()
        try:
            hint_rollback_window_tokens = int(args.hint_rollback_window_tokens)
        except Exception:
            hint_rollback_window_tokens = 512
        try:
            pending_hint_tail_decode_tokens = int(args.pending_hint_tail_decode_tokens)
        except Exception:
            pending_hint_tail_decode_tokens = 128

    eos_token_ids: set[int] = {int(x) for x in (stop_token_ids or []) if isinstance(x, int)}
    try:
        if tokenizer.eos_token_id is not None:
            eos_token_ids.add(int(tokenizer.eos_token_id))
    except Exception:
        pass
    try:
        if tokenizer.pad_token_id is not None:
            eos_token_ids.add(int(tokenizer.pad_token_id))
    except Exception:
        pass

    actor_transport = str(args.actor_transport)
    actor_api_base = str(args.actor_api_base or "").strip()
    actor_api_model = str(args.actor_api_model or "").strip()
    actor_api_mode = str(args.actor_api_mode or "auto").strip().lower()
    if actor_api_mode not in ("auto", "completions", "chat"):
        actor_api_mode = "auto"
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
        actor_api_model = _wait_openai_ready(
            actor_api_base,
            timeout_s=int(args.actor_api_timeout),
            preferred_model=actor_api_model,
        )
    elif actor_transport == "openai":
        if not actor_api_base:
            raise SystemExit("--actor-transport=openai requires --actor-api-base (or --start-actor-api-server).")
        actor_api_model = _wait_openai_ready(
            actor_api_base,
            timeout_s=int(args.actor_api_timeout),
            preferred_model=actor_api_model,
        )

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
    actor_api_mode_state: Dict[str, str] = {"mode": actor_api_mode if actor_transport == "openai" else "pipeline"}

    run_param_fingerprint = {
        "base_model": base_model_dir,
        "verifier_model": verifier_model_dir if run_exp else "",
        "verifier_lora": verifier_lora_dir if run_exp else "",
        "mode": args.mode,
        "actor_transport": actor_transport,
        "actor_api_base": actor_api_base if actor_transport == "openai" else "",
        "actor_api_model": actor_api_model if actor_transport == "openai" else "",
        "actor_api_mode": actor_api_mode if actor_transport == "openai" else "",
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "seed": int(args.seed),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "max_response_tokens": int(args.max_response_tokens),
        "token_check_interval": int(args.token_check_interval),
        "min_step_tokens": int(args.min_step_tokens),
        "max_interventions": int(args.max_interventions),
        "lmdeploy_backend": str(args.lmdeploy_backend),
        "lmdeploy_session_len": int(args.lmdeploy_session_len),
        "lmdeploy_max_batch_size": int(args.lmdeploy_max_batch_size),
        "use_verifier_system_prompt": bool(args.use_verifier_system_prompt),
        "online_math_prompt": bool(args.online_math_prompt),
        "degeneration_guard": bool(args.degeneration_guard),
        "stop_token_ids": [int(x) for x in stop_token_ids],
        "stop_words": list(stop_words),
    }
    run_param_hash = hashlib.sha1(
        json.dumps(run_param_fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    # Verifier model (Transformers + PEFT).
    print(f"[config] actor_base_model={base_model_dir}", flush=True)
    verifier_model = None
    if run_exp:
        print(f"[config] verifier_model={verifier_model_dir}", flush=True)
        print(f"[config] verifier_lora={(verifier_lora_dir or '(none)')}", flush=True)
        verifier_model = _load_verifier_model(
            verifier_model_dir, verifier_lora_dir or None, dtype=dtype, device_map=device_map
        )

    prompt_items = _iter_prompt_items_from_file(prompts_file, args.prompt_key)
    if not prompt_items:
        raise SystemExit(f"No prompts found in: {prompts_file}")

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

            prompt_obj_eval = prompt_obj
            if args.online_math_prompt:
                prompt_obj_eval = _append_user_prompt_suffix(prompt_obj_eval, ONLINE_MATH_BOXED_REMINDER)

            base_prompt_raw = _prompt_obj_to_raw_prompt(tokenizer, prompt_obj_eval, system_prompt=system_prompt)
            base_prompt_raw = _tail_truncate_to_tokens(tokenizer, base_prompt_raw, int(args.max_prompt_tokens))
            actor_base_messages = _prompt_obj_to_chat_messages(prompt_obj_eval, system_prompt=system_prompt)

            rec: Dict[str, Any] = {
                "idx": int(i),
                "prompt": prompt_obj,
                "prompt_eval": prompt_obj_eval if prompt_obj_eval != prompt_obj else None,
                "prompt_style": "online_math" if args.online_math_prompt else "raw",
                "prompt_text": base_prompt_raw,
                "origin_info": origin_info,
                "base_model_dir": base_model_dir,
                "verifier_model_dir": verifier_model_dir if run_exp else None,
                "verifier_lora_dir": verifier_lora_dir if run_exp else None,
                "backend": "lmdeploy",
                "mode": args.mode,
                "system_prompt": "VERIFIER_SYSTEM_PROMPT" if args.use_verifier_system_prompt else None,
                "max_model_len": int(args.lmdeploy_session_len),
                "actor_transport": actor_transport,
                "actor_api_mode": actor_api_mode_state.get("mode", "") if actor_transport == "openai" else "",
                "run_param_hash": run_param_hash,
            }

            if run_control:
                # Chunked generation with degeneration guard (mirrors EXP loop but without verifier calls).
                response_text = ""
                model_gen_tokens = 0
                error: Optional[str] = None
                complete_reason = "unknown"
                control_steps = 0
                # If token_check_interval covers the entire requested generation budget, treat control
                # as a single-shot generation (matches typical OpenCompass behavior and avoids
                # re-sending a prompt after the model internally hit EOS, which would otherwise
                # cause runaway continuation).
                control_single_shot = int(args.token_check_interval) >= int(args.max_response_tokens)

                t0 = time.time()
                while True:
                    openai_finish_reason = ""
                    if int(args.max_sample_seconds) > 0 and (time.time() - float(t0)) > float(args.max_sample_seconds):
                        error = f"timeout>{int(args.max_sample_seconds)}s"
                        complete_reason = "timeout"
                        break
                    if model_gen_tokens >= int(args.max_response_tokens):
                        complete_reason = "max_response_tokens"
                        break

                    # Context length check (best-effort, based on tokenizer).
                    try:
                        ctx_len = len(tokenizer.encode(base_prompt_raw + response_text, add_special_tokens=False))
                    except Exception:
                        ctx_len = 0
                    if ctx_len >= int(args.lmdeploy_session_len):
                        complete_reason = "context_full"
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
                            new_text, openai_finish_reason = _openai_generate_chunk_auto(
                                api_mode_state=actor_api_mode_state,
                                api_base=actor_api_base,
                                model=actor_api_model,
                                base_prompt_raw=base_prompt_raw,
                                base_messages=actor_base_messages,
                                response_so_far=response_text,
                                max_tokens=int(step_tokens),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p) if args.top_p is not None else 1.0,
                                top_k=int(args.top_k) if int(args.top_k) > 0 else None,
                                seed=int(args.seed) if int(args.seed) != 0 else None,
                                stop=stop_words,
                                timeout_s=int(args.actor_api_timeout),
                                extra_body=actor_extra_body,
                            )
                        else:
                            assert actor_pipe is not None and GenerationConfig is not None
                            gen_kwargs = dict(
                                max_new_tokens=int(step_tokens),
                                min_new_tokens=1,
                                do_sample=True,
                                temperature=float(args.temperature),
                                top_p=float(args.top_p) if args.top_p is not None else 1.0,
                                top_k=int(args.top_k) if int(args.top_k) > 0 else 0,
                            )
                            # Make pipeline stop behavior match OpenAI server (and online vLLM) as much as possible.
                            if eos_token_ids:
                                gen_kwargs["stop_token_ids"] = sorted({int(x) for x in eos_token_ids})
                            try:
                                if int(args.seed) != 0:
                                    gen_kwargs["random_seed"] = int(args.seed)
                            except Exception:
                                pass
                            if supports_stop_words and stop_words:
                                gen_kwargs["stop_words"] = stop_words
                            gen_cfg = GenerationConfig(**gen_kwargs)
                            out = actor_pipe([base_prompt_raw + response_text], gen_config=gen_cfg, do_preprocess=False)[0]
                            new_text = str(getattr(out, "text", "") or "")
                    except Exception as e:
                        error = f"actor_generate_error: {e}"
                        complete_reason = "error"
                        break
                    new_text, hit_stop = _apply_stop_words(new_text, stop_words)
                    if not new_text:
                        complete_reason = "empty_output"
                        break
                    response_text += new_text
                    model_gen_tokens += len(tokenizer.encode(new_text, add_special_tokens=False))
                    control_steps += 1
                    if _has_complete_boxed_answer(response_text):
                        complete_reason = "final_answer"
                        break
                    if str(openai_finish_reason).lower() in {"stop", "eos", "content_filter"}:
                        complete_reason = "eos_or_stop"
                        break
                    if hit_stop:
                        complete_reason = "eos_or_stop"
                        break
                    if args.degeneration_guard and _too_repetitive_suffix(response_text):
                        complete_reason = "degeneration"
                        break
                    if control_single_shot and control_steps >= 1:
                        if str(openai_finish_reason).lower() == "length":
                            complete_reason = "max_response_tokens"
                        else:
                            complete_reason = "single_shot_limit"
                        break

                dt = float(time.time() - t0)
                # Token counts after any post-cut.
                tok_total = len(tokenizer.encode(response_text, add_special_tokens=False)) if response_text else 0
                if complete_reason == "unknown":
                    complete_reason = "finished"
                rec["control"] = {
                    "response_text": response_text,
                    "response_tokens_model_gen": int(model_gen_tokens),
                    "response_tokens_total": int(tok_total),
                    "interventions_count": 0,
                    "actor_transport": actor_transport,
                    "actor_api_mode": actor_api_mode_state.get("mode", "") if actor_transport == "openai" else "",
                    "last_openai_finish_reason": str(openai_finish_reason or "") if actor_transport == "openai" else "",
                    "termination_reason": complete_reason,
                    "gen_s": dt,
                    "error": error,
                }

            if run_exp:
                if verifier_model is None or verifier_tokenizer is None or inj is None:
                    raise SystemExit("Internal error: verifier not initialized for exp mode.")

                response_text = ""
                interventions: List[Dict[str, Any]] = []
                previous_hints: List[str] = []
                model_gen_tokens = 0  # model-generated tokens only (hints excluded)
                hint_token_total = 0
                prethink_rollback_used = False
                last_hint_end_keep = 0
                last_hint_tokens: Optional[List[int]] = None
                error: Optional[str] = None
                complete_reason = "unknown"

                t0 = time.time()
                while True:
                    openai_finish_reason = ""
                    if int(args.max_sample_seconds) > 0 and (time.time() - float(t0)) > float(args.max_sample_seconds):
                        error = f"timeout>{int(args.max_sample_seconds)}s"
                        complete_reason = "timeout"
                        break
                    try:
                        tok_total = len(tokenizer.encode(response_text, add_special_tokens=False)) if response_text else 0
                    except Exception:
                        tok_total = 0
                    model_gen_tokens = max(0, int(tok_total) - int(hint_token_total))
                    if model_gen_tokens >= int(args.max_response_tokens):
                        complete_reason = "max_response_tokens"
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
                        complete_reason = "context_full"
                        break

                    gen_remaining = int(args.max_response_tokens) - int(model_gen_tokens)
                    step_tokens = min(int(args.token_check_interval), int(gen_remaining))
                    # Also respect session length.
                    if ctx_len > 0:
                        ctx_remaining = int(args.lmdeploy_session_len) - int(ctx_len)
                        step_tokens = min(int(step_tokens), int(ctx_remaining))
                    step_tokens = max(1, int(step_tokens))

                    pending_complete_reason: Optional[str] = None
                    try:
                        if actor_transport == "openai":
                            new_text, openai_finish_reason = _openai_generate_chunk_auto(
                                api_mode_state=actor_api_mode_state,
                                api_base=actor_api_base,
                                model=actor_api_model,
                                base_prompt_raw=base_prompt_raw,
                                base_messages=actor_base_messages,
                                response_so_far=response_text,
                                max_tokens=int(step_tokens),
                                temperature=float(args.temperature),
                                top_p=float(args.top_p) if args.top_p is not None else 1.0,
                                top_k=int(args.top_k) if int(args.top_k) > 0 else None,
                                seed=int(args.seed) if int(args.seed) != 0 else None,
                                stop=stop_words,
                                timeout_s=int(args.actor_api_timeout),
                                extra_body=actor_extra_body,
                            )
                        else:
                            assert actor_pipe is not None and GenerationConfig is not None
                            gen_kwargs = dict(
                                max_new_tokens=int(step_tokens),
                                min_new_tokens=1,
                                do_sample=True,
                                temperature=float(args.temperature),
                                top_p=float(args.top_p) if args.top_p is not None else 1.0,
                                top_k=int(args.top_k) if int(args.top_k) > 0 else 0,
                            )
                            if eos_token_ids:
                                gen_kwargs["stop_token_ids"] = sorted({int(x) for x in eos_token_ids})
                            try:
                                if int(args.seed) != 0:
                                    gen_kwargs["random_seed"] = int(args.seed)
                            except Exception:
                                pass
                            if supports_stop_words and stop_words:
                                gen_kwargs["stop_words"] = stop_words
                            gen_cfg = GenerationConfig(**gen_kwargs)
                            out = actor_pipe([base_prompt_raw + response_text], gen_config=gen_cfg, do_preprocess=False)[0]
                            new_text = str(getattr(out, "text", "") or "")
                    except Exception as e:
                        error = f"actor_generate_error: {e}"
                        complete_reason = "error"
                        break
                    new_text, hit_stop = _apply_stop_words(new_text, stop_words)
                    if new_text:
                        response_text += new_text
                    else:
                        pending_complete_reason = "empty_output"
                    if str(openai_finish_reason).lower() in {"stop", "eos", "content_filter"}:
                        pending_complete_reason = "eos"
                    if hit_stop:
                        pending_complete_reason = "eos"
                    if _has_complete_boxed_answer(response_text):
                        pending_complete_reason = pending_complete_reason or "final_answer"
                    if args.degeneration_guard and _too_repetitive_suffix(response_text):
                        pending_complete_reason = pending_complete_reason or "degeneration"

                    if len(interventions) >= int(args.max_interventions):
                        if pending_complete_reason is not None:
                            complete_reason = str(pending_complete_reason)
                            break
                        continue

                    # Always call verifier after each chunk; on EOS/degeneration, allow ONE final verifier.
                    try:
                        response_tokens = (
                            tokenizer.encode(response_text, add_special_tokens=False) if response_text else []
                        )
                    except Exception:
                        response_tokens = []

                    hint_anchor_keep = len(response_tokens)
                    tail_keep = inj.compute_tail_rollback_keep_len(
                        response_tokens, tokenizer, hint_rollback_window_tokens
                    )
                    if tail_keep is not None:
                        hint_anchor_keep = int(tail_keep)

                    # Never roll back past an already-inserted hint span.
                    if last_hint_tokens:
                        try:
                            pos = inj.find_last_subsequence_index(response_tokens, last_hint_tokens)
                        except Exception:
                            pos = None
                        if pos is not None:
                            last_hint_end_keep = int(pos) + int(len(last_hint_tokens))
                    if int(last_hint_end_keep) > int(hint_anchor_keep):
                        hint_anchor_keep = int(last_hint_end_keep)

                    use_prethink_anchor = False
                    anchor_tokens = response_tokens[:hint_anchor_keep]
                    if pending_complete_reason is not None and (not prethink_rollback_used):
                        try:
                            has_safe_think_anchor = inj.find_think_close_pos(anchor_tokens, tokenizer) is not None
                        except Exception:
                            has_safe_think_anchor = False
                        try:
                            post_final = inj.is_post_think_finalized(
                                anchor_tokens,
                                tokenizer,
                                decode_tail_tokens=pending_hint_tail_decode_tokens,
                            )
                        except Exception:
                            post_final = False
                        if (not has_safe_think_anchor) and post_final:
                            prethink_keep = inj.find_prethink_rollback_keep_len(anchor_tokens, tokenizer)
                            if prethink_keep is not None and int(prethink_keep) < int(hint_anchor_keep):
                                hint_anchor_keep = int(prethink_keep)
                                use_prethink_anchor = True
                                # After rolling back to prethink, re-apply tail rollback within the new anchor.
                                tail_keep = inj.compute_tail_rollback_keep_len(
                                    response_tokens[:hint_anchor_keep],
                                    tokenizer,
                                    hint_rollback_window_tokens,
                                )
                                if tail_keep is not None:
                                    hint_anchor_keep = int(tail_keep)
                                if int(last_hint_end_keep) > int(hint_anchor_keep):
                                    hint_anchor_keep = int(last_hint_end_keep)

                    anchor_tokens = response_tokens[:hint_anchor_keep]
                    try:
                        has_safe_think_anchor = inj.find_think_close_pos(anchor_tokens, tokenizer) is not None
                    except Exception:
                        has_safe_think_anchor = False
                    try:
                        post_final = inj.is_post_think_finalized(
                            anchor_tokens,
                            tokenizer,
                            decode_tail_tokens=pending_hint_tail_decode_tokens,
                        )
                    except Exception:
                        post_final = False
                    anchor_insertable = has_safe_think_anchor or (not post_final)
                    if not anchor_insertable:
                        if pending_complete_reason is not None:
                            break
                        continue

                    try:
                        reasoning_for_verifier = tokenizer.decode(anchor_tokens, skip_special_tokens=True)
                    except Exception:
                        reasoning_for_verifier = ""

                    try:
                        verifier_text, wait_confidence, wait_avg_logprob = _verifier_generate(
                            model=verifier_model,
                            tokenizer=verifier_tokenizer,
                            question=str(question),
                            current_reasoning=reasoning_for_verifier,
                            verifier_max_prompt_length=int(args.verifier_max_prompt_length),
                            verifier_max_new_tokens=int(args.verifier_max_new_tokens),
                            max_model_len=int(args.verifier_max_prompt_length) + int(args.verifier_max_new_tokens),
                            compute_wait_confidence=float(args.confidence_threshold) > 0,
                            wait_conf_tail_tokens=int(args.wait_conf_tail_tokens),
                        )
                    except Exception as e:
                        error = f"verifier_generate_error: {e}"
                        break
                    decision = _parse_verifier_decision(verifier_text)

                    hint = (decision.hint or "").strip()
                    if not hint or decision.action != "Intervene":
                        if pending_complete_reason is not None:
                            break
                        continue
                    if float(args.confidence_threshold) > 0:
                        if wait_confidence is None or float(wait_confidence) < float(args.confidence_threshold):
                            if pending_complete_reason is not None:
                                break
                            continue
                    # Truncate hint tokens.
                    if int(args.verifier_max_hint_tokens) > 0:
                        hint_ids = tokenizer.encode(hint, add_special_tokens=False)
                        if len(hint_ids) > int(args.verifier_max_hint_tokens):
                            hint = tokenizer.decode(hint_ids[: int(args.verifier_max_hint_tokens)], skip_special_tokens=True).strip()
                    if not _hint_allowed(hint, previous_hints):
                        if pending_complete_reason is not None:
                            break
                        continue

                    # Apply hint at the same anchor used for verifier context (online-aligned).
                    state: Dict[str, Any] = {
                        "response_tokens": list(anchor_tokens),
                        "loss_masks": [1] * len(anchor_tokens),
                        "_prethink_rollback_used": bool(prethink_rollback_used),
                    }
                    if use_prethink_anchor:
                        state["_prethink_rollback_used"] = True
                        prethink_rollback_used = True

                    # If anchor is still post-final and not a safe think-close insertion, allow one
                    # best-effort rollback to prethink boundary (once per sample).
                    if pending_complete_reason is not None:
                        try:
                            still_safe_think = inj.find_think_close_pos(state["response_tokens"], tokenizer) is not None
                        except Exception:
                            still_safe_think = False
                        try:
                            still_post_final = inj.is_post_think_finalized(
                                state["response_tokens"],
                                tokenizer,
                                decode_tail_tokens=pending_hint_tail_decode_tokens,
                            )
                        except Exception:
                            still_post_final = False
                        if (not still_safe_think) and still_post_final:
                            try:
                                rolled = bool(inj.rollback_prethink_once(state, tokenizer))
                            except Exception:
                                rolled = False
                            if rolled:
                                prethink_rollback_used = True

                    try:
                        tail = tokenizer.decode(
                            state["response_tokens"][-int(pending_hint_tail_decode_tokens) :],
                            skip_special_tokens=True,
                        )
                    except Exception:
                        tail = ""
                    hint_text = inj.format_hint_text(hint=hint, tail_text=tail)
                    hint_tokens: List[int] = []
                    try:
                        hint_tokens = tokenizer.encode(hint_text, add_special_tokens=False) if hint_text else []
                    except Exception:
                        hint_tokens = []

                    hint_applied = False
                    if hint_tokens:
                        inserted = inj.insert_hint_tokens(state, hint_tokens, tokenizer, eos_token_ids)
                        hint_applied = bool(inserted)

                    if hint_applied:
                        try:
                            response_text = tokenizer.decode(state["response_tokens"], skip_special_tokens=True)
                        except Exception:
                            response_text = response_text + hint_text
                        hint_token_total += int(len(hint_tokens) if hint_tokens else 0)
                        last_hint_tokens = list(hint_tokens)
                        try:
                            pos = inj.find_last_subsequence_index(state["response_tokens"], hint_tokens)
                        except Exception:
                            pos = None
                        if pos is not None:
                            last_hint_end_keep = int(pos) + int(len(hint_tokens))
                        previous_hints.append(hint)
                        interventions.append(
                            {
                                "hint": hint,
                                "hint_text": hint_text,
                                "verifier_action": decision.action,
                                "verifier_critique": verifier_text,
                                "wait_confidence": wait_confidence,
                                "wait_avg_logprob": wait_avg_logprob,
                                "hint_token_count": int(len(hint_tokens) if hint_tokens else 0),
                                "anchor_keep": int(hint_anchor_keep),
                                "prethink_anchor": bool(use_prethink_anchor),
                                "response_tokens_after": len(tokenizer.encode(response_text, add_special_tokens=False)),
                            }
                        )
                        # If we were about to finish, keep it running so the hint can affect continuation.
                        pending_complete_reason = None

                    if pending_complete_reason is not None:
                        complete_reason = str(pending_complete_reason)
                        break

                try:
                    tok_total = len(tokenizer.encode(response_text, add_special_tokens=False)) if response_text else 0
                except Exception:
                    tok_total = 0
                model_gen_tokens = max(0, int(tok_total) - int(hint_token_total))
                dt = float(time.time() - t0)
                if complete_reason == "unknown":
                    complete_reason = "finished"
                rec["exp"] = {
                    "response_text": response_text,
                    "response_tokens_total": len(tokenizer.encode(response_text, add_special_tokens=False)),
                    "response_tokens_model_gen": int(model_gen_tokens),
                    "interventions": interventions,
                    "num_interventions": len(interventions),
                    "interventions_count": len(interventions),
                    "actor_transport": actor_transport,
                    "actor_api_mode": actor_api_mode_state.get("mode", "") if actor_transport == "openai" else "",
                    "last_openai_finish_reason": str(openai_finish_reason or "") if actor_transport == "openai" else "",
                    "termination_reason": complete_reason,
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
