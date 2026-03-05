"""
Shared Verifier hint injection helpers.

This module contains the lightweight, tokenizer-only logic used by online CoGRPO
by_step rollouts to:
  - choose safe insertion positions (prefer before </think>)
  - detect post-</think> finalized segments ("马后炮" guard)
  - roll back to natural boundaries for effective interventions
  - format hint text with natural newlines (NO "[Guide]:" marker)

It is intentionally self-contained (stdlib only) so offline tools can load it
via file-path imports without pulling in heavy training dependencies.

It also hosts the canonical Verifier prompting templates used across:
  - online RL (CoGRPO by_step Verifier)
  - offline eval scripts
  - Verifier cold-start data pipelines (step5 prompt wrapping)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


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


def find_last_eos_index(tokens: List[int], eos_token_ids: set[int]) -> Optional[int]:
    if not tokens or not eos_token_ids:
        return None
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in eos_token_ids:
            return i
    return None


def find_last_subsequence_index(tokens: List[int], subseq: List[int]) -> Optional[int]:
    """Return the start index of the last occurrence of subseq in tokens."""
    if not tokens or not subseq or len(subseq) > len(tokens):
        return None
    for start in range(len(tokens) - len(subseq), -1, -1):
        if tokens[start : start + len(subseq)] == subseq:
            return start
    return None


def find_last_marker_span(
    tokens: List[int],
    tokenizer: Any,
    markers: Tuple[str, ...],
) -> Optional[Tuple[int, int]]:
    """Return (start, end) of the last marker occurrence."""
    if not tokens:
        return None

    best_span: Optional[Tuple[int, int]] = None
    for marker in markers:
        try:
            marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        except Exception:
            marker_ids = []
        if not marker_ids:
            continue
        pos = find_last_subsequence_index(tokens, marker_ids)
        if pos is None:
            continue
        span = (pos, pos + len(marker_ids))
        if best_span is None or span[0] > best_span[0]:
            best_span = span
    if best_span is not None:
        return best_span

    # Fallback: marker tokenization can differ depending on surrounding text.
    # Example (Qwen2.*): "</think>" encodes to ids for "</", "think", ">", but
    # in-context it may become "</", "think", ">\\n\\n" and thus the exact marker
    # id subsequence is absent. Recover by searching over decoded token text.

    def _get_decode_cache(tok: Any) -> Optional[Dict[int, str]]:
        try:
            cache = getattr(tok, "_verl_token_decode_cache", None)
            if isinstance(cache, dict):
                return cache
        except Exception:
            return None
        try:
            cache = {}
            setattr(tok, "_verl_token_decode_cache", cache)
            return cache
        except Exception:
            return None

    def _decode_token(tok: Any, tid: int) -> str:
        try:
            tid = int(tid)
        except Exception:
            tid = 0
        cache = _get_decode_cache(tok)
        if cache is not None:
            cached = cache.get(tid, None)
            if cached is not None:
                return cached
        try:
            text = tok.decode([tid], skip_special_tokens=True)
        except Exception:
            text = ""
        if cache is not None:
            # Avoid unbounded growth in long runs.
            if len(cache) < 200_000:
                cache[tid] = text
        return text

    try:
        chunk_tokens = 2048
        if chunk_tokens < 64:
            chunk_tokens = 64
    except Exception:
        chunk_tokens = 2048

    end = len(tokens)
    start = max(0, end - int(chunk_tokens))
    decoded_parts = [_decode_token(tokenizer, t) for t in tokens[start:end]]
    decoded_text = "".join(decoded_parts)
    decoded_text_lower = decoded_text.lower()
    markers_lower = tuple((m or "").lower() for m in markers)

    while True:
        best_marker = None
        best_pos = -1
        best_len = 0
        for m, m_low in zip(markers, markers_lower):
            if not m_low:
                continue
            pos = decoded_text_lower.rfind(m_low)
            if pos > best_pos:
                best_pos = int(pos)
                best_marker = m
                best_len = len(m)
        if best_marker is not None and best_pos >= 0:
            marker_start_char = best_pos
            marker_end_char = best_pos + int(best_len)

            cum = 0
            start_tok = None
            end_tok_excl = None
            for i, part in enumerate(decoded_parts):
                next_cum = cum + len(part)
                if start_tok is None and next_cum > marker_start_char:
                    start_tok = int(i)
                if end_tok_excl is None and next_cum >= marker_end_char:
                    end_tok_excl = int(i + 1)
                    break
                cum = next_cum
            if start_tok is None:
                start_tok = max(0, len(decoded_parts) - 1)
            if end_tok_excl is None:
                end_tok_excl = len(decoded_parts)
            return (int(start + start_tok), int(start + end_tok_excl))

        if start <= 0:
            break

        new_start = max(0, start - int(chunk_tokens))
        new_parts = [_decode_token(tokenizer, t) for t in tokens[new_start:start]]
        decoded_parts = new_parts + decoded_parts
        decoded_text = "".join(new_parts) + decoded_text
        decoded_text_lower = decoded_text.lower()
        start = new_start

    return None


def find_last_think_close_span(tokens: List[int], tokenizer: Any) -> Optional[Tuple[int, int]]:
    return find_last_marker_span(
        tokens,
        tokenizer,
        (
            "</think>",
            "<｜end of thought｜>",
            "<|end of thought|>",
            "<|end_of_thought|>",
        ),
    )


def find_prethink_rollback_keep_len(tokens: List[int], tokenizer: Any) -> Optional[int]:
    """Return keep_len to roll back before last </think> when suffix is substantive."""
    think_span = find_last_think_close_span(tokens, tokenizer)
    if think_span is None:
        return None
    start, end = think_span
    if start <= 0:
        return None

    try:
        suffix_text = tokenizer.decode(tokens[end : end + 256], skip_special_tokens=True)
    except Exception:
        suffix_text = ""
    if not suffix_text.strip():
        return None
    return start


def is_post_think_finalized(tokens: List[int], tokenizer: Any, decode_tail_tokens: int = 512) -> bool:
    """Return True iff we've already entered the post-``</think>`` segment."""
    think_span = find_last_think_close_span(tokens, tokenizer)
    if think_span is None:
        return False
    _, end = think_span
    suffix_tokens = tokens[end:]
    if not suffix_tokens:
        return False
    try:
        tail_len = max(64, int(decode_tail_tokens))
    except Exception:
        tail_len = 512
    try:
        suffix_text = tokenizer.decode(suffix_tokens[:tail_len], skip_special_tokens=True)
    except Exception:
        suffix_text = ""
    return bool(suffix_text.strip())


def rollback_state_to_keep_len(state: Dict[str, Any], keep_len: int) -> bool:
    response_tokens = list(state.get("response_tokens") or [])
    loss_masks = list(state.get("loss_masks") or [])
    if len(response_tokens) != len(loss_masks):
        return False
    try:
        keep = int(keep_len)
    except Exception:
        return False
    if keep < 0 or keep > len(response_tokens):
        return False
    if keep == len(response_tokens):
        return False

    removed_masks = loss_masks[keep:]
    state["response_tokens"] = response_tokens[:keep]
    state["loss_masks"] = loss_masks[:keep]

    # Best-effort: keep fast counters in sync when present.
    # - gen_len : number of policy-generated tokens (loss_mask==1)
    # - hint_len: number of hint tokens (loss_mask==0)
    if removed_masks:
        try:
            removed_gen = sum(1 for m in removed_masks if int(m) == 1)
            removed_hint = sum(1 for m in removed_masks if int(m) == 0)
        except Exception:
            removed_gen = None
            removed_hint = None

        if removed_gen is not None and "gen_len" in state:
            try:
                state["gen_len"] = max(0, int(state.get("gen_len") or 0) - int(removed_gen))
            except Exception:
                pass
        if removed_hint is not None and "hint_len" in state:
            try:
                state["hint_len"] = max(0, int(state.get("hint_len") or 0) - int(removed_hint))
            except Exception:
                pass

    # Best-effort: keep cached prompt+response token list in sync when present.
    # This avoids rebuilding `prompt_tokens + response_tokens` on every by_step loop.
    input_tokens = state.get("input_tokens")
    if isinstance(input_tokens, list):
        prompt_tokens = state.get("prompt_tokens") or []
        prompt_len = len(prompt_tokens)
        expected_total = prompt_len + keep
        try:
            if input_tokens[:prompt_len] != list(prompt_tokens):
                # Prefix mismatch (should be rare); rebuild.
                state["input_tokens"] = list(prompt_tokens) + state["response_tokens"]
            elif expected_total < len(input_tokens):
                del input_tokens[expected_total:]
        except Exception:
            # As a last resort, rebuild.
            try:
                state["input_tokens"] = list(prompt_tokens) + state["response_tokens"]
            except Exception:
                pass

    return True


def last_hint_end(loss_masks: List[int]) -> int:
    """Return the exclusive end index of the last inserted hint token span."""
    if not loss_masks:
        return 0
    last = -1
    for i in range(len(loss_masks) - 1, -1, -1):
        if int(loss_masks[i]) == 0:
            last = i
            break
    return 0 if last < 0 else int(last + 1)


def compute_tail_rollback_keep_len(
    response_tokens: List[int],
    tokenizer: Any,
    window_tokens: int,
) -> Optional[int]:
    try:
        window_tokens = int(window_tokens)
    except Exception:
        return None
    if window_tokens <= 0 or len(response_tokens) < window_tokens + 1:
        return None

    tail = response_tokens[-window_tokens:]
    keep_in_tail = find_last_trim_boundary(tail, tokenizer)
    if keep_in_tail is None:
        return None

    keep = len(response_tokens) - window_tokens + int(keep_in_tail)
    if keep <= 0 or keep >= len(response_tokens):
        return None
    return keep


def rollback_prethink_once(state: Dict[str, Any], tokenizer: Any) -> bool:
    """Rollback before </think> once per sample."""
    if state.get("_prethink_rollback_used", False):
        return False
    keep = find_prethink_rollback_keep_len(state.get("response_tokens") or [], tokenizer)
    if keep is None:
        return False
    if not rollback_state_to_keep_len(state, keep):
        return False
    state["_prethink_rollback_used"] = True
    return True


def find_think_close_pos(tokens: List[int], tokenizer: Any) -> Optional[int]:
    """Find a safe insertion position before the last think-closure marker."""
    think_span = find_last_think_close_span(tokens, tokenizer)
    if think_span is None:
        return None
    start, end = think_span

    suffix_ids = tokens[end : end + 128]
    try:
        suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=True)
    except Exception:
        suffix_text = ""
    if suffix_text.strip():
        return None

    return start


def insert_hint_tokens(
    state: Dict[str, Any],
    hint_tokens: List[int],
    tokenizer: Any,
    eos_token_ids: set[int],
) -> bool:
    """Insert hint_tokens into response_tokens/loss_masks, preferring before </think>."""
    if not hint_tokens:
        return False

    response_tokens = state.get("response_tokens") or []
    loss_masks = state.get("loss_masks") or []
    if len(response_tokens) != len(loss_masks):
        return False

    think_pos = find_think_close_pos(response_tokens, tokenizer)
    if think_pos is not None:
        insert_pos = think_pos
    else:
        if is_post_think_finalized(response_tokens, tokenizer):
            state["_hint_skipped_late_stage"] = True
            return False
        insert_pos = find_last_eos_index(response_tokens, eos_token_ids)

    if insert_pos is None:
        response_tokens.extend(hint_tokens)
        loss_masks.extend([0] * len(hint_tokens))
    else:
        response_tokens[insert_pos:insert_pos] = hint_tokens
        loss_masks[insert_pos:insert_pos] = [0] * len(hint_tokens)

    state["response_tokens"] = response_tokens
    state["loss_masks"] = loss_masks
    state["_hint_inserted_this_step"] = True

    # Best-effort: keep fast counters and cached full-seq tokens in sync when present.
    try:
        if "hint_len" in state:
            state["hint_len"] = int(state.get("hint_len") or 0) + int(len(hint_tokens))
    except Exception:
        pass

    input_tokens = state.get("input_tokens")
    if isinstance(input_tokens, list):
        prompt_tokens = state.get("prompt_tokens") or []
        prompt_len = len(prompt_tokens)
        try:
            if input_tokens[:prompt_len] != list(prompt_tokens):
                input_tokens = list(prompt_tokens) + list(state.get("response_tokens") or [])
                state["input_tokens"] = input_tokens
                return True
            if insert_pos is None:
                input_tokens.extend(hint_tokens)
            else:
                abs_pos = prompt_len + int(insert_pos)
                input_tokens[abs_pos:abs_pos] = hint_tokens
        except Exception:
            try:
                state["input_tokens"] = list(prompt_tokens) + list(state.get("response_tokens") or [])
            except Exception:
                pass
    return True


def rollback_tail_for_hint(state: Dict[str, Any], tokenizer: Any, window_tokens: int) -> bool:
    """Trim a trailing fragment so hint can affect regenerated content."""
    response_tokens = list(state.get("response_tokens") or [])
    keep = compute_tail_rollback_keep_len(response_tokens, tokenizer, window_tokens)
    if keep is None:
        return False
    return rollback_state_to_keep_len(state, keep)


def find_last_trim_boundary(tokens: List[int], tokenizer: Any) -> Optional[int]:
    """Find a natural boundary inside `tokens` to trim to (return keep_len)."""
    if not tokens:
        return None

    boundary_strs = [
        "\n\n",
        "。",
        ".",
        "！",
        "!",
        "？",
        "?",
        "；",
        ";",
        "：",
        ":",
        "，",
        ",",
    ]
    span = find_last_marker_span(tokens, tokenizer, tuple(boundary_strs))
    if span is None:
        return None
    _, end = span
    keep = int(end)
    if keep <= 0 or keep >= len(tokens):
        return None
    return keep


def hint_prefix_for_tail(tail_text: str) -> str:
    if tail_text.endswith("\n\n"):
        return ""
    if tail_text.endswith("\n"):
        return "\n"
    return "\n\n"


def format_hint_text(*, hint: str, tail_text: str) -> str:
    hint = (hint or "").strip()
    if not hint:
        return ""
    prefix = hint_prefix_for_tail(tail_text)
    return f"{prefix}{hint}\n\n"
