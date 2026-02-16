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
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


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
    return best_span


def find_last_think_close_span(tokens: List[int], tokenizer: Any) -> Optional[Tuple[int, int]]:
    return find_last_marker_span(tokens, tokenizer, ("</think>", "<｜end of thought｜>"))


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

    state["response_tokens"] = response_tokens[:keep]
    state["loss_masks"] = loss_masks[:keep]
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

    best = None
    for s in boundary_strs:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
        except Exception:
            ids = []
        if not ids:
            continue
        pos = find_last_subsequence_index(tokens, ids)
        if pos is None:
            continue
        keep = pos + len(ids)
        if keep <= 0 or keep >= len(tokens):
            continue
        if best is None or keep > best:
            best = keep
    return best


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

