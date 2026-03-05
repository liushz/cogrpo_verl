#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def _iter_concatenated_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Parse a file consisting of multiple pretty-printed JSON dicts concatenated together.

    XTuner trajectory dumps use `json.dump(..., indent=2)` per item and append `\\n`,
    so each object spans multiple lines and the file is NOT real jsonl.
    """

    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = decoder.raw_decode(text, i)
        i = j
        if isinstance(obj, dict):
            yield obj


@dataclass(frozen=True)
class Intervention:
    action_id: str
    event_uid: str | None
    inserted_before_think: bool | None
    insert_pos: int | None
    think_insert_pos: int | None
    hint_token_count: int | None
    wait_confidence: float | None


def _as_int(x: Any) -> int | None:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    try:
        return int(x)
    except Exception:
        return None


def _as_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect XTuner trajectory dump (optionally with CoGRPO extra_info).")
    ap.add_argument("path", type=Path, help="Path to rollout_idx_*_trajectory.jsonl")
    ap.add_argument("--expect-tail-tokens", type=int, default=None, help="Expect cf reward tail tokens (e.g., 2048).")
    ap.add_argument("--show-bad-inserts", type=int, default=5, help="Print up to N bad insert examples.")
    ap.add_argument("--show-hints", type=int, default=5, help="Print up to N hint examples.")
    args = ap.parse_args()

    text = args.path.read_text(encoding="utf-8")
    objs = list(_iter_concatenated_json_objects(text))
    if not objs:
        print(f"[inspect] ERROR: no JSON objects parsed: {args.path}")
        return 2

    summary = objs[0]
    items = objs[1:]
    print(f"[inspect] path={args.path}")
    print(f"[inspect] items={len(items)}")
    if isinstance(summary, dict) and summary:
        print(
            f"[inspect] reward: mean={summary.get('reward_mean')} std={summary.get('reward_std')} "
            f"min={summary.get('reward_min')} max={summary.get('reward_max')}"
        )
        print(
            f"[inspect] response_len: mean={summary.get('response_len_mean')} std={summary.get('response_len_std')} "
            f"min={summary.get('response_len_min')} max={summary.get('response_len_max')}"
        )

    finish_reason = Counter()
    cogrpo_present = 0

    last_finish_reason = Counter()
    context_exhausted = Counter()
    num_interventions = Counter()
    inserted_before_think = Counter()

    cf_tail_tokens = Counter()
    cf_baseline_tail_len_max_values: list[float] = []
    cf_baseline_original_len_max_values: list[float] = []
    cf_delta_values: dict[str, list[float]] = defaultdict(list)

    hint_texts: list[str] = []
    bad_inserts: list[Intervention] = []

    for it in items:
        finish_reason[str(it.get("finish_reason") or "")] += 1

        cogrpo = it.get("cogrpo")
        if not isinstance(cogrpo, dict):
            continue
        cogrpo_present += 1

        last_finish_reason[str(cogrpo.get("cogrpo_last_finish_reason") or "")] += 1
        context_exhausted[str(bool(cogrpo.get("cogrpo_context_exhausted")))] += 1

        n_int = _as_int(cogrpo.get("cogrpo_num_interventions"))
        if n_int is not None:
            num_interventions[n_int] += 1

        tail_tok = _as_int(cogrpo.get("cogrpo_cf_reward_tail_tokens"))
        if tail_tok is not None:
            cf_tail_tokens[tail_tok] += 1

        tail_len_max = _as_float(cogrpo.get("cogrpo_cf_baseline_tail_len_max"))
        if tail_len_max is not None:
            cf_baseline_tail_len_max_values.append(tail_len_max)
        orig_len_max = _as_float(cogrpo.get("cogrpo_cf_baseline_original_len_max"))
        if orig_len_max is not None:
            cf_baseline_original_len_max_values.append(orig_len_max)

        for k, v in cogrpo.items():
            if not isinstance(k, str):
                continue
            if k.startswith("cogrpo_cf_delta_"):
                fv = _as_float(v)
                if fv is not None:
                    cf_delta_values[k].append(fv)

        hints = cogrpo.get("cogrpo_hints")
        if isinstance(hints, list):
            for h in hints:
                if isinstance(h, str) and h.strip():
                    hint_texts.append(h.strip())

        interventions = cogrpo.get("cogrpo_interventions")
        if isinstance(interventions, list):
            for inv in interventions:
                if not isinstance(inv, dict):
                    continue
                b = inv.get("inserted_before_think")
                inserted_before_think[str(b)] += 1
                if b is False and len(bad_inserts) < max(0, int(args.show_bad_inserts)):
                    bad_inserts.append(
                        Intervention(
                            action_id=str(it.get("action_id", "")),
                            event_uid=inv.get("event_uid"),
                            inserted_before_think=bool(b) if isinstance(b, bool) else None,
                            insert_pos=_as_int(inv.get("insert_pos")),
                            think_insert_pos=_as_int(inv.get("think_insert_pos")),
                            hint_token_count=_as_int(inv.get("hint_token_count")),
                            wait_confidence=_as_float(inv.get("wait_confidence")),
                        )
                    )

    print(f"[inspect] finish_reason={dict(finish_reason)}")

    if cogrpo_present == 0:
        print("[inspect] cogrpo=missing (set COGRPO_DUMP_TRAJECTORY_EXTRA=1 to embed rollout extra_info)")
        return 0

    print(f"[inspect] cogrpo_present={cogrpo_present}/{len(items)}")
    if last_finish_reason:
        print(f"[inspect] cogrpo_last_finish_reason={dict(last_finish_reason)}")
    if context_exhausted:
        print(f"[inspect] cogrpo_context_exhausted={dict(context_exhausted)}")
    if num_interventions:
        print(f"[inspect] cogrpo_num_interventions={dict(num_interventions)}")
    if inserted_before_think:
        print(f"[inspect] inserted_before_think={dict(inserted_before_think)}")

    if cf_tail_tokens:
        print(f"[inspect] cogrpo_cf_reward_tail_tokens={dict(cf_tail_tokens)}")
        if args.expect_tail_tokens is not None and args.expect_tail_tokens not in cf_tail_tokens:
            print(f"[inspect] WARNING: expected tail_tokens={args.expect_tail_tokens}, got {sorted(cf_tail_tokens)}")

    if cf_baseline_tail_len_max_values:
        mx = max(cf_baseline_tail_len_max_values)
        mean = sum(cf_baseline_tail_len_max_values) / len(cf_baseline_tail_len_max_values)
        print(f"[inspect] cogrpo_cf_baseline_tail_len_max: max={mx} mean={mean:.2f}")
    if cf_baseline_original_len_max_values:
        mx = max(cf_baseline_original_len_max_values)
        mean = sum(cf_baseline_original_len_max_values) / len(cf_baseline_original_len_max_values)
        print(f"[inspect] cogrpo_cf_baseline_original_len_max: max={mx} mean={mean:.2f}")

    if cf_delta_values:
        for k in sorted(cf_delta_values):
            vals = cf_delta_values[k]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            print(f"[inspect] {k}: last={vals[-1]} min={min(vals)} max={max(vals)} mean={mean:.4f}")

    if hint_texts and args.show_hints > 0:
        print(f"[inspect] hints_count={len(hint_texts)} (showing up to {args.show_hints})")
        for h in hint_texts[: int(args.show_hints)]:
            print(f"[inspect] hint: {h[:240]}")

    if bad_inserts:
        print(f"[inspect] bad_inserts={len(bad_inserts)} (showing up to {args.show_bad_inserts})")
        for inv in bad_inserts:
            print(
                f"[inspect] bad_insert action_id={inv.action_id} event_uid={inv.event_uid} "
                f"insert_pos={inv.insert_pos} think_insert_pos={inv.think_insert_pos} "
                f"hint_tokens={inv.hint_token_count} wait_conf={inv.wait_confidence}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

