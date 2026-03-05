#!/usr/bin/env python3
"""
Step 0b: Cap rollout pool size per question (keep top-N by quality).

Why:
  - Old rollout dumps often contain many duplicate questions with many responses.
  - Downstream verifier cold-start pipelines may assume a bounded number of responses per question
    (e.g., <=8), both for quality and for memory/runtime on step1/step2.

Input:
  - A single jsonl file in the "rollout_w_answer" format (one JSON object per line), produced by:
      scripts/verifier_data_gen/step0_convert_trajectory_rollouts.py

Output:
  - A filtered jsonl file with at most `--max-per-question` records per question.

Selection policy (default):
  - Prefer keeping at least 1 correct response per question when available.
  - Fill the remaining slots with incorrect responses.
  - Rank responses by a simple quality key:
      (finish_reason == "stop", response_len, output_char_len)  # higher is better

This script is stdlib-only and does two passes:
  1) Scan and decide which line indices to keep (bounded memory).
  2) Re-scan and write only selected lines (stable original order).
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_USER_RE = re.compile(r"\nuser\n(.*?)(?=\nassistant\n|$)", re.DOTALL | re.IGNORECASE)


def _extract_question_from_input(input_text: str) -> str:
    if not input_text:
        return ""
    m = _USER_RE.search(input_text)
    if m:
        return (m.group(1) or "").strip()
    return input_text.strip()


def _quality_key(rec: Dict[str, Any]) -> Tuple[int, int, int]:
    stop = 1 if str(rec.get("finish_reason") or "").lower() == "stop" else 0
    try:
        rlen = int(rec.get("response_len") or 0)
    except Exception:
        rlen = 0
    out_len = len(str(rec.get("output") or ""))
    return (stop, rlen, out_len)


@dataclass
class _HeapItem:
    quality: Tuple[int, int, int]
    line_idx: int


def _push_topk(
    heap: List[Tuple[Tuple[int, int, int], int]],
    item: _HeapItem,
    k: int,
) -> None:
    if k <= 0:
        return
    if len(heap) < k:
        heapq.heappush(heap, (item.quality, item.line_idx))
        return
    if heap and item.quality > heap[0][0]:
        heapq.heapreplace(heap, (item.quality, item.line_idx))


def _iter_jsonl_lines(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except Exception:
                continue


def main() -> int:
    ap = argparse.ArgumentParser(description="Cap rollout pool per question (two-pass, stdlib-only).")
    ap.add_argument("--input_jsonl", type=str, required=True, help="Input jsonl (rollout_w_answer format).")
    ap.add_argument("--output_jsonl", type=str, required=True, help="Output jsonl (filtered).")
    ap.add_argument("--max-per-question", type=int, default=8)
    ap.add_argument(
        "--min-correct-per-question",
        type=int,
        default=1,
        help="Try to keep at least this many correct responses per question when available.",
    )
    ap.add_argument(
        "--min-incorrect-per-question",
        type=int,
        default=1,
        help="Try to keep at least this many incorrect responses per question when available.",
    )
    ap.add_argument(
        "--bucket-k",
        type=int,
        default=8,
        help="Per question, keep top-K candidates per class during scan (must be >= max-per-question).",
    )
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[ERR] input_jsonl not found: {in_path}", file=sys.stderr)
        return 2

    max_per_q = max(1, int(args.max_per_question))
    bucket_k = max(max_per_q, int(args.bucket_k))
    min_c = max(0, int(args.min_correct_per_question))
    min_i = max(0, int(args.min_incorrect_per_question))

    # Pass 1: decide indices to keep.
    by_q_correct: Dict[str, List[Tuple[Tuple[int, int, int], int]]] = {}
    by_q_incorrect: Dict[str, List[Tuple[Tuple[int, int, int], int]]] = {}
    total = 0
    bad = 0

    for i, rec in _iter_jsonl_lines(in_path):
        total += 1
        try:
            question = rec.get("question")
            if not question:
                question = _extract_question_from_input(str(rec.get("input") or ""))
            question = str(question).strip()
        except Exception:
            question = ""
        if not question:
            bad += 1
            continue

        acc = rec.get("acc", None)
        is_correct = bool(acc) if acc is not None else False
        qkey = question
        item = _HeapItem(quality=_quality_key(rec), line_idx=int(i))

        if is_correct:
            heap = by_q_correct.setdefault(qkey, [])
            _push_topk(heap, item, bucket_k)
        else:
            heap = by_q_incorrect.setdefault(qkey, [])
            _push_topk(heap, item, bucket_k)

    selected: set[int] = set()
    num_questions = len(set(by_q_correct.keys()) | set(by_q_incorrect.keys()))

    for q in set(by_q_correct.keys()) | set(by_q_incorrect.keys()):
        c_heap = by_q_correct.get(q) or []
        i_heap = by_q_incorrect.get(q) or []
        c_sorted = sorted(c_heap, key=lambda x: x[0], reverse=True)
        i_sorted = sorted(i_heap, key=lambda x: x[0], reverse=True)

        keep: List[int] = []
        if min_c > 0:
            keep.extend([idx for _, idx in c_sorted[:min_c]])

        remaining = max_per_q - len(keep)
        if remaining > 0:
            keep.extend([idx for _, idx in i_sorted[:remaining]])

        remaining = max_per_q - len(keep)
        if remaining > 0:
            extra_c = [idx for _, idx in c_sorted[min_c : min_c + remaining]]
            keep.extend(extra_c)

        # Ensure min incorrect if requested and available (swap with extra correct if needed).
        if min_i > 0 and i_sorted:
            kept_incorrect = sum(1 for idx in keep if idx in {j for _, j in i_sorted})
            need = max(0, min_i - kept_incorrect)
            if need > 0:
                want_i = [idx for _, idx in i_sorted if idx not in keep][:need]
                if want_i:
                    # Replace from the tail of keep (lowest priority).
                    keep = keep[: max(0, len(keep) - len(want_i))] + want_i

        for idx in keep[:max_per_q]:
            selected.add(int(idx))

    # Pass 2: write selected indices in original order.
    kept = 0
    with in_path.open("r", encoding="utf-8") as rf, out_path.open("w", encoding="utf-8") as wf:
        for i, line in enumerate(rf):
            if i not in selected:
                continue
            wf.write(line.rstrip("\n") + "\n")
            kept += 1

    print(f"[OK] input_lines_total={total} bad_question_lines={bad} questions={num_questions} kept={kept}")
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

