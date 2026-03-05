#!/usr/bin/env python3
"""
Summarize CoGRPO offline eval results (merged.cv.jsonl) into OpenCompass-like metrics:
- accuracy (k runs average): mean cv_score over all repeats
- G-Pass@k_0.0: per-question pass@k (any repeat has cv_score > threshold), averaged over questions

This works with `scripts/compassverifier_eval_tail_jsonl.py` outputs where:
  record["control"]["cv_score"] / record["exp"]["cv_score"] is 0/1 (or None on error).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional


def _maybe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _question_id(rec: Dict[str, Any]) -> Optional[int]:
    origin = rec.get("origin_info") or {}
    if isinstance(origin, dict):
        qid = _maybe_int(origin.get("_orig_line_idx"))
        if qid is None:
            nested = origin.get("origin_info")
            if isinstance(nested, dict):
                qid = _maybe_int(nested.get("_orig_line_idx"))
        if qid is not None:
            return qid
    return _maybe_int(rec.get("_orig_line_idx"))


def _get_cv_score(rec: Dict[str, Any], side: str) -> Optional[float]:
    obj = rec.get(side)
    if not isinstance(obj, dict):
        return None
    s = obj.get("cv_score")
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _infer_k(per_q_counts: Dict[int, int]) -> int:
    if not per_q_counts:
        return 0
    cnt = Counter(per_q_counts.values())
    # Prefer the mode; tie-break by larger k.
    k, _ = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))
    return int(k)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize merged.cv.jsonl into accuracy + G-Pass@k metrics.")
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--mode", default="both", choices=["control", "exp", "both"])
    ap.add_argument("--threshold", type=float, default=0.0, help="Pass threshold for G-Pass@k (uses score > threshold).")
    ap.add_argument("--out-json", default="", help="Optional output json path.")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    if not in_path.exists():
        raise SystemExit(f"Missing --in-jsonl: {in_path}")

    want_control = args.mode in ("control", "both")
    want_exp = args.mode in ("exp", "both")

    sum_control = 0.0
    scored_control = 0
    missing_control = 0
    per_q_control_scores: dict[int, list[float]] = defaultdict(list)
    per_q_control_total: dict[int, int] = defaultdict(int)

    sum_exp = 0.0
    scored_exp = 0
    missing_exp = 0
    per_q_exp_scores: dict[int, list[float]] = defaultdict(list)
    per_q_exp_total: dict[int, int] = defaultdict(int)

    records = 0
    qid_missing = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records += 1
            rec = json.loads(line)
            qid = _question_id(rec)
            if qid is None:
                qid_missing += 1

            if want_control:
                s = _get_cv_score(rec, "control")
                if s is None:
                    missing_control += 1
                else:
                    sum_control += float(s)
                    scored_control += 1
                if qid is not None:
                    per_q_control_total[int(qid)] += 1
                    per_q_control_scores[int(qid)].append(float(s or 0.0))

            if want_exp:
                s = _get_cv_score(rec, "exp")
                if s is None:
                    missing_exp += 1
                else:
                    sum_exp += float(s)
                    scored_exp += 1
                if qid is not None:
                    per_q_exp_total[int(qid)] += 1
                    per_q_exp_scores[int(qid)].append(float(s or 0.0))

    def _acc(sum_s: float, n: int) -> float:
        return float(sum_s / n) if n else 0.0

    def _pass_at_k(per_q_scores: Dict[int, list[float]]) -> float:
        if not per_q_scores:
            return 0.0
        thr = float(args.threshold)
        passes = 0
        for scores in per_q_scores.values():
            best = max(scores) if scores else 0.0
            if float(best) > thr:
                passes += 1
        return float(passes / len(per_q_scores))

    control_acc = _acc(sum_control, scored_control) if want_control else 0.0
    exp_acc = _acc(sum_exp, scored_exp) if want_exp else 0.0
    control_pass = _pass_at_k(per_q_control_scores) if want_control else 0.0
    exp_pass = _pass_at_k(per_q_exp_scores) if want_exp else 0.0

    out: Dict[str, Any] = {
        "records": records,
        "n_questions": len(set(per_q_control_scores) | set(per_q_exp_scores)),
        "qid_missing": qid_missing,
        "threshold": float(args.threshold),
    }

    if want_control:
        k_control = _infer_k(per_q_control_total)
        out["control"] = {
            "k": int(k_control),
            "accuracy": float(control_acc),
            "g_pass_at_k": float(control_pass),
            "scored": int(scored_control),
            "missing_cv_score": int(missing_control),
            "repeats_min": int(min(per_q_control_total.values()) if per_q_control_total else 0),
            "repeats_p50": int(statistics.median(per_q_control_total.values()) if per_q_control_total else 0),
            "repeats_max": int(max(per_q_control_total.values()) if per_q_control_total else 0),
        }

    if want_exp:
        k_exp = _infer_k(per_q_exp_total)
        out["exp"] = {
            "k": int(k_exp),
            "accuracy": float(exp_acc),
            "g_pass_at_k": float(exp_pass),
            "scored": int(scored_exp),
            "missing_cv_score": int(missing_exp),
            "repeats_min": int(min(per_q_exp_total.values()) if per_q_exp_total else 0),
            "repeats_p50": int(statistics.median(per_q_exp_total.values()) if per_q_exp_total else 0),
            "repeats_max": int(max(per_q_exp_total.values()) if per_q_exp_total else 0),
        }

    if want_control and want_exp:
        out["delta"] = {
            "accuracy": float(exp_acc - control_acc),
            "g_pass_at_k": float(exp_pass - control_pass),
        }

    raw = json.dumps(out, ensure_ascii=False)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(raw + "\n", encoding="utf-8")
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

