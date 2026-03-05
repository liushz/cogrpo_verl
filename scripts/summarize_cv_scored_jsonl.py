#!/usr/bin/env python3
"""
Summarize already-scored CoGRPO eval JSONL (e.g. merged.cv.jsonl).

Why this exists:
- `compassverifier_eval_tail_jsonl.py` computes metrics while scoring online.
- Historical outputs may have correct `*.cv_score` fields but incorrect `cv_metrics.json`
  aggregation (e.g., missing question ids due to nested origin_info layout).

This script does *not* call any reward endpoints. It only reads existing cv_score fields
and computes micro/macro averages (repeat-aware when _orig_line_idx is available).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    """
    Repeat-aware question id used by local eval runners.

    Supports both layouts:
    1) origin_info["_orig_line_idx"]
    2) origin_info["origin_info"]["_orig_line_idx"]
    """
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


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize scored eval jsonl (merged.cv.jsonl).")
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--mode", default="both", choices=["control", "exp", "both"])
    ap.add_argument("--out-json", default="", help="Optional output path for metrics json.")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    if not in_path.exists():
        raise SystemExit(f"Missing --in-jsonl: {in_path}")

    want_control = args.mode in ("control", "both")
    want_exp = args.mode in ("exp", "both")

    scored_control = 0
    scored_exp = 0
    sum_control = 0.0
    sum_exp = 0.0

    per_q_control: dict[int, list[float]] = defaultdict(list)
    per_q_exp: dict[int, list[float]] = defaultdict(list)

    records = 0
    missing_control = 0
    missing_exp = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records += 1
            rec = json.loads(line)
            qid = _question_id(rec)

            if want_control:
                s = None
                side = rec.get("control")
                if isinstance(side, dict):
                    s = side.get("cv_score")
                if s is None:
                    missing_control += 1
                else:
                    s = float(s)
                    scored_control += 1
                    sum_control += s
                    if qid is not None:
                        per_q_control[int(qid)].append(s)

            if want_exp:
                s = None
                side = rec.get("exp")
                if isinstance(side, dict):
                    s = side.get("cv_score")
                if s is None:
                    missing_exp += 1
                else:
                    s = float(s)
                    scored_exp += 1
                    sum_exp += s
                    if qid is not None:
                        per_q_exp[int(qid)].append(s)

    control_micro = (sum_control / scored_control) if scored_control else 0.0
    exp_micro = (sum_exp / scored_exp) if scored_exp else 0.0
    control_macro = _mean([_mean(v) for v in per_q_control.values()])
    exp_macro = _mean([_mean(v) for v in per_q_exp.values()])

    metrics = {
        "records": records,
        "n_questions": len(set(per_q_control) | set(per_q_exp)),
        "control_scored": scored_control,
        "control_micro_acc": float(control_micro),
        "control_macro_acc": float(control_macro),
        "control_missing_cv_score": int(missing_control),
        "exp_scored": scored_exp,
        "exp_micro_acc": float(exp_micro),
        "exp_macro_acc": float(exp_macro),
        "exp_missing_cv_score": int(missing_exp),
        "delta_micro": float(exp_micro - control_micro),
        "delta_macro": float(exp_macro - control_macro),
    }

    raw = json.dumps(metrics, ensure_ascii=False)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(raw + "\n", encoding="utf-8")
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

