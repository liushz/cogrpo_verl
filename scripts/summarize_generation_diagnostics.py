#!/usr/bin/env python3
"""
Summarize generation diagnostics from merged JSONL outputs.

Works for outputs from eval_co_grpo_with_verifier_lmdeploy64k.py where
records may contain:
  - control / exp side payloads
  - response token counts
  - termination_reason
  - interventions metadata (exp)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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


def _repeat_id(rec: Dict[str, Any]) -> Optional[int]:
    origin = rec.get("origin_info") or {}
    if isinstance(origin, dict):
        rid = _maybe_int(origin.get("_repeat_id"))
        if rid is None:
            nested = origin.get("origin_info")
            if isinstance(nested, dict):
                rid = _maybe_int(nested.get("_repeat_id"))
        if rid is not None:
            return rid
    return _maybe_int(rec.get("_repeat_id"))


def _percent(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(100.0 * numer / denom)


def _quantile(values: List[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    if len(s) == 1:
        return int(s[0])
    pos = max(0, min(len(s) - 1, int(round((len(s) - 1) * float(q)))))
    return int(s[pos])


def _tail_repeat_ngram_count(text: str, *, ngram_n: int = 20, tail_words: int = 1500) -> int:
    words = re.findall(r"\w+", (text or "").lower())
    if tail_words > 0 and len(words) > tail_words:
        words = words[-tail_words:]
    if len(words) < ngram_n:
        return 0
    counts: Counter = Counter()
    for i in range(0, len(words) - ngram_n + 1):
        counts[tuple(words[i : i + ngram_n])] += 1
    return int(max(counts.values()) if counts else 0)


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _summarize_side(records: List[Dict[str, Any]], side: str) -> Dict[str, Any]:
    side_records: List[Dict[str, Any]] = []
    token_model_gen: List[int] = []
    token_total: List[int] = []
    durations: List[float] = []
    termination_counts: Counter = Counter()
    error_count = 0
    long32 = 0
    long60 = 0
    long65 = 0
    repeat_ge2 = 0
    repeat_ge3 = 0
    repeat_ge4 = 0

    interventions_nonzero = 0
    wait_records = 0
    zero_interventions = 0
    per_q_long: Dict[int, List[int]] = defaultdict(list)

    for rec in records:
        obj = rec.get(side)
        if not isinstance(obj, dict):
            continue
        side_records.append(rec)
        t_model = _maybe_int(obj.get("response_tokens_model_gen"))
        t_total = _maybe_int(obj.get("response_tokens_total"))
        if t_model is None:
            t_model = t_total if t_total is not None else 0
        if t_total is None:
            t_total = t_model
        t_model = int(t_model or 0)
        t_total = int(t_total or 0)
        token_model_gen.append(t_model)
        token_total.append(t_total)
        durations.append(float(obj.get("gen_s") or 0.0))

        if t_model >= 32000:
            long32 += 1
        if t_model >= 60000:
            long60 += 1
        if t_model >= 65000:
            long65 += 1

        reason = str(obj.get("termination_reason") or "unknown")
        termination_counts[reason] += 1
        if obj.get("error"):
            error_count += 1

        text = str(obj.get("response_text") or "")
        rep = _tail_repeat_ngram_count(text)
        if rep >= 2:
            repeat_ge2 += 1
        if rep >= 3:
            repeat_ge3 += 1
        if rep >= 4:
            repeat_ge4 += 1

        qid = _question_id(rec)
        if qid is not None:
            per_q_long[int(qid)].append(1 if t_model >= 32000 else 0)

        if side == "exp":
            interventions_count = _maybe_int(obj.get("interventions_count"))
            if interventions_count is None:
                interventions_count = _maybe_int(obj.get("num_interventions")) or 0
            interventions_count = int(interventions_count)
            if interventions_count > 0:
                interventions_nonzero += 1
            else:
                zero_interventions += 1

            interventions = obj.get("interventions")
            has_wait = False
            if isinstance(interventions, list):
                for it in interventions:
                    if not isinstance(it, dict):
                        continue
                    action = str(it.get("verifier_action") or "").lower()
                    hint = str(it.get("hint") or "").strip()
                    if action == "intervene" or bool(hint):
                        has_wait = True
                        break
            if has_wait:
                wait_records += 1

    n = len(side_records)
    per_q_long_rate = []
    for qid, flags in per_q_long.items():
        if not flags:
            continue
        rate = float(sum(flags) / len(flags))
        per_q_long_rate.append(
            {
                "question_id": int(qid),
                "long32_rate": float(rate),
                "n": int(len(flags)),
            }
        )
    per_q_long_rate.sort(key=lambda x: (x["long32_rate"], x["n"]), reverse=True)

    out: Dict[str, Any] = {
        "present": bool(n > 0),
        "records": int(n),
        "tokens_model_gen": {
            "mean": float(sum(token_model_gen) / n) if n > 0 else 0.0,
            "p50": int(_quantile(token_model_gen, 0.50)),
            "p90": int(_quantile(token_model_gen, 0.90)),
            "p99": int(_quantile(token_model_gen, 0.99)),
            "max": int(max(token_model_gen) if token_model_gen else 0),
            "ge_32000_pct": float(_percent(long32, n)),
            "ge_60000_pct": float(_percent(long60, n)),
            "ge_65000_pct": float(_percent(long65, n)),
        },
        "tokens_total": {
            "mean": float(sum(token_total) / n) if n > 0 else 0.0,
            "p50": int(_quantile(token_total, 0.50)),
            "p90": int(_quantile(token_total, 0.90)),
            "p99": int(_quantile(token_total, 0.99)),
            "max": int(max(token_total) if token_total else 0),
        },
        "duration_s": {
            "mean": float(sum(durations) / n) if n > 0 else 0.0,
            "p50": int(_quantile([int(round(x)) for x in durations], 0.50)),
            "p90": int(_quantile([int(round(x)) for x in durations], 0.90)),
            "max": int(max([int(round(x)) for x in durations]) if durations else 0),
        },
        "termination_reason_counts": dict(termination_counts),
        "error_pct": float(_percent(error_count, n)),
        "tail_repeat_ngram20": {
            "ge_2_pct": float(_percent(repeat_ge2, n)),
            "ge_3_pct": float(_percent(repeat_ge3, n)),
            "ge_4_pct": float(_percent(repeat_ge4, n)),
        },
        "per_question_long32_top10": per_q_long_rate[:10],
    }

    if side == "exp":
        out["intervention"] = {
            "nonzero_pct": float(_percent(interventions_nonzero, n)),
            "zero_pct": float(_percent(zero_interventions, n)),
            "wait_like_pct": float(_percent(wait_records, n)),
        }

    return out


def _collect_audit_samples(records: List[Dict[str, Any]], max_n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        exp = rec.get("exp")
        if not isinstance(exp, dict):
            continue
        interventions = exp.get("interventions")
        if not isinstance(interventions, list) or not interventions:
            continue
        first = interventions[0] if isinstance(interventions[0], dict) else {}
        item = {
            "idx": _maybe_int(rec.get("idx")),
            "question_id": _question_id(rec),
            "repeat_id": _repeat_id(rec),
            "interventions_count": _maybe_int(exp.get("interventions_count") or exp.get("num_interventions")) or 0,
            "first_verifier_action": str(first.get("verifier_action") or ""),
            "first_wait_confidence": first.get("wait_confidence"),
            "first_hint": str(first.get("hint") or ""),
            "termination_reason": str(exp.get("termination_reason") or ""),
        }
        out.append(item)
        if len(out) >= max_n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize generation diagnostics from merged jsonl.")
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--mode", default="both", choices=["control", "exp", "both"])
    ap.add_argument("--audit-sample-n", type=int, default=20)
    ap.add_argument("--audit-out", default="", help="Optional output jsonl for exp intervention audit samples.")
    ap.add_argument("--out-json", default="", help="Optional output json path.")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    if not in_path.exists():
        raise SystemExit(f"Missing --in-jsonl: {in_path}")

    records = list(_iter_records(in_path))
    out: Dict[str, Any] = {
        "records": int(len(records)),
        "mode": str(args.mode),
    }

    if args.mode in ("control", "both"):
        out["control"] = _summarize_side(records, "control")
    if args.mode in ("exp", "both"):
        out["exp"] = _summarize_side(records, "exp")

    if args.audit_out and args.audit_sample_n > 0:
        samples = _collect_audit_samples(records, int(args.audit_sample_n))
        audit_path = Path(args.audit_out)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8") as f:
            for item in samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        out["exp_intervention_audit_samples"] = {
            "count": int(len(samples)),
            "path": str(audit_path),
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
