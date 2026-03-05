#!/usr/bin/env python3
"""
Report OpenCompass baseline metrics vs offline "control/exp" (verifier intervention) metrics.

Inputs:
- OpenCompass report root + baseline model abbr (reads results/{abbr}/{dataset}.json)
- Offline run dirs (each contains merged.cv.jsonl from compassverifier_eval_tail_jsonl.py)

Outputs:
- Markdown table (stdout or --out-md)
- Optional JSON summary (--out-json)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


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
    k, _ = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))
    return int(k)


def _summarize_offline_cv_jsonl(path: Path, *, threshold: float = 0.0) -> Dict[str, Any]:
    want_sides = ("control", "exp")
    sum_s: dict[str, float] = {s: 0.0 for s in want_sides}
    scored: dict[str, int] = {s: 0 for s in want_sides}
    missing: dict[str, int] = {s: 0 for s in want_sides}
    per_q_scores: dict[str, dict[int, list[float]]] = {s: defaultdict(list) for s in want_sides}  # type: ignore[assignment]
    per_q_total: dict[str, dict[int, int]] = {s: defaultdict(int) for s in want_sides}  # type: ignore[assignment]

    records = 0
    qid_missing = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records += 1
            rec = json.loads(line)
            qid = _question_id(rec)
            if qid is None:
                qid_missing += 1

            for side in want_sides:
                side_obj = rec.get(side)
                if not isinstance(side_obj, dict):
                    # Side not produced (e.g., MODE=exp). Treat as missing side, not incorrect.
                    continue

                if qid is not None:
                    per_q_total[side][int(qid)] += 1

                s = _get_cv_score(rec, side)
                if s is None:
                    # A produced side but missing score: treat as incorrect for acc/pass.
                    missing[side] += 1
                    s_val = 0.0
                else:
                    scored[side] += 1
                    sum_s[side] += float(s)
                    s_val = float(s)

                if qid is not None:
                    per_q_scores[side][int(qid)].append(float(s_val))

    def _acc(sumv: float, n: int) -> float:
        return float(sumv / n) if n else 0.0

    def _pass_at_k(scores_by_q: Dict[int, list[float]]) -> float:
        if not scores_by_q:
            return 0.0
        passes = 0
        for scores in scores_by_q.values():
            best = max(scores) if scores else 0.0
            if float(best) > float(threshold):
                passes += 1
        return float(passes / len(scores_by_q))

    out: Dict[str, Any] = {
        "records": records,
        "n_questions": len(set(per_q_scores["control"]) | set(per_q_scores["exp"])),
        "qid_missing": qid_missing,
        "threshold": float(threshold),
    }
    for side in want_sides:
        k = _infer_k(per_q_total[side])
        counts = list(per_q_total[side].values())
        attempted = int(sum(counts)) if counts else 0
        present = bool(counts)
        out[side] = {
            "k": int(k),
            "present": bool(present),
            "attempted": int(attempted),
            "accuracy": float(_acc(sum_s[side], scored[side])) if present else float("nan"),
            "g_pass_at_k": float(_pass_at_k(per_q_scores[side])) if present else float("nan"),
            "scored": int(scored[side]),
            "missing_cv_score": int(missing[side]),
            "repeats_min": int(min(counts) if counts else 0),
            "repeats_p50": int(statistics.median(counts) if counts else 0),
            "repeats_max": int(max(counts) if counts else 0),
        }

    return out


def _dataset_to_oc_file(dataset: str) -> Tuple[str, int]:
    d = (dataset or "").strip().lower()
    if d in {"gpqa_diamond", "gpqa"}:
        return ("GPQA_diamond", 8)
    if d == "aime2024":
        return ("aime2024", 32)
    if d == "aime2025":
        return ("aime2025", 32)
    raise ValueError(f"Unsupported dataset={dataset!r}. Supported: GPQA_diamond, aime2024, aime2025")


def _read_oc_baseline(oc_root: Path, oc_model_abbr: str, dataset: str) -> Dict[str, float]:
    oc_name, k = _dataset_to_oc_file(dataset)
    p = oc_root / "results" / oc_model_abbr / f"{oc_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing OpenCompass results file: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    acc_key = f"accuracy ({k} runs average)"
    pass_key = f"G-Pass@{k}_0.0"
    acc = float(obj.get(acc_key)) if obj.get(acc_key) is not None else float("nan")
    gp = float(obj.get(pass_key)) if obj.get(pass_key) is not None else float("nan")
    return {"k": float(k), "accuracy": acc, "g_pass_at_k_0.0": gp, "results_file": str(p)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Report verifier intervention metrics vs OpenCompass baseline.")
    ap.add_argument("--oc-root", required=True)
    ap.add_argument("--oc-model-abbr", required=True, help="Baseline model abbr under OpenCompass results/")
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="Dataset run mapping: dataset=/path/to/offline/run_dir (contains merged.cv.jsonl).",
    )
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    oc_root = Path(args.oc_root).expanduser().resolve()
    oc_model_abbr = str(args.oc_model_abbr)

    runs: Dict[str, Path] = {}
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"Bad --run {item!r} (expected dataset=/path/to/run_dir)")
        ds, p = item.split("=", 1)
        ds = ds.strip()
        run_dir = Path(p).expanduser().resolve()
        if not run_dir.exists():
            raise SystemExit(f"Missing run_dir for {ds}: {run_dir}")
        runs[ds] = run_dir

    if not runs:
        raise SystemExit("No --run provided.")

    def _fmt(x: float) -> str:
        try:
            if x is None or (isinstance(x, float) and math.isnan(float(x))):
                return "-"
        except Exception:
            return "-"
        return f"{float(x):.2f}"

    def _fmt_delta(x: float) -> str:
        try:
            if x is None or (isinstance(x, float) and math.isnan(float(x))):
                return "-"
        except Exception:
            return "-"
        return f"{float(x):+.2f}"

    rows_md: list[str] = []
    rows_md.append("| Dataset | Metric | OpenCompass baseline | Offline control | Offline exp | Delta vs baseline |")
    rows_md.append("| --- | --- | ---: | ---: | ---: | ---: |")

    summary: Dict[str, Any] = {"oc_root": str(oc_root), "oc_model_abbr": oc_model_abbr, "datasets": {}}

    for dataset, run_dir in runs.items():
        merged = run_dir / "merged.cv.jsonl"
        if not merged.exists():
            raise SystemExit(f"Missing {merged} (run {dataset} must have DO_CV_EVAL=1).")

        offline = _summarize_offline_cv_jsonl(merged, threshold=float(args.threshold))
        baseline = _read_oc_baseline(oc_root, oc_model_abbr, dataset)

        # Percent formatting: OpenCompass already uses percent; offline is [0,1].
        k_expected = int(baseline["k"])
        k_note = ""
        k_parts = []
        if bool(offline.get("control", {}).get("present")):
            k_control = int(offline["control"]["k"])
            if k_control != k_expected:
                k_parts.append(f"control:{k_control}")
        if bool(offline.get("exp", {}).get("present")):
            k_exp = int(offline["exp"]["k"])
            if k_exp != k_expected:
                k_parts.append(f"exp:{k_exp}")
        if k_parts:
            k_note = f" (offline k={','.join(k_parts)})"

        acc_base = float(baseline["accuracy"])
        pass_base = float(baseline["g_pass_at_k_0.0"])

        # Offline metrics in percent. Delta is always vs OpenCompass baseline (control).
        acc_c = float("nan")
        pass_c = float("nan")
        if bool(offline.get("control", {}).get("present")):
            acc_c = float(offline["control"]["accuracy"]) * 100.0
            pass_c = float(offline["control"]["g_pass_at_k"]) * 100.0
        acc_e = float("nan")
        pass_e = float("nan")
        if bool(offline.get("exp", {}).get("present")):
            acc_e = float(offline["exp"]["accuracy"]) * 100.0
            pass_e = float(offline["exp"]["g_pass_at_k"]) * 100.0

        acc_d = acc_e - acc_base if math.isfinite(acc_e) and math.isfinite(acc_base) else float("nan")
        pass_d = pass_e - pass_base if math.isfinite(pass_e) and math.isfinite(pass_base) else float("nan")

        rows_md.append(
            f"| {dataset}{k_note} | accuracy ({k_expected} avg) | {_fmt(acc_base)} | {_fmt(acc_c)} | {_fmt(acc_e)} | {_fmt_delta(acc_d)} |"
        )
        rows_md.append(
            f"| {dataset}{k_note} | G-Pass@{k_expected}_0.0 | {_fmt(pass_base)} | {_fmt(pass_c)} | {_fmt(pass_e)} | {_fmt_delta(pass_d)} |"
        )

        summary["datasets"][dataset] = {"baseline": baseline, "offline": offline}

    md = "\n".join(rows_md) + "\n"
    if args.out_md:
        out_md = Path(args.out_md).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
    else:
        print(md)

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
