#!/usr/bin/env python3
"""
Build a small JSONL dataset (question+answer) from an existing OpenCompass report's predictions.

Why:
- OpenCompass prediction shards already contain the exact prompt template and gold labels.
- They also include multi-run repeats (e.g., AIME n=32, GPQA n=8). We deduplicate to 1 record per question.

Output JSONL schema (per line):
  {"question": "...", "answer": "..."}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _dataset_prefix(dataset: str) -> str:
    d = (dataset or "").strip()
    d_lower = d.lower()
    if d_lower in {"gpqa_diamond", "gpqa"}:
        return "GPQA_diamond"
    if d_lower in {"aime2024"}:
        return "aime2024"
    if d_lower in {"aime2025"}:
        return "aime2025"
    raise ValueError(f"Unsupported dataset={dataset!r}. Supported: GPQA_diamond, aime2024, aime2025")


_SUFFIX_RE = re.compile(r"_(\d+)\.json$")


def _sort_key_json(path: Path) -> Tuple[int, str]:
    m = _SUFFIX_RE.search(path.name)
    if m:
        return (int(m.group(1)), path.name)
    return (10**9, path.name)


def _iter_prediction_files(pred_dir: Path, prefix: str) -> List[Path]:
    files = list(pred_dir.glob(f"{prefix}_*.json"))
    files.sort(key=_sort_key_json)
    if not files:
        raise FileNotFoundError(f"No prediction shards found under {pred_dir} for prefix {prefix!r}")
    return files


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_human_prompt(origin_prompt: Any) -> str:
    if isinstance(origin_prompt, list):
        # Prefer the last HUMAN/user message.
        for msg in reversed(origin_prompt):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            api_role = str(msg.get("api_role") or "")
            if role.upper() in {"HUMAN", "USER"} or api_role.upper() == "HUMAN":
                text = msg.get("prompt")
                if text is None:
                    text = msg.get("content")
                if text is not None:
                    return str(text)

        # Fallback: first dict with prompt/content.
        for msg in origin_prompt:
            if not isinstance(msg, dict):
                continue
            text = msg.get("prompt")
            if text is None:
                text = msg.get("content")
            if text is not None:
                return str(text)

    # Scalar fallback.
    if origin_prompt is None:
        return ""
    return str(origin_prompt)


def _iter_records_from_shard(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        # Common OpenCompass format: {"0": {...}, "1": {...}, ...}
        keys = list(obj.keys())
        try:
            keys.sort(key=lambda x: int(x))
        except Exception:
            keys.sort()
        for k in keys:
            rec = obj.get(k)
            if isinstance(rec, dict):
                yield rec
        return
    if isinstance(obj, list):
        for rec in obj:
            if isinstance(rec, dict):
                yield rec
        return
    return


def main() -> int:
    ap = argparse.ArgumentParser(description="Build JSONL dataset from OpenCompass predictions (dedup to unique questions).")
    ap.add_argument("--oc-root", required=True, help="OpenCompass report root (contains predictions/).")
    ap.add_argument("--pred-abbr", required=True, help="Predictions subdir name under oc-root/predictions/ (model abbr).")
    ap.add_argument("--dataset", required=True, help="Dataset name: GPQA_diamond|aime2024|aime2025")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--no-dedup", action="store_true", help="Keep all records (including multi-run repeats).")
    ap.add_argument("--max-items", type=int, default=0, help="Optional cap on number of written items (0=no cap).")
    args = ap.parse_args()

    oc_root = Path(args.oc_root).expanduser().resolve()
    pred_dir = oc_root / "predictions" / str(args.pred_abbr)
    if not pred_dir.exists():
        raise SystemExit(f"Missing predictions dir: {pred_dir}")

    prefix = _dataset_prefix(args.dataset)
    shard_files = _iter_prediction_files(pred_dir, prefix)

    out_path = Path(args.out_jsonl).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[Tuple[str, str]] = set()
    written = 0
    total_seen = 0
    total_records = 0

    with out_path.open("w", encoding="utf-8") as w:
        for shard in shard_files:
            obj = _load_json(shard)
            for rec in _iter_records_from_shard(obj):
                total_records += 1
                origin_prompt = rec.get("origin_prompt")
                gold = rec.get("gold")
                q = _extract_human_prompt(origin_prompt)
                a = "" if gold is None else str(gold).strip()

                if not q:
                    continue

                key = (q, a)
                total_seen += 1
                if (not args.no_dedup) and key in seen:
                    continue
                seen.add(key)

                w.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
                written += 1
                if int(args.max_items) > 0 and written >= int(args.max_items):
                    break
            if int(args.max_items) > 0 and written >= int(args.max_items):
                break

    print(
        json.dumps(
            {
                "dataset_prefix": prefix,
                "pred_dir": str(pred_dir),
                "shards": len(shard_files),
                "records_total": total_records,
                "records_considered": total_seen,
                "written": written,
                "dedup": (not bool(args.no_dedup)),
                "out_jsonl": str(out_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
