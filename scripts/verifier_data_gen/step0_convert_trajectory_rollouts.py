#!/usr/bin/env python3
"""
Step 0: Convert "pretty" rollout trajectory dumps into the jsonl format expected by
Verifier cold-start pipelines (question/output/ground_truth + acc/score).

Input format (common in old rollout dumps):
  - Files named like: rollout_idx_1_trajectory.jsonl ... rollout_idx_10_trajectory.jsonl
  - Each file contains *multiple JSON objects*, pretty-printed (multi-line), concatenated:
      { ...stats... }
      { ...sample1... }
      { ...sample2... }
      ...
  - Sample objects usually contain:
      action_id, prompt, response, label, reward, origin_data_source, finish_reason, response_len, entropy

Output format (one json object per line):
  {
    "input": "system\\n...\\nuser\\n{question}\\nassistant\\n",
    "output": "<think>...model response...</think> ...",
    "ground_truth": "...",
    "acc": true/false,
    "score": 1.0/0.0,
    "reward": ...,
    "step": 0,
    "origin_data_source": "...",
    "action_id": "...",
    "response_len": 1234,
    "finish_reason": "stop",
    "entropy": 0.123,
    "source_file": "/abs/path/to/rollout_idx_x_trajectory.jsonl"
  }

This makes the result directly consumable by:
  - /mnt/.../verifier_llmit/scripts/lora_cold_data/run_verifier_data_pipeline_v4_checkpoint.sh
  - scripts/verifier_data_gen/run_verifier_data_pipeline_v2.sh
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert reasoner with extensive experience in all areas. You approach problems through systematic "
    "thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, "
    "making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags."
)


_QWEN_USER_RE = re.compile(r"<\|im_start\|>user\s*\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)
_GENERIC_USER_RE = re.compile(r"\nuser\n(.*?)(?=\nassistant\n|$)", re.DOTALL | re.IGNORECASE)


def _extract_question_from_prompt(prompt: str) -> str:
    if not prompt:
        return ""

    m = _QWEN_USER_RE.search(prompt)
    if m:
        return m.group(1).strip()

    m = _GENERIC_USER_RE.search(prompt)
    if m:
        return m.group(1).strip()

    # Fallback: best-effort stripping of known wrappers.
    text = prompt.strip()
    text = re.sub(r"^<\|im_start\|>user\s*\n", "", text)
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return text.strip()


def _clean_response_text(response: str) -> str:
    if response is None:
        return ""

    text = str(response)
    # Remove common special tokens from various chat templates.
    for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>"):
        text = text.replace(tok, "")
    return text.strip()


def _maybe_prepend_think(prompt: str, response: str) -> str:
    """
    Many old dumps put '<think>' at the end of the prompt, so the stored `response`
    begins *inside* the think tag. Our pipelines usually expect output to start with
    '<think>' and the prompt to end with 'assistant\\n'.
    """
    prompt_tail = (prompt or "").rstrip()
    resp = response.lstrip()
    if prompt_tail.endswith("<think>") and not resp.startswith("<think>"):
        return "<think>" + response
    return response


def _build_input_text(question: str, system_prompt: str) -> str:
    q = (question or "").strip()
    sp = (system_prompt or "").strip()
    if sp:
        return f"system\n{sp}\nuser\n{q}\nassistant\n"
    return f"user\n{q}\nassistant\n"


def _iter_pretty_json_objects(path: Path, *, max_objects: int = 0) -> Iterator[Dict[str, Any]]:
    """
    Iterate over concatenated JSON objects in a file.

    Supports:
    - "pretty" multi-line objects where top-level '{' and '}' are at column 0
    - standard jsonl where each line is a JSON object
    """
    num = 0
    buf: List[str] = []
    in_obj = False

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not in_obj:
                if not line.startswith("{"):
                    continue
                # Fast path: one-line JSON object.
                if line.rstrip().endswith("}"):
                    try:
                        yield json.loads(line)
                        num += 1
                        if max_objects and num >= max_objects:
                            return
                    except json.JSONDecodeError:
                        # Fall back to buffered mode.
                        in_obj = True
                        buf = [line]
                    continue

                in_obj = True
                buf = [line]
                continue

            # Buffered mode.
            buf.append(line)
            if line.startswith("}"):
                raw = "".join(buf)
                buf = []
                in_obj = False
                try:
                    yield json.loads(raw)
                    num += 1
                    if max_objects and num >= max_objects:
                        return
                except json.JSONDecodeError:
                    # Skip bad objects but keep going.
                    continue


def _find_trajectory_files(
    root_dir: Path,
    *,
    filename_pattern: str,
    skip_substrings: List[str],
    skip_rollout_subdir: bool,
    max_files: int,
) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(str(root_dir)):
        for fn in filenames:
            if not fnmatch.fnmatch(fn, filename_pattern):
                continue
            p = Path(dirpath) / fn
            p_str = str(p)
            if skip_rollout_subdir and "/rollout/" in p_str:
                continue
            if any(s and (s in p_str) for s in skip_substrings):
                continue
            out.append(p)
            if max_files and len(out) >= max_files:
                return sorted(out)
    return sorted(out)


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_bool_acc(reward: Any) -> Optional[bool]:
    r = _as_float(reward)
    if r is None:
        return None
    # Common: reward in {-1, +1} or [0, 1].
    return r > 0.0


def _quality_key(rec: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Higher is better.
    Prefer:
      1) finish_reason == stop
      2) larger response_len (token length if provided)
      3) larger output char length
    """
    stop = 1 if str(rec.get("finish_reason") or "").lower() == "stop" else 0
    rl = rec.get("response_len")
    try:
        rl_i = int(rl) if rl is not None else 0
    except Exception:
        rl_i = 0
    out_len = len(str(rec.get("output") or ""))
    return (stop, rl_i, out_len)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert pretty rollout trajectory dumps to rollout_w_answer jsonl.")
    ap.add_argument("--source_dir", type=str, required=True, help="Root dir to search for rollout_idx_*_trajectory.jsonl")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory")

    ap.add_argument("--filename_pattern", type=str, default="rollout_idx_*_trajectory.jsonl")
    ap.add_argument("--data_source", type=str, default="math", help="Filter by origin_data_source (empty disables)")
    ap.add_argument(
        "--skip_substrings",
        type=str,
        default="resume,mask,run",
        help="Comma-separated substrings; files with these in path will be skipped",
    )
    ap.add_argument("--skip_rollout_subdir", action="store_true", help="Skip any path containing '/rollout/'")

    ap.add_argument("--require_finish_reason_stop", action="store_true", help="Keep only finish_reason == 'stop'")
    ap.add_argument("--min_output_chars", type=int, default=1, help="Drop samples with output shorter than this")
    ap.add_argument("--min_response_len", type=int, default=0, help="Drop samples with response_len smaller than this")
    ap.add_argument("--max_response_len", type=int, default=0, help="Drop samples with response_len larger than this (0=off)")

    ap.add_argument(
        "--dedup_per_action",
        action="store_true",
        help="Keep at most one correct and one incorrect sample per action_id (best by response_len/output_len)",
    )
    ap.add_argument(
        "--stream_output",
        action="store_true",
        help="Write output incrementally (low memory). Not compatible with --dedup_per_action.",
    )

    ap.add_argument("--max_files", type=int, default=0, help="Only process first N files (0=all)")
    ap.add_argument("--max_objects_per_file", type=int, default=0, help="Only parse first N JSON objects per file (0=all)")
    ap.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--output_name", type=str, default="converted_from_trajectory.jsonl")

    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"[ERR] source_dir not found: {source_dir}", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    out_dir = output_root / "rollout_w_answer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    skip_substrings = [s.strip() for s in (args.skip_substrings or "").split(",") if s.strip()]
    files = _find_trajectory_files(
        source_dir,
        filename_pattern=args.filename_pattern,
        skip_substrings=skip_substrings,
        skip_rollout_subdir=bool(args.skip_rollout_subdir),
        max_files=int(args.max_files or 0),
    )
    if not files:
        print(f"[ERR] No files matched under: {source_dir} (pattern={args.filename_pattern})", file=sys.stderr)
        return 2

    print(f"[INFO] Found {len(files)} trajectory files")
    for p in files[:5]:
        print(f"  - {p}")
    if len(files) > 5:
        print("  ...")

    drop_reasons: Counter[str] = Counter()
    total_objects = 0
    total_samples = 0

    if args.stream_output and args.dedup_per_action:
        print("[ERR] --stream_output is not compatible with --dedup_per_action", file=sys.stderr)
        return 2

    candidates: Dict[Tuple[str, bool], Dict[str, Any]] = {}
    kept: List[Dict[str, Any]] = []
    kept_count = 0
    pos_count = 0
    neg_count = 0
    out_f = None
    if args.stream_output:
        out_f = out_path.open("w", encoding="utf-8")

    data_source_filter = (args.data_source or "").strip()
    require_stop = bool(args.require_finish_reason_stop)
    min_chars = int(args.min_output_chars or 0)
    min_rlen = int(args.min_response_len or 0)
    max_rlen = int(args.max_response_len or 0)

    for fp in files:
        for obj in _iter_pretty_json_objects(fp, max_objects=int(args.max_objects_per_file or 0)):
            total_objects += 1

            # Skip file-level stats objects
            if "action_id" not in obj:
                drop_reasons["no_action_id"] += 1
                continue

            origin_ds = obj.get("origin_data_source")
            if data_source_filter and str(origin_ds) != data_source_filter:
                drop_reasons["origin_data_source_mismatch"] += 1
                continue

            prompt = obj.get("prompt") or obj.get("input") or ""
            response = obj.get("response") or obj.get("output") or ""
            label = obj.get("label")
            if label is None:
                label = obj.get("ground_truth")
            if label is None:
                label = obj.get("answer")

            if not prompt or not response or label is None:
                drop_reasons["missing_fields"] += 1
                continue

            finish_reason = str(obj.get("finish_reason") or "").lower() if obj.get("finish_reason") is not None else ""
            if require_stop and finish_reason and finish_reason != "stop":
                drop_reasons["finish_reason_not_stop"] += 1
                continue

            question = _extract_question_from_prompt(str(prompt))
            if not question:
                drop_reasons["question_empty"] += 1
                continue

            out_text = _clean_response_text(str(response))
            out_text = _maybe_prepend_think(str(prompt), out_text)
            if len(out_text) < min_chars:
                drop_reasons["output_too_short"] += 1
                continue

            response_len = obj.get("response_len")
            try:
                response_len_i = int(response_len) if response_len is not None else 0
            except Exception:
                response_len_i = 0

            if min_rlen and response_len_i and response_len_i < min_rlen:
                drop_reasons["response_len_too_short"] += 1
                continue
            if max_rlen and response_len_i and response_len_i > max_rlen:
                drop_reasons["response_len_too_long"] += 1
                continue

            reward = obj.get("reward")
            if reward is None and obj.get("score") is not None:
                reward = obj.get("score")
            acc = _to_bool_acc(reward)
            if acc is None:
                # Some dumps store correctness elsewhere; for our usage we need an acc signal.
                drop_reasons["acc_unknown"] += 1
                continue

            record: Dict[str, Any] = {
                "input": _build_input_text(question, str(args.system_prompt or "")),
                "output": out_text,
                "ground_truth": str(label),
                "acc": bool(acc),
                "score": 1.0 if acc else 0.0,
                "reward": reward,
                "step": 0,
                "origin_data_source": origin_ds,
                "action_id": str(obj.get("action_id")),
                "response_len": response_len_i,
                "finish_reason": obj.get("finish_reason"),
                "entropy": obj.get("entropy"),
                "source_file": str(fp),
            }

            total_samples += 1

            if args.dedup_per_action:
                key = (record["action_id"], bool(record["acc"]))
                prev = candidates.get(key)
                if prev is None or _quality_key(record) > _quality_key(prev):
                    candidates[key] = record
                continue

            if out_f is not None:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept_count += 1
                if bool(record.get("acc")):
                    pos_count += 1
                else:
                    neg_count += 1
            else:
                kept.append(record)

    if out_f is not None:
        try:
            out_f.close()
        except Exception:
            pass

    if args.dedup_per_action:
        kept = list(candidates.values())

    if out_f is None:
        kept.sort(key=lambda r: (r.get("action_id") or "", 0 if r.get("acc") else 1), reverse=False)
        kept_count = len(kept)
        pos_count = sum(1 for r in kept if r.get("acc"))
        neg_count = kept_count - pos_count

    print(f"[INFO] Parsed objects: {total_objects}")
    print(f"[INFO] Candidate samples (pre-dedup): {total_samples}")
    print(f"[INFO] Kept samples: {kept_count}")
    if kept_count:
        print(
            f"[INFO] acc distribution: correct={pos_count} incorrect={neg_count} correct_ratio={pos_count/kept_count:.3f}"
        )

    if drop_reasons:
        print("[INFO] Drop reasons (top 12):")
        for k, v in drop_reasons.most_common(12):
            print(f"  - {k}: {v}")

    # Write output jsonl (non-streaming mode).
    if out_f is None:
        with out_path.open("w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats_path = out_path.with_suffix(".stats.json")
    stats = {
        "source_dir": str(source_dir),
        "num_files": len(files),
        "parsed_objects": total_objects,
        "candidate_samples": total_samples,
        "kept_samples": kept_count,
        "drop_reasons": dict(drop_reasons),
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    manifest_path = out_path.with_suffix(".manifest.txt")
    with manifest_path.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(str(p) + "\n")

    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Stats: {stats_path}")
    print(f"[OK] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
