#!/usr/bin/env python3
"""
Analyze hint-induced truncation and hint gain from dual_rollout_data dumps.

We estimate:
  1) hint-induced truncation ratio:
       control is NOT truncated, but EXP becomes truncated after inserting hints
  2) hint gain health on untruncated pairs:
       Δreward = reward_exp - reward_control on (control_untrunc & exp_untrunc)

This script expects a dump directory like:
  dual_rollout_data/<exp_name>/
    ├── control/batch_<tag>.json
    ├── exp/batch_<tag>.json
    └── cf_control/batch_<tag>.json   (ignored)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TRUNC_FINISH_REASONS = {
    "gen_budget_exhausted",
    "context_exhausted",
    "max_steps_exhausted",
}


def _safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return bool(v)
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _is_truncated(sample: dict[str, Any]) -> bool:
    if _safe_bool(sample.get("context_exhausted", False)):
        return True
    finish = sample.get("last_finish_reason", None)
    if finish in TRUNC_FINISH_REASONS:
        return True
    return False


def _hint_inserted(exp_sample: dict[str, Any]) -> bool:
    hint_len = exp_sample.get("hint_len", None)
    if hint_len is not None:
        try:
            return int(hint_len) > 0
        except Exception:
            pass
    hints = exp_sample.get("hints", None)
    if isinstance(hints, str) and hints.strip():
        return True
    return False


def _get_join_key(sample: dict[str, Any]) -> str:
    # Prefer sample_uid (unique per repeated sample). Fall back to sample_idx.
    for k in ("sample_uid", "sample_idx"):
        if k in sample and sample[k] not in (None, ""):
            return f"{k}:{sample[k]}"
    return f"row:{id(sample)}"


def _load_batch(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class BatchStats:
    batch_tag: str
    exp_samples: int
    control_samples: int
    suspect_overwritten_control_dump: bool
    total_pairs: int
    matched_pairs: int
    unmatched_control: int
    unmatched_exp: int

    control_trunc_ratio: float
    exp_trunc_ratio: float

    hint_inserted_ratio: float

    hint_induced_trunc_ratio_all: float
    hint_induced_trunc_ratio_given_hint: float

    delta_mean_untrunc: float
    delta_pos_ratio_untrunc: float
    delta_p10_untrunc: float
    delta_p50_untrunc: float
    delta_p90_untrunc: float

    delta_mean_untrunc_hint: float
    delta_pos_ratio_untrunc_hint: float


def _quantiles(x: np.ndarray, qs: Iterable[float]) -> list[float]:
    if x.size == 0:
        return [float("nan")] * len(list(qs))
    return [float(np.quantile(x, q)) for q in qs]


def analyze_one_batch(exp_path: Path, control_path: Path) -> BatchStats:
    exp = _load_batch(exp_path)
    ctrl = _load_batch(control_path)

    exp_samples = exp.get("samples") or []
    ctrl_samples = ctrl.get("samples") or []
    exp_n = int(len(exp_samples))
    ctrl_n = int(len(ctrl_samples))
    suspect_overwritten = False
    if exp_n > 0 and ctrl_n > 0:
        # In normal dual-stream dumps, exp/control should have the same sample count.
        # If control is much larger, it's usually overwritten by cf_control-tail reward dumps.
        if (ctrl_n > exp_n * 1.1) or (ctrl_n < exp_n * 0.9):
            suspect_overwritten = True

    exp_by_key: dict[str, dict[str, Any]] = {}
    for s in exp_samples:
        exp_by_key[_get_join_key(s)] = s

    ctrl_by_key: dict[str, dict[str, Any]] = {}
    for s in ctrl_samples:
        ctrl_by_key[_get_join_key(s)] = s

    keys = sorted(set(exp_by_key.keys()) | set(ctrl_by_key.keys()))
    matched = []
    unmatched_ctrl = 0
    unmatched_exp = 0
    for k in keys:
        e = exp_by_key.get(k)
        c = ctrl_by_key.get(k)
        if e is None:
            unmatched_exp += 1
            continue
        if c is None:
            unmatched_ctrl += 1
            continue
        matched.append((e, c))

    total_pairs = max(len(exp_samples), len(ctrl_samples))
    matched_pairs = len(matched)

    exp_trunc = np.array([_is_truncated(e) for e, _ in matched], dtype=bool)
    ctrl_trunc = np.array([_is_truncated(c) for _, c in matched], dtype=bool)
    hint_ins = np.array([_hint_inserted(e) for e, _ in matched], dtype=bool)

    # "control was fine, but exp got truncated after hint insertion"
    hint_induced_trunc = (~ctrl_trunc) & exp_trunc & hint_ins

    denom_all = matched_pairs
    denom_hint = int((~ctrl_trunc & hint_ins).sum())

    exp_rewards = np.array([float(e.get("reward", np.nan)) for e, _ in matched], dtype=float)
    ctrl_rewards = np.array([float(c.get("reward", np.nan)) for _, c in matched], dtype=float)
    delta = exp_rewards - ctrl_rewards

    both_untrunc = (~ctrl_trunc) & (~exp_trunc)
    delta_untrunc = delta[both_untrunc & np.isfinite(delta)]
    p10, p50, p90 = _quantiles(delta_untrunc, [0.1, 0.5, 0.9])

    delta_untrunc_hint = delta[both_untrunc & hint_ins & np.isfinite(delta)]

    def _pos_ratio(x: np.ndarray) -> float:
        if x.size == 0:
            return float("nan")
        return float((x > 0).mean())

    def _mean(x: np.ndarray) -> float:
        if x.size == 0:
            return float("nan")
        return float(np.mean(x))

    return BatchStats(
        batch_tag=str(exp.get("batch_tag") or exp_path.stem.replace("batch_", "")),
        exp_samples=exp_n,
        control_samples=ctrl_n,
        suspect_overwritten_control_dump=bool(suspect_overwritten),
        total_pairs=int(total_pairs),
        matched_pairs=int(matched_pairs),
        unmatched_control=int(unmatched_ctrl),
        unmatched_exp=int(unmatched_exp),
        control_trunc_ratio=float(ctrl_trunc.mean()) if matched_pairs else float("nan"),
        exp_trunc_ratio=float(exp_trunc.mean()) if matched_pairs else float("nan"),
        hint_inserted_ratio=float(hint_ins.mean()) if matched_pairs else float("nan"),
        hint_induced_trunc_ratio_all=float(hint_induced_trunc.sum() / denom_all) if denom_all else float("nan"),
        hint_induced_trunc_ratio_given_hint=float(hint_induced_trunc.sum() / denom_hint) if denom_hint else float("nan"),
        delta_mean_untrunc=_mean(delta_untrunc),
        delta_pos_ratio_untrunc=_pos_ratio(delta_untrunc),
        delta_p10_untrunc=float(p10),
        delta_p50_untrunc=float(p50),
        delta_p90_untrunc=float(p90),
        delta_mean_untrunc_hint=_mean(delta_untrunc_hint),
        delta_pos_ratio_untrunc_hint=_pos_ratio(delta_untrunc_hint),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dual-dir",
        type=str,
        required=True,
        help="Path to dual_rollout_data/<exp_name> (contains exp/control subdirs).",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Output CSV path (default: <dual-dir>/hint_truncation_gain.csv).",
    )
    ap.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit number of batch tags to analyze (0 = all).",
    )
    args = ap.parse_args()

    dual_dir = Path(args.dual_dir)
    exp_dir = dual_dir / "exp"
    ctrl_dir = dual_dir / "control"
    if not exp_dir.exists() or not ctrl_dir.exists():
        raise SystemExit(f"Missing exp/control under {dual_dir}")

    exp_files = sorted(exp_dir.glob("batch_*.json"))
    ctrl_files = sorted(ctrl_dir.glob("batch_*.json"))

    exp_by_tag = {p.stem.replace("batch_", ""): p for p in exp_files}
    ctrl_by_tag = {p.stem.replace("batch_", ""): p for p in ctrl_files}
    tags = sorted(set(exp_by_tag.keys()) & set(ctrl_by_tag.keys()), key=lambda x: int(x) if x.isdigit() else x)

    if args.max_batches and args.max_batches > 0:
        tags = tags[-int(args.max_batches) :]

    rows: list[dict[str, Any]] = []
    for tag in tags:
        st = analyze_one_batch(exp_by_tag[tag], ctrl_by_tag[tag])
        rows.append(st.__dict__)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No matched batch tags found.")
        return 0

    out_csv = Path(args.out_csv) if args.out_csv else (dual_dir / "hint_truncation_gain.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Print a concise summary for quick debugging.
    latest = df.sort_values("batch_tag").iloc[-1].to_dict()
    print(f"[OK] wrote: {out_csv}")
    if bool(latest.get("suspect_overwritten_control_dump", False)):
        print(
            "[WARN] control dump sample count mismatched exp; likely overwritten by cf_control tail dumps. "
            "Hint-induced truncation / Δreward pairing is NOT reliable for this run. "
            "Re-run after updating to stream_type=cf_control dump fix.",
            file=sys.stderr,
        )
    print(
        "latest batch_tag={batch_tag} matched={matched_pairs} "
        "hint_induced_trunc(all)={hint_induced_trunc_ratio_all:.4f} "
        "hint_induced_trunc|hint={hint_induced_trunc_ratio_given_hint:.4f} "
        "delta_mean_untrunc={delta_mean_untrunc:.4f} "
        "delta_pos_untrunc={delta_pos_ratio_untrunc:.4f}".format(**latest)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
