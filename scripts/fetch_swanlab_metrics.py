#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def _default_swanlab_root() -> Optional[Path]:
    env_dir = os.environ.get("SWANLAB_LOG_DIR")
    if env_dir:
        p = Path(env_dir).expanduser()
        if p.exists():
            return p

    candidates = [
        Path("/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/swanlab"),
        Path("/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/swanlab"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _pick_latest_run_dir(swanlab_root: Path) -> Path:
    run_dirs = [p for p in swanlab_root.iterdir() if p.is_dir() and p.name.startswith("run-")]
    if not run_dirs:
        raise FileNotFoundError(f"No run-* directories found under {swanlab_root}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _iter_swanlab_json_records(backup_path: Path) -> Iterator[dict]:
    """
    SwanLab `backup.swanlab` is a binary container that embeds JSON records.
    We stream-decode those records via `strings`, which is robust and fast.
    """
    proc = subprocess.Popen(
        ["strings", "-n", "8", str(backup_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        errors="ignore",
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def _build_key_to_kid(backup_path: Path) -> Dict[str, str]:
    key_to_kid: Dict[str, str] = {}
    # SwanLab Column records are JSON, but `strings` may occasionally split long
    # records across multiple lines, making JSON parsing fail. We therefore do a
    # best-effort scan over `strings` output and extract key/kid via:
    #   1) JSON decode (preferred)
    #   2) Regex on partial Column lines (fallback)
    col_key_re = re.compile(r'"key"\s*:\s*"([^"]+)"')
    col_kid_re = re.compile(r'"kid"\s*:\s*"([^"]+)"')
    proc = subprocess.Popen(
        ["strings", "-n", "8", str(backup_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        errors="ignore",
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            s = line.strip()
            if not s:
                continue

            # Preferred: parse full JSON records.
            if s.startswith("{"):
                try:
                    rec = json.loads(s)
                except Exception:
                    rec = None
                if rec and rec.get("model_type") == "Column":
                    data = rec.get("data") or {}
                    key = data.get("key")
                    kid = data.get("kid")
                    if isinstance(key, str) and isinstance(kid, str):
                        key_to_kid.setdefault(key, kid)
                        continue

            # Fallback: parse partial Column records.
            if '"model_type"' in s and '"Column"' in s:
                m_key = col_key_re.search(s)
                m_kid = col_kid_re.search(s)
                if m_key and m_kid:
                    key_to_kid.setdefault(m_key.group(1), m_kid.group(1))
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
    return key_to_kid


def _read_latest_scalar(run_dir: Path, kid: str) -> Optional[Tuple[int, float]]:
    log_path = run_dir / "logs" / kid / "1000.log"
    if not log_path.exists():
        return None

    # Read last non-empty line (tail-like).
    last: Optional[str] = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    if not last:
        return None

    try:
        obj = json.loads(last)
        return int(obj.get("index", -1)), float(obj.get("data"))
    except Exception:
        return None


DEFAULT_KEYS = [
    # training progress
    "co_grpo/training_progress",
    # cf-branch specific
    "co_grpo/cf_delta_mean",
    "co_grpo/cf_delta_std",
    "co_grpo/cf_delta_pos_ratio",
    "co_grpo/cf_missing_ratio",
    "co_grpo/cf_cost_mean",
    # general co_grpo (newer SwanLab key schema)
    "co_grpo/exp_reward/mean",
    "co_grpo/control_reward/mean",
    "co_grpo/cf_diff_mean",
    "co_grpo/cf_diff_pos_ratio",
    "co_grpo/verifier_help_rate",
    "co_grpo/exp_hint_len_mean",
    "co_grpo/exp_response_len_mean",
    "co_grpo/control_response_len_mean",
    "co_grpo/effective_exp_weight",
    "co_grpo/effective_control_weight",
    # perf/timing
    "timing_s/step",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/reward",
    "perf/throughput",
    "perf/mfu/actor",
]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch latest scalar metrics from a SwanLab run directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Path to swanlab/run-*/ directory. If empty, auto-pick latest.",
    )
    parser.add_argument(
        "--swanlab-root",
        type=str,
        default="",
        help="Root directory containing run-* dirs (default: auto-detect).",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Metric keys to fetch (default: curated co_grpo keys).",
    )
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="List available keys and exit.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional key prefix filter when listing keys.",
    )

    args = parser.parse_args(argv)

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
    else:
        swanlab_root = Path(args.swanlab_root).expanduser() if args.swanlab_root else _default_swanlab_root()
        if swanlab_root is None:
            raise FileNotFoundError(
                "Cannot locate SwanLab root. Set SWANLAB_LOG_DIR or pass --swanlab-root."
            )
        run_dir = _pick_latest_run_dir(swanlab_root)

    backup_path = run_dir / "backup.swanlab"
    if not backup_path.exists():
        raise FileNotFoundError(f"Missing backup.swanlab under {run_dir}")

    key_to_kid = _build_key_to_kid(backup_path)
    if not key_to_kid:
        raise RuntimeError(f"Failed to parse any Column mappings from {backup_path}")

    if args.list_keys:
        keys = sorted(k for k in key_to_kid.keys() if (not args.prefix or k.startswith(args.prefix)))
        for k in keys:
            print(k)
        return 0

    keys = args.keys if args.keys is not None else DEFAULT_KEYS
    print(f"[swanlab] run_dir={run_dir}")
    for key in keys:
        kid = key_to_kid.get(key)
        if kid is None:
            print(f"{key}\t<missing>")
            continue
        latest = _read_latest_scalar(run_dir, kid)
        if latest is None:
            print(f"{key}\t<no-data>")
            continue
        idx, val = latest
        print(f"{key}\t{val}\t(index={idx}, kid={kid})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        raise SystemExit(0)
