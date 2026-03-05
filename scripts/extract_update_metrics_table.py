#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


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
    run_dirs = [
        p for p in swanlab_root.iterdir() if p.is_dir() and p.name.startswith("run-")
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run-* directories found under {swanlab_root}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _iter_swanlab_strings(backup_path: Path) -> Iterator[str]:
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
            yield line.strip()
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def _build_key_to_kid(backup_path: Path) -> Dict[str, str]:
    key_to_kid: Dict[str, str] = {}
    col_key_re = re.compile(r'"key"\s*:\s*"([^"]+)"')
    col_kid_re = re.compile(r'"kid"\s*:\s*"([^"]+)"')
    for s in _iter_swanlab_strings(backup_path):
        if not s:
            continue

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

        if '"model_type"' in s and '"Column"' in s:
            m_key = col_key_re.search(s)
            m_kid = col_kid_re.search(s)
            if m_key and m_kid:
                key_to_kid.setdefault(m_key.group(1), m_kid.group(1))
    return key_to_kid


def _read_series(run_dir: Path, kid: str) -> Dict[int, float]:
    log_path = run_dir / "logs" / kid / "1000.log"
    if not log_path.exists():
        return {}
    out: Dict[int, float] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                idx = int(obj.get("index", -1))
                val = obj.get("data", None)
                if idx < 0:
                    continue
                out[idx] = float(val)
            except Exception:
                continue
    return out


DEFAULT_KEYS = [
    # global step / progress
    "training/global_step",
    # actor update metrics
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/grad_norm",
    "actor/entropy",
    # verifier update metrics (prefixed by fsdp_workers.update_verifier)
    "verifier/actor/ppo_kl",
    "verifier/actor/pg_clipfrac",
    "verifier/actor/grad_norm",
    "verifier/lr",
    "verifier/lr_scale",
    "verifier/enabled",
    "verifier/update_base",
    # shared-base conflict debug (optional)
    "conflict/cos_actor_verifier",
    "conflict/norm_ratio_verifier_over_actor",
]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Export per-step SwanLab metrics to a CSV table (actor vs verifier).",
    )
    parser.add_argument("--run-dir", default="", help="Path to swanlab/run-*/ directory.")
    parser.add_argument(
        "--swanlab-root",
        default="",
        help="Root directory containing run-* dirs (default: auto-detect).",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Metric keys to export (default: curated actor/verifier keys).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output CSV path (default: <run_dir>/update_metrics.csv).",
    )
    args = parser.parse_args(argv)

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
    else:
        swanlab_root = (
            Path(args.swanlab_root).expanduser()
            if args.swanlab_root
            else _default_swanlab_root()
        )
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

    keys = args.keys if args.keys is not None else DEFAULT_KEYS
    key_to_series: Dict[str, Dict[int, float]] = {}
    indices = set()
    for key in keys:
        kid = key_to_kid.get(key, None)
        if not kid:
            continue
        series = _read_series(run_dir, kid)
        if not series:
            continue
        key_to_series[key] = series
        indices.update(series.keys())

    out_path = Path(args.out).expanduser() if args.out else (run_dir / "update_metrics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["index"] + list(keys)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for idx in sorted(indices):
            row = [idx]
            for key in keys:
                val = key_to_series.get(key, {}).get(idx, "")
                row.append(val)
            w.writerow(row)

    print(f"[export] run_dir={run_dir}")
    print(f"[export] wrote {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        raise SystemExit(0)
