#!/usr/bin/env python3
"""
Generate tables/plots comparing Verifier LoRA vs shared-base (no-LoRA) runs.

This script works off existing artifacts:
  - dual_rollout_data JSON dumps (exp/control)
  - rank0 training logs (step metrics + no-decision/preempt warnings)

Example:
  python repos/repro/scripts/report_lora_vs_base.py \
    --base-exp cogrpo32-hf181-nolora-dapo17k-pl2048-cfK2E2-int2-32k-v4 \
    --lora-exp cogrpo32-hf170-lora1705-dapo17k-pl2048-cfK2E2-int2-32k-v4 \
    --base-batches 5 10 15 20 \
    --lora-batches 5 10 15
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class DumpPaths:
    ckpt_root: Path

    def dump_file(self, exp_name: str, stream: str, batch_tag: int) -> Path:
        return (
            self.ckpt_root
            / exp_name
            / "dual_rollout_data"
            / exp_name
            / stream
            / f"batch_{batch_tag}.json"
        )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    idx = int(q * (len(xs_sorted) - 1))
    return float(xs_sorted[idx])


def _safe_mean(xs: Iterable[float]) -> float:
    xs_list = list(xs)
    if not xs_list:
        return float("nan")
    return float(statistics.fmean(xs_list))


def _safe_std(xs: Iterable[float]) -> float:
    xs_list = list(xs)
    if len(xs_list) < 2:
        return float("nan")
    try:
        return float(statistics.pstdev(xs_list))
    except Exception:
        return float("nan")


def _pearson(xs: List[float], ys: List[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return float("nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = statistics.fmean(xvals)
    my = statistics.fmean(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return float(cov / (vx * vy) ** 0.5)


def _rank(xs: List[float]) -> List[float]:
    # Average rank for ties.
    pairs = sorted((float(x), i) for i, x in enumerate(xs))
    out = [0.0] * len(xs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for _, idx in pairs[i:j]:
            out[idx] = avg_rank
        i = j
    return out


def _spearman(xs: List[float], ys: List[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return float("nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    return _pearson(_rank(xvals), _rank(yvals))


def _finish_reason_ratios(reasons: List[str], prefix: str) -> Dict[str, float]:
    if not reasons:
        return {
            f"{prefix}_finish_reason_eos_ratio": float("nan"),
            f"{prefix}_finish_reason_length_ratio": float("nan"),
            f"{prefix}_finish_reason_stop_ratio": float("nan"),
            f"{prefix}_finish_reason_other_ratio": float("nan"),
        }
    normed = []
    for r in reasons:
        r2 = str(r) if r is not None else ""
        r2 = r2.strip().lower()
        normed.append(r2 or "none")
    c = collections.Counter(normed)
    n = float(sum(c.values()))
    eos = float(c.get("eos", 0))
    length = float(c.get("length", 0))
    stop = float(c.get("stop", 0))
    other = float(n - eos - length - stop)
    return {
        f"{prefix}_finish_reason_eos_ratio": eos / n,
        f"{prefix}_finish_reason_length_ratio": length / n,
        f"{prefix}_finish_reason_stop_ratio": stop / n,
        f"{prefix}_finish_reason_other_ratio": other / n,
    }


def _parse_step_kvs(step_line: str) -> Dict[str, float]:
    # Step line uses " - k:v - k:v ..." format.
    # Values are mostly numeric; keep numeric prefix.
    out: Dict[str, float] = {}
    parts = step_line.split(" - ")
    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        m = re.match(r"([-+]?\d+(?:\.\d+)?)", value)
        if not m:
            continue
        try:
            out[key] = float(m.group(1))
        except Exception:
            continue
    return out


def load_dual_dump_stats(
    dump_paths: DumpPaths,
    exp_name: str,
    batch_tags: List[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for tag in batch_tags:
        exp_path = dump_paths.dump_file(exp_name, "exp", tag)
        control_path = dump_paths.dump_file(exp_name, "control", tag)
        if not exp_path.exists() or not control_path.exists():
            rows.append(
                {
                    "exp_name": exp_name,
                    "batch_tag": tag,
                    "missing": True,
                    "exp_path": str(exp_path),
                    "control_path": str(control_path),
                }
            )
            continue

        exp_obj = _read_json(exp_path)
        control_obj = _read_json(control_path)
        exp_samples = exp_obj.get("samples") or []
        control_samples = control_obj.get("samples") or []

        main_n = len(exp_samples)
        control_main = [
            s for s in control_samples if int(s.get("sample_idx", -1)) < main_n
        ]
        control_main_by_idx = {int(s["sample_idx"]): s for s in control_main}

        deltas: List[float] = []
        hint_lens: List[int] = []
        resp_lens: List[int] = []
        ctx_exhausted: List[int] = []
        control_ctx_exhausted: List[int] = []
        exp_finish_reasons: List[str] = []
        control_finish_reasons: List[str] = []
        for s in exp_samples:
            idx = int(s.get("sample_idx", -1))
            c = control_main_by_idx.get(idx)
            if c is None:
                continue
            deltas.append(float(s.get("reward", 0.0)) - float(c.get("reward", 0.0)))
            hint_lens.append(int(s.get("hint_len", 0) or 0))
            resp_lens.append(int(s.get("valid_response_length") or 0))
            ctx_exhausted.append(1 if s.get("context_exhausted") else 0)
            if "context_exhausted" in c:
                control_ctx_exhausted.append(1 if c.get("context_exhausted") else 0)
            if "last_finish_reason" in s:
                exp_finish_reasons.append(str(s.get("last_finish_reason") or ""))
            if "last_finish_reason" in c:
                control_finish_reasons.append(str(c.get("last_finish_reason") or ""))

        hint_pos = [d for d, h in zip(deltas, hint_lens) if h > 0]
        hint_zero = [d for d, h in zip(deltas, hint_lens) if h == 0]
        deltas_p10 = _quantile(deltas, 0.1)
        deltas_p50 = _quantile(deltas, 0.5)
        deltas_p90 = _quantile(deltas, 0.9)

        pearson_delta_hint = _pearson(deltas, [float(h) for h in hint_lens])
        spearman_delta_hint = _spearman(deltas, [float(h) for h in hint_lens])
        pearson_delta_resp = _pearson(deltas, [float(r) for r in resp_lens])
        spearman_delta_resp = _spearman(deltas, [float(r) for r in resp_lens])

        rows.append(
            {
                "exp_name": exp_name,
                "batch_tag": tag,
                "missing": False,
                "exp_total": len(exp_samples),
                "control_total": len(control_samples),
                "control_main_total": len(control_main),
                "control_extra_total": len(control_samples) - len(control_main),
                "hint_rate": len([h for h in hint_lens if h > 0])
                / max(1, len(hint_lens)),
                "hint_len_mean": _safe_mean(hint_lens),
                "hint_len_p50": _quantile([float(x) for x in hint_lens], 0.5),
                "hint_len_p90": _quantile([float(x) for x in hint_lens], 0.9),
                "resp_len_mean": _safe_mean(resp_lens),
                "resp_len_p50": _quantile([float(x) for x in resp_lens], 0.5),
                "resp_len_p90": _quantile([float(x) for x in resp_lens], 0.9),
                "ctx_exhausted_ratio": sum(ctx_exhausted) / max(1, len(ctx_exhausted)),
                "control_ctx_exhausted_ratio": sum(control_ctx_exhausted)
                / max(1, len(control_ctx_exhausted))
                if control_ctx_exhausted
                else float("nan"),
                "delta_mean": _safe_mean(deltas),
                "delta_std": _safe_std(deltas),
                "delta_p10": deltas_p10,
                "delta_p50": deltas_p50,
                "delta_p90": deltas_p90,
                "delta_pos_ratio": sum(d > 0 for d in deltas) / max(1, len(deltas)),
                "delta_neg_ratio": sum(d < 0 for d in deltas) / max(1, len(deltas)),
                "delta_zero_ratio": sum(d == 0 for d in deltas) / max(1, len(deltas)),
                "delta_mean_hint_pos": _safe_mean(hint_pos),
                "delta_mean_hint_zero": _safe_mean(hint_zero),
                "delta_pos_ratio_hint_pos": sum(d > 0 for d in hint_pos)
                / max(1, len(hint_pos)),
                "delta_neg_ratio_hint_pos": sum(d < 0 for d in hint_pos)
                / max(1, len(hint_pos)),
                "corr_delta_hint_len_pearson": pearson_delta_hint,
                "corr_delta_hint_len_spearman": spearman_delta_hint,
                "corr_delta_resp_len_pearson": pearson_delta_resp,
                "corr_delta_resp_len_spearman": spearman_delta_resp,
                **_finish_reason_ratios(exp_finish_reasons, prefix="exp"),
                **_finish_reason_ratios(control_finish_reasons, prefix="control"),
            }
        )

    return pd.DataFrame(rows).sort_values(["exp_name", "batch_tag"])


def load_dual_dump_samples(
    dump_paths: DumpPaths,
    exp_name: str,
    batch_tags: List[int],
    max_samples_per_tag: int = 0,
) -> pd.DataFrame:
    """Load per-sample exp/control deltas for plotting/debugging."""
    rows: List[Dict[str, Any]] = []
    for tag in batch_tags:
        exp_path = dump_paths.dump_file(exp_name, "exp", tag)
        control_path = dump_paths.dump_file(exp_name, "control", tag)
        if not exp_path.exists() or not control_path.exists():
            continue

        exp_obj = _read_json(exp_path)
        control_obj = _read_json(control_path)
        exp_samples = exp_obj.get("samples") or []
        control_samples = control_obj.get("samples") or []

        main_n = len(exp_samples)
        control_main = [
            s for s in control_samples if int(s.get("sample_idx", -1)) < main_n
        ]
        control_main_by_idx = {int(s["sample_idx"]): s for s in control_main}

        n_keep = int(max_samples_per_tag) if max_samples_per_tag else len(exp_samples)
        for s in exp_samples[:n_keep]:
            idx = int(s.get("sample_idx", -1))
            c = control_main_by_idx.get(idx)
            if c is None:
                continue
            exp_reward = float(s.get("reward", 0.0))
            ctrl_reward = float(c.get("reward", 0.0))
            rows.append(
                {
                    "exp_name": exp_name,
                    "batch_tag": int(tag),
                    "sample_idx": idx,
                    "delta": exp_reward - ctrl_reward,
                    "exp_reward": exp_reward,
                    "control_reward": ctrl_reward,
                    "hint_len": int(s.get("hint_len", 0) or 0),
                    "exp_resp_len": int(s.get("valid_response_length") or 0),
                    "control_resp_len": int(c.get("valid_response_length") or 0),
                    "exp_context_exhausted": bool(s.get("context_exhausted"))
                    if "context_exhausted" in s
                    else None,
                    "control_context_exhausted": bool(c.get("context_exhausted"))
                    if "context_exhausted" in c
                    else None,
                    "exp_finish_reason": s.get("last_finish_reason")
                    if "last_finish_reason" in s
                    else None,
                    "control_finish_reason": c.get("last_finish_reason")
                    if "last_finish_reason" in c
                    else None,
                }
            )
    return pd.DataFrame(rows)


def load_log_step_stats(log_path: Path, exp_name: str) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(
            [{"exp_name": exp_name, "missing": True, "log_path": str(log_path)}]
        )

    lines = log_path.read_text(errors="ignore").splitlines()

    step_re = re.compile(r"step:(\d+)\s+-\s+")
    no_dec_re = re.compile(r"High no-decision rate:\s+(\d+)\s*/\s*(\d+)")

    step_entries: List[Tuple[int, int, str]] = []
    for i, ln in enumerate(lines):
        m = step_re.search(ln)
        if not m:
            continue
        step_entries.append((int(m.group(1)), i, ln))

    if not step_entries:
        return pd.DataFrame(
            [{"exp_name": exp_name, "missing": True, "log_path": str(log_path)}]
        )

    rows: List[Dict[str, Any]] = []
    prev_i = 0
    for step, i, ln in step_entries:
        segment = lines[prev_i : i + 1]
        prev_i = i + 1

        numer = 0
        denom = 0
        for s in segment:
            m = no_dec_re.search(s)
            if not m:
                continue
            numer += int(m.group(1))
            denom += int(m.group(2))

        preempt = sum("PreemptionMode.RECOMPUTE" in s for s in segment)
        kvs = _parse_step_kvs(ln)

        row: Dict[str, Any] = {"exp_name": exp_name, "missing": False, "step": step}
        row.update(kvs)
        row["no_decision_numer"] = numer
        row["no_decision_denom"] = denom
        row["no_decision_rate"] = (numer / denom) if denom else float("nan")
        row["preempt_recompute_count"] = preempt
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["exp_name", "step"])


def bucket_delta_stats(
    samples_df: pd.DataFrame,
    exp_name: str,
    xcol: str,
    q: int = 5,
) -> pd.DataFrame:
    df = samples_df[samples_df["exp_name"] == exp_name].copy()
    if df.empty or xcol not in df.columns or "delta" not in df.columns:
        return pd.DataFrame()

    df = df[[xcol, "delta"]].dropna()
    if df.empty:
        return pd.DataFrame()

    df[xcol] = df[xcol].astype(float)
    df["delta"] = df["delta"].astype(float)
    try:
        df["bucket"] = pd.qcut(df[xcol], q=int(q), duplicates="drop")
    except Exception:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for bucket, g in df.groupby("bucket"):
        n = int(len(g))
        if n <= 0:
            continue
        rows.append(
            {
                "exp_name": exp_name,
                "xcol": xcol,
                "bucket": str(bucket),
                "n": n,
                "x_mean": float(g[xcol].mean()),
                "delta_mean": float(g["delta"].mean()),
                "delta_pos_ratio": float((g["delta"] > 0).mean()),
                "delta_zero_ratio": float((g["delta"] == 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["exp_name", "xcol", "bucket"])


def _plot_compare_series(
    out_dir: Path,
    base_df: pd.DataFrame,
    lora_df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    fname: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.figure(figsize=(10, 4))
    for label, df in [("base", base_df), ("lora", lora_df)]:
        if df.empty or y not in df.columns or x not in df.columns:
            continue
        df2 = df[[x, y]].dropna()
        if df2.empty:
            continue
        plt.plot(
            df2[x].to_list(), df2[y].to_list(), marker="o", linewidth=1.5, label=label
        )
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-exp", required=True, help="Experiment name (shared-base / no LoRA)."
    )
    parser.add_argument("--lora-exp", required=True, help="Experiment name (verifier LoRA).")
    parser.add_argument("--base-batches", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--lora-batches", type=int, nargs="+", default=[5, 10, 15])

    parser.add_argument(
        "--ckpt-root",
        default="/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/checkpoints/co_grpo_v2",
    )
    parser.add_argument(
        "--log-root",
        default="/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/logs",
    )
    parser.add_argument("--out-dir", default="", help="Output dir.")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (repo_root / "outputs" / f"lora_vs_base_{args.base_exp}_vs_{args.lora_exp}")
    )

    dump_paths = DumpPaths(ckpt_root=Path(args.ckpt_root))
    base_dual = load_dual_dump_stats(dump_paths, args.base_exp, args.base_batches)
    lora_dual = load_dual_dump_stats(dump_paths, args.lora_exp, args.lora_batches)

    out_dir.mkdir(parents=True, exist_ok=True)
    dual_csv = out_dir / "dual_batches.csv"
    dual_df = pd.concat([base_dual, lora_dual], ignore_index=True)
    dual_df.to_csv(dual_csv, index=False)
    # Alias for report naming used in analysis docs.
    reward_csv = out_dir / "reward_decomposition.csv"
    dual_df.to_csv(reward_csv, index=False)

    base_samples = load_dual_dump_samples(
        dump_paths, args.base_exp, args.base_batches, max_samples_per_tag=0
    )
    lora_samples = load_dual_dump_samples(
        dump_paths, args.lora_exp, args.lora_batches, max_samples_per_tag=0
    )
    samples_df = pd.concat([base_samples, lora_samples], ignore_index=True)
    samples_csv = out_dir / "sample_deltas.csv"
    samples_df.to_csv(samples_csv, index=False)

    # Bucket tables: delta statistics by hint_len / response_len quantiles.
    bucket_frames = []
    for exp_name in (args.base_exp, args.lora_exp):
        for xcol in ("hint_len", "exp_resp_len"):
            bt = bucket_delta_stats(samples_df, exp_name=exp_name, xcol=xcol, q=5)
            if not bt.empty:
                bucket_frames.append(bt)
    bucket_csv = out_dir / "bucket_delta_stats.csv"
    if bucket_frames:
        pd.concat(bucket_frames, ignore_index=True).to_csv(bucket_csv, index=False)

    base_log = load_log_step_stats(
        Path(args.log_root) / f"verl_log_{args.base_exp}_rank0.txt", args.base_exp
    )
    lora_log = load_log_step_stats(
        Path(args.log_root) / f"verl_log_{args.lora_exp}_rank0.txt", args.lora_exp
    )
    steps_csv = out_dir / "log_steps.csv"
    pd.concat([base_log, lora_log], ignore_index=True).to_csv(steps_csv, index=False)

    _plot_compare_series(
        out_dir,
        base_log,
        lora_log,
        x="training/global_step",
        y="co_grpo/cf_delta_mean",
        title="cf_delta_mean vs global_step",
        fname="cf_delta_mean.png",
    )
    _plot_compare_series(
        out_dir,
        base_log,
        lora_log,
        x="training/global_step",
        y="co_grpo/verifier_help_rate",
        title="verifier_help_rate vs global_step",
        fname="verifier_help_rate.png",
    )
    _plot_compare_series(
        out_dir,
        base_log,
        lora_log,
        x="training/global_step",
        y="no_decision_rate",
        title="no_decision_rate (from log warnings) vs global_step",
        fname="no_decision_rate.png",
    )
    _plot_compare_series(
        out_dir,
        base_dual,
        lora_dual,
        x="batch_tag",
        y="delta_mean_hint_pos",
        title="mean(exp-control) over samples WITH hint_len>0",
        fname="delta_mean_hint_pos_by_batch.png",
    )

    # Per-sample plots (hist/scatter) for quick diagnosis.
    try:
        import matplotlib.pyplot as plt

        if not base_samples.empty or not lora_samples.empty:
            plt.figure(figsize=(8, 4))
            for label, df in [("base", base_samples), ("lora", lora_samples)]:
                if df.empty:
                    continue
                xs = df["delta"].dropna().astype(float).to_list()
                if not xs:
                    continue
                plt.hist(xs, bins=41, alpha=0.45, density=True, label=label)
            plt.title("delta histogram (exp-control)")
            plt.xlabel("delta")
            plt.ylabel("density")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "delta_hist.png", dpi=160)
            plt.close()

            for xcol, fname, xlabel in [
                ("hint_len", "delta_vs_hintlen.png", "hint_len"),
                ("exp_resp_len", "delta_vs_resplen.png", "exp_response_len"),
            ]:
                plt.figure(figsize=(8, 4))
                for label, df in [("base", base_samples), ("lora", lora_samples)]:
                    if df.empty or xcol not in df.columns:
                        continue
                    df2 = df[[xcol, "delta"]].dropna()
                    if df2.empty:
                        continue
                    plt.scatter(
                        df2[xcol].astype(float),
                        df2["delta"].astype(float),
                        s=8,
                        alpha=0.25,
                        label=label,
                    )
                plt.title(f"delta vs {xlabel}")
                plt.xlabel(xlabel)
                plt.ylabel("delta")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / fname, dpi=160)
                plt.close()
    except Exception:
        pass

    print(f"[report] wrote {dual_csv}")
    print(f"[report] wrote {reward_csv}")
    print(f"[report] wrote {samples_csv}")
    if bucket_frames:
        print(f"[report] wrote {bucket_csv}")
    print(f"[report] wrote {steps_csv}")
    print(f"[report] plots: {out_dir}/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
