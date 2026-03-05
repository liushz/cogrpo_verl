#!/usr/bin/env python3
"""
Report offline OC-alignment runs with multi-arm metrics and diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dataset_meta(dataset: str) -> Tuple[str, int]:
    d = dataset.strip().lower()
    if d in {"gpqa_diamond", "gpqa"}:
        return ("GPQA_diamond", 8)
    if d == "aime2024":
        return ("aime2024", 32)
    if d == "aime2025":
        return ("aime2025", 32)
    raise ValueError(f"Unsupported dataset={dataset}")


def _fmt(x: Any) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return "-"
        return f"{v:.2f}"
    except Exception:
        return "-"


def _fmt_delta(x: Any) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return "-"
        return f"{v:+.2f}"
    except Exception:
        return "-"


def _pick_metric(obj: Dict[str, Any], key_candidates: List[str]) -> float:
    for k in key_candidates:
        if k in obj and obj.get(k) is not None:
            try:
                return float(obj[k])
            except Exception:
                continue
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Report offline OC-alignment benchmark.")
    ap.add_argument("--oc-root", required=True)
    ap.add_argument("--oc-model-abbr", required=True)
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run mapping: dataset:arm=/path/to/run_dir. Example: aime2024:control_oc_exact=/tmp/run",
    )
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    runs: Dict[str, Dict[str, Path]] = {}
    for item in args.run:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise SystemExit(f"Bad --run: {item!r}, expect dataset:arm=/path/to/run_dir")
        left, p = item.split("=", 1)
        dataset, arm = left.split(":", 1)
        dataset = dataset.strip()
        arm = arm.strip()
        run_dir = Path(p).expanduser().resolve()
        runs.setdefault(dataset, {})[arm] = run_dir

    if not runs:
        raise SystemExit("No --run provided.")

    summary: Dict[str, Any] = {
        "oc_root": str(Path(args.oc_root).expanduser().resolve()),
        "oc_model_abbr": str(args.oc_model_abbr),
        "datasets": {},
    }

    lines: List[str] = []
    lines.append("# Offline OC Alignment Report")
    lines.append("")

    for dataset, arm_dirs in runs.items():
        oc_name, k = _dataset_meta(dataset)
        baseline_path = Path(args.oc_root).expanduser().resolve() / "results" / args.oc_model_abbr / f"{oc_name}.json"
        baseline = _load_json(baseline_path)
        base_acc = _pick_metric(baseline, [f"accuracy ({k} runs average)"])
        base_pass = _pick_metric(baseline, [f"G-Pass@{k}_0.0"])

        dataset_out: Dict[str, Any] = {
            "baseline": {
                "results_file": str(baseline_path),
                "accuracy": base_acc,
                "g_pass": base_pass,
                "k": int(k),
            },
            "arms": {},
        }

        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Arm | OC accuracy | OC G-Pass | CV accuracy | CV G-Pass | long>=32k% | interventions_nonzero% |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

        arm_metric: Dict[str, Dict[str, float]] = {}
        for arm, run_dir in sorted(arm_dirs.items()):
            run_dir = Path(run_dir)
            oc_json = _load_json(run_dir / f"oc_metrics.{arm}.json")
            cv_json = _load_json(run_dir / f"cv_passk.{arm}.json")
            diag_json = _load_json(run_dir / f"diag.{arm}.json")

            oc_acc = _pick_metric(oc_json, [f"accuracy ({k} runs average)"])
            oc_pass = _pick_metric(oc_json, [f"G-Pass@{k}_0.0"])
            cv_acc = float("nan")
            cv_pass = float("nan")
            if "control" in cv_json:
                cv_acc = _pick_metric(cv_json["control"], ["accuracy"])
                cv_pass = _pick_metric(cv_json["control"], ["g_pass_at_k"])
            if "exp" in cv_json:
                cv_acc = _pick_metric(cv_json["exp"], ["accuracy"])
                cv_pass = _pick_metric(cv_json["exp"], ["g_pass_at_k"])
            if not math.isnan(cv_acc):
                cv_acc *= 100.0
            if not math.isnan(cv_pass):
                cv_pass *= 100.0

            long32 = float("nan")
            interventions_nonzero = float("nan")
            for side_key in ("control", "exp"):
                if side_key in diag_json and isinstance(diag_json[side_key], dict):
                    side = diag_json[side_key]
                    tokens = side.get("tokens_model_gen") or {}
                    if math.isnan(long32):
                        long32 = _pick_metric(tokens, ["ge_32000_pct"])
                    if side_key == "exp":
                        itv = side.get("intervention") or {}
                        interventions_nonzero = _pick_metric(itv, ["nonzero_pct"])

            dataset_out["arms"][arm] = {
                "run_dir": str(run_dir),
                "oc_metrics": oc_json,
                "cv_metrics": cv_json,
                "diag": diag_json,
            }
            arm_metric[arm] = {
                "oc_acc": oc_acc,
                "oc_pass": oc_pass,
                "cv_acc": cv_acc,
                "cv_pass": cv_pass,
                "long32": long32,
                "interventions_nonzero": interventions_nonzero,
            }

            lines.append(
                f"| {arm} | {_fmt(oc_acc)} | {_fmt(oc_pass)} | {_fmt(cv_acc)} | {_fmt(cv_pass)} | {_fmt(long32)} | {_fmt(interventions_nonzero)} |"
            )

        lines.append("")
        lines.append(
            f"- Baseline (OC): accuracy={_fmt(base_acc)}, G-Pass@{k}_0.0={_fmt(base_pass)}"
        )

        delta: Dict[str, Any] = {}
        if "exp_chunked" in arm_metric and "control_chunked" in arm_metric:
            delta["intervention_pure"] = {
                "oc_acc": arm_metric["exp_chunked"]["oc_acc"] - arm_metric["control_chunked"]["oc_acc"],
                "oc_pass": arm_metric["exp_chunked"]["oc_pass"] - arm_metric["control_chunked"]["oc_pass"],
                "cv_acc": arm_metric["exp_chunked"]["cv_acc"] - arm_metric["control_chunked"]["cv_acc"],
                "cv_pass": arm_metric["exp_chunked"]["cv_pass"] - arm_metric["control_chunked"]["cv_pass"],
            }
            lines.append(
                f"- Delta intervention(pure): OC acc={_fmt_delta(delta['intervention_pure']['oc_acc'])}, "
                f"OC pass={_fmt_delta(delta['intervention_pure']['oc_pass'])}, "
                f"CV acc={_fmt_delta(delta['intervention_pure']['cv_acc'])}, "
                f"CV pass={_fmt_delta(delta['intervention_pure']['cv_pass'])}"
            )

        if "exp_chunked" in arm_metric and "control_oc_exact" in arm_metric:
            delta["vs_oc_control"] = {
                "oc_acc": arm_metric["exp_chunked"]["oc_acc"] - arm_metric["control_oc_exact"]["oc_acc"],
                "oc_pass": arm_metric["exp_chunked"]["oc_pass"] - arm_metric["control_oc_exact"]["oc_pass"],
                "cv_acc": arm_metric["exp_chunked"]["cv_acc"] - arm_metric["control_oc_exact"]["cv_acc"],
                "cv_pass": arm_metric["exp_chunked"]["cv_pass"] - arm_metric["control_oc_exact"]["cv_pass"],
            }
            lines.append(
                f"- Delta vs OC-exact-control: OC acc={_fmt_delta(delta['vs_oc_control']['oc_acc'])}, "
                f"OC pass={_fmt_delta(delta['vs_oc_control']['oc_pass'])}, "
                f"CV acc={_fmt_delta(delta['vs_oc_control']['cv_acc'])}, "
                f"CV pass={_fmt_delta(delta['vs_oc_control']['cv_pass'])}"
            )

        verdict = "可信"
        control_oc = arm_metric.get("control_oc_exact")
        exp_chunked = arm_metric.get("exp_chunked")
        if exp_chunked is None:
            verdict = "不可用"
        elif not math.isnan(exp_chunked.get("interventions_nonzero", float("nan"))) and exp_chunked[
            "interventions_nonzero"
        ] <= 0.0:
            verdict = "不可用"
        elif control_oc is not None:
            d_acc = abs(control_oc["oc_acc"] - base_acc) if math.isfinite(control_oc["oc_acc"]) and math.isfinite(base_acc) else float("nan")
            d_pass = abs(control_oc["oc_pass"] - base_pass) if math.isfinite(control_oc["oc_pass"]) and math.isfinite(base_pass) else float("nan")
            if (math.isfinite(d_acc) and d_acc > 1.0) or (math.isfinite(d_pass) and d_pass > 1.0):
                verdict = "需复跑"
        dataset_out["delta"] = delta
        dataset_out["verdict"] = verdict
        lines.append(f"- 结论标签: **{verdict}**")
        lines.append("")

        summary["datasets"][dataset] = dataset_out

    md = "\n".join(lines).rstrip() + "\n"
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
