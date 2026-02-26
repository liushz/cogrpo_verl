#!/usr/bin/env python3
"""
CompassVerifier-style evaluation for eval_co_grpo_with_verifier_v2 outputs.

Scoring policy matches CompassVerifier:
- Only judge the final answer (tail-only), ignoring reasoning.
- Tail extraction is done by verl.utils.reward_score.compassverifier.compute_score().

This script adds cv_score fields under record["control"] / record["exp"].
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


# Repeat-aware aggregation helpers.
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
        if qid is not None:
            return qid
    return _maybe_int(rec.get("_orig_line_idx"))


def _avg(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _macro_avg(per: Dict[int, list[float]]) -> float:
    return _avg([_avg(v) for v in per.values()]) if per else 0.0


# Allow running this script without installing the repo as a package.
# NOTE: We avoid importing the `verl` package directly because its `__init__`
# may require heavy deps (e.g., ray) that are not present in local eval envs.
_REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPRO_ROOT))


def _load_compute_score():
    p = _REPRO_ROOT / "verl" / "utils" / "reward_score" / "compassverifier.py"
    if not p.exists():
        raise FileNotFoundError(f"Missing CompassVerifier module file: {p}")
    spec = importlib.util.spec_from_file_location("_cv_module", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[attr-defined]
    compute_score = getattr(m, "compute_score", None)
    if compute_score is None:
        raise AttributeError(f"Module {p} does not export compute_score")
    return compute_score


def _count_nonempty_lines(p: Path) -> int:
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _strip_injected_hints(text: str, interventions: Any) -> str:
    if not text:
        return ""

    out = str(text)

    # Preferred: remove exactly what was injected, as recorded by the pipeline.
    if isinstance(interventions, list):
        for itv in interventions:
            if not isinstance(itv, dict):
                continue
            hint_text = itv.get("hint_text")
            if hint_text:
                out = out.replace(str(hint_text), "\n\n")
                continue

            # Backward compatibility for older records.
            hint = itv.get("hint")
            if hint:
                hint = str(hint).strip()
                out = out.replace(f"\n\n[Guide]: {hint}\n\n", "\n\n")
                out = out.replace(f"\n\n{hint}\n\n", "\n\n")

    # Fallback: strip old "[Guide]:" blocks even when interventions are missing.
    out = re.sub(r"\n\n\[Guide\]:.*?(\n\n|$)", "\n\n", out, flags=re.DOTALL)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _get_side_text(rec: Dict[str, Any], side_key: str) -> str:
    side = rec.get(side_key)
    if not isinstance(side, dict):
        return ""
    return _strip_injected_hints(str(side.get("response_text") or ""), side.get("interventions"))


def _extract_question_and_gold(origin_info: Dict[str, Any], prompt_key: str, ground_truth_key: str) -> Tuple[Optional[str], Optional[str]]:
    q = origin_info.get(prompt_key) or origin_info.get("question")
    gt = origin_info.get(ground_truth_key)

    # Common alt layouts (parquet rows / training rows).
    if gt is None:
        rm = origin_info.get("reward_model") or {}
        if isinstance(rm, dict):
            gt = rm.get("ground_truth") or rm.get("answer")
    if q is None:
        extra = origin_info.get("extra_info") or {}
        if isinstance(extra, dict):
            q = extra.get("question") or extra.get(prompt_key)
    if gt is None:
        for k in ("ground_truth", "answer", "final_answer", "gold", "label"):
            if k in origin_info and origin_info.get(k) is not None:
                gt = origin_info.get(k)
                break

    return (str(q) if q is not None else None), (str(gt) if gt is not None else None)


def _parse_eval_urls(eval_urls: str) -> list[str]:
    urls = [u.strip() for u in (eval_urls or "").split(",") if u.strip()]
    if not urls:
        raise ValueError("Empty --eval-urls; provide comma-separated http(s)://.../v1 endpoints.")
    bad = [u for u in urls if not (u.startswith("http://") or u.startswith("https://"))]
    if bad:
        raise ValueError(f"--eval-urls must include scheme http(s):// ; bad={bad}")
    return urls


def _load_clients(urls: list[str]):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        OpenAI = None  # type: ignore[assignment]
        openai_import_error = e
    else:
        openai_import_error = None

    api_key = os.environ.get("OPENAI_API_KEY", "NONE")

    if OpenAI is not None:
        return [OpenAI(base_url=u, api_key=api_key) for u in urls]

    # Fallback: minimal OpenAI-compatible client using stdlib urllib.
    # This matches the subset used by verl.utils.reward_score.compassverifier.compute_score().
    import json as _json
    import urllib.request
    from types import SimpleNamespace
    from urllib.error import HTTPError, URLError

    class _MiniChatCompletions:
        def __init__(self, base_url: str, api_key: str):
            self._base_url = base_url.rstrip("/")
            self._api_key = api_key
            # Force direct connections to internal eval services even when the environment
            # exports HTTP(S)_PROXY. These model URLs are typically cluster-internal and
            # should not go through a corporate proxy (which often returns HTML 503).
            self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def create(self, *, model: str, messages, temperature: float = 0.0, **kwargs):
            url = f"{self._base_url}/chat/completions"
            payload = {"model": model, "messages": messages, "temperature": temperature}
            # Allow extra fields (e.g., max_tokens) if caller passes them.
            payload.update({k: v for k, v in kwargs.items() if v is not None})
            data = _json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url=url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {self._api_key}")
            try:
                with self._opener.open(req, timeout=120) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {e.code} from {url}: {body[:400]}") from e
            except URLError as e:
                raise RuntimeError(f"Network error calling {url}: {e}") from e

            obj = _json.loads(raw)
            # Convert dict payload to attribute-like structure.
            try:
                content = obj["choices"][0]["message"]["content"]
            except Exception as e:
                raise RuntimeError(f"Unexpected response schema from {url}: keys={list(obj)[:20]}") from e
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class _MiniChat:
        def __init__(self, base_url: str, api_key: str):
            self.completions = _MiniChatCompletions(base_url, api_key)

    class MiniOpenAI:
        def __init__(self, base_url: str, api_key: str):
            # compute_score checks client._client.base_url
            self._client = SimpleNamespace(base_url=base_url)
            self.chat = _MiniChat(base_url, api_key)

    if openai_import_error is not None:
        print(
            f"[WARN] openai SDK not available ({openai_import_error}); using MiniOpenAI urllib fallback for CompassVerifier eval.",
            flush=True,
        )
    return [MiniOpenAI(base_url=u, api_key=api_key) for u in urls]


def _score_one(*, question: str, gold: str, response: str, compute_score, client) -> Tuple[Optional[float], Optional[str]]:
    try:
        s = float(compute_score(question, gold, response, [client]))
        return (s, None)
    except Exception as e:
        return (None, str(e))


def main() -> int:
    ap = argparse.ArgumentParser(description="CompassVerifier tail-only eval for CoGRPO eval jsonl outputs.")
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--prompt-key", default="question")
    ap.add_argument("--ground-truth-key", default="ground_truth")
    ap.add_argument("--mode", default="both", choices=["control", "exp", "both"])
    ap.add_argument("--eval-urls", required=True, help="Comma-separated CompassVerifier model URLs (http(s)://.../v1).")
    ap.add_argument("--workers", type=int, default=0, help="Concurrent workers (default: number of eval URLs).")
    ap.add_argument("--batch-size", type=int, default=64, help="In-flight scoring batch size (default: 64).")
    ap.add_argument("--progress", action="store_true", help="Show progress (tqdm if available, else periodic logs).")
    ap.add_argument("--log-every", type=int, default=20, help="Progress log frequency when tqdm is unavailable.")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = _count_nonempty_lines(in_path) if args.progress else 0
    urls = _parse_eval_urls(args.eval_urls)
    clients = _load_clients(urls)

    compute_score = _load_compute_score()

    n = 0
    scored_control = 0
    scored_exp = 0
    sum_control = 0.0
    sum_exp = 0.0
    per_q_control: Dict[int, list[float]] = {}
    per_q_exp: Dict[int, list[float]] = {}
    missing_gt = 0
    errors = 0

    pbar = None
    if args.progress and tqdm is not None and sys.stderr.isatty():
        pbar = tqdm(total=total, desc="cv_eval", unit="rec")

    with in_path.open("r", encoding="utf-8") as r, out_path.open("w", encoding="utf-8") as w:
        workers = int(args.workers) if int(args.workers) > 0 else len(clients)
        workers = max(1, workers)
        batch_size = max(1, int(args.batch_size))

        rr = count(0)

        def _flush_batch(batch: list[dict]) -> None:
            nonlocal n, scored_control, scored_exp, sum_control, sum_exp, missing_gt, errors
            if not batch:
                return

            def _choose_client():
                idx = next(rr) % len(clients)
                return clients[idx]

            # Attach origin_info normalization and detect missing gold early.
            to_score: list[dict] = []
            for rec in batch:
                n += 1
                origin = rec.get("origin_info") or {}
                if not isinstance(origin, dict):
                    origin = {"origin_info": origin}
                    rec["origin_info"] = origin
                question, gold = _extract_question_and_gold(origin, args.prompt_key, args.ground_truth_key)
                if not question or gold is None or gold == "None":
                    rec["_cv_skip"] = True
                    missing_gt += 1
                else:
                    rec["_cv_skip"] = False
                    rec["_cv_question"] = question
                    rec["_cv_gold"] = gold
                    to_score.append(rec)

            if workers <= 1 or not to_score:
                for rec in batch:
                    if not rec.get("_cv_skip"):
                        question = str(rec.pop("_cv_question"))
                        gold = str(rec.pop("_cv_gold"))
                        if args.mode in ("control", "both") and isinstance(rec.get("control"), dict):
                            resp = _get_side_text(rec, "control")
                            s, err = _score_one(
                                question=question,
                                gold=gold,
                                response=resp,
                                compute_score=compute_score,
                                client=_choose_client(),
                            )
                            rec["control"]["cv_score"] = s
                            if err is not None:
                                rec["control"]["cv_error"] = err
                                errors += 1
                            else:
                                scored_control += 1
                                sum_control += float(s or 0.0)
                                qid = _question_id(rec)
                                if qid is not None:
                                    per_q_control.setdefault(qid, []).append(float(s or 0.0))
                        if args.mode in ("exp", "both") and isinstance(rec.get("exp"), dict):
                            resp = _get_side_text(rec, "exp")
                            s, err = _score_one(
                                question=question,
                                gold=gold,
                                response=resp,
                                compute_score=compute_score,
                                client=_choose_client(),
                            )
                            rec["exp"]["cv_score"] = s
                            if err is not None:
                                rec["exp"]["cv_error"] = err
                                errors += 1
                            else:
                                scored_exp += 1
                                sum_exp += float(s or 0.0)
                                qid = _question_id(rec)
                                if qid is not None:
                                    per_q_exp.setdefault(qid, []).append(float(s or 0.0))

                    rec.pop("_cv_skip", None)
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if pbar is not None:
                        pbar.update(1)
                    elif args.progress:
                        every = max(1, int(args.log_every))
                        if (n % every) == 0:
                            print(
                                f"[cv_eval] {n}/{total or '?'} processed (missing_gt={missing_gt}, errors={errors})",
                                file=sys.stderr,
                                flush=True,
                            )
                return

            # Concurrent scoring: submit jobs per side, and distribute across all provided URLs.
            future_to_meta: dict = {}
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for rec in to_score:
                    question = str(rec["_cv_question"])
                    gold = str(rec["_cv_gold"])
                    if args.mode in ("control", "both") and isinstance(rec.get("control"), dict):
                        resp = _get_side_text(rec, "control")
                        fut = ex.submit(
                            _score_one,
                            question=question,
                            gold=gold,
                            response=resp,
                            compute_score=compute_score,
                            client=_choose_client(),
                        )
                        future_to_meta[fut] = (rec, "control")
                    if args.mode in ("exp", "both") and isinstance(rec.get("exp"), dict):
                        resp = _get_side_text(rec, "exp")
                        fut = ex.submit(
                            _score_one,
                            question=question,
                            gold=gold,
                            response=resp,
                            compute_score=compute_score,
                            client=_choose_client(),
                        )
                        future_to_meta[fut] = (rec, "exp")

                for fut in as_completed(future_to_meta):
                    rec, side = future_to_meta[fut]
                    s, err = fut.result()
                    if isinstance(rec.get(side), dict):
                        rec[side]["cv_score"] = s
                        if err is not None:
                            rec[side]["cv_error"] = err
                            errors += 1
                        else:
                            if side == "control":
                                scored_control += 1
                                sum_control += float(s or 0.0)
                                qid = _question_id(rec)
                                if qid is not None:
                                    per_q_control.setdefault(qid, []).append(float(s or 0.0))
                            elif side == "exp":
                                scored_exp += 1
                                sum_exp += float(s or 0.0)
                                qid = _question_id(rec)
                                if qid is not None:
                                    per_q_exp.setdefault(qid, []).append(float(s or 0.0))

            # Write out in original order.
            for rec in batch:
                rec.pop("_cv_skip", None)
                rec.pop("_cv_question", None)
                rec.pop("_cv_gold", None)
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if pbar is not None:
                    pbar.update(1)
                elif args.progress:
                    every = max(1, int(args.log_every))
                    if (n % every) == 0:
                        print(
                            f"[cv_eval] {n}/{total or '?'} processed (missing_gt={missing_gt}, errors={errors})",
                            file=sys.stderr,
                            flush=True,
                        )

        # Read + process in batches.
        batch: list[dict] = []
        for line in r:
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                # flush
                _flush_batch(batch)
                batch = []

        if batch:
            _flush_batch(batch)

    if pbar is not None:
        pbar.close()

    def _micro(sum_s: float, k: int) -> float:
        return float(sum_s / k) if k else 0.0

    control_micro = _micro(sum_control, scored_control)
    exp_micro = _micro(sum_exp, scored_exp)
    control_macro = _macro_avg(per_q_control)
    exp_macro = _macro_avg(per_q_exp)

    print(
        json.dumps(
            {
                "records": n,
                "missing_ground_truth": missing_gt,
                "errors": errors,
                "n_questions": len(set(per_q_control) | set(per_q_exp)),
                "control_scored": scored_control,
                # Backward compatible key (micro average across all records).
                "control_acc": control_micro,
                "control_micro_acc": control_micro,
                "control_macro_acc": control_macro,
                "exp_scored": scored_exp,
                # Backward compatible key (micro average across all records).
                "exp_acc": exp_micro,
                "exp_micro_acc": exp_micro,
                "exp_macro_acc": exp_macro,
                "delta_micro": exp_micro - control_micro,
                "delta_macro": exp_macro - control_macro,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
