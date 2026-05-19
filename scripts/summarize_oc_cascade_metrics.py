#!/usr/bin/env python3
"""
Summarize merged.jsonl with OpenCompass-style cascade evaluator metrics.

This script evaluates one side (`control` or `exp`) from merged.jsonl and
outputs:
  - accuracy (k runs average)
  - G-Pass@k_0.0

It uses OpenCompass evaluators (rule + optional LLM fallback) to keep metric
semantics aligned with OpenCompass result JSONs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


DEFAULT_OC_REPO_ROOT = "/mnt/shared-storage-user/opencompass-shared/qa-llm-cicd/opencompass-main2/opencompass"


def _iter_url_like_strings(obj: Any) -> List[str]:
    urls: List[str] = []
    if isinstance(obj, str):
        text = obj.strip()
        if text.startswith("http://") or text.startswith("https://"):
            urls.append(text)
        return urls
    if isinstance(obj, dict):
        for value in obj.values():
            urls.extend(_iter_url_like_strings(value))
        return urls
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            urls.extend(_iter_url_like_strings(value))
    return urls


def _collect_internal_no_proxy_hosts(*objs: Any) -> List[str]:
    hosts: List[str] = []
    seen = set()

    def _add(host: str) -> None:
        host = str(host or "").strip()
        if not host or host in seen:
            return
        seen.add(host)
        hosts.append(host)

    for obj in objs:
        for url in _iter_url_like_strings(obj):
            try:
                parsed = urlparse(url)
            except Exception:
                continue
            host = str(parsed.hostname or "").strip()
            if not host:
                continue
            _add(host)
            if host.endswith(".svc"):
                _add(".svc")
            parts = host.split(".")
            for i in range(1, len(parts)):
                suffix = "." + ".".join(parts[i:])
                if suffix.endswith(".svc") or suffix.endswith(".cluster.local") or suffix.endswith(".local"):
                    _add(suffix)
    return hosts


def _extend_no_proxy(hosts: List[str]) -> List[str]:
    if not hosts:
        return []

    existing_raw = (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").strip()
    existing = [item.strip() for item in existing_raw.split(",") if item.strip()]
    merged: List[str] = []
    seen = set()
    for item in existing + list(hosts):
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    merged_value = ",".join(merged)
    os.environ["NO_PROXY"] = merged_value
    os.environ["no_proxy"] = merged_value
    return merged


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    return os.access(str(path), os.W_OK)


def _is_readable_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and os.access(str(path), os.R_OK)


def _resolve_runtime_cache_dir(env_name: str, fallback: Path) -> Path:
    cur = (os.environ.get(env_name) or "").strip()
    candidate = Path(cur).expanduser() if cur else fallback
    if not _is_writable_dir(candidate):
        candidate = fallback
        if not _is_writable_dir(candidate):
            raise RuntimeError(f"Cannot create writable cache dir for {env_name}: {candidate}")
    os.environ[env_name] = str(candidate)
    return candidate


def _setup_runtime_cache_env() -> Dict[str, str]:
    root_env = (os.environ.get("OC_RUNTIME_CACHE_ROOT") or "").strip()
    if root_env:
        root = Path(root_env).expanduser()
    else:
        root = Path("/tmp") / f"oc_runtime_cache_{os.getuid()}"
    if not _is_writable_dir(root):
        root = Path.cwd() / ".oc_runtime_cache"
    if not _is_writable_dir(root):
        raise RuntimeError(f"Cannot prepare OC runtime cache root: {root}")
    os.environ["OC_RUNTIME_CACHE_ROOT"] = str(root)

    resolved = {
        "XDG_CACHE_HOME": _resolve_runtime_cache_dir("XDG_CACHE_HOME", root / "xdg_cache"),
        "HF_HOME": _resolve_runtime_cache_dir("HF_HOME", root / "hf_home"),
        "HF_MODULES_CACHE": _resolve_runtime_cache_dir("HF_MODULES_CACHE", root / "hf_modules"),
        "EVALUATE_CACHE": _resolve_runtime_cache_dir("EVALUATE_CACHE", root / "evaluate"),
        "HF_DATASETS_CACHE": _resolve_runtime_cache_dir("HF_DATASETS_CACHE", root / "datasets"),
    }
    # HF_HUB_CACHE can be readonly shared cache (good for offline tokenizer/model lookup).
    hf_home = resolved["HF_HOME"]
    shared_hub = Path("/mnt/shared-storage-user/large-model-center-share-weights/hf_hub")
    hf_hub_env = (os.environ.get("HF_HUB_CACHE") or "").strip()
    candidates = []
    if hf_hub_env:
        candidates.append(Path(hf_hub_env).expanduser())
    candidates.append(shared_hub)
    candidates.append(hf_home / "hub")

    chosen_hub: Optional[Path] = None
    for c in candidates:
        if _is_readable_dir(c):
            chosen_hub = c
            break
    if chosen_hub is None:
        chosen_hub = _resolve_runtime_cache_dir("HF_HUB_CACHE", hf_home / "hub")
    else:
        os.environ["HF_HUB_CACHE"] = str(chosen_hub)

    os.environ.setdefault("TRANSFORMERS_CACHE", str(chosen_hub))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")

    tiktoken_cache_env = (os.environ.get("TIKTOKEN_CACHE_DIR") or "").strip()
    if tiktoken_cache_env:
        tiktoken_cache = Path(tiktoken_cache_env).expanduser()
    else:
        shared_tiktoken = Path("/mnt/shared-storage-user/auto-eval-pipeline/opencompass/llmeval/share_tiktoken")
        if _is_readable_dir(shared_tiktoken):
            tiktoken_cache = shared_tiktoken
        else:
            tiktoken_cache = root / "tiktoken"
            _is_writable_dir(tiktoken_cache)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache)
    return {k: str(v) for k, v in resolved.items()}


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


def _extract_question_gold(rec: Dict[str, Any], prompt_key: str, answer_key: str) -> Tuple[Optional[str], Optional[str]]:
    origin = rec.get("origin_info") or {}
    if not isinstance(origin, dict):
        return (None, None)
    question = origin.get(prompt_key) or origin.get("question")
    answer = origin.get(answer_key)
    if answer is None:
        answer = origin.get("answer") or origin.get("gold") or origin.get("ground_truth")
    if isinstance(question, str):
        question = question.strip()
    if isinstance(answer, str):
        answer = answer.strip()
    return (str(question) if question is not None else None, str(answer) if answer is not None else None)


_GPQA_OPT_RE = re.compile(
    r"(?s)(?P<q>.*?)\nA\)\s*(?P<A>.*?)\nB\)\s*(?P<B>.*?)\nC\)\s*(?P<C>.*?)\nD\)\s*(?P<D>.*)"
)


def _parse_gpqa_options(question: str) -> Dict[str, str]:
    out = {"question": question, "A": "", "B": "", "C": "", "D": ""}
    q = (question or "").strip()
    if not q:
        return out
    m = _GPQA_OPT_RE.match(q)
    if not m:
        return out
    out["question"] = str(m.group("q") or "").strip()
    out["A"] = str(m.group("A") or "").strip()
    out["B"] = str(m.group("B") or "").strip()
    out["C"] = str(m.group("C") or "").strip()
    out["D"] = str(m.group("D") or "").strip()
    return out


def _is_usable_oc_repo(root: Path) -> bool:
    return root.is_dir() and (root / "configs").is_dir() and (root / "__init__.py").is_file()


def _load_opencompass_repo(oc_repo_root: str) -> Path:
    candidates: List[Path] = []
    if oc_repo_root:
        candidates.append(Path(oc_repo_root).expanduser())
    env_root = (os.environ.get("OC_REPO_ROOT") or "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            Path(DEFAULT_OC_REPO_ROOT),
            Path("/mnt/shared-storage-user/auto-eval-pipeline/opencompass@f1e50d4/opencompass"),
            Path("/mnt/shared-storage-user/auto-eval-pipeline/opencompass@f1e50d4.bak20260226-new/opencompass"),
            Path("/mnt/shared-storage-user/opencompass-shared/liushudong/opencompass/opencompass"),
        ]
    )

    checked: List[str] = []
    for cand in candidates:
        root = cand.resolve()
        checked.append(str(root))
        if _is_usable_oc_repo(root):
            parent = str(root.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return root

    raise FileNotFoundError(
        "Missing usable OpenCompass repo. Checked: " + ", ".join(dict.fromkeys(checked))
    )


def _find_oc_eval_config(oc_root: str, oc_model_abbr: str) -> Optional[Path]:
    cfg_dir = Path(oc_root).expanduser().resolve() / "configs"
    if not cfg_dir.exists():
        return None
    matches: List[Path] = []
    for p in cfg_dir.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        pattern1 = f"abbr='{oc_model_abbr}'"
        pattern2 = f'abbr="{oc_model_abbr}"'
        if pattern1 in txt or pattern2 in txt:
            matches.append(p)
    if not matches:
        return None
    matches.sort(key=lambda x: x.stat().st_mtime)
    return matches[-1]


def _load_judge_cfg(oc_eval_config: Path) -> Optional[Dict[str, Any]]:
    try:
        from mmengine import Config  # type: ignore
    except Exception:
        from mmengine.config import Config  # type: ignore

    cfg = Config.fromfile(str(oc_eval_config))
    value = cfg.get("obj_llm_judge_cfg", None)
    if value is None:
        return None
    return dict(value)


def _load_dataset_evaluator_cfg(oc_eval_config: Path, dataset_name: str) -> Optional[Dict[str, Any]]:
    try:
        from mmengine import Config  # type: ignore
    except Exception:
        from mmengine.config import Config  # type: ignore

    cfg = Config.fromfile(str(oc_eval_config))
    candidates: List[Any] = []
    ds_list = cfg.get("datasets", None)
    if isinstance(ds_list, (list, tuple)):
        candidates.extend(list(ds_list))
    for k in ("dataset", "d"):
        v = cfg.get(k, None)
        if isinstance(v, dict):
            candidates.append(v)

    for item in candidates:
        if not isinstance(item, dict):
            continue
        if str(item.get("abbr", "")) != str(dataset_name):
            continue
        eval_cfg = item.get("eval_cfg", None)
        if not isinstance(eval_cfg, dict):
            continue
        evaluator_cfg = eval_cfg.get("evaluator", None)
        if isinstance(evaluator_cfg, dict):
            return copy.deepcopy(dict(evaluator_cfg))
    return None


def _load_model_pred_postprocessor_cfg(oc_eval_config: Path) -> Optional[Dict[str, Any]]:
    try:
        from mmengine import Config  # type: ignore
    except Exception:
        from mmengine.config import Config  # type: ignore

    cfg = Config.fromfile(str(oc_eval_config))
    base_model = cfg.get("base_model", None)
    if not isinstance(base_model, dict):
        return None
    pred_pp = base_model.get("pred_postprocessor", None)
    if isinstance(pred_pp, dict):
        return copy.deepcopy(dict(pred_pp))
    return None


def _extract_judge_cfg_from_evaluator_cfg(evaluator_cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(evaluator_cfg, dict):
        return None
    llm_eval = evaluator_cfg.get("llm_evaluator", None)
    if not isinstance(llm_eval, dict):
        return None
    judge_cfg = llm_eval.get("judge_cfg", None)
    if isinstance(judge_cfg, dict):
        return dict(judge_cfg)
    return None


def _apply_pred_postprocessor(
    predictions: List[str], pred_postprocessor_cfg: Optional[Dict[str, Any]]
) -> List[str]:
    if not predictions or not isinstance(pred_postprocessor_cfg, dict):
        return predictions
    pp_type = str(pred_postprocessor_cfg.get("type", "")).strip()
    if not pp_type:
        return predictions

    # Most OC configs in this project use extract_non_reasoning_content.
    if "extract_non_reasoning_content" in pp_type or "extract-non-reasoning-content" in pp_type:
        try:
            from opencompass.utils.text_postprocessors import extract_non_reasoning_content  # type: ignore
        except Exception:
            return predictions
        return [str(extract_non_reasoning_content(str(p or ""))) for p in predictions]
    return predictions


def _apply_judge_overrides(
    *,
    judge_cfg: Optional[Dict[str, Any]],
    dataset_evaluator_cfg: Optional[Dict[str, Any]],
    query_per_second: int,
    max_workers: int,
    batch_size: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if query_per_second <= 0 and max_workers <= 0 and batch_size <= 0:
        return judge_cfg, dataset_evaluator_cfg

    out_judge_cfg = dict(judge_cfg) if isinstance(judge_cfg, dict) else {}
    if query_per_second > 0:
        out_judge_cfg["query_per_second"] = int(query_per_second)
    if max_workers > 0:
        out_judge_cfg["max_workers"] = int(max_workers)
    if batch_size > 0:
        out_judge_cfg["batch_size"] = int(batch_size)

    out_eval_cfg = copy.deepcopy(dataset_evaluator_cfg) if isinstance(dataset_evaluator_cfg, dict) else None
    if isinstance(out_eval_cfg, dict):
        llm_eval = out_eval_cfg.get("llm_evaluator", None)
        if isinstance(llm_eval, dict):
            llm_judge_cfg = llm_eval.get("judge_cfg", None)
            if not isinstance(llm_judge_cfg, dict):
                llm_judge_cfg = {}
            else:
                llm_judge_cfg = dict(llm_judge_cfg)
            if query_per_second > 0:
                llm_judge_cfg["query_per_second"] = int(query_per_second)
            if max_workers > 0:
                llm_judge_cfg["max_workers"] = int(max_workers)
            if batch_size > 0:
                llm_judge_cfg["batch_size"] = int(batch_size)
            llm_eval["judge_cfg"] = llm_judge_cfg
            out_eval_cfg["llm_evaluator"] = llm_eval

    if not out_judge_cfg:
        out_judge_cfg = None
    return out_judge_cfg, out_eval_cfg


def _make_cascade_evaluator(
    *,
    dataset_name: str,
    judge_cfg: Optional[Dict[str, Any]],
    enable_llm_judge: bool,
    dataset_evaluator_cfg: Optional[Dict[str, Any]] = None,
) -> Any:
    if isinstance(dataset_evaluator_cfg, dict):
        try:
            from opencompass.registry import ICL_EVALUATORS  # type: ignore

            if enable_llm_judge:
                return ICL_EVALUATORS.build(copy.deepcopy(dataset_evaluator_cfg))

            evaluator_cfg = copy.deepcopy(dataset_evaluator_cfg)
            evaluator_type = str(evaluator_cfg.get("type", ""))
            if "CascadeEvaluator" in evaluator_type and isinstance(evaluator_cfg.get("rule_evaluator"), dict):
                return ICL_EVALUATORS.build(copy.deepcopy(evaluator_cfg["rule_evaluator"]))
            return ICL_EVALUATORS.build(evaluator_cfg)
        except Exception as e:
            print(
                f"[warn] failed to build evaluator from OC config ({e}); fallback to local evaluator template.",
                file=sys.stderr,
            )

    from opencompass.evaluator import CascadeEvaluator, GenericLLMEvaluator  # type: ignore
    from opencompass.openicl.icl_evaluator import AccEvaluator  # type: ignore
    from opencompass.openicl.icl_prompt_template import PromptTemplate  # type: ignore

    if dataset_name in ("aime2024", "aime2025"):
        from opencompass.evaluator import MATHVerifyEvaluator  # type: ignore
        if not enable_llm_judge:
            return MATHVerifyEvaluator()
        if dataset_name == "aime2024":
            from opencompass.datasets import Aime2024Dataset as _DummyDataset  # type: ignore
            dummy_dataset_cfg = dict(
                type=_DummyDataset,
                path="opencompass/aime2024",
                reader_cfg=dict(input_columns=["question"], output_column="answer"),
            )
        else:
            from opencompass.datasets import CustomDataset as _DummyDataset  # type: ignore
            dummy_dataset_cfg = dict(
                type=_DummyDataset,
                path="opencompass/aime2025",
                reader_cfg=dict(input_columns=["question"], output_column="answer"),
            )

        grader_template = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given.
2. Compare candidate answer and standard answer semantically and mathematically.
3. If the prediction is given with \\boxed{}, ignore \\boxed{} and judge consistency only.

Grade as:
A: CORRECT
B: INCORRECT
Return only A or B.

<Original Question Begin>:
{question}
<Original Question End>
<Gold Target Begin>:
{answer}
<Gold Target End>
<Predicted Answer Begin>:
{prediction}
<Predicted End>
""".strip()

        llm_evaluator_cfg = dict(
            type=GenericLLMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role="SYSTEM",
                            fallback_role="HUMAN",
                            prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                        )
                    ],
                    round=[dict(role="HUMAN", prompt=grader_template)],
                ),
            ),
            dataset_cfg=dummy_dataset_cfg,
            judge_cfg=dict(judge_cfg or {}),
            dict_postprocessor=dict(type="opencompass.datasets.generic_llmjudge_postprocess"),
        )
        return CascadeEvaluator(
            rule_evaluator=dict(type=MATHVerifyEvaluator),
            llm_evaluator=llm_evaluator_cfg,
            parallel=False,
        )

    if dataset_name == "GPQA_diamond":
        from opencompass.datasets import GPQADataset  # type: ignore
        if not enable_llm_judge:
            return AccEvaluator(pred_postprocessor=dict(type="match_answer_pattern", answer_pattern=r"(?i)ANSWER\s*:\s*([A-D])"))

        grader_template = """
Please as a grading expert, judge whether candidate answer is consistent with the standard answer.
Return only A (CORRECT) or B (INCORRECT).

<Original Question Begin>:
{question}
A) {A}
B) {B}
C) {C}
D) {D}
<Original Question End>
<Gold Target Begin>:
{answer}
<Gold Target End>
<Predicted Answer Begin>:
{prediction}
<Predicted End>
""".strip()

        llm_evaluator_cfg = dict(
            type=GenericLLMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role="SYSTEM",
                            fallback_role="HUMAN",
                            prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                        )
                    ],
                    round=[dict(role="HUMAN", prompt=grader_template)],
                ),
            ),
            dataset_cfg=dict(
                type=GPQADataset,
                path="./data/gpqa/",
                name="gpqa_diamond.csv",
                reader_cfg=dict(input_columns=["question", "A", "B", "C", "D"], output_column="answer"),
            ),
            judge_cfg=dict(judge_cfg or {}),
            dict_postprocessor=dict(type="opencompass.datasets.generic_llmjudge_postprocess"),
        )
        return CascadeEvaluator(
            rule_evaluator=dict(
                type=AccEvaluator,
                pred_postprocessor=dict(type="match_answer_pattern", answer_pattern=r"(?i)ANSWER\s*:\s*([A-D])"),
            ),
            llm_evaluator=llm_evaluator_cfg,
            parallel=False,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _extract_correct_from_detail(detail: Dict[str, Any]) -> bool:
    if not isinstance(detail, dict):
        return False
    if detail.get("correct") is not None:
        return bool(detail.get("correct"))
    if detail.get("is_correct") is not None:
        return bool(detail.get("is_correct"))
    cascade_correct = detail.get("cascade_correct")
    if cascade_correct is not None:
        return bool(cascade_correct)

    rule_eval = detail.get("rule_evaluation")
    rule_correct = False
    if isinstance(rule_eval, dict):
        rule_correct = bool(rule_eval.get("correct", False))

    llm_eval = detail.get("llm_evaluation")
    llm_correct = False
    if isinstance(llm_eval, dict):
        llm_correct = bool(llm_eval.get("llm_correct", False))
        if not llm_correct:
            prediction = str(llm_eval.get("prediction") or "").strip().upper()
            if prediction == "A" or prediction.startswith("CORRECT"):
                llm_correct = True

    return bool(rule_correct or llm_correct)


def _is_llm_judge_runtime_deps_error(err: Exception) -> bool:
    msg = str(err)
    keys = (
        "Could not automatically map",
        "to a tokeniser",
        "LocalEntryNotFoundError",
        "Cannot find the requested files in the disk cache",
        "outgoing traffic has been disabled",
        "tiktoken.get_encoding",
    )
    return any(k in msg for k in keys)


def _infer_repeat(counts: List[int], fallback: int) -> int:
    if not counts:
        return int(fallback)
    cnt = Counter(counts)
    repeat, _ = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))
    return int(repeat)


def _load_side_rows(in_jsonl: Path, side: str, prompt_key: str, answer_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with in_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            side_obj = rec.get(side)
            if not isinstance(side_obj, dict):
                continue
            qid = _question_id(rec)
            rid = _repeat_id(rec)
            if qid is None or rid is None:
                continue
            question, gold = _extract_question_gold(rec, prompt_key=prompt_key, answer_key=answer_key)
            if not question or gold is None:
                continue
            rows.append(
                {
                    "qid": int(qid),
                    "rid": int(rid),
                    "question": str(question),
                    "gold": str(gold),
                    "prediction": str(side_obj.get("response_text") or ""),
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize merged.jsonl with OpenCompass cascade metrics.")
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--dataset", required=True, choices=["GPQA_diamond", "aime2024", "aime2025"])
    ap.add_argument("--side", default="control", choices=["control", "exp"])
    ap.add_argument("--prompt-key", default="question")
    ap.add_argument("--answer-key", default="answer")
    ap.add_argument("--repeat", type=int, default=0, help="Expected repeats per question (0=infer from records).")
    ap.add_argument("--k", type=int, default=0, help="Pass@k value (0=dataset default).")
    ap.add_argument("--llm-judge", default="auto", choices=["auto", "on", "off"])
    ap.add_argument(
        "--oc-repo-root",
        default=DEFAULT_OC_REPO_ROOT,
        help="OpenCompass repo root directory.",
    )
    ap.add_argument("--oc-root", default="", help="OC report root containing configs/ (optional).")
    ap.add_argument("--oc-model-abbr", default="", help="Model abbr used to locate OC config for judge cfg.")
    ap.add_argument("--oc-eval-config", default="", help="Explicit OC config .py path for obj_llm_judge_cfg.")
    ap.add_argument("--out-json", default="", help="Optional output json path.")
    ap.add_argument(
        "--llm-judge-out-dir",
        default="",
        help="Optional writable directory for LLM-judge detail files (overrides evaluator default).",
    )
    ap.add_argument("--judge-query-per-second", type=int, default=0, help="Override judge_cfg.query_per_second (>0).")
    ap.add_argument("--judge-max-workers", type=int, default=0, help="Override judge_cfg.max_workers (>0).")
    ap.add_argument("--judge-batch-size", type=int, default=0, help="Override judge_cfg.batch_size (>0).")
    ap.add_argument(
        "--replica-workers",
        type=int,
        default=1,
        help="Reserved. Replica scoring currently runs sequentially for OC runtime compatibility.",
    )
    ap.add_argument(
        "--require-oc-judge-prompt",
        type=int,
        default=1,
        help="When llm judge is enabled, require evaluator prompt/template to come from OC eval config.",
    )
    args = ap.parse_args()
    if int(args.replica_workers) <= 0:
        raise SystemExit("--replica-workers must be >=1")
    if int(args.require_oc_judge_prompt) not in (0, 1):
        raise SystemExit("--require-oc-judge-prompt must be 0 or 1")

    cache_env = _setup_runtime_cache_env()
    print(
        f"[cache] OC_RUNTIME_CACHE_ROOT={os.environ.get('OC_RUNTIME_CACHE_ROOT')} "
        f"HF_MODULES_CACHE={cache_env.get('HF_MODULES_CACHE')}",
        file=sys.stderr,
    )

    in_path = Path(args.in_jsonl).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Missing --in-jsonl: {in_path}")

    oc_repo_root_resolved = _load_opencompass_repo(args.oc_repo_root)
    print(f"[oc] repo_root={oc_repo_root_resolved}", file=sys.stderr)
    try:
        from datasets import Dataset  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Missing python dependency 'datasets'. Run this script in the OpenCompass eval env, e.g. `conda activate oc`."
        ) from exc
    try:
        from opencompass.openicl.icl_evaluator import compute_g_pass_at_k  # type: ignore
    except Exception:
        from opencompass.openicl.icl_evaluator.icl_base_evaluator import compute_g_pass_at_k  # type: ignore

    side_rows = _load_side_rows(in_path, side=args.side, prompt_key=args.prompt_key, answer_key=args.answer_key)
    if not side_rows:
        raise SystemExit(f"No valid side rows found for side={args.side} in {in_path}")

    qids = sorted({int(r["qid"]) for r in side_rows})
    rid_counts = Counter(int(r["qid"]) for r in side_rows)
    inferred_repeat = _infer_repeat(list(rid_counts.values()), fallback=1)
    n_repeat = int(args.repeat) if int(args.repeat) > 0 else inferred_repeat
    dataset_k_default = 8 if args.dataset == "GPQA_diamond" else 32
    k_value = int(args.k) if int(args.k) > 0 else int(dataset_k_default)

    by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in side_rows:
        by_key[(int(r["qid"]), int(r["rid"]))] = r

    missing_predictions = 0
    predictions_all: List[str] = []
    references_all: List[str] = []
    test_rows: Dict[str, List[Any]] = {
        "idx": [],
        "subdivision": [],
        "question": [],
        "answer": [],
    }
    if args.dataset == "GPQA_diamond":
        test_rows["A"] = []
        test_rows["B"] = []
        test_rows["C"] = []
        test_rows["D"] = []

    q_meta: Dict[int, Dict[str, Any]] = {}
    for qid in qids:
        first: Optional[Dict[str, Any]] = None
        for rid in range(0, n_repeat):
            first = by_key.get((qid, rid))
            if first is not None:
                break
        if first is None:
            continue
        q_meta[qid] = first

    qids = [qid for qid in qids if qid in q_meta]
    if not qids:
        raise SystemExit("No valid questions with metadata found.")

    for rid in range(0, n_repeat):
        for qid in qids:
            item = by_key.get((qid, rid))
            if item is None:
                item = dict(q_meta[qid])
                item["prediction"] = ""
                missing_predictions += 1
            predictions_all.append(str(item["prediction"]))
            references_all.append(str(item["gold"]))

            test_rows["idx"].append(int(qid))
            test_rows["subdivision"].append(str(args.dataset))
            test_rows["question"].append(str(item["question"]))
            test_rows["answer"].append(str(item["gold"]))
            if args.dataset == "GPQA_diamond":
                opts = _parse_gpqa_options(str(item["question"]))
                test_rows["question"][-1] = str(opts["question"])
                test_rows["A"].append(str(opts["A"]))
                test_rows["B"].append(str(opts["B"]))
                test_rows["C"].append(str(opts["C"]))
                test_rows["D"].append(str(opts["D"]))

    test_set = Dataset.from_dict(test_rows)

    oc_eval_config: Optional[Path] = None
    if args.oc_eval_config:
        oc_eval_config = Path(args.oc_eval_config).expanduser().resolve()
        if not oc_eval_config.exists():
            raise SystemExit(f"Missing --oc-eval-config: {oc_eval_config}")
    elif args.oc_root and args.oc_model_abbr:
        oc_eval_config = _find_oc_eval_config(args.oc_root, args.oc_model_abbr)

    dataset_evaluator_cfg: Optional[Dict[str, Any]] = None
    model_pred_postprocessor_cfg: Optional[Dict[str, Any]] = None
    if oc_eval_config is not None:
        try:
            dataset_evaluator_cfg = _load_dataset_evaluator_cfg(oc_eval_config, args.dataset)
        except Exception as e:
            print(
                f"[warn] failed to load dataset evaluator cfg from {oc_eval_config}: {e}",
                file=sys.stderr,
            )
            dataset_evaluator_cfg = None
        try:
            model_pred_postprocessor_cfg = _load_model_pred_postprocessor_cfg(oc_eval_config)
        except Exception as e:
            print(
                f"[warn] failed to load base_model pred_postprocessor from {oc_eval_config}: {e}",
                file=sys.stderr,
            )
            model_pred_postprocessor_cfg = None

    judge_cfg = None
    if oc_eval_config is not None:
        try:
            judge_cfg = _load_judge_cfg(oc_eval_config)
        except Exception as e:
            if args.llm_judge == "on":
                raise SystemExit(f"Failed to load judge cfg from {oc_eval_config}: {e}")
            print(
                f"[warn] failed to load judge cfg from {oc_eval_config}: {e}; fallback to rule-only.",
                file=sys.stderr,
            )
            judge_cfg = None
    if judge_cfg is None:
        judge_cfg = _extract_judge_cfg_from_evaluator_cfg(dataset_evaluator_cfg)

    judge_cfg, dataset_evaluator_cfg = _apply_judge_overrides(
        judge_cfg=judge_cfg,
        dataset_evaluator_cfg=dataset_evaluator_cfg,
        query_per_second=int(args.judge_query_per_second),
        max_workers=int(args.judge_max_workers),
        batch_size=int(args.judge_batch_size),
    )

    judge_no_proxy_hosts: List[str] = []
    if args.llm_judge != "off":
        judge_no_proxy_hosts = _collect_internal_no_proxy_hosts(judge_cfg, dataset_evaluator_cfg)
        if judge_no_proxy_hosts:
            _extend_no_proxy(judge_no_proxy_hosts)
            print(
                f"[env] extended NO_PROXY for llm-judge: {','.join(judge_no_proxy_hosts)}",
                file=sys.stderr,
            )

    enable_llm_judge = (args.llm_judge == "on") or (args.llm_judge == "auto" and judge_cfg is not None)
    if args.llm_judge == "off":
        enable_llm_judge = False
    if enable_llm_judge and judge_cfg is None:
        raise SystemExit("LLM judge requested but judge cfg is missing. Set --oc-eval-config or --oc-root/--oc-model-abbr.")
    if enable_llm_judge and int(args.require_oc_judge_prompt) == 1 and not isinstance(dataset_evaluator_cfg, dict):
        raise SystemExit(
            "LLM judge prompt alignment requires OC evaluator cfg, but none was loaded. "
            "Set --oc-eval-config explicitly (or ensure --oc-root/--oc-model-abbr resolves to a config containing this dataset)."
        )

    llm_judge_requested = bool(enable_llm_judge)
    llm_judge_runtime_fallback = False
    llm_judge_runtime_error: Optional[str] = None
    evaluator = _make_cascade_evaluator(
        dataset_name=args.dataset,
        judge_cfg=judge_cfg,
        enable_llm_judge=bool(enable_llm_judge),
        dataset_evaluator_cfg=dataset_evaluator_cfg,
    )
    if enable_llm_judge:
        forced_out_dir = (args.llm_judge_out_dir or os.environ.get("OC_CASCADE_EVAL_DIR", "")).strip()
        if forced_out_dir:
            target_out_dir = Path(forced_out_dir).expanduser()
            target_out_dir.mkdir(parents=True, exist_ok=True)
            evaluator._out_dir = str(target_out_dir)
        elif not getattr(evaluator, "_out_dir", None):
            tmp_root = Path(os.environ.get("TMPDIR", "/tmp")) / "oc_cascade_eval"
            tmp_root.mkdir(parents=True, exist_ok=True)
            evaluator._out_dir = str(tmp_root / f"{args.dataset}_{args.side}_{os.getpid()}")
        print(f"[llm_judge] out_dir={getattr(evaluator, '_out_dir', '')}", file=sys.stderr)
    llm_eval_base_out_dir = str(getattr(evaluator, "_out_dir", "")) if enable_llm_judge else ""

    # Score per replica for stable replica-level bookkeeping.
    all_correct: List[int] = []
    per_q_success: Dict[int, int] = {int(qid): 0 for qid in qids}
    llm_evaluated = 0
    llm_correct = 0

    q_count = len(qids)
    predictions_all = _apply_pred_postprocessor(predictions_all, model_pred_postprocessor_cfg)
    def _score_one_replica(rid: int) -> Dict[str, Any]:
        local_enable_llm = bool(enable_llm_judge)
        local_fallback = False
        local_error: Optional[str] = None
        local_llm_evaluated = 0
        local_llm_correct = 0

        local_evaluator = _make_cascade_evaluator(
            dataset_name=args.dataset,
            judge_cfg=judge_cfg,
            enable_llm_judge=bool(local_enable_llm),
            dataset_evaluator_cfg=dataset_evaluator_cfg,
        )
        if local_enable_llm and llm_eval_base_out_dir:
            rep_out_dir = Path(f"{llm_eval_base_out_dir}_rep{rid}")
            rep_out_dir.mkdir(parents=True, exist_ok=True)
            local_evaluator._out_dir = str(rep_out_dir)

        start = rid * q_count
        end = (rid + 1) * q_count
        preds = predictions_all[start:end]
        refs = references_all[start:end]
        replica_set = test_set.select(range(start, end))
        try:
            result = local_evaluator.score(predictions=preds, references=refs, test_set=replica_set)
        except Exception as e:
            if local_enable_llm and _is_llm_judge_runtime_deps_error(e):
                local_fallback = True
                local_error = str(e)
                print(
                    f"[warn] llm-judge runtime failed on replica={rid} ({e}); fallback to rule-only evaluator.",
                    file=sys.stderr,
                )
                local_enable_llm = False
                local_evaluator = _make_cascade_evaluator(
                    dataset_name=args.dataset,
                    judge_cfg=None,
                    enable_llm_judge=False,
                    dataset_evaluator_cfg=dataset_evaluator_cfg,
                )
                result = local_evaluator.score(predictions=preds, references=refs, test_set=replica_set)
            else:
                raise

        details = result.get("details") or []
        if isinstance(details, dict):
            details = list(details.values())

        stats = result.get("cascade_stats")
        if isinstance(stats, dict):
            local_llm_evaluated = int(stats.get("llm_evaluated", 0) or 0)
            local_llm_correct = int(stats.get("llm_correct", 0) or 0)

        replica_flags: List[int] = []
        if len(details) == q_count:
            for i, detail in enumerate(details):
                _ = qids[i]
                ok = 1 if _extract_correct_from_detail(detail) else 0
                replica_flags.append(ok)
        else:
            # Fallback path for evaluators that do not emit per-sample details
            # (e.g., rule-only AccEvaluator).
            for i in range(0, q_count):
                pred = str(preds[i] or "")
                ref = str(refs[i] or "").strip().upper()
                if args.dataset == "GPQA_diamond":
                    m = re.search(r"(?i)ANSWER\s*:\s*([A-D])", pred)
                    picked = (m.group(1) if m else pred.strip()).strip().upper()
                    ok = 1 if picked == ref else 0
                else:
                    ok = 1 if pred.strip() == refs[i].strip() else 0
                replica_flags.append(ok)

        return {
            "rid": int(rid),
            "flags": replica_flags,
            "llm_used": bool(local_enable_llm),
            "llm_evaluated": int(local_llm_evaluated),
            "llm_correct": int(local_llm_correct),
            "fallback": bool(local_fallback),
            "error": local_error,
        }

    replica_workers = max(1, int(args.replica_workers))
    if replica_workers > 1:
        print(
            "[warn] --replica-workers>1 requested, but OC LLM judge is not thread-safe in this runtime; forcing sequential scoring.",
            file=sys.stderr,
        )
        replica_workers = 1
    replica_results: List[Dict[str, Any]] = []
    for rid in range(0, n_repeat):
        replica_results.append(_score_one_replica(rid))

    replica_results.sort(key=lambda x: int(x.get("rid", 0)))
    used_llm_any = False
    for rr in replica_results:
        flags = rr.get("flags") or []
        if len(flags) != q_count:
            raise SystemExit(
                f"Replica scoring length mismatch: rid={rr.get('rid')} flags={len(flags)} q_count={q_count}"
            )
        for i, ok in enumerate(flags):
            qid = int(qids[i])
            ok_i = int(ok)
            all_correct.append(ok_i)
            per_q_success[qid] += ok_i
        llm_evaluated += int(rr.get("llm_evaluated", 0) or 0)
        llm_correct += int(rr.get("llm_correct", 0) or 0)
        used_llm_any = used_llm_any or bool(rr.get("llm_used", False))
        if bool(rr.get("fallback", False)):
            llm_judge_runtime_fallback = True
            if llm_judge_runtime_error is None:
                llm_judge_runtime_error = str(rr.get("error") or "")

    enable_llm_judge = bool(used_llm_any)

    accuracy = float(100.0 * sum(all_correct) / len(all_correct)) if all_correct else 0.0
    g_pass = 0.0
    if per_q_success:
        g_values = [
            float(compute_g_pass_at_k(n=n_repeat, c=int(c), k=k_value, t=0.0)) for c in per_q_success.values()
        ]
        g_pass = float(100.0 * sum(g_values) / len(g_values))

    out = {
        "dataset": str(args.dataset),
        "side": str(args.side),
        "records": int(len(side_rows)),
        "n_questions": int(len(qids)),
        "n_repeats": int(n_repeat),
        "k": int(k_value),
        f"accuracy ({k_value} runs average)": float(accuracy),
        f"G-Pass@{k_value}_0.0": float(g_pass),
        "llm_judge_requested": bool(llm_judge_requested),
        "llm_judge_enabled": bool(enable_llm_judge),
        "llm_judge_out_dir": llm_eval_base_out_dir,
        "llm_judge_runtime_fallback": bool(llm_judge_runtime_fallback),
        "llm_judge_runtime_error": llm_judge_runtime_error,
        "judge_no_proxy_hosts": judge_no_proxy_hosts,
        "llm_evaluated": int(llm_evaluated),
        "llm_correct": int(llm_correct),
        "require_oc_judge_prompt": int(args.require_oc_judge_prompt),
        "oc_evaluator_cfg_loaded": bool(isinstance(dataset_evaluator_cfg, dict)),
        "missing_predictions_filled": int(missing_predictions),
        "oc_eval_config": str(oc_eval_config) if oc_eval_config is not None else None,
        "replica_workers": int(max(1, int(args.replica_workers))),
        "judge_query_per_second_override": int(args.judge_query_per_second),
        "judge_max_workers_override": int(args.judge_max_workers),
        "judge_batch_size_override": int(args.judge_batch_size),
        "model_pred_postprocessor_type": (
            str(model_pred_postprocessor_cfg.get("type"))
            if isinstance(model_pred_postprocessor_cfg, dict)
            else None
        ),
    }

    raw = json.dumps(out, ensure_ascii=False)
    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(raw + "\n", encoding="utf-8")
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
