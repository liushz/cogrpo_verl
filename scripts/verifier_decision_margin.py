#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch


def _load_hint_injection_module():
    """
    Load the canonical verifier prompt helpers without importing the full `verl` package.

    Rationale: importing `verl` may require heavy deps (e.g., ray) which are not always
    installed in lightweight offline-eval environments.
    """
    import importlib.util

    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    p = repo_root / "verl" / "workers" / "rollout" / "vllm_rollout" / "verifier_hint_injection.py"
    if not p.exists():
        raise FileNotFoundError(f"verifier_hint_injection.py not found: {p}")

    spec = importlib.util.spec_from_file_location("_verifier_hint_injection", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[attr-defined]
    return m


def _safe_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _find_last_subsequence(tokens: Sequence[int], subseq: Sequence[int]) -> Optional[int]:
    if not tokens or not subseq or len(subseq) > len(tokens):
        return None
    for start in range(len(tokens) - len(subseq), -1, -1):
        if list(tokens[start : start + len(subseq)]) == list(subseq):
            return int(start)
    return None


def _parse_decision_tag(text: str) -> Optional[str]:
    if not text:
        return None
    go_re = re.compile(r"^\s*<go>\s*$", re.IGNORECASE)
    wait_re = re.compile(r"^\s*<wait>\b", re.IGNORECASE)
    tag = None
    for ln in text.splitlines():
        if go_re.match(ln):
            tag = "GO"
        elif wait_re.match(ln):
            tag = "WAIT"
    return tag


def _has_malformed_final_like(text: str) -> bool:
    low = (text or "").lower()
    return ("final answer" in low) or ("\\boxed" in (text or ""))


def _log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x - torch.logsumexp(x, dim=dim, keepdim=True)


def _sequence_logprob(
    model: torch.nn.Module,
    context_ids: List[int],
    cont_ids: List[int],
    device: torch.device,
    max_context_len: int = 0,
) -> float:
    if max_context_len and max_context_len > 0 and len(context_ids) > max_context_len:
        context_ids = context_ids[-max_context_len:]
    ids = torch.tensor([context_ids + cont_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids)
        logits = out.logits  # (1, seq_len, vocab)
    logprobs = _log_softmax(logits, dim=-1)
    ctx = len(context_ids)
    total = 0.0
    for i, tok in enumerate(cont_ids):
        pos = ctx + i - 1
        if pos < 0:
            continue
        total += float(logprobs[0, pos, int(tok)].item())
    return float(total)


@dataclass(frozen=True)
class DumpSample:
    sample_idx: int
    question: str
    student_response: str


def _iter_dump_samples(paths: List[Path], max_samples: int = 0) -> List[DumpSample]:
    out: List[DumpSample] = []
    for p in paths:
        obj = json.loads(p.read_text())
        samples = obj.get("samples") or []
        for s in samples:
            idx = int(s.get("sample_idx", -1))
            q = str(s.get("question") or "").strip()
            if not q:
                # Fallback for old dumps.
                q = str(s.get("prompt") or "").strip()
            resp = str(
                s.get("student_response_policy")
                or s.get("student_response_full")
                or s.get("response")
                or s.get("full_response")
                or ""
            )
            out.append(DumpSample(sample_idx=idx, question=q, student_response=resp))
            if max_samples and len(out) >= max_samples:
                return out
    return out


def _collect_dump_files(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for s in inputs:
        p = Path(s).expanduser()
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("**/batch_*.json")))
        else:
            # Allow glob-like inputs.
            files.extend(sorted(Path(".").glob(s)))
    uniq = []
    seen = set()
    for f in files:
        rp = str(f.resolve())
        if rp not in seen:
            uniq.append(f)
            seen.add(rp)
    return uniq


def _build_verifier_prompt_ids(
    tokenizer: Any,
    question: str,
    student_response: str,
    verifier_prompt_budget: int,
) -> Tuple[List[int], Dict[str, Any]]:
    # Keep consistent with online `_run_verifier_inference` structured truncation.
    inj = _load_hint_injection_module()
    VERIFIER_SYSTEM_PROMPT = getattr(inj, "VERIFIER_SYSTEM_PROMPT")
    build_verifier_user_prompt = getattr(inj, "build_verifier_user_prompt")

    verifier_prompt = build_verifier_user_prompt(question, student_response)
    prefix_text, sep, student_resp = verifier_prompt.partition(
        "**Student Response (So Far):**"
    )
    if sep:
        user_prefix = f"{prefix_text}**Student Response (So Far):**\n"
    else:
        user_prefix = verifier_prompt
        student_resp = ""

    base_messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prefix},
    ]
    try:
        base_ids = tokenizer.apply_chat_template(
            base_messages, tokenize=True, add_generation_prompt=True
        )
    except Exception:
        base_text = tokenizer.apply_chat_template(
            base_messages, tokenize=False, add_generation_prompt=True
        )
        base_ids = tokenizer.encode(base_text, add_special_tokens=False)

    remaining = verifier_prompt_budget - len(base_ids)
    truncated_tokens = 0
    if remaining > 0 and student_resp:
        tail_ids = tokenizer.encode(student_resp, add_special_tokens=False)
        if len(tail_ids) > remaining:
            truncated_tokens = len(tail_ids) - remaining
        tail_ids = tail_ids[-remaining:]
        truncated_student_resp = tokenizer.decode(tail_ids, skip_special_tokens=True)
        user_full = user_prefix + truncated_student_resp
    else:
        truncated_tokens = len(tokenizer.encode(student_resp, add_special_tokens=False)) if student_resp else 0
        user_full = user_prefix

    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_full},
    ]
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    except Exception:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    if len(prompt_ids) > verifier_prompt_budget:
        prompt_ids = prompt_ids[-verifier_prompt_budget:]

    meta = {
        "verifier_prompt_budget": int(verifier_prompt_budget),
        "prompt_len": int(len(prompt_ids)),
        "truncated_student_resp_tokens": int(truncated_tokens),
    }
    return list(prompt_ids), meta


def _load_model(
    model_path: str,
    tokenizer_path: str,
    lora_path: str = "",
    dtype: str = "bf16",
    device: str = "cuda",
) -> Tuple[Any, torch.nn.Module]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trust_remote_code = True
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)

    torch_dtype = torch.bfloat16 if dtype in ("bf16", "bfloat16") else torch.float16
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.to(dev)
    model.eval()

    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path)
        model.to(dev)
        model.eval()

    return tok, model


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Offline verifier decision analysis: no-decision + GO/WAIT margin/entropy (LoRA vs base).",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Dump batch json paths or directories containing batch_*.json.",
    )
    parser.add_argument("--max-samples", type=int, default=64, help="Max samples to process.")
    parser.add_argument("--base-model", required=True, help="Base model path (e.g., hf-170 or hf-181).")
    parser.add_argument(
        "--lora-adapter",
        default="",
        help="Optional LoRA adapter path. If set, runs an extra 'lora' variant.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="Tokenizer path (default: --base-model).",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=35840,
        help="Online-like max_model_len; used to derive verifier prompt budget.",
    )
    parser.add_argument(
        "--verifier-max-new-tokens",
        type=int,
        default=4096,
        help="Verifier generation max_new_tokens (online default: 4096).",
    )
    parser.add_argument(
        "--verifier-max-prompt-length",
        type=int,
        default=16384,
        help="Verifier max_prompt_length (online default: 16384).",
    )
    parser.add_argument(
        "--score-context-len",
        type=int,
        default=8192,
        help="Max context tokens kept when scoring GO/WAIT tags (0 disables truncation).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: ./outputs/verifier_margin_<timestamp>/).",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generation; only score GO/WAIT immediately after prompt (approx).",
    )
    args = parser.parse_args(argv)

    dump_files = _collect_dump_files(args.inputs)
    if not dump_files:
        raise FileNotFoundError(f"No dump files found from inputs={args.inputs}")

    max_samples = int(args.max_samples)
    samples = _iter_dump_samples(dump_files, max_samples=max_samples)
    if not samples:
        raise RuntimeError("No samples loaded from dumps.")

    tokenizer_path = args.tokenizer_path or args.base_model
    tok, base_model = _load_model(
        model_path=args.base_model,
        tokenizer_path=tokenizer_path,
        lora_path="",
        dtype=args.dtype,
        device=args.device,
    )
    lora_model = None
    if args.lora_adapter:
        _, lora_model = _load_model(
            model_path=args.base_model,
            tokenizer_path=tokenizer_path,
            lora_path=args.lora_adapter,
            dtype=args.dtype,
            device=args.device,
        )

    max_model_len = int(args.max_model_len)
    verifier_max_new_tokens = int(args.verifier_max_new_tokens)
    verifier_max_prompt_length = int(args.verifier_max_prompt_length)
    verifier_prompt_budget = max(
        1, min(verifier_max_prompt_length, max_model_len - verifier_max_new_tokens)
    )
    go_ids = tok.encode("<GO>", add_special_tokens=False)
    wait_ids = tok.encode("<WAIT>", add_special_tokens=False)
    eos_id = tok.eos_token_id
    device = next(base_model.parameters()).device

    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else (Path(__file__).resolve().parents[1] / "outputs" / "verifier_margin")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    def _run_variant(variant: str, model: torch.nn.Module) -> None:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnDecision(StoppingCriteria):
            def __init__(self, seqs: List[List[int]]):
                self.seqs = [list(s) for s in seqs if s]

            def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
                ids = input_ids[0].tolist()
                for s in self.seqs:
                    if len(ids) >= len(s) and ids[-len(s) :] == s:
                        return True
                return False

        stopping = StoppingCriteriaList([StopOnDecision([go_ids, wait_ids])])

        for sample in samples:
            prompt_ids, meta = _build_verifier_prompt_ids(
                tok,
                sample.question,
                sample.student_response,
                verifier_prompt_budget=verifier_prompt_budget,
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            gen_ids: List[int] = []
            finish_reason = "unknown"
            decision_tag = None
            malformed_final_like = False

            if args.skip_generate:
                # Approx: treat prompt end as decision point (may not match true verifier format).
                finish_reason = "skip_generate"
                decision_tag = None
            else:
                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=verifier_max_new_tokens,
                        do_sample=False,
                        top_k=1,
                        top_p=1.0,
                        temperature=1.0,
                        stopping_criteria=stopping,
                        pad_token_id=tok.pad_token_id or eos_id,
                        eos_token_id=eos_id,
                        use_cache=True,
                    )
                full = out[0].tolist()
                gen_ids = full[len(prompt_ids) :]
                text = tok.decode(gen_ids, skip_special_tokens=True)
                decision_tag = _parse_decision_tag(text)
                malformed_final_like = _has_malformed_final_like(text)
                if eos_id is not None and gen_ids and int(gen_ids[-1]) == int(eos_id):
                    finish_reason = "eos"
                elif len(gen_ids) >= verifier_max_new_tokens:
                    finish_reason = "length"
                elif decision_tag in ("GO", "WAIT"):
                    finish_reason = "stop"
                else:
                    finish_reason = "unknown"

            # Score GO/WAIT tag log-prob at the decision point.
            logp_go = float("nan")
            logp_wait = float("nan")
            margin = float("nan")
            entropy = float("nan")

            context_ids = prompt_ids + gen_ids
            # Prefer the last occurrence so "<GO>" inside the shadow verification doesn't confuse us.
            pos_go = _find_last_subsequence(context_ids, go_ids)
            pos_wait = _find_last_subsequence(context_ids, wait_ids)
            decision_pos = None
            if pos_go is not None and (pos_wait is None or pos_go > pos_wait):
                decision_pos = pos_go
            elif pos_wait is not None:
                decision_pos = pos_wait

            if decision_pos is None:
                # Fallback: score at prompt end.
                decision_pos = len(prompt_ids)

            prefix_ids = context_ids[: int(decision_pos)]
            try:
                logp_go = _sequence_logprob(
                    model,
                    prefix_ids,
                    go_ids,
                    device=device,
                    max_context_len=int(args.score_context_len),
                )
                logp_wait = _sequence_logprob(
                    model,
                    prefix_ids,
                    wait_ids,
                    device=device,
                    max_context_len=int(args.score_context_len),
                )
                margin = float(logp_go - logp_wait)
                # Binary entropy over {GO, WAIT}.
                m = max(logp_go, logp_wait)
                p_go = math.exp(logp_go - m)
                p_wait = math.exp(logp_wait - m)
                z = p_go + p_wait
                p_go /= z
                p_wait /= z
                entropy = float(
                    -p_go * math.log(max(1e-12, p_go))
                    - p_wait * math.log(max(1e-12, p_wait))
                )
            except Exception:
                pass

            results.append(
                {
                    "variant": variant,
                    "sample_idx": int(sample.sample_idx),
                    "prompt_len": int(meta["prompt_len"]),
                    "prompt_budget": int(meta["verifier_prompt_budget"]),
                    "truncated_student_resp_tokens": int(meta["truncated_student_resp_tokens"]),
                    "gen_len": int(len(gen_ids)),
                    "finish_reason": finish_reason,
                    "decision_tag": decision_tag or "",
                    "no_decision": 0 if decision_tag in ("GO", "WAIT") else 1,
                    "malformed_final_like": 1 if malformed_final_like else 0,
                    "logp_go": logp_go,
                    "logp_wait": logp_wait,
                    "margin_go_minus_wait": margin,
                    "entropy_go_wait": entropy,
                }
            )

    _run_variant("base", base_model)
    if lora_model is not None:
        _run_variant("lora", lora_model)

    df = pd.DataFrame(results)
    out_csv = out_dir / "decision_margin_per_sample.csv"
    df.to_csv(out_csv, index=False)

    summary = []
    for variant, g in df.groupby("variant"):
        n = len(g)
        no_dec = float(g["no_decision"].mean()) if n else float("nan")
        summary.append(
            {
                "variant": variant,
                "n": n,
                "no_decision_rate": no_dec,
                "malformed_final_like_rate": float(g["malformed_final_like"].mean()) if n else float("nan"),
                "finish_reason_stop_ratio": float((g["finish_reason"] == "stop").mean()) if n else float("nan"),
                "finish_reason_length_ratio": float((g["finish_reason"] == "length").mean()) if n else float("nan"),
                "margin_mean": float(g["margin_go_minus_wait"].mean()) if n else float("nan"),
                "margin_std": float(g["margin_go_minus_wait"].std()) if n else float("nan"),
                "entropy_mean": float(g["entropy_go_wait"].mean()) if n else float("nan"),
                "prompt_len_mean": float(g["prompt_len"].mean()) if n else float("nan"),
                "trunc_tokens_mean": float(g["truncated_student_resp_tokens"].mean()) if n else float("nan"),
            }
        )
    summary_df = pd.DataFrame(summary)
    out_summary = out_dir / "decision_margin_summary.csv"
    summary_df.to_csv(out_summary, index=False)

    # Optional plots.
    try:
        import matplotlib.pyplot as plt

        if "margin_go_minus_wait" in df.columns:
            plt.figure(figsize=(8, 4))
            for variant, g in df.groupby("variant"):
                xs = g["margin_go_minus_wait"].dropna().astype(float).to_list()
                if xs:
                    plt.hist(xs, bins=41, alpha=0.45, density=True, label=variant)
            plt.title("margin: logp(<GO>) - logp(<WAIT>)")
            plt.xlabel("margin")
            plt.ylabel("density")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "margin_hist.png", dpi=160)
            plt.close()

        if "entropy_go_wait" in df.columns:
            plt.figure(figsize=(8, 4))
            for variant, g in df.groupby("variant"):
                xs = g["entropy_go_wait"].dropna().astype(float).to_list()
                if xs:
                    plt.hist(xs, bins=41, alpha=0.45, density=True, label=variant)
            plt.title("entropy over {GO, WAIT}")
            plt.xlabel("entropy")
            plt.ylabel("density")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "entropy_hist.png", dpi=160)
            plt.close()
    except Exception:
        pass

    print(f"[margin] wrote {out_csv}")
    print(f"[margin] wrote {out_summary}")
    print(f"[margin] out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
