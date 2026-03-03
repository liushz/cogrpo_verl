#!/usr/bin/env bash
# Local 8-GPU offline eval for a CoGRPO checkpoint (actor + verifier LoRA), aligned with online vLLM by_step rollout.
#
# Default target:
# - Actor: global_step_20 HF weights (huggingface export)
# - Verifier: verifier_lora/global_step_20 (RL-updated adapter snapshot)
#
# What it runs:
# 1) vLLM inference via scripts/eval_co_grpo_with_verifier_v2.py (--backend vllm, --mode both)
# 2) CompassVerifier tail-only scoring via scripts/compassverifier_eval_tail_jsonl.py
# 3) Repeat-aware micro/macro summary via scripts/summarize_cv_scored_jsonl.py
#
# Usage:
#   cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro
#   bash run_local_eval_cogrpo_gs20_vllm_8gpu.sh
#
# Common overrides:
#   DATA_JSONL=... PROMPT_KEY=question REPEAT=4 MODE=both bash run_local_eval_cogrpo_gs20_vllm_8gpu.sh
#   GPU_IDS="0 1 2 3" SHARDS=4 bash run_local_eval_cogrpo_gs20_vllm_8gpu.sh
#   DEBUG_N=8 REPEAT=1 DO_CV_EVAL=0 bash run_local_eval_cogrpo_gs20_vllm_8gpu.sh
#   MAX_SEQ_LEN=65536 MAX_OUT_LEN=65536 bash run_local_eval_cogrpo_gs20_vllm_8gpu.sh
#   BACKEND=lmdeploy CONDA_ENV=oc bash run_local_eval_cogrpo_gs20_vllm_8gpu.sh
#   DO_CV_EVAL=0 bash run_local_eval_cogrpo_gs20_vllm_8gpu.sh
set -eEuo pipefail

_die() { echo "ERROR: $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backend:
# - vllm: fastest, but can show EOS/degeneration mismatches vs online eval.
# - lmdeploy: aligns with OpenCompass-style lmdeploy serving (recommended for apples-to-apples).
BACKEND="${BACKEND:-vllm}"  # vllm|lmdeploy|hf (hf unsupported for exp in v2)

EVAL_PY_VLLM="${REPO_ROOT}/scripts/eval_co_grpo_with_verifier_v2.py"
EVAL_PY_LMDEPLOY="${REPO_ROOT}/scripts/eval_co_grpo_with_verifier_lmdeploy64k.py"
if [[ "${BACKEND}" == "lmdeploy" ]]; then
  EVAL_PY="${EVAL_PY_LMDEPLOY}"
else
  EVAL_PY="${EVAL_PY_VLLM}"
fi

CV_EVAL_PY="${REPO_ROOT}/scripts/compassverifier_eval_tail_jsonl.py"
SUM_PY="${REPO_ROOT}/scripts/summarize_cv_scored_jsonl.py"

[[ -f "${EVAL_PY}" ]] || _die "Missing: ${EVAL_PY}"
[[ -f "${CV_EVAL_PY}" ]] || _die "Missing: ${CV_EVAL_PY}"
[[ -f "${SUM_PY}" ]] || _die "Missing: ${SUM_PY}"

# --------------------------
# Optional conda activation
# --------------------------
# Default conda env depends on backend:
# - vllm: `repro`
# - lmdeploy: `oc` (matches most offline OpenCompass eval configs)
if [[ -z "${CONDA_ENV:-}" ]]; then
  if [[ "${BACKEND}" == "lmdeploy" ]]; then
    CONDA_ENV="oc"
  else
    CONDA_ENV="repro"
  fi
fi
CONDA_SH="${CONDA_SH:-/mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh}"
if [[ "${CONDA_ENV}" != "none" ]]; then
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    echo "[env] conda_env=${CONDA_ENV}"
    python3 -V || true
    which python3 || true
  else
    _die "conda.sh not found at ${CONDA_SH} (set CONDA_ENV=none to skip)"
  fi
fi

# --------------------------
# Eval knobs that affect deps
# --------------------------
python3 - <<'PY'
import importlib.util
import os

backend = (os.environ.get("BACKEND") or "vllm").strip().lower()
req = ["torch", "transformers"]
if backend == "vllm":
    req += ["vllm"]
elif backend == "lmdeploy":
    req += ["lmdeploy", "peft"]
else:
    # keep it permissive; the script will fail-fast later if exp mode is unsupported.
    req += []

missing = [m for m in req if importlib.util.find_spec(m) is None]
if missing:
    hint = ""
    if backend == "lmdeploy":
        hint = " (hint: try CONDA_ENV=oc for lmdeploy)"
    raise SystemExit(f"[env][ERR] Missing modules: {missing}.{hint}")
print(f"[env] deps_ok backend={backend}")
PY

# --------------------------
# Paths (actor ckpt + verifier lora snapshot)
# --------------------------
CKPT_ROOT="${CKPT_ROOT:-/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/checkpoints/co_grpo_v2/cogrpo32-hf170-lora2445-dapo17k-pl2048-cfK2E2-int2-32k-v5}"
ACTOR_MODEL="${ACTOR_MODEL:-${CKPT_ROOT}/global_step_20/actor/huggingface}"
VERIFIER_LORA="${VERIFIER_LORA:-${CKPT_ROOT}/verifier_lora/global_step_20}"

[[ -f "${ACTOR_MODEL}/config.json" ]] || _die "Missing actor config.json under ${ACTOR_MODEL}"
[[ -f "${VERIFIER_LORA}/adapter_config.json" ]] || _die "Missing verifier adapter_config.json under ${VERIFIER_LORA}"

# Dataset (jsonl with prompt_key + answer).
DATA_JSONL="${DATA_JSONL:-/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/data/test/aime2025-I.jsonl}"
PROMPT_KEY="${PROMPT_KEY:-question}"
[[ -f "${DATA_JSONL}" ]] || _die "Missing DATA_JSONL: ${DATA_JSONL}"

# Output root.
OUT_ROOT="${OUT_ROOT:-/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/eval_local_vllm}"
TAG="${TAG:-cogrpo32-hf170-lora2445-gs20-vllm}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$RANDOM}"
RUN_DIR="${OUT_ROOT}/${TAG}/${RUN_ID}"
IN_DIR="${RUN_DIR}/inputs"
OUT_DIR="${RUN_DIR}/results"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${IN_DIR}" "${OUT_DIR}" "${LOG_DIR}"

echo "[run] run_dir=${RUN_DIR}"
echo "[run] actor_model=${ACTOR_MODEL}"
echo "[run] verifier_lora=${VERIFIER_LORA}"
echo "[run] data_jsonl=${DATA_JSONL} prompt_key=${PROMPT_KEY}"

# --------------------------
# Parallelism
# --------------------------
GPU_IDS_STR="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_IDS_ARR <<<"${GPU_IDS_STR}"
[[ "${#GPU_IDS_ARR[@]}" -gt 0 ]] || _die "Empty GPU_IDS"
SHARDS="${SHARDS:-${#GPU_IDS_ARR[@]}}"
[[ "${SHARDS}" -le "${#GPU_IDS_ARR[@]}" ]] || _die "SHARDS(${SHARDS}) > number of GPU_IDS(${#GPU_IDS_ARR[@]})"

# Repeat to reduce variance (adds origin_info._orig_line_idx/_repeat_id).
REPEAT="${REPEAT:-4}"
MODE="${MODE:-both}"  # control|exp|both

# Progress printing (since workers run in background and logs are redirected).
MONITOR_PROGRESS="${MONITOR_PROGRESS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

# Only run first N original items from DATA_JSONL (before repeat). 0 disables.
DEBUG_N="${DEBUG_N:-0}"

# --------------------------
# Online-aligned eval knobs (copied from the training override for this run)
# --------------------------
# Sequence/output lengths (OpenCompass-style naming):
# - MAX_SEQ_LEN: cap total prompt+response tokens (vLLM max_model_len).
# - MAX_OUT_LEN: cap model generation tokens (SamplingParams.max_tokens).
# NOTE: for vLLM, actual out tokens will be <= MAX_SEQ_LEN - prompt_len.
if [[ "${BACKEND}" == "lmdeploy" ]]; then
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
  MAX_OUT_LEN="${MAX_OUT_LEN:-65536}"
else
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-35840}"
  MAX_OUT_LEN="${MAX_OUT_LEN:-32768}"
fi

VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${MAX_SEQ_LEN}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

if [[ "${BACKEND}" == "lmdeploy" ]]; then
  TEMPERATURE="${TEMPERATURE:-0.7}"
else
  TEMPERATURE="${TEMPERATURE:-1.0}"
fi
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
SEED="${SEED:-0}"

if [[ "${BACKEND}" == "lmdeploy" ]]; then
  MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-65536}"
else
  MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-2048}"
fi
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-${MAX_OUT_LEN}}"  # model-gen tokens (hints excluded)
if [[ "${BACKEND}" == "lmdeploy" ]]; then
  TOKEN_CHECK_INTERVAL="${TOKEN_CHECK_INTERVAL:-4096}"
  MIN_STEP_TOKENS="${MIN_STEP_TOKENS:-4096}"
  MAX_INTERVENTIONS="${MAX_INTERVENTIONS:-5}"
else
  TOKEN_CHECK_INTERVAL="${TOKEN_CHECK_INTERVAL:-2048}"
  MIN_STEP_TOKENS="${MIN_STEP_TOKENS:-2048}"
  MAX_INTERVENTIONS="${MAX_INTERVENTIONS:-2}"
fi
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.0}"
WAIT_CONF_TAIL_TOKENS="${WAIT_CONF_TAIL_TOKENS:-64}"
VERIFIER_MAX_PROMPT_LENGTH="${VERIFIER_MAX_PROMPT_LENGTH:-16384}"
VERIFIER_MAX_NEW_TOKENS="${VERIFIER_MAX_NEW_TOKENS:-4096}"
VERIFIER_MAX_HINT_TOKENS="${VERIFIER_MAX_HINT_TOKENS:-512}"
STOP_TOKEN_ID="${STOP_TOKEN_ID:-151645}"
if [[ "${BACKEND}" == "lmdeploy" ]]; then
  USE_VERIFIER_SYSTEM_PROMPT="${USE_VERIFIER_SYSTEM_PROMPT:-1}"
else
  USE_VERIFIER_SYSTEM_PROMPT="${USE_VERIFIER_SYSTEM_PROMPT:-0}"
fi

# vLLM engine selection:
# - Default to V0 for stability (matches training launcher).
# - Some base images export VLLM_USE_V1=1 globally; override unless VERL_VLLM_USE_V1=1.
export VLLM_USE_V1="${VERL_VLLM_USE_V1:-0}"
if [[ "${VLLM_USE_V1}" == "1" ]]; then
  # Avoid "Cannot re-initialize CUDA in forked subprocess" in vLLM V1.
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
fi
echo "[env] VLLM_USE_V1=${VLLM_USE_V1}"

# LMDeploy backend knobs (only used when BACKEND=lmdeploy).
LMDEPLOY_BACKEND="${LMDEPLOY_BACKEND:-pytorch}"
LMDEPLOY_SESSION_LEN="${LMDEPLOY_SESSION_LEN:-${MAX_SEQ_LEN}}"
LMDEPLOY_MAX_BATCH_SIZE="${LMDEPLOY_MAX_BATCH_SIZE:-128}"
LMDEPLOY_LOG_LEVEL="${LMDEPLOY_LOG_LEVEL:-WARNING}"
DEGENERATION_GUARD="${DEGENERATION_GUARD:-1}"

# LMDeploy actor transport (optional, for OpenCompass-style OpenAISDK alignment).
# - pipeline (default): run lmdeploy pipeline() in-process
# - openai: call an OpenAI-compatible /v1/chat/completions endpoint
ACTOR_TRANSPORT="${ACTOR_TRANSPORT:-pipeline}"  # pipeline|openai
ACTOR_API_BASE="${ACTOR_API_BASE:-}"
ACTOR_API_TIMEOUT="${ACTOR_API_TIMEOUT:-600}"
START_ACTOR_API_SERVER="${START_ACTOR_API_SERVER:-0}"
ACTOR_API_PORT="${ACTOR_API_PORT:-0}"
ACTOR_API_TP="${ACTOR_API_TP:-1}"
ACTOR_API_WORKER_NUM="${ACTOR_API_WORKER_NUM:-1}"
ACTOR_API_EXTRA_CLI="${ACTOR_API_EXTRA_CLI:-}"
ACTOR_DISABLE_THINKING="${ACTOR_DISABLE_THINKING:-0}"
ONLINE_MATH_PROMPT="${ONLINE_MATH_PROMPT:-0}"

# --------------------------
# CompassVerifier tail scoring
# --------------------------
DO_CV_EVAL="${DO_CV_EVAL:-1}"
EVAL_URLS="${EVAL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"
CV_WORKERS="${CV_WORKERS:-4}"
CV_BATCH_SIZE="${CV_BATCH_SIZE:-64}"

_setup_no_proxy() {
  local urls="$1"
  local hosts=""
  hosts="$(python3 - "$urls" <<'PY'
import re, sys
urls=sys.argv[1]
hosts=[]
for u in urls.split(","):
    u=u.strip()
    if not u:
        continue
    u=re.sub(r"^https?://","",u)
    u=u.split("/",1)[0]
    h=u.split(":",1)[0]
    if h:
        hosts.append(h)
print(",".join(sorted(set(hosts))))
PY
)"
  export NO_PROXY="${hosts}${NO_PROXY:+,${NO_PROXY}}"
  export no_proxy="${NO_PROXY}"
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
  echo "[env] NO_PROXY=${NO_PROXY}"
}

_setup_no_proxy "${EVAL_URLS}"

# --------------------------
# Prepare inputs: repeat-expand then shard
# --------------------------
ALL_JSONL="${IN_DIR}/all.repeat${REPEAT}.jsonl"
python3 - "${DATA_JSONL}" "${ALL_JSONL}" "${REPEAT}" "${DEBUG_N}" <<'PY'
import json, sys
src, dst, repeat, debug_n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out = open(dst, "w", encoding="utf-8")
kept = 0
with open(src, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        if debug_n > 0 and kept >= debug_n:
            break
        obj = json.loads(line)
        for r in range(repeat):
            if isinstance(obj, dict):
                rec = dict(obj)
            else:
                rec = {"prompt": obj}
            origin = rec.get("origin_info")
            if not isinstance(origin, dict):
                origin = {"origin_info": origin}
            else:
                origin = dict(origin)
            origin["_orig_line_idx"] = int(line_idx)
            origin["_repeat_id"] = int(r)
            rec["origin_info"] = origin
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1
out.close()
print(f"[inputs] wrote {dst}")
if debug_n > 0:
    print(f"[inputs] DEBUG_N={debug_n} (kept_orig_items={kept})")
PY

python3 - "${ALL_JSONL}" "${IN_DIR}" "${SHARDS}" <<'PY'
import os, sys
src, out_dir, shards = sys.argv[1], sys.argv[2], int(sys.argv[3])
paths=[os.path.join(out_dir, f"shard_{i:02d}.jsonl") for i in range(shards)]
outs=[open(p, "w", encoding="utf-8") for p in paths]
n=0
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        outs[n % shards].write(line + "\n")
        n += 1
for o in outs:
    o.close()
print(f"[inputs] total={n} shards={shards} out_dir={out_dir}")
PY

# --------------------------
# Launch per-GPU workers (1 shard per GPU)
# --------------------------
echo "[launch] shards=${SHARDS} gpus=${GPU_IDS_STR}"
pids=()
failed=0
declare -a launched_shards=()
declare -a shard_in_paths=()
declare -a shard_out_paths=()
declare -a shard_tmp_paths=()
declare -a shard_log_paths=()

for ((i=0; i<SHARDS; i++)); do
  gpu="${GPU_IDS_ARR[$i]}"
  in_shard="${IN_DIR}/shard_$(printf '%02d' "$i").jsonl"
  out_shard="${OUT_DIR}/shard_$(printf '%02d' "$i").jsonl"
  tmp_shard="${out_shard}.tmp"
  log_file="${LOG_DIR}/worker_${i}.log"

  # Skip empty shards.
  if [[ ! -s "${in_shard}" ]]; then
    echo "[skip] empty shard: ${in_shard}"
    continue
  fi

  # Resume: if output exists and has >= input lines, skip.
  if [[ -s "${out_shard}" ]]; then
    in_n=$(wc -l < "${in_shard}" | tr -d ' ')
    out_n=$(wc -l < "${out_shard}" | tr -d ' ')
    if [[ "${out_n}" -ge "${in_n}" ]]; then
      echo "[resume] shard_${i} already done (${out_n}/${in_n}), skipping"
      continue
    fi
  fi

  echo "[launch] shard_${i} gpu=${gpu} -> ${out_shard}"
  launched_shards+=("${i}")
  shard_in_paths+=("${in_shard}")
  shard_out_paths+=("${out_shard}")
  shard_tmp_paths+=("${tmp_shard}")
  shard_log_paths+=("${log_file}")
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1
    rm -f "${tmp_shard}"
    if [[ "${BACKEND}" == "lmdeploy" ]]; then
      # LMDeploy eval (actor=lmdeploy, verifier=HF+PEFT on the same GPU).
      python3 "${EVAL_PY}" \
        --base-model "${ACTOR_MODEL}" \
        --verifier-lora "${VERIFIER_LORA}" \
        --prompts-file "${in_shard}" \
        --prompt-key "${PROMPT_KEY}" \
        --out-jsonl "${tmp_shard}" \
        --mode "${MODE}" \
        $(if [[ "${USE_VERIFIER_SYSTEM_PROMPT}" == "1" ]]; then echo "--use-verifier-system-prompt"; fi) \
        $(if [[ "${ONLINE_MATH_PROMPT}" == "1" ]]; then echo "--online-math-prompt"; fi) \
        --actor-transport "${ACTOR_TRANSPORT}" \
        --actor-api-base "${ACTOR_API_BASE}" \
        --actor-api-timeout "${ACTOR_API_TIMEOUT}" \
        $(if [[ "${START_ACTOR_API_SERVER}" == "1" ]]; then echo "--start-actor-api-server"; fi) \
        --actor-api-port "${ACTOR_API_PORT}" \
        --actor-api-tp "${ACTOR_API_TP}" \
        --actor-api-worker-num "${ACTOR_API_WORKER_NUM}" \
        --actor-api-extra-cli "${ACTOR_API_EXTRA_CLI}" \
        $(if [[ "${ACTOR_DISABLE_THINKING}" == "1" ]]; then echo "--actor-disable-thinking"; fi) \
        $(if [[ "${DEGENERATION_GUARD}" == "1" ]]; then echo "--degeneration-guard"; fi) \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}" \
        --seed "${SEED}" \
        --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
        --max-response-tokens "${MAX_RESPONSE_TOKENS}" \
        --token-check-interval "${TOKEN_CHECK_INTERVAL}" \
        --min-step-tokens "${MIN_STEP_TOKENS}" \
        --max-interventions "${MAX_INTERVENTIONS}" \
        --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
        --wait-conf-tail-tokens "${WAIT_CONF_TAIL_TOKENS}" \
        --verifier-max-prompt-length "${VERIFIER_MAX_PROMPT_LENGTH}" \
        --verifier-max-new-tokens "${VERIFIER_MAX_NEW_TOKENS}" \
        --verifier-max-hint-tokens "${VERIFIER_MAX_HINT_TOKENS}" \
        --stop-token-id "${STOP_TOKEN_ID}" \
        --lmdeploy-backend "${LMDEPLOY_BACKEND}" \
        --lmdeploy-session-len "${LMDEPLOY_SESSION_LEN}" \
        --lmdeploy-max-batch-size "${LMDEPLOY_MAX_BATCH_SIZE}" \
        --lmdeploy-log-level "${LMDEPLOY_LOG_LEVEL}" \
        --progress
    else
      # vLLM eval (fastest; interventions via vLLM LoRA).
      python3 "${EVAL_PY}" \
        --backend "${BACKEND}" \
        --base-model "${ACTOR_MODEL}" \
        --verifier-lora "${VERIFIER_LORA}" \
        --prompts-file "${in_shard}" \
        --prompt-key "${PROMPT_KEY}" \
        --out-jsonl "${tmp_shard}" \
        --mode "${MODE}" \
        $(if [[ "${USE_VERIFIER_SYSTEM_PROMPT}" == "1" ]]; then echo "--use-verifier-system-prompt"; fi) \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}" \
        --seed "${SEED}" \
        --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
        --max-response-tokens "${MAX_RESPONSE_TOKENS}" \
        --token-check-interval "${TOKEN_CHECK_INTERVAL}" \
        --min-step-tokens "${MIN_STEP_TOKENS}" \
        --max-interventions "${MAX_INTERVENTIONS}" \
        --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
        --wait-conf-tail-tokens "${WAIT_CONF_TAIL_TOKENS}" \
        --verifier-max-prompt-length "${VERIFIER_MAX_PROMPT_LENGTH}" \
        --verifier-max-new-tokens "${VERIFIER_MAX_NEW_TOKENS}" \
        --verifier-max-hint-tokens "${VERIFIER_MAX_HINT_TOKENS}" \
        --stop-token-id "${STOP_TOKEN_ID}" \
        --vllm-tp "${VLLM_TP}" \
        --vllm-gpu-mem-util "${VLLM_GPU_MEM_UTIL}" \
        --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --batch-size "${EVAL_BATCH_SIZE}" \
        --progress
    fi
    mv -f "${tmp_shard}" "${out_shard}"
  ) >"${log_file}" 2>&1 &
  pids+=("$!")
done

monitor_pid=""
_print_progress_once() {
  local total_in=0
  local total_out=0
  local running=0
  local n="${#pids[@]}"
  if [[ "${n}" -eq 0 ]]; then
    echo "[progress] no active workers"
    return
  fi
  echo "========== $(date '+%F %T') Progress =========="
  for idx in "${!pids[@]}"; do
    local pid="${pids[$idx]}"
    local shard="${launched_shards[$idx]}"
    local in_path="${shard_in_paths[$idx]}"
    local out_path="${shard_out_paths[$idx]}"
    local tmp_path="${shard_tmp_paths[$idx]}"
    local in_n out_n state
    in_n=$(wc -l < "${in_path}" | tr -d ' ')
    out_n=0
    if [[ -s "${out_path}" ]]; then
      out_n=$(wc -l < "${out_path}" | tr -d ' ')
    elif [[ -s "${tmp_path}" ]]; then
      out_n=$(wc -l < "${tmp_path}" | tr -d ' ')
    fi
    if kill -0 "${pid}" 2>/dev/null; then
      state="running"
      running=$((running + 1))
    else
      state="done"
    fi
    total_in=$((total_in + in_n))
    total_out=$((total_out + out_n))
    printf "  [shard_%02d pid=%s] %-7s %6s/%s\n" "${shard}" "${pid}" "${state}" "${out_n}" "${in_n}"
  done
  printf -- "-----------------------------------------------------------\n"
  printf "  Overall: %s%% (%s/%s lines) | active_workers=%s | interval=%ss\n" \
    "$(( (total_out * 100) / (total_in > 0 ? total_in : 1) ))" \
    "${total_out}" "${total_in}" "${running}" "${PROGRESS_INTERVAL}"
  echo "==========================================================="
}

if [[ "${MONITOR_PROGRESS}" == "1" ]] && [[ "${#pids[@]}" -gt 0 ]]; then
  echo "[progress] enabled (MONITOR_PROGRESS=1, interval=${PROGRESS_INTERVAL}s). Logs: ${LOG_DIR}/worker_*.log"
  (
    while true; do
      _print_progress_once || true
      alive=0
      for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          alive=1
          break
        fi
      done
      if [[ "${alive}" == "0" ]]; then
        break
      fi
      sleep "${PROGRESS_INTERVAL}"
    done
  ) &
  monitor_pid="$!"
fi

cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
[[ "${failed}" == "0" ]] || _die "Some workers failed; check ${LOG_DIR}/worker_*.log"

MERGED_JSONL="${RUN_DIR}/merged.jsonl"
cat "${OUT_DIR}"/shard_*.jsonl > "${MERGED_JSONL}"
echo "[merge] ${MERGED_JSONL}"

if [[ "${DO_CV_EVAL}" == "1" ]]; then
  [[ -n "${EVAL_URLS}" ]] || _die "DO_CV_EVAL=1 but EVAL_URLS is empty"
  _setup_no_proxy "${EVAL_URLS}"

  SCORED_JSONL="${RUN_DIR}/merged.cv.jsonl"
  CV_METRICS_JSON="${RUN_DIR}/cv_metrics.json"
  CV_METRICS_FIXED_JSON="${RUN_DIR}/cv_metrics.fixed.json"

  echo "[cv_eval] scoring -> ${SCORED_JSONL}"
  python3 "${CV_EVAL_PY}" \
    --in-jsonl "${MERGED_JSONL}" \
    --out-jsonl "${SCORED_JSONL}" \
    --prompt-key "${PROMPT_KEY}" \
    --mode "${MODE}" \
    --eval-urls "${EVAL_URLS}" \
    --workers "${CV_WORKERS}" \
    --batch-size "${CV_BATCH_SIZE}" \
    --progress >"${CV_METRICS_JSON}"

  # Repeat-aware summary without re-calling endpoints (uses existing cv_score fields).
  python3 "${SUM_PY}" \
    --in-jsonl "${SCORED_JSONL}" \
    --mode "${MODE}" \
    --out-json "${CV_METRICS_FIXED_JSON}" >/dev/null

  echo "[cv_eval] metrics(raw): ${CV_METRICS_JSON}"
  cat "${CV_METRICS_JSON}"
  echo "[cv_eval] metrics(fixed): ${CV_METRICS_FIXED_JSON}"
  cat "${CV_METRICS_FIXED_JSON}"
else
  echo "[cv_eval] skipped (DO_CV_EVAL=${DO_CV_EVAL})"
fi

echo ""
echo "[done] run_dir=${RUN_DIR}"
