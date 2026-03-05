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

_sanitize_eval_urls() {
  local raw="$1"
  raw="${raw#<}"
  raw="${raw%>}"
  local out=""
  local part=""
  IFS=',' read -r -a _parts <<<"${raw}"
  for part in "${_parts[@]}"; do
    part="$(echo "${part}" | sed -e 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    part="${part#<}"
    part="${part%>}"
    [[ -n "${part}" ]] || continue
    if [[ -n "${out}" ]]; then
      out+=","
    fi
    out+="${part}"
  done
  echo "${out}"
}

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
# Paths (actor + verifier)
# --------------------------
CKPT_ROOT="${CKPT_ROOT:-/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/checkpoints/co_grpo_v2/cogrpo32-hf170-lora2445-dapo17k-pl2048-cfK2E2-int2-32k-v5}"
ACTOR_MODEL="${ACTOR_MODEL:-${CKPT_ROOT}/global_step_20/actor/huggingface}"
# Optional verifier base model (HF dir). If empty, default verifier base = ACTOR_MODEL.
VERIFIER_MODEL="${VERIFIER_MODEL:-}"
# Optional verifier LoRA adapter dir (PEFT). If empty, verifier runs as a full model.
# NOTE: use ${VAR-default} (not :-) so callers can explicitly disable via VERIFIER_LORA="".
VERIFIER_LORA="${VERIFIER_LORA-${CKPT_ROOT}/verifier_lora/global_step_20}"

[[ -f "${ACTOR_MODEL}/config.json" ]] || _die "Missing actor config.json under ${ACTOR_MODEL}"
if [[ -n "${VERIFIER_MODEL}" ]]; then
  [[ -f "${VERIFIER_MODEL}/config.json" ]] || _die "Missing verifier config.json under ${VERIFIER_MODEL}"
fi
if [[ -n "${VERIFIER_LORA}" ]]; then
  [[ -f "${VERIFIER_LORA}/adapter_config.json" ]] || _die "Missing verifier adapter_config.json under ${VERIFIER_LORA}"
fi

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
echo "[run] verifier_model=${VERIFIER_MODEL:-${ACTOR_MODEL}}"
echo "[run] verifier_lora=${VERIFIER_LORA:-'(none)'}"
echo "[run] data_jsonl=${DATA_JSONL} prompt_key=${PROMPT_KEY}"
echo "[run] actor_transport=${ACTOR_TRANSPORT:-pipeline} actor_api_mode=${ACTOR_API_MODE:-auto} actor_server_pool=${ACTOR_SERVER_POOL:-auto} start_actor_api_server=${START_ACTOR_API_SERVER:-0}"

# --------------------------
# Parallelism
# --------------------------
if [[ -n "${GPU_IDS:-}" ]]; then
  GPU_IDS_STR="${GPU_IDS}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    _gpu_n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${_gpu_n}" =~ ^[0-9]+$ ]] && [[ "${_gpu_n}" -gt 0 ]]; then
      GPU_IDS_STR="$(seq 0 $((_gpu_n - 1)) | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
    else
      GPU_IDS_STR="0"
    fi
  else
    GPU_IDS_STR="0"
  fi
fi
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

# Best-effort shard-level resume from partial `${out}.tmp` (append remaining lines instead of restarting a shard).
# Note: this assumes inputs are unchanged between runs.
RESUME_FROM_TMP="${RESUME_FROM_TMP:-1}"

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
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${MAX_SEQ_LEN}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

if [[ "${BACKEND}" == "lmdeploy" ]]; then
  TEMPERATURE="${TEMPERATURE:-0.7}"
else
  TEMPERATURE="${TEMPERATURE:-1.0}"
fi
TOP_P="${TOP_P:-1.0}"
if [[ "${BACKEND}" == "lmdeploy" ]]; then
  # Align with LMDeploy OpenAI server default.
  TOP_K="${TOP_K:-40}"
else
  TOP_K="${TOP_K:--1}"
fi
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
MAX_SAMPLE_SECONDS="${MAX_SAMPLE_SECONDS:-0}"

# LMDeploy actor transport (optional, for OpenCompass-style OpenAISDK alignment).
# - pipeline (default): run lmdeploy pipeline() in-process
# - openai: call an OpenAI-compatible /v1/completions endpoint (raw prompt continuation)
ACTOR_TRANSPORT="${ACTOR_TRANSPORT:-pipeline}"  # pipeline|openai
ACTOR_API_BASE="${ACTOR_API_BASE:-}"
ACTOR_API_MODEL="${ACTOR_API_MODEL:-}"
ACTOR_API_MODE="${ACTOR_API_MODE:-auto}"        # auto|completions|chat
ACTOR_API_TIMEOUT="${ACTOR_API_TIMEOUT:-600}"
START_ACTOR_API_SERVER="${START_ACTOR_API_SERVER:-0}"
ACTOR_API_PORT="${ACTOR_API_PORT:-0}"
ACTOR_API_TP="${ACTOR_API_TP:-1}"
ACTOR_API_WORKER_NUM="${ACTOR_API_WORKER_NUM:-1}"
ACTOR_API_EXTRA_CLI="${ACTOR_API_EXTRA_CLI:-}"
ACTOR_DISABLE_THINKING="${ACTOR_DISABLE_THINKING:-0}"
ONLINE_MATH_PROMPT="${ONLINE_MATH_PROMPT:-0}"
ACTOR_SERVER_POOL="${ACTOR_SERVER_POOL:-auto}"  # auto|0|1
ACTOR_API_PORT_BASE="${ACTOR_API_PORT_BASE:-0}" # used when ACTOR_SERVER_POOL=1
# When using script-managed actor pool, prestart and wait all actor servers before any worker starts.
ACTOR_POOL_WAIT_ALL="${ACTOR_POOL_WAIT_ALL:-1}"  # 0|1
if [[ "${ACTOR_TRANSPORT}" == "openai" && "${START_ACTOR_API_SERVER}" != "1" && -z "${ACTOR_API_BASE}" ]]; then
  echo "[auto] ACTOR_TRANSPORT=openai but ACTOR_API_BASE is empty; auto enable START_ACTOR_API_SERVER=1"
  START_ACTOR_API_SERVER=1
fi
case "${ACTOR_API_MODE}" in
  auto|completions|chat) ;;
  *) _die "Invalid ACTOR_API_MODE=${ACTOR_API_MODE} (expected: auto|completions|chat)" ;;
esac
case "${ACTOR_SERVER_POOL}" in
  auto|0|1) ;;
  *) _die "Invalid ACTOR_SERVER_POOL=${ACTOR_SERVER_POOL} (expected: auto|0|1)" ;;
esac
case "${ACTOR_POOL_WAIT_ALL}" in
  0|1) ;;
  *) _die "Invalid ACTOR_POOL_WAIT_ALL=${ACTOR_POOL_WAIT_ALL} (expected: 0|1)" ;;
esac

ACTOR_SERVER_POOL_ENABLED=0
if [[ "${ACTOR_SERVER_POOL}" == "1" ]]; then
  ACTOR_SERVER_POOL_ENABLED=1
elif [[ "${ACTOR_SERVER_POOL}" == "auto" ]]; then
  if [[ "${BACKEND}" == "lmdeploy" && "${ACTOR_TRANSPORT}" == "openai" && "${START_ACTOR_API_SERVER}" == "1" && -z "${ACTOR_API_BASE}" ]]; then
    ACTOR_SERVER_POOL_ENABLED=1
  fi
fi
if [[ "${ACTOR_SERVER_POOL_ENABLED}" == "1" && "${ACTOR_TRANSPORT}" != "openai" ]]; then
  _die "ACTOR_SERVER_POOL requires ACTOR_TRANSPORT=openai"
fi

# Support comma-separated external actor API bases (e.g. 4 endpoints on 2 nodes).
# Each shard is pinned to one endpoint by shard index to improve throughput and avoid
# sending all workers to a single URL.
declare -a ACTOR_API_BASES=()
if [[ -n "${ACTOR_API_BASE}" ]]; then
  ACTOR_API_BASE="$(_sanitize_eval_urls "${ACTOR_API_BASE}")"
  IFS=',' read -r -a ACTOR_API_BASES <<<"${ACTOR_API_BASE}"
fi

# --------------------------
# CompassVerifier tail scoring
# --------------------------
DO_CV_EVAL="${DO_CV_EVAL:-1}"
EVAL_URLS="${EVAL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"
EVAL_URLS="$(_sanitize_eval_urls "${EVAL_URLS}")"
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
  local fixed="localhost,127.0.0.1,::1,.svc,.pjlab.org.cn"
  if [[ -n "${hosts}" ]]; then
    export NO_PROXY="${hosts},${fixed}${NO_PROXY:+,${NO_PROXY}}"
  else
    export NO_PROXY="${fixed}${NO_PROXY:+,${NO_PROXY}}"
  fi
  export no_proxy="${NO_PROXY}"
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
  echo "[env] NO_PROXY=${NO_PROXY}"
}

_no_proxy_inputs="${EVAL_URLS}"
if [[ -n "${ACTOR_API_BASE}" ]]; then
  _no_proxy_inputs="${_no_proxy_inputs},${ACTOR_API_BASE}"
fi
_setup_no_proxy "${_no_proxy_inputs}"

# --------------------------
# Run metadata + resume safety
# --------------------------
RUN_META_TXT="${RUN_DIR}/run_param_meta.txt"
RUN_PARAM_STRING="$(cat <<EOF
backend=${BACKEND}
actor_model=${ACTOR_MODEL}
verifier_model=${VERIFIER_MODEL:-${ACTOR_MODEL}}
verifier_lora=${VERIFIER_LORA}
data_jsonl=${DATA_JSONL}
prompt_key=${PROMPT_KEY}
repeat=${REPEAT}
mode=${MODE}
debug_n=${DEBUG_N}
max_seq_len=${MAX_SEQ_LEN}
max_out_len=${MAX_OUT_LEN}
max_response_tokens=${MAX_RESPONSE_TOKENS}
token_check_interval=${TOKEN_CHECK_INTERVAL}
min_step_tokens=${MIN_STEP_TOKENS}
max_interventions=${MAX_INTERVENTIONS}
temperature=${TEMPERATURE}
top_p=${TOP_P}
top_k=${TOP_K}
seed=${SEED}
stop_token_id=${STOP_TOKEN_ID}
actor_transport=${ACTOR_TRANSPORT}
actor_api_base=${ACTOR_API_BASE}
actor_api_model=${ACTOR_API_MODEL}
actor_api_mode=${ACTOR_API_MODE}
actor_server_pool=${ACTOR_SERVER_POOL}
actor_api_port_base=${ACTOR_API_PORT_BASE}
actor_disable_thinking=${ACTOR_DISABLE_THINKING}
use_verifier_system_prompt=${USE_VERIFIER_SYSTEM_PROMPT}
online_math_prompt=${ONLINE_MATH_PROMPT}
degeneration_guard=${DEGENERATION_GUARD}
max_sample_seconds=${MAX_SAMPLE_SECONDS}
EOF
)"
RUN_PARAM_HASH="$(printf '%s' "${RUN_PARAM_STRING}" | sha1sum | awk '{print $1}')"
echo "[run] run_param_hash=${RUN_PARAM_HASH}"

if [[ -f "${RUN_META_TXT}" ]]; then
  prev_hash="$(awk -F= '/^run_param_hash=/{print $2; exit}' "${RUN_META_TXT}" || true)"
  if [[ -n "${prev_hash}" && "${prev_hash}" != "${RUN_PARAM_HASH}" ]]; then
    _die "run_param_hash mismatch for existing RUN_DIR (${RUN_DIR}). Use a new RUN_ID or keep parameters identical."
  fi
elif [[ "${RESUME_FROM_TMP}" == "1" ]]; then
  if compgen -G "${OUT_DIR}/shard_*.jsonl*" > /dev/null; then
    _die "RESUME_FROM_TMP=1 but missing ${RUN_META_TXT}; cannot verify parameter consistency safely."
  fi
fi

{
  echo "run_param_hash=${RUN_PARAM_HASH}"
  echo "${RUN_PARAM_STRING}"
} > "${RUN_META_TXT}"

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
# Optional actor service pool (one API server per shard/GPU)
# --------------------------
_pick_free_local_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

_wait_openai_ready_pick_model() {
  local api_base="$1"
  local timeout_s="$2"
  local preferred_model="${3:-}"
  python3 - "${api_base}" "${timeout_s}" "${preferred_model}" <<'PY'
import json
import sys
import time
from urllib.request import urlopen

api_base = (sys.argv[1] or "").rstrip("/")
timeout_s = int(sys.argv[2])
preferred = (sys.argv[3] or "").strip()
if not api_base:
    raise SystemExit("empty api_base")

deadline = time.time() + max(1, timeout_s)
last_err = None
while time.time() < deadline:
    try:
        with urlopen(f"{api_base}/models", timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        ids = []
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    mid = str(item.get("id") or "").strip()
                    if mid:
                        ids.append(mid)
        models = payload.get("models")
        if isinstance(models, list):
            for item in models:
                if isinstance(item, dict):
                    mid = str(item.get("id") or "").strip()
                    if mid:
                        ids.append(mid)
        if preferred:
            if preferred in ids:
                print(preferred)
                raise SystemExit(0)
            raise RuntimeError(f"preferred_model_not_found preferred={preferred} ids={ids[:8]}")
        if ids:
            print(ids[0])
            raise SystemExit(0)
        raise RuntimeError(f"no_model_ids keys={list(payload.keys())}")
    except Exception as e:
        last_err = e
        time.sleep(2)

raise SystemExit(f"timeout_wait_ready api_base={api_base} last_err={last_err}")
PY
}

declare -a actor_service_pids=()
declare -a actor_service_urls=()
declare -a actor_service_models=()
declare -a actor_service_logs=()

_start_actor_service_for_shard() {
  local shard_idx="$1"
  local gpu="$2"
  local log_file="${LOG_DIR}/actor_server_$(printf '%02d' "${shard_idx}").log"
  local port=0
  local port_base="${ACTOR_API_PORT_BASE}"
  local preferred_model="${ACTOR_API_MODEL:-}"
  local extra_cli="${ACTOR_API_EXTRA_CLI}"
  local model_id=""

  if [[ -n "${actor_service_urls[$shard_idx]:-}" ]] && [[ -n "${actor_service_pids[$shard_idx]:-}" ]]; then
    return 0
  fi

  if [[ -z "${extra_cli}" ]]; then
    extra_cli="--backend ${LMDEPLOY_BACKEND} --session-len ${LMDEPLOY_SESSION_LEN} --max-batch-size ${LMDEPLOY_MAX_BATCH_SIZE}"
  fi

  if [[ "${port_base}" =~ ^[0-9]+$ ]] && [[ "${port_base}" -gt 0 ]]; then
    port=$((port_base + shard_idx))
  else
    port="$(_pick_free_local_port)"
  fi

  echo "[actor_pool] shard_${shard_idx} gpu=${gpu} start api_server port=${port} log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export LMDEPLOY_SKIP_WARMUP=1
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    exec lmdeploy serve api_server "${ACTOR_MODEL}" \
      --tp "${ACTOR_API_TP}" \
      --server-port "${port}" \
      ${extra_cli}
  ) >"${log_file}" 2>&1 &
  local pid="$!"

  local api_base="http://127.0.0.1:${port}/v1"
  if ! model_id="$(_wait_openai_ready_pick_model "${api_base}" "${ACTOR_API_TIMEOUT}" "${preferred_model}")"; then
    echo "[actor_pool][ERR] shard_${shard_idx} failed to become ready at ${api_base}" >&2
    tail -n 80 "${log_file}" >&2 || true
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    return 1
  fi

  actor_service_pids[$shard_idx]="${pid}"
  actor_service_urls[$shard_idx]="${api_base}"
  actor_service_models[$shard_idx]="${model_id}"
  actor_service_logs[$shard_idx]="${log_file}"
  echo "[actor_pool] shard_${shard_idx} ready api_base=${api_base} model_id=${model_id} pid=${pid}"
}

_stop_actor_service_pool() {
  local pid=""
  for pid in "${actor_service_pids[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${actor_service_pids[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    wait "${pid}" 2>/dev/null || true
  done
}

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
declare -a shard_part_paths=()
declare -a shard_log_paths=()
declare -a shard_status_paths=()
monitor_pid=""

if [[ "${ACTOR_SERVER_POOL_ENABLED}" == "1" && "${ACTOR_POOL_WAIT_ALL}" == "1" ]]; then
  declare -a _pool_target_shards=()
  for ((i=0; i<SHARDS; i++)); do
    in_shard="${IN_DIR}/shard_$(printf '%02d' "$i").jsonl"
    if [[ -s "${in_shard}" ]]; then
      _pool_target_shards+=("${i}")
    fi
  done
  if [[ "${#_pool_target_shards[@]}" -gt 0 ]]; then
    echo "[actor_pool] prestart enabled; waiting all services ready before launching workers..."
    for i in "${_pool_target_shards[@]}"; do
      gpu="${GPU_IDS_ARR[$i]}"
      _start_actor_service_for_shard "${i}" "${gpu}" || _die "Failed to start actor service for shard_${i}"
    done
    echo "[actor_pool] all services ready (${#_pool_target_shards[@]} shards)"
  fi
fi

cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
  fi
  _stop_actor_service_pool || true
}
trap cleanup EXIT

for ((i=0; i<SHARDS; i++)); do
  gpu="${GPU_IDS_ARR[$i]}"
  in_shard="${IN_DIR}/shard_$(printf '%02d' "$i").jsonl"
  out_shard="${OUT_DIR}/shard_$(printf '%02d' "$i").jsonl"
  tmp_shard="${out_shard}.tmp"
  part_shard="${tmp_shard}.part"
  log_file="${LOG_DIR}/worker_${i}.log"

  # Skip empty shards.
  if [[ ! -s "${in_shard}" ]]; then
    echo "[skip] empty shard: ${in_shard}"
    continue
  fi

  # With RESUME_FROM_TMP=0, always rerun this shard from scratch.
  if [[ "${RESUME_FROM_TMP}" != "1" ]]; then
    rm -f "${out_shard}" "${tmp_shard}" "${part_shard}"
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

  worker_actor_api_base="${ACTOR_API_BASE}"
  worker_actor_api_model="${ACTOR_API_MODEL}"
  worker_start_actor_api_server="${START_ACTOR_API_SERVER}"
  if [[ "${worker_start_actor_api_server}" != "1" && "${#ACTOR_API_BASES[@]}" -gt 0 ]]; then
    worker_actor_api_base="${ACTOR_API_BASES[$(( i % ${#ACTOR_API_BASES[@]} ))]}"
  fi
  if [[ "${ACTOR_SERVER_POOL_ENABLED}" == "1" ]]; then
    _start_actor_service_for_shard "${i}" "${gpu}" || _die "Failed to start actor service for shard_${i}"
    worker_actor_api_base="${actor_service_urls[$i]}"
    worker_actor_api_model="${actor_service_models[$i]}"
    worker_start_actor_api_server="0"
  fi

  status_file="${LOG_DIR}/worker_${i}.exitcode"
  rm -f "${status_file}"

  echo "[launch] shard_${i} gpu=${gpu} actor_api_base=${worker_actor_api_base:-'(in-proc)'} -> ${out_shard}"
  launched_shards+=("${i}")
  shard_in_paths+=("${in_shard}")
  shard_out_paths+=("${out_shard}")
  shard_tmp_paths+=("${tmp_shard}")
  shard_part_paths+=("${part_shard}")
  shard_log_paths+=("${log_file}")
  shard_status_paths+=("${status_file}")
  {
  set +e
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1

    # --------------------------
    # Resume within shard (best-effort): if `${tmp_shard}` exists, append missing outputs.
    # --------------------------
    resume_in_shard="${in_shard}"
    resume_part_out=""
    # If a previous run was interrupted before appending `${tmp}.part` into
    # `${tmp}`, merge complete lines first to maximize resume continuity.
    if [[ "${RESUME_FROM_TMP}" == "1" ]] && [[ -e "${part_shard}" ]]; then
      python3 - "${part_shard}" <<'PY'
import os
import sys

path = sys.argv[1]
try:
    st = os.stat(path)
except FileNotFoundError:
    raise SystemExit(0)

size = int(st.st_size)
if size <= 0:
    raise SystemExit(0)

with open(path, "rb+") as f:
    try:
        f.seek(-1, os.SEEK_END)
    except OSError:
        f.truncate(0)
        raise SystemExit(0)
    last = f.read(1)
    if last == b"\n":
        raise SystemExit(0)

    block = 1024 * 1024
    pos = size
    last_nl = -1
    while pos > 0:
        read_sz = min(block, pos)
        pos -= read_sz
        f.seek(pos, os.SEEK_SET)
        data = f.read(read_sz)
        idx = data.rfind(b"\n")
        if idx >= 0:
            last_nl = pos + idx
            break
    if last_nl >= 0:
        f.truncate(last_nl + 1)
    else:
        f.truncate(0)
PY
      part_done=$(wc -l < "${part_shard}" | tr -d ' ')
      if [[ "${part_done}" -gt 0 ]]; then
        echo "[resume] shard_${i} merging stale part (${part_done} lines) into tmp"
        touch "${tmp_shard}"
        cat "${part_shard}" >> "${tmp_shard}"
      fi
      rm -f "${part_shard}"
    fi
    if [[ "${RESUME_FROM_TMP}" == "1" ]] && [[ -s "${tmp_shard}" ]]; then
      in_n=$(wc -l < "${in_shard}" | tr -d ' ')
      # If the previous process was killed mid-write, `${tmp_shard}` can end with a partial JSON line.
      # Truncate it back to the last newline so we can safely append resumed outputs.
      python3 - "${tmp_shard}" <<'PY'
import os
import sys

path = sys.argv[1]
try:
    st = os.stat(path)
except FileNotFoundError:
    raise SystemExit(0)

size = int(st.st_size)
if size <= 0:
    raise SystemExit(0)

with open(path, "rb+") as f:
    try:
        f.seek(-1, os.SEEK_END)
    except OSError:
        # very small file
        f.truncate(0)
        raise SystemExit(0)
    last = f.read(1)
    if last == b"\n":
        raise SystemExit(0)

    # Search backwards for the last newline and truncate after it.
    block = 1024 * 1024
    pos = size
    last_nl = -1
    while pos > 0:
        read_sz = min(block, pos)
        pos -= read_sz
        f.seek(pos, os.SEEK_SET)
        data = f.read(read_sz)
        idx = data.rfind(b"\n")
        if idx >= 0:
            last_nl = pos + idx
            break
    if last_nl >= 0:
        f.truncate(last_nl + 1)
    else:
        f.truncate(0)
PY
      tmp_done=$(wc -l < "${tmp_shard}" | tr -d ' ')
      if [[ "${tmp_done}" -ge "${in_n}" ]]; then
        echo "[resume] shard_${i} tmp complete (${tmp_done}/${in_n}), moving into place"
        mv -f "${tmp_shard}" "${out_shard}"
        exit 0
      fi
      if [[ "${tmp_done}" -gt 0 ]]; then
        echo "[resume] shard_${i} resuming from tmp (${tmp_done}/${in_n})"
        resume_in_shard="${IN_DIR}/shard_$(printf '%02d' "$i").resume_from_${tmp_done}.jsonl"
        tail -n "+$((tmp_done + 1))" "${in_shard}" > "${resume_in_shard}"
        resume_part_out="${part_shard}"
        rm -f "${resume_part_out}"
      else
        # tmp exists but has no valid lines; restart from scratch.
        rm -f "${tmp_shard}"
      fi
    else
      rm -f "${tmp_shard}"
    fi

    if [[ "${BACKEND}" == "lmdeploy" ]]; then
      # LMDeploy eval (actor=lmdeploy, verifier=HF+PEFT on the same GPU).
      python3 "${EVAL_PY}" \
        --base-model "${ACTOR_MODEL}" \
        $(if [[ -n "${VERIFIER_MODEL}" ]]; then echo "--verifier-model" "${VERIFIER_MODEL}"; fi) \
        $(if [[ -n "${VERIFIER_LORA}" ]]; then echo "--verifier-lora" "${VERIFIER_LORA}"; fi) \
        --prompts-file "${resume_in_shard}" \
        --prompt-key "${PROMPT_KEY}" \
        --out-jsonl "${resume_part_out:-${tmp_shard}}" \
        --mode "${MODE}" \
        $(if [[ "${USE_VERIFIER_SYSTEM_PROMPT}" == "1" ]]; then echo "--use-verifier-system-prompt"; fi) \
        $(if [[ "${ONLINE_MATH_PROMPT}" == "1" ]]; then echo "--online-math-prompt"; fi) \
        --actor-transport "${ACTOR_TRANSPORT}" \
        --actor-api-base "${worker_actor_api_base}" \
        $(if [[ -n "${worker_actor_api_model}" ]]; then echo "--actor-api-model" "${worker_actor_api_model}"; fi) \
        --actor-api-mode "${ACTOR_API_MODE}" \
        --actor-api-timeout "${ACTOR_API_TIMEOUT}" \
        $(if [[ "${worker_start_actor_api_server}" == "1" ]]; then echo "--start-actor-api-server"; fi) \
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
        --max-sample-seconds "${MAX_SAMPLE_SECONDS}" \
        --progress
    else
      # vLLM eval (fastest; interventions via vLLM LoRA).
      python3 "${EVAL_PY}" \
        --backend "${BACKEND}" \
        --base-model "${ACTOR_MODEL}" \
        $(if [[ -n "${VERIFIER_LORA}" ]]; then echo "--verifier-lora" "${VERIFIER_LORA}"; fi) \
        --prompts-file "${resume_in_shard}" \
        --prompt-key "${PROMPT_KEY}" \
        --out-jsonl "${resume_part_out:-${tmp_shard}}" \
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

    # If we resumed from tmp, append the "part" outputs and keep tmp as the canonical partial file.
    if [[ -n "${resume_part_out}" ]]; then
      cat "${resume_part_out}" >> "${tmp_shard}"
      rm -f "${resume_part_out}"
      rm -f "${resume_in_shard}" || true
    fi

    # Validate shard completion before publishing.
    in_n=$(wc -l < "${in_shard}" | tr -d ' ')
    out_n=$(wc -l < "${tmp_shard}" | tr -d ' ')
    if [[ "${out_n}" -lt "${in_n}" ]]; then
      echo "ERROR: shard_${i} incomplete (${out_n}/${in_n}) after run; keep tmp for resume: ${tmp_shard}" >&2
      exit 2
    fi
    mv -f "${tmp_shard}" "${out_shard}"
  ) >"${log_file}" 2>&1
  rc=$?
  set -e
  echo "${rc}" > "${status_file}"
  exit "${rc}"
  } &
  pids+=("$!")
done

monitor_pid=""
monitor_start_ts=""
monitor_prev_ts=""
monitor_prev_out=""
_print_progress_once() {
  local total_in=0
  local total_out=0
  local running=0
  local failed_cnt=0
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
    local part_path="${shard_part_paths[$idx]}"
    local status_path="${shard_status_paths[$idx]}"
    local in_n out_n part_n state rc
    in_n=$(wc -l < "${in_path}" | tr -d ' ')
    out_n=0
    part_n=0
    if [[ -s "${out_path}" ]]; then
      out_n=$(wc -l < "${out_path}" | tr -d ' ')
    elif [[ -s "${tmp_path}" ]]; then
      out_n=$(wc -l < "${tmp_path}" | tr -d ' ')
    fi
    if [[ -s "${part_path}" ]]; then
      part_n=$(wc -l < "${part_path}" | tr -d ' ')
      out_n=$((out_n + part_n))
    fi
    if kill -0 "${pid}" 2>/dev/null; then
      state="running"
      running=$((running + 1))
    else
      if [[ -s "${status_path}" ]]; then
        rc="$(tr -d ' \t\r\n' < "${status_path}" || true)"
        if [[ "${rc}" =~ ^[0-9]+$ ]] && [[ "${rc}" -ne 0 ]]; then
          state="failed(${rc})"
          failed_cnt=$((failed_cnt + 1))
        else
          state="done"
        fi
      else
        state="done"
      fi
    fi
    total_in=$((total_in + in_n))
    total_out=$((total_out + out_n))
    printf "  [shard_%02d pid=%s] %-7s %6s/%s\n" "${shard}" "${pid}" "${state}" "${out_n}" "${in_n}"
  done
  printf -- "-----------------------------------------------------------\n"
  local now_ts elapsed_s dt_s delta_out inst_rate avg_rate pct remaining eta_s eta_str
  now_ts="$(date +%s)"
  if [[ -z "${monitor_start_ts}" ]]; then
    monitor_start_ts="${now_ts}"
    monitor_prev_ts="${now_ts}"
    monitor_prev_out="${total_out}"
  fi
  elapsed_s=$(( now_ts - monitor_start_ts ))
  dt_s=$(( now_ts - monitor_prev_ts ))
  delta_out=$(( total_out - monitor_prev_out ))
  inst_rate="$(awk -v d="${dt_s}" -v x="${delta_out}" 'BEGIN{ if(d>0) printf("%.3f", x/d); else printf("0.000") }')"
  avg_rate="$(awk -v d="${elapsed_s}" -v x="${total_out}" 'BEGIN{ if(d>0) printf("%.3f", x/d); else printf("0.000") }')"
  pct="$(awk -v x="${total_out}" -v y="${total_in}" 'BEGIN{ if(y>0) printf("%.1f", (x*100.0)/y); else printf("0.0") }')"
  remaining=$(( total_in - total_out ))
  if [[ "${remaining}" -lt 0 ]]; then remaining=0; fi
  eta_s="$(awk -v r="${avg_rate}" -v left="${remaining}" 'BEGIN{ if(r>0) printf("%.0f", left/r); else printf("-1") }')"
  if [[ "${eta_s}" -ge 0 ]]; then
    # Format ETA as HH:MM:SS.
    local h m s
    h=$(( eta_s / 3600 ))
    m=$(( (eta_s % 3600) / 60 ))
    s=$(( eta_s % 60 ))
    if [[ "${h}" -gt 0 ]]; then
      eta_str="$(printf '%02dh%02dm%02ds' "${h}" "${m}" "${s}")"
    else
      eta_str="$(printf '%02dm%02ds' "${m}" "${s}")"
    fi
  else
    eta_str="-"
  fi

  printf "  Overall: %s%% (%s/%s lines) | active_workers=%s failed_workers=%s | speed=%s lines/s (avg=%s) | eta=%s | interval=%ss\n" \
    "${pct}" "${total_out}" "${total_in}" "${running}" "${failed_cnt}" "${inst_rate}" "${avg_rate}" "${eta_str}" "${PROGRESS_INTERVAL}"
  echo "==========================================================="

  monitor_prev_ts="${now_ts}"
  monitor_prev_out="${total_out}"
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

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${MONITOR_PROGRESS}" == "1" ]] && [[ "${#pids[@]}" -gt 0 ]]; then
  _print_progress_once || true
fi
[[ "${failed}" == "0" ]] || _die "Some workers failed; check ${LOG_DIR}/worker_*.log"

MERGED_JSONL="${RUN_DIR}/merged.jsonl"
cat "${OUT_DIR}"/shard_*.jsonl > "${MERGED_JSONL}"
echo "[merge] ${MERGED_JSONL}"

# Sanity check: detect catastrophic runs where every sample is an error.
python3 - "${MERGED_JSONL}" "${MODE}" <<'PY'
import json
import sys
from collections import Counter

path = sys.argv[1]
mode = sys.argv[2]
total = 0
err = 0
err_counter = Counter()

with open(path, "r", encoding="utf-8") as f:
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        total += 1
        rec = json.loads(raw)
        sides = []
        if mode in ("control", "both"):
            sides.append(rec.get("control") or {})
        if mode in ("exp", "both"):
            sides.append(rec.get("exp") or {})
        side_err = False
        for s in sides:
            e = str((s or {}).get("error") or "").strip()
            if e:
                side_err = True
                err_counter[e] += 1
        if side_err:
            err += 1

ratio = (float(err) / float(total)) if total > 0 else 0.0
print(f"[sanity] records={total} error_records={err} error_ratio={ratio:.4f}")
if err_counter:
    msg, cnt = err_counter.most_common(1)[0]
    print(f"[sanity] top_error_count={cnt} top_error={msg[:240]}")
if total > 0 and err == total:
    raise SystemExit(
        "ALL_RECORDS_ERROR: every sample failed. "
        "Likely endpoint/model/session-len mismatch or invalid API params."
    )
PY

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
