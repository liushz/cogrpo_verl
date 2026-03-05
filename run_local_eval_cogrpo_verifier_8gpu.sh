#!/usr/bin/env bash
# Local (single-node) multi-GPU eval runner for Co-GRPO actor + verifier interventions.
#
# Goal: replace rjob queueing with a single-node 8-GPU run.
# - Each shard runs as an independent 1-GPU process (CUDA_VISIBLE_DEVICES=...).
# - Produces merged outputs and CompassVerifier tail-only scores.
#
# Default configs align with the previously submitted rjob runs:
# - infer_engine=lmdeploy (actor)
# - mode=exp (no control)
# - repeat=4
# - OpenCompass-style 64k settings (max_prompt/max_response/session_len=65536)
# - verifier_max_new_tokens=4096
#
# Usage:
#   bash repos/repro/run_local_eval_cogrpo_verifier_8gpu.sh
#
# Optional overrides (env vars):
#   GPU_IDS="0 1 2 3 4 5 6 7"   # which GPUs to use
#   RUN_FULL=1 RUN_HEAD4=1     # toggle datasets
#   OUT_ROOT=/path/to/out
#   RESUME=1                   # resume from previous runs (default: 1)
#   PROGRESS_INTERVAL=30       # seconds between progress reports (default: 30)
set -eEuo pipefail

_die() { echo "ERROR: $*" >&2; exit 1; }

# --------------------------
# Optional conda activation
# --------------------------
#
# The local node may not have lmdeploy/vllm in system python.
# We auto-activate a shared-storage conda env by default (same spirit as rjob scripts).
CONDA_ENV="${CONDA_ENV:-oc}"  # oc is the most stable lmdeploy env on shared storage
CONDA_SH="${CONDA_SH:-/mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh}"
if [[ "${CONDA_ENV}" != "none" ]]; then
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    echo "[env] conda_env=${CONDA_ENV}"
    conda --version || true
    python3 -V || true
    which python3 || true
  else
    _die "conda.sh not found at ${CONDA_SH} (set CONDA_ENV=none to skip)"
  fi
fi

# Fail-fast on common missing deps (avoids 8 shards failing immediately).
python3 - <<'PY'
import importlib.util
import sys

required = ("torch", "lmdeploy", "transformers", "peft", "openai")
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(
        f"[env][ERR] Missing modules: {missing} (python={sys.executable}). "
        "Try: CONDA_ENV=oc (lmdeploy) or use an env with torch+lmdeploy."
    )
print(f"[env] deps_ok python={sys.executable}")
PY

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${REPO_ROOT}/scripts/eval_co_grpo_with_verifier_lmdeploy64k.py"
CV_EVAL_PY="${REPO_ROOT}/scripts/compassverifier_eval_tail_jsonl.py"

[[ -f "${EVAL_PY}" ]] || _die "Missing eval script: ${EVAL_PY}"
[[ -f "${CV_EVAL_PY}" ]] || _die "Missing cv eval script: ${CV_EVAL_PY}"

# --------------------------
# Common (shared) config
# --------------------------

# Actor base model (cispo hf-170).
BASE_MODEL="${BASE_MODEL:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"

# Verifier LoRA roots (two ablations).
FPFIX_ROOT="${FPFIX_ROOT:-/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-verifier_sft_train_55_waitx2_fpfix_plus_v4ckptwait_20260223-02241157}"
DROP_ROOT="${DROP_ROOT:-/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-verifier_sft_train_55_waitx2_drop_plus_v4ckptwait_20260223-02241756}"

# Full mixed verifier model (no LoRA).
FULLMIX_VERIFIER_MODEL="${FULLMIX_VERIFIER_MODEL:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/ckpt/cold_start_full_qwen2d5-7b/20260223230703_mixed_full/20260224105827/hf-181}"
# Fullmix actor model: default to hf-181 as well (expected for "full mix").
FULLMIX_ACTOR_MODEL="${FULLMIX_ACTOR_MODEL:-${FULLMIX_VERIFIER_MODEL}}"

# Datasets.
DATA_FULL="${DATA_FULL:-/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/data/test/aime2025-I.jsonl}"
DATA_HEAD4="${DATA_HEAD4:-/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/data/test/aime2025-I.head4.jsonl}"
PROMPT_KEY="${PROMPT_KEY:-question}"

# CompassVerifier URLs (match online newest).
EVAL_URLS="${EVAL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"

# Output root (shared storage suggested).
OUT_ROOT="${OUT_ROOT:-/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/eval_local}"

# GPU selection.
GPU_IDS_STR="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_IDS_ARR <<<"${GPU_IDS_STR}"
[[ "${#GPU_IDS_ARR[@]}" -gt 0 ]] || _die "Empty GPU_IDS"
SHARDS="${SHARDS:-${#GPU_IDS_ARR[@]}}"
[[ "${SHARDS}" -le "${#GPU_IDS_ARR[@]}" ]] || _die "SHARDS(${SHARDS}) > number of GPU_IDS(${#GPU_IDS_ARR[@]})"

# Eval knobs (keep aligned with rjob defaults / previous runs).
REPEAT="${REPEAT:-4}"
MODE="${MODE:-exp}"
BASE_TEMPERATURE="${BASE_TEMPERATURE:-0.8}"

MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-65536}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-65536}"
TOKEN_CHECK_INTERVAL="${TOKEN_CHECK_INTERVAL:-4096}"
MIN_STEP_TOKENS="${MIN_STEP_TOKENS:-4096}"
MAX_INTERVENTIONS="${MAX_INTERVENTIONS:-5}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.0}"
WAIT_CONF_TAIL_TOKENS="${WAIT_CONF_TAIL_TOKENS:-64}"
VERIFIER_MAX_PROMPT_LENGTH="${VERIFIER_MAX_PROMPT_LENGTH:-16384}"
VERIFIER_MAX_NEW_TOKENS="${VERIFIER_MAX_NEW_TOKENS:-4096}"
VERIFIER_MAX_HINT_TOKENS="${VERIFIER_MAX_HINT_TOKENS:-512}"
STOP_TOKEN_ID="${STOP_TOKEN_ID:-151645}"

LMDEPLOY_BACKEND="${LMDEPLOY_BACKEND:-turbomind}"
LMDEPLOY_SESSION_LEN="${LMDEPLOY_SESSION_LEN:-65536}"
LMDEPLOY_MAX_BATCH_SIZE="${LMDEPLOY_MAX_BATCH_SIZE:-128}"
LMDEPLOY_LOG_LEVEL="${LMDEPLOY_LOG_LEVEL:-WARNING}"

RUN_FULL="${RUN_FULL:-1}"
RUN_HEAD4="${RUN_HEAD4:-1}"
RUN_BASE="${RUN_BASE:-1}"
RUN_FPFIX="${RUN_FPFIX:-1}"
RUN_DROP="${RUN_DROP:-1}"
RUN_FULLMIX="${RUN_FULLMIX:-1}"
RESUME="${RESUME:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

export LMDEPLOY_SKIP_WARMUP="${LMDEPLOY_SKIP_WARMUP:-1}"

# Prompt/system alignment:
# - USE_VERIFIER_SYSTEM_PROMPT=1: prepend VERIFIER_SYSTEM_PROMPT as system msg for the actor
#   (useful when you want <think> tags / match some OpenCompass configs).
# - Set to 0 to evaluate on raw dataset prompts only (closer to RL parquet prompts).
USE_VERIFIER_SYSTEM_PROMPT="${USE_VERIFIER_SYSTEM_PROMPT:-1}"
# Safety guards for offline eval (avoid runaway 65k generations on degenerate samples).
DEGENERATION_GUARD="${DEGENERATION_GUARD:-1}"
STOP_AT_THINK_END="${STOP_AT_THINK_END:-0}"

# Prompt alignment with online OpenCompass math eval.
ONLINE_MATH_PROMPT="${ONLINE_MATH_PROMPT:-0}"
EVAL_PROMPT_FLAGS="${EVAL_PROMPT_FLAGS:-}"
if [[ -z "${EVAL_PROMPT_FLAGS}" ]] && [[ "${ONLINE_MATH_PROMPT}" == "1" ]]; then
  EVAL_PROMPT_FLAGS="--online-math-prompt"
fi

# Optional: align actor transport with online OpenAISDK decode service.
# - pipeline (default): lmdeploy pipeline() in-process.
# - openai: OpenAI-compatible /v1/chat/completions (either external or started in-job).
ACTOR_TRANSPORT="${ACTOR_TRANSPORT:-pipeline}"  # pipeline|openai
ACTOR_API_BASE="${ACTOR_API_BASE:-}"
ACTOR_API_TIMEOUT="${ACTOR_API_TIMEOUT:-600}"
START_ACTOR_API_SERVER="${START_ACTOR_API_SERVER:-0}"
ACTOR_API_PORT="${ACTOR_API_PORT:-0}"
ACTOR_API_TP="${ACTOR_API_TP:-1}"
ACTOR_API_WORKER_NUM="${ACTOR_API_WORKER_NUM:-1}"
ACTOR_API_EXTRA_CLI="${ACTOR_API_EXTRA_CLI:-}"
ACTOR_DISABLE_THINKING="${ACTOR_DISABLE_THINKING:-0}"

_setup_no_proxy() {
  local urls="$1"
  local hosts=""
  # Convert "http://a:1/v1,http://b:2/v1" -> "a,b"
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

_split_jsonl_repeat() {
  local jsonl="$1"
  local out_dir="$2"
  local repeat="$3"
  mkdir -p "${out_dir}"
  JSONL="${jsonl}" OUT_DIR="${out_dir}" REPEAT="${repeat}" python3 - <<'PY'
import os, json

src = os.environ["JSONL"]
out_dir = os.environ["OUT_DIR"]
repeat = int(os.environ.get("REPEAT", "1"))

# Write individual item files (as single-line JSONL) for dynamic scheduling
item_idx = 0
with open(src, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        obj = json.loads(line)
        for r in range(repeat):
            rec = obj
            if repeat > 1:
                if isinstance(obj, dict):
                    rec = dict(obj)
                    origin = rec.get("origin_info")
                    if not isinstance(origin, dict):
                        origin = {"origin_info": origin}
                    else:
                        origin = dict(origin)
                    origin["_orig_line_idx"] = line_idx
                    origin["_repeat_id"] = r
                    rec["origin_info"] = origin
                else:
                    rec = {"prompt": obj, "origin_info": {"_orig_line_idx": line_idx, "_repeat_id": r}}
            # Write as single-line JSONL (eval script expects JSONL format)
            item_file = os.path.join(out_dir, f"item_{item_idx:05d}.jsonl")
            with open(item_file, "w", encoding="utf-8") as of:
                of.write(json.dumps(rec, ensure_ascii=False) + "\n")
            item_idx += 1

# Create immutable full task list (for resume) + mutable working queue.
queue_all_file = os.path.join(out_dir, "task_queue_all.txt")
with open(queue_all_file, "w") as qf:
    for i in range(item_idx):
        qf.write(f"{i:05d}\n")
queue_file = os.path.join(out_dir, "task_queue.txt")
with open(queue_file, "w") as qf:
    for i in range(item_idx):
        qf.write(f"{i:05d}\n")

print(f"OK: {item_idx} items, queue at {queue_file}")
PY
}

_rebuild_task_queue() {
  local items_dir="$1"
  local results_dir="$2"
  [[ -d "${items_dir}" ]] || _die "Missing items_dir: ${items_dir}"
  [[ -d "${results_dir}" ]] || mkdir -p "${results_dir}"

  ITEMS_DIR="${items_dir}" RESULTS_DIR="${results_dir}" python3 - <<'PY'
import os
import re
from pathlib import Path

items_dir = Path(os.environ["ITEMS_DIR"])
results_dir = Path(os.environ["RESULTS_DIR"])

pat = re.compile(r"^item_(\d+)\.jsonl$")
indices = []
for name in os.listdir(items_dir):
    m = pat.match(name)
    if m:
        indices.append(m.group(1))

if not indices:
    raise SystemExit(f"[queue][ERR] No item_*.jsonl found under {items_dir}")

indices = sorted(set(indices), key=int)

queue_all = items_dir / "task_queue_all.txt"
queue_tmp = items_dir / "task_queue.txt.tmp"
queue = items_dir / "task_queue.txt"

done = 0
todo = 0
with queue_tmp.open("w", encoding="utf-8") as f:
    for idx in indices:
        out_file = results_dir / f"item_{idx}.jsonl"
        if out_file.is_file() and out_file.stat().st_size > 0:
            done += 1
            continue
        todo += 1
        f.write(f"{idx}\n")

# Always refresh the immutable full list for sanity/debug.
with queue_all.open("w", encoding="utf-8") as f:
    for idx in indices:
        f.write(f"{idx}\n")

queue_tmp.replace(queue)
print(f"[queue] total={len(indices)} done={done} todo={todo} -> {queue}")
PY
}

# Find the most recent run_dir for a given tag
_find_latest_run_dir() {
  local tag="$1"
  local stem="$2"
  local parent_dir="${OUT_ROOT}/${tag}"
  if [[ ! -d "${parent_dir}" ]]; then
    return 1
  fi
  # Find dirs matching stem.* and get the most recent one
  local latest=""
  for d in "${parent_dir}"/${stem}.*; do
    if [[ -d "$d" ]]; then
      latest="$d"
    fi
  done
  if [[ -n "${latest}" ]]; then
    echo "${latest}"
    return 0
  fi
  return 1
}

# Check if a run is complete (has merged.cv.jsonl and cv_metrics.json)
_is_run_complete() {
  local run_dir="$1"
  [[ -f "${run_dir}/merged.cv.jsonl" ]] && [[ -f "${run_dir}/cv_metrics.json" ]] || return 1
  local items_dir="${run_dir}/items"
  local results_dir="${run_dir}/results"
  if [[ -d "${items_dir}" ]] && [[ -d "${results_dir}" ]]; then
    shopt -s nullglob
    local item_files=("${items_dir}"/item_*.jsonl)
    local result_files=("${results_dir}"/item_*.jsonl)
    shopt -u nullglob
    local total="${#item_files[@]}"
    if [[ "${total}" -gt 0 ]]; then
      local done=0
      for f in "${result_files[@]}"; do
        [[ -s "$f" ]] && ((done++)) || true
      done
      [[ "${done}" -ge "${total}" ]] || return 1
    fi
  fi
  return 0
}

# Check if a shard result is complete (has valid jsonl with expected line count)
_is_shard_complete() {
  local shard_input="$1"
  local shard_output="$2"
  if [[ ! -s "${shard_output}" ]]; then
    return 1
  fi
  local input_lines output_lines
  input_lines=$(wc -l < "${shard_input}")
  output_lines=$(wc -l < "${shard_output}")
  # Output should have at least as many lines as input (may have more due to multi-turn)
  [[ "${output_lines}" -ge "${input_lines}" ]]
}

# Progress monitor function - prints latest log lines periodically
_progress_monitor() {
  local logs_dir="$1"
  local num_workers="$2"
  local monitor_pid_file="$3"
  local total_items="$4"
  local start_time_file="${logs_dir}/../.start_time"

  # Record start time
  date +%s > "${start_time_file}"

  while [[ -f "${monitor_pid_file}" ]]; do
    echo ""
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') Progress =========="

    # Count completed items (one result file per item)
    local completed=0
    local result_dir="${logs_dir}/../results"
    if [[ -d "${result_dir}" ]]; then
      shopt -s nullglob
      local result_files=("${result_dir}"/item_*.jsonl)
      shopt -u nullglob
      for f in "${result_files[@]}"; do
        [[ -s "$f" ]] && ((completed++)) || true
      done
    fi

    local overall_pct=0
    if [[ $total_items -gt 0 ]]; then
      overall_pct=$((completed * 100 / total_items))
    fi

    # Calculate speed and ETA
    local start_time elapsed speed items_per_min eta_str
    start_time=$(cat "${start_time_file}" 2>/dev/null || echo "$(date +%s)")
    elapsed=$(($(date +%s) - start_time))
    if [[ $elapsed -gt 0 ]] && [[ $completed -gt 0 ]]; then
      # Items per minute
      items_per_min=$(python3 -c "print(f'{$completed / $elapsed * 60:.2f}')")
      # ETA calculation
      local remaining=$((total_items - completed))
      if [[ $remaining -gt 0 ]]; then
        local eta_secs=$(python3 -c "print(int($remaining / ($completed / $elapsed)))")
        # Format ETA as HH:MM:SS
        eta_str=$(python3 -c "
eta = $eta_secs
h, m, s = eta // 3600, (eta % 3600) // 60, eta % 60
print(f'{h:d}:{m:02d}:{s:02d}')
")
      else
        eta_str="00:00:00"
      fi
    else
      items_per_min="..."
      eta_str="..."
    fi

    # Show per-GPU status
    for ((i=0; i<num_workers; i++)); do
      local log_file="${logs_dir}/worker_${i}.log"
      local log_content=""
      if [[ -f "${log_file}" ]]; then
        log_content=$(cat "${log_file}" 2>/dev/null || echo "")
      fi

      # Check if this worker is done
      if echo "${log_content}" | grep -q "WORKER_DONE"; then
        echo "  [GPU_${i}] ✅ FINISHED"
      elif echo "${log_content}" | grep -q "Saved"; then
        local last_item
        last_item=$(echo "${log_content}" | grep -oE 'item_[0-9]+' | tail -1 | sed 's/item_//' || echo "?")
        echo "  [GPU_${i}] ✅ DONE (last: item_${last_item})"
      else
        # Get current progress - check for eval progress bar first
        local eval_line
        eval_line=$(echo "${log_content}" | tr '\r' '\n' | grep -E 'eval:.*[0-9]+%' | tail -1 || echo "")
        if [[ -n "$eval_line" ]]; then
          local speed
          speed=$(echo "$eval_line" | grep -oE '[0-9]+\.[0-9]+s/rec' | tail -1 || echo "...")
          local worker_pct
          worker_pct=$(echo "$eval_line" | grep -oE '[0-9]+%' | head -1 || echo "?%")
          echo "  [GPU_${i}] running ${worker_pct} | speed: ${speed}"
        elif echo "${log_content}" | grep -q "Loading checkpoint"; then
          echo "  [GPU_${i}] loading model..."
        else
          echo "  [GPU_${i}] starting..."
        fi
      fi
    done

    echo "-----------------------------------------------------------"
    printf "  Progress: %d%% (%d/%d) | Speed: %s items/min | ETA: %s\n" "${overall_pct}" "${completed}" "${total_items}" "${items_per_min}" "${eta_str}"
    echo "==========================================================="
    sleep "${PROGRESS_INTERVAL}"
  done
}

_run_one_eval() {
  local tag="$1"          # e.g. fpfix/checkpoint-2445
  local jsonl="$2"        # dataset jsonl
  local verifier_lora="$3"
  local verifier_model="$4"
  local run_mode="${5:-${MODE}}"
  local actor_base_model="${6:-${BASE_MODEL}}"

  [[ -f "${jsonl}" ]] || _die "Missing dataset jsonl: ${jsonl}"
  [[ -f "${actor_base_model}/config.json" ]] || _die "Missing actor base model config.json: ${actor_base_model}"
  if [[ -n "${verifier_lora}" ]]; then
    [[ -f "${verifier_lora}/adapter_config.json" ]] || _die "Missing verifier LoRA adapter_config.json: ${verifier_lora}"
  fi
  if [[ -n "${verifier_model}" ]]; then
    [[ -f "${verifier_model}/config.json" ]] || _die "Missing verifier model config.json: ${verifier_model}"
  fi

  local base_name
  base_name="$(basename "${jsonl}")"
  local stem="${base_name%.jsonl}"

  # RESUME: check for existing complete run
  local run_dir=""
  local items_dir=""
  local results_dir=""
  local logs_dir=""

  if [[ "${RESUME}" == "1" ]]; then
    local existing_run
    existing_run=$(_find_latest_run_dir "${tag}" "${stem}") || existing_run=""
    if [[ -n "${existing_run}" ]] && _is_run_complete "${existing_run}"; then
      echo ""
      echo "============================================================"
      echo "[resume] tag=${tag} already complete, skipping"
      echo "[resume] existing: ${existing_run}"
      echo "============================================================"
      return 0
    fi
    if [[ -n "${existing_run}" ]]; then
      run_dir="${existing_run}"
      items_dir="${run_dir}/items"
      results_dir="${run_dir}/results"
      logs_dir="${run_dir}/logs"
      echo ""
      echo "============================================================"
      echo "[resume] tag=${tag} resuming from existing run"
      echo "[resume] run_dir=${run_dir}"
      echo "============================================================"
    fi
  fi

  # Create new run_dir if not resuming
  if [[ -z "${run_dir}" ]]; then
    local run_id
    run_id="$(date +%Y%m%d_%H%M%S)_$RANDOM"
    run_dir="${OUT_ROOT}/${tag}/${stem}.${run_id}"
    items_dir="${run_dir}/items"
    results_dir="${run_dir}/results"
    logs_dir="${run_dir}/logs"
    mkdir -p "${results_dir}" "${logs_dir}"

    echo ""
    echo "============================================================"
    echo "[run] tag=${tag}"
    echo "[run] dataset=${jsonl}"
    echo "[run] run_dir=${run_dir}"
    echo "[run] actor_base_model=${actor_base_model}"
    echo "[run] verifier_model=${verifier_model:-${actor_base_model}}"
    echo "[run] verifier_lora=${verifier_lora:-}"
    echo "[run] workers=${SHARDS} repeat=${REPEAT} mode=${run_mode}"
    echo "============================================================"
  fi

  # Create items if not exist
  if [[ ! -d "${items_dir}" ]] || [[ ! -f "${items_dir}/task_queue_all.txt" ]]; then
    _split_jsonl_repeat "${jsonl}" "${items_dir}" "${REPEAT}"
  fi

  # Rebuild task queue from items + existing results (robust resume even if the queue was already consumed).
  _rebuild_task_queue "${items_dir}" "${results_dir}"

  local total_items remaining_items completed_items
  total_items=$(wc -l < "${items_dir}/task_queue_all.txt")
  remaining_items=$(wc -l < "${items_dir}/task_queue.txt")
  completed_items=$((total_items - remaining_items))

  if [[ "${remaining_items}" -eq 0 ]]; then
    echo "[resume] all items already complete"
  else
    local prev_mode="${MODE}"
    export MODE="${run_mode}"

    # Ensure direct connections to internal endpoints (CompassVerifier + optional actor OpenAI API).
    local no_proxy_urls="${EVAL_URLS}"
    if [[ "${ACTOR_TRANSPORT}" == "openai" ]] && [[ -n "${ACTOR_API_BASE}" ]]; then
      no_proxy_urls="${no_proxy_urls},${ACTOR_API_BASE}"
    fi
    _setup_no_proxy "${no_proxy_urls}"

    # Create worker script for dynamic task fetching
    local worker_script="${run_dir}/worker.sh"
    cat > "${worker_script}" << 'WORKER_EOF'
#!/bin/bash
set -eE
trap 'echo "[ERROR] Worker ${WORKER_ID} failed at line $LINENO" >> "${LOGS_DIR}/worker_${WORKER_ID}.log"' ERR
WORKER_ID="$1"
ITEMS_DIR="$2"
RESULTS_DIR="$3"
LOGS_DIR="$4"
QUEUE_FILE="${ITEMS_DIR}/task_queue.txt"
LOCK_FILE="${ITEMS_DIR}/.queue.lock"

get_next_task() {
  local task_file="${ITEMS_DIR}/.task_${WORKER_ID}.tmp"
  (
    flock -x 200
    if [[ -s "${QUEUE_FILE}" ]]; then
      head -1 "${QUEUE_FILE}" > "${task_file}"
      tail -n +2 "${QUEUE_FILE}" > "${QUEUE_FILE}.tmp"
      mv "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
    else
      rm -f "${task_file}"
    fi
  ) 200>"${LOCK_FILE}"
  if [[ -f "${task_file}" ]]; then
    cat "${task_file}"
    rm -f "${task_file}"
  fi
}

while true; do
  task_idx=$(get_next_task)
  if [[ -z "$task_idx" ]]; then
    echo "WORKER_DONE" >> "${LOGS_DIR}/worker_${WORKER_ID}.log"
    break
  fi

  item_file="${ITEMS_DIR}/item_${task_idx}.jsonl"
  out_file="${RESULTS_DIR}/item_${task_idx}.jsonl"
  tmp_file="${out_file}.tmp"

  # Skip if already done (resume case)
  if [[ -s "${out_file}" ]]; then
    echo "[skip] item_${task_idx} already done" >> "${LOGS_DIR}/worker_${WORKER_ID}.log"
    continue
  fi

  echo "[task] item_${task_idx}" >> "${LOGS_DIR}/worker_${WORKER_ID}.log"

  # Actor transport alignment with online OpenAISDK:
  # - pipeline: in-process lmdeploy pipeline
  # - openai: OpenAI-compatible /v1/chat/completions (external or started in-job)
  actor_args=(--actor-transport "${ACTOR_TRANSPORT}" --actor-api-timeout "${ACTOR_API_TIMEOUT}")
  if [[ "${ACTOR_DISABLE_THINKING}" == "1" ]]; then
    actor_args+=(--actor-disable-thinking)
  fi
  if [[ "${ACTOR_TRANSPORT}" == "openai" ]]; then
    if [[ -z "${ACTOR_API_BASE}" ]]; then
      echo "[ERROR] ACTOR_API_BASE is required when ACTOR_TRANSPORT=openai" >> "${LOGS_DIR}/worker_${WORKER_ID}.log"
      exit 2
    fi
    actor_args+=(--actor-api-base "${ACTOR_API_BASE}")
  fi
  if [[ "${START_ACTOR_API_SERVER}" == "1" ]]; then
    actor_args+=(--start-actor-api-server)
    actor_args+=(--actor-api-port "${ACTOR_API_PORT}")
    actor_args+=(--actor-api-tp "${ACTOR_API_TP}")
    actor_args+=(--actor-api-worker-num "${ACTOR_API_WORKER_NUM}")
    if [[ -n "${ACTOR_API_EXTRA_CLI}" ]]; then
      actor_args+=(--actor-api-extra-cli "${ACTOR_API_EXTRA_CLI}")
    fi
  fi

  # Run eval on single item
  python3 "${EVAL_PY}" \
    --base-model "${ACTOR_BASE_MODEL}" \
    ${VERIFIER_MODEL:+--verifier-model "${VERIFIER_MODEL}"} \
    ${VERIFIER_LORA:+--verifier-lora "${VERIFIER_LORA}"} \
    --prompts-file "${item_file}" \
    --prompt-key "${PROMPT_KEY}" \
    --out-jsonl "${tmp_file}" \
    "${actor_args[@]}" \
    ${EVAL_PROMPT_FLAGS} \
    --mode "${MODE}" \
    $(if [[ "${USE_VERIFIER_SYSTEM_PROMPT}" == "1" ]]; then echo "--use-verifier-system-prompt"; fi) \
    $(if [[ "${DEGENERATION_GUARD}" == "1" ]]; then echo "--degeneration-guard"; fi) \
    $(if [[ "${STOP_AT_THINK_END}" == "1" ]]; then echo "--stop-at-think-end"; fi) \
    --temperature "${BASE_TEMPERATURE}" \
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
    --progress >> "${LOGS_DIR}/worker_${WORKER_ID}.log" 2>&1

  mv -f "${tmp_file}" "${out_file}"
  echo "Saved -> ${out_file}" >> "${LOGS_DIR}/worker_${WORKER_ID}.log"
done
WORKER_EOF
    chmod +x "${worker_script}"

    # Export needed variables for worker script
    # Note: local variables cannot be exported, so we copy to uppercase versions
    export EVAL_PY PROMPT_KEY MODE EVAL_PROMPT_FLAGS USE_VERIFIER_SYSTEM_PROMPT
    export DEGENERATION_GUARD STOP_AT_THINK_END
    export ACTOR_TRANSPORT ACTOR_API_BASE ACTOR_API_TIMEOUT START_ACTOR_API_SERVER
    export ACTOR_API_PORT ACTOR_API_TP ACTOR_API_WORKER_NUM ACTOR_API_EXTRA_CLI ACTOR_DISABLE_THINKING
    export BASE_TEMPERATURE MAX_PROMPT_TOKENS MAX_RESPONSE_TOKENS
    export TOKEN_CHECK_INTERVAL MIN_STEP_TOKENS MAX_INTERVENTIONS
    export CONFIDENCE_THRESHOLD WAIT_CONF_TAIL_TOKENS
    export VERIFIER_MAX_PROMPT_LENGTH VERIFIER_MAX_NEW_TOKENS VERIFIER_MAX_HINT_TOKENS
    export STOP_TOKEN_ID LMDEPLOY_BACKEND LMDEPLOY_SESSION_LEN
    export LMDEPLOY_MAX_BATCH_SIZE LMDEPLOY_LOG_LEVEL
    # Copy local variables to exportable names (must be after the above exports)
    export ACTOR_BASE_MODEL="${actor_base_model}"
    export VERIFIER_MODEL="${verifier_model}"
    export VERIFIER_LORA="${verifier_lora}"

    # Launch workers
    local pids=()
    for ((i=0; i<SHARDS; i++)); do
      local gpu="${GPU_IDS_ARR[$i]}"
      echo "[launch] GPU_${gpu} worker_${i}"
      (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        bash "${worker_script}" "${i}" "${items_dir}" "${results_dir}" "${logs_dir}"
      ) &
      pids+=("$!")
    done

    # Start progress monitor
    local monitor_pid_file="${run_dir}/.monitor.pid"
    echo $$ > "${monitor_pid_file}"
    _progress_monitor "${logs_dir}" "${SHARDS}" "${monitor_pid_file}" "${total_items}" &
    local monitor_pid=$!

    # Wait for all workers
    local failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done

    # Stop progress monitor
    rm -f "${monitor_pid_file}"
    wait "${monitor_pid}" 2>/dev/null || true

    [[ "${failed}" == "0" ]] || _die "Some worker jobs failed for run_dir=${run_dir} (check logs/)"

    export MODE="${prev_mode}"
  fi

  local merged_jsonl="${run_dir}/merged.jsonl"
  # Merge all result files (handle case with no matches)
  shopt -s nullglob
  local result_files=("${results_dir}"/item_*.jsonl)
  shopt -u nullglob
  if [[ ${#result_files[@]} -eq 0 ]]; then
    _die "No result files found in ${results_dir}"
  fi
  cat "${result_files[@]}" > "${merged_jsonl}"
  echo "[merge] ${merged_jsonl} (${#result_files[@]} files)"

  _setup_no_proxy "${EVAL_URLS}"
  local scored_jsonl="${run_dir}/merged.cv.jsonl"
  local metrics_json="${run_dir}/cv_metrics.json"
  echo "[cv_eval] scoring -> ${scored_jsonl}"
  python3 "${CV_EVAL_PY}" \
    --in-jsonl "${merged_jsonl}" \
    --out-jsonl "${scored_jsonl}" \
    --prompt-key "${PROMPT_KEY}" \
    --mode "${run_mode}" \
    --eval-urls "${EVAL_URLS}" \
    --workers 4 \
    --batch-size 64 \
    --progress >"${metrics_json}"
  echo "[cv_eval] metrics: ${metrics_json}"
  cat "${metrics_json}"
}

# --------------------------
# Task list (latest ckpt per group + base)
# --------------------------

_latest_checkpoint_dir() {
  local root="$1"
  local override="${2:-}"

  if [[ -z "${root}" ]]; then
    _die "Empty root for _latest_checkpoint_dir"
  fi

  # If caller passed a direct checkpoint dir, accept it.
  if [[ -d "${root}" ]] && [[ "$(basename "${root}")" == checkpoint-* ]] && [[ -f "${root}/adapter_config.json" ]]; then
    echo "${root}"
    return 0
  fi

  if [[ -n "${override}" ]]; then
    local p="${root}/checkpoint-${override}"
    [[ -d "${p}" ]] || _die "Override ckpt not found: ${p}"
    echo "${p}"
    return 0
  fi

  local best=""
  local best_n=-1
  shopt -s nullglob
  local d
  for d in "${root}"/checkpoint-*; do
    [[ -d "${d}" ]] || continue
    local bn n
    bn="$(basename "${d}")"
    n="${bn#checkpoint-}"
    if [[ "${n}" =~ ^[0-9]+$ ]]; then
      if (( n > best_n )); then
        best_n="${n}"
        best="${d}"
      fi
    fi
  done
  shopt -u nullglob

  [[ -n "${best}" ]] || _die "No checkpoint-* dirs found under ${root}"
  echo "${best}"
}

main() {
  mkdir -p "${OUT_ROOT}"

  local fpfix_ckpt_dir
  fpfix_ckpt_dir="$(_latest_checkpoint_dir "${FPFIX_ROOT}" "${FPFIX_CKPT:-}")"
  local fpfix_ckpt_name
  fpfix_ckpt_name="$(basename "${fpfix_ckpt_dir}")"

  local drop_ckpt_dir
  drop_ckpt_dir="$(_latest_checkpoint_dir "${DROP_ROOT}" "${DROP_CKPT:-}")"
  local drop_ckpt_name
  drop_ckpt_name="$(basename "${drop_ckpt_dir}")"

  echo "[ckpt] fpfix_latest=${fpfix_ckpt_name} (${fpfix_ckpt_dir})"
  echo "[ckpt] drop_latest=${drop_ckpt_name} (${drop_ckpt_dir})"

  # To avoid RESUME mistakenly skipping a run after prompt/style changes, encode prompt style in tag.
  local prompt_tag=""
  if [[ "${ONLINE_MATH_PROMPT}" == "1" ]]; then
    prompt_tag="_online_math"
  fi

  if [[ "${RUN_HEAD4}" == "1" ]]; then
    if [[ "${RUN_BASE}" == "1" ]]; then
      _run_one_eval \
        "aime2025-I_head4_lmdeploy64k_local/base/control${prompt_tag}" \
        "${DATA_HEAD4}" \
        "" \
        "" \
        "control"
    fi

    if [[ "${RUN_FPFIX}" == "1" ]]; then
      _run_one_eval \
        "aime2025-I_head4_lmdeploy64k_local/fpfix/${fpfix_ckpt_name}${prompt_tag}" \
        "${DATA_HEAD4}" \
        "${fpfix_ckpt_dir}" \
        "" \
        "${MODE}"
    fi

    if [[ "${RUN_DROP}" == "1" ]]; then
      _run_one_eval \
        "aime2025-I_head4_lmdeploy64k_local/drop/${drop_ckpt_name}${prompt_tag}" \
        "${DATA_HEAD4}" \
        "${drop_ckpt_dir}" \
        "" \
        "${MODE}"
    fi

    if [[ "${RUN_FULLMIX}" == "1" ]]; then
      _run_one_eval \
        "aime2025-I_head4_lmdeploy64k_local/fullmix/hf-181_actor181${prompt_tag}" \
        "${DATA_HEAD4}" \
        "" \
        "${FULLMIX_VERIFIER_MODEL}" \
        "${MODE}" \
        "${FULLMIX_ACTOR_MODEL}"
    fi
  fi

  if [[ "${RUN_FULL}" == "1" ]]; then
    if [[ "${RUN_BASE}" == "1" ]]; then
      _run_one_eval \
        "aime2025-I_lmdeploy64k_local/base/control${prompt_tag}" \
        "${DATA_FULL}" \
        "" \
        "" \
        "control"
    fi

    if [[ "${RUN_FPFIX}" == "1" ]]; then
      _run_one_eval \
        "aime2025-I_lmdeploy64k_local/fpfix/${fpfix_ckpt_name}${prompt_tag}" \
        "${DATA_FULL}" \
        "${fpfix_ckpt_dir}" \
        "" \
        "${MODE}"
    fi

    if [[ "${RUN_DROP}" == "1" ]]; then
      _run_one_eval \
        "aime2025-I_lmdeploy64k_local/drop/${drop_ckpt_name}${prompt_tag}" \
        "${DATA_FULL}" \
        "${drop_ckpt_dir}" \
        "" \
        "${MODE}"
    fi

    if [[ "${RUN_FULLMIX}" == "1" ]]; then
      _run_one_eval \
        "aime2025-I_lmdeploy64k_local/fullmix/hf-181_actor181${prompt_tag}" \
        "${DATA_FULL}" \
        "" \
        "${FULLMIX_VERIFIER_MODEL}" \
        "${MODE}" \
        "${FULLMIX_ACTOR_MODEL}"
    fi
  fi
}

main "$@"
