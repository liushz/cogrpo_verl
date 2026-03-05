#!/usr/bin/env bash
set -euo pipefail

# Local (single-node) XTuner run script for CoGRPO v2 by_step + cf_branch on an 8-GPU machine.
#
# Supports:
# - nodec: verifier=actor (no verifier LoRA training)
# - dec:   enable verifier LoRA online training path
#
# Defaults align with VERL `cfK2E2` settings for local 8-GPU debug:
# - GLOBAL_BATCH_SIZE=8
# - PROMPT_REPEAT_K=16 (dual-stream: control 8 + exp 8)
# - ROLLOUT_STEPS=4
# - TRAIN_OPTIMIZER_STEPS=4
# - MAX_RESPONSE_LENGTH=32768 (do not change long output length)
# - ACTOR_UPDATE_STREAMS=both
# - USE_CURRICULUM_WEIGHTING=1 (0.3 -> 0.5)
# - COGRPO_CF_BRANCH_K=2 / COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE=2
#
# Usage:
#   bash run_local_xtuner_cf_branch_8gpu_bsz8_roll4_step4.sh nodec
#   bash run_local_xtuner_cf_branch_8gpu_bsz8_roll4_step4.sh dec
#
# Override common paths/knobs via env:
#   XTUNER_DIR, MODEL_PATH, DATASET_PATH, CONDA_ENV_PATH, LMDEPLOY_PATH_OVERRIDE, INFER_BACKEND
#   REWARD_MODEL_URLS, REWARD_NUM_ACTORS, REWARD_TIMEOUT_S, REWARD_RETRIES_PER_URL

mode="${1:-nodec}"
if [ "${mode}" != "nodec" ] && [ "${mode}" != "dec" ]; then
  echo "[xtuner-local8] ERROR: mode must be 'nodec' or 'dec', got: ${mode}" >&2
  exit 2
fi

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

is_truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"
export REPRO_ROOT="${REPRO_ROOT:-${repo_dir}}"

xtuner_dir="${XTUNER_DIR:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner}"
xtuner_branch_expected="${XTUNER_BRANCH_EXPECTED:-cogrpo-v2-xtuner-control-exp}"

if ! git -C "${xtuner_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[xtuner-local8] ERROR: not a git repo: ${xtuner_dir}" >&2
  exit 1
fi

xtuner_branch="$(git -C "${xtuner_dir}" rev-parse --abbrev-ref HEAD)"
xtuner_commit="$(git -C "${xtuner_dir}" rev-parse --short HEAD)"
if [ "${xtuner_branch}" != "${xtuner_branch_expected}" ]; then
  echo "[xtuner-local8] ERROR: expected xtuner branch ${xtuner_branch_expected}, got ${xtuner_branch}" >&2
  echo "[xtuner-local8] Hint: cd ${xtuner_dir} && git checkout ${xtuner_branch_expected}" >&2
  exit 1
fi

# Paths (override as needed)
model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"
dataset_path="${DATASET_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-dapo-math-17k.jsonl}"
conda_env_path="${CONDA_ENV_PATH:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro}"
lmdeploy_path="${LMDEPLOY_PATH_OVERRIDE:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/lmdeploy_manually_fp8}"
infer_backend="${INFER_BACKEND:-lmdeploy}"

# If CUDA_VISIBLE_DEVICES is not set, default to 8 GPUs.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi

# Prefer running under the given conda env (can be disabled).
if [ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ] && [ -f /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
  conda activate "${conda_env_path}"
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "[xtuner-local8] ERROR: ray not found in current env. Please activate your conda env first (or set CONDA_ENV_PATH)." >&2
  exit 127
fi

export LMDEPLOY_PATH="${lmdeploy_path}"
if [ "${infer_backend}" = "lmdeploy" ] && [ ! -d "${LMDEPLOY_PATH}" ]; then
  echo "[xtuner-local8] ERROR: LMDEPLOY_PATH not found: ${LMDEPLOY_PATH}" >&2
  exit 2
fi

# Core run knobs
export NODE_COUNT=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export NUM_WORKERS="${NUM_WORKERS:-8}"
export RAY_MAX_CONCURRENCY="${RAY_MAX_CONCURRENCY:-1024}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

# Async rollout (DataFlow partial rollout)
# Set ENABLE_PARTIAL_ROLLOUT=0 to disable and fallback to synchronous mode.
export ENABLE_PARTIAL_ROLLOUT="${ENABLE_PARTIAL_ROLLOUT:-1}"
export STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0.0}"
export TAIL_BATCH_CANDIDATE_STEPS="${TAIL_BATCH_CANDIDATE_STEPS:-4}"
export TAIL_BATCH_TRIGGER_SIZE="${TAIL_BATCH_TRIGGER_SIZE:-0}"

export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export PROMPT_REPEAT_K="${PROMPT_REPEAT_K:-16}"
export ROLLOUT_STEPS="${ROLLOUT_STEPS:-4}"
export TRAIN_OPTIMIZER_STEPS="${TRAIN_OPTIMIZER_STEPS:-4}"
export HF_INTERVAL="${HF_INTERVAL:-0}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
export PACK_MAX_LENGTH="${PACK_MAX_LENGTH:-35840}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"

# Reward endpoints (CompassVerifier OpenAI-compatible /v1). Default is 0/1 reward in xtuner judger.
export REWARD_MODEL_URLS="${REWARD_MODEL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"
export REWARD_MODEL_KEY="${REWARD_MODEL_KEY:-EMPTY}"
if [ -z "$(printf '%s' "${REWARD_MODEL_KEY}" | tr -d '[:space:]')" ]; then
  export REWARD_MODEL_KEY="EMPTY"
fi
export REWARD_NUM_ACTORS="${REWARD_NUM_ACTORS:-4}"
export REWARD_TIMEOUT_S="${REWARD_TIMEOUT_S:-120}"
export REWARD_RETRIES_PER_URL="${REWARD_RETRIES_PER_URL:-2}"
export COMPASSVERIFIER_DEBUG="${COMPASSVERIFIER_DEBUG:-1}"
export COMPASSVERIFIER_DEBUG_MAX_CALLS="${COMPASSVERIFIER_DEBUG_MAX_CALLS:-20}"

# CoGRPO rollout (by_step + cf_branch)
export COGRPO_ENABLE="${COGRPO_ENABLE:-1}"
export COGRPO_TOKEN_CHECK_INTERVAL="${COGRPO_TOKEN_CHECK_INTERVAL:-2048}"
export COGRPO_MIN_STEP_TOKENS="${COGRPO_MIN_STEP_TOKENS:-2048}"
export COGRPO_MAX_INTERVENTIONS="${COGRPO_MAX_INTERVENTIONS:-2}"
export COGRPO_CONFIDENCE_THRESHOLD="${COGRPO_CONFIDENCE_THRESHOLD:-0.0}"
export ACTOR_UPDATE_STREAMS="${ACTOR_UPDATE_STREAMS:-both}"
export USE_CURRICULUM_WEIGHTING="${USE_CURRICULUM_WEIGHTING:-1}"
export CONTROL_GROUP_WEIGHT="${CONTROL_GROUP_WEIGHT:-0.5}"
export CURRICULUM_START_WEIGHT="${CURRICULUM_START_WEIGHT:-0.3}"
export CURRICULUM_END_WEIGHT="${CURRICULUM_END_WEIGHT:-0.5}"

export COGRPO_CF_BRANCH_PROB="${COGRPO_CF_BRANCH_PROB:-1.0}"
export COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE="${COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE:-2}"
export COGRPO_CF_BRANCH_STATE_HASH_MOD="${COGRPO_CF_BRANCH_STATE_HASH_MOD:-1024}"
export COGRPO_CF_BRANCH_K="${COGRPO_CF_BRANCH_K:-2}"
export COGRPO_VERIFIER_REWARD_MODE="${COGRPO_VERIFIER_REWARD_MODE:-headroom}"
export COGRPO_VERIFIER_REWARD_HEADROOM_MIN="${COGRPO_VERIFIER_REWARD_HEADROOM_MIN:-0.05}"
export COGRPO_VERIFIER_REWARD_IMPROVE_COEF="${COGRPO_VERIFIER_REWARD_IMPROVE_COEF:-1.0}"
export COGRPO_STRICT_DUAL_STREAM="${COGRPO_STRICT_DUAL_STREAM:-1}"
export COGRPO_VERIFIER_UPDATE_BASE="${COGRPO_VERIFIER_UPDATE_BASE:-0}"
export COGRPO_STRICT_VERIFIER_SYNC="${COGRPO_STRICT_VERIFIER_SYNC:-1}"
export COGRPO_VERIFIER_SERVER_URLS="${COGRPO_VERIFIER_SERVER_URLS:-${VERIFIER_SERVER_URLS:-}}"
export COGRPO_VERIFIER_SERVER_URL_DICT="${COGRPO_VERIFIER_SERVER_URL_DICT:-${VERIFIER_SERVER_URL_DICT:-}}"

export COGRPO_KEEP_CF_ROLLOUTS="${COGRPO_KEEP_CF_ROLLOUTS:-0}"
export COGRPO_VERIFIER_SKIP_TRUNCATED="${COGRPO_VERIFIER_SKIP_TRUNCATED:-1}"

# Debug dumps / prints (enabled by default for debug phase).
export COGRPO_DUMP_TRAJECTORY_EXTRA="${COGRPO_DUMP_TRAJECTORY_EXTRA:-1}"
export COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES="${COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES:-16}"
export COGRPO_DUMP_VERL_COMPAT_FIELDS="${COGRPO_DUMP_VERL_COMPAT_FIELDS:-1}"
export COGRPO_PRINT_TRAJECTORY_DEBUG="${COGRPO_PRINT_TRAJECTORY_DEBUG:-1}"
export COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES="${COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES:-4}"
export COGRPO_DEBUG_HINT_CONTEXT="${COGRPO_DEBUG_HINT_CONTEXT:-1}"
export COGRPO_DEBUG_HINT_CONTEXT_TOKENS="${COGRPO_DEBUG_HINT_CONTEXT_TOKENS:-96}"
export XTUNER_PRETTY_TRAJECTORY_JSONL="${XTUNER_PRETTY_TRAJECTORY_JSONL:-0}"

# Verifier debug (off by default): logs verifier decision/parsing/insertion gates.
# In this local debug script, keep verifier debug on by default.
export COGRPO_VERIFIER_DEBUG="${COGRPO_VERIFIER_DEBUG:-1}"
export COGRPO_VERIFIER_DEBUG_MAX_CALLS="${COGRPO_VERIFIER_DEBUG_MAX_CALLS:-200}"
export COGRPO_VERIFIER_DEBUG_TEXT_CHARS="${COGRPO_VERIFIER_DEBUG_TEXT_CHARS:-640}"
export COGRPO_VERIFIER_DEBUG_DUMP="${COGRPO_VERIFIER_DEBUG_DUMP:-1}"
export COGRPO_VERIFIER_DEBUG_DUMP_MAX_CALLS="${COGRPO_VERIFIER_DEBUG_DUMP_MAX_CALLS:-200}"
export COGRPO_VERIFIER_DEBUG_DUMP_DIR="${COGRPO_VERIFIER_DEBUG_DUMP_DIR:-/tmp/xtuner_verifier_debug}"
export COGRPO_VERIFIER_DEBUG_DUMP_FULL_TEXT="${COGRPO_VERIFIER_DEBUG_DUMP_FULL_TEXT:-0}"
# Verifier parser mode:
# - strict      : use XTuner strict tail-line parser only
# - auto        : prefer external VERL parser, fallback to strict parser
# - verl_compat : explicit permissive fallback (for debugging only)
export COGRPO_VERIFIER_PARSER_MODE="${COGRPO_VERIFIER_PARSER_MODE:-auto}"
export COGRPO_VERL_HINT_INJECTION_PY="${COGRPO_VERL_HINT_INJECTION_PY:-${REPRO_ROOT}/verl/workers/rollout/vllm_rollout/verifier_hint_injection.py}"
export COGRPO_VERL_ROLLOUT_SPMD_PY="${COGRPO_VERL_ROLLOUT_SPMD_PY:-${REPRO_ROOT}/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py}"

# Dual-stream split preflight (CONTROL + EXP).
if ! [[ "${PROMPT_REPEAT_K}" =~ ^[0-9]+$ ]] || [ "${PROMPT_REPEAT_K}" -le 0 ]; then
  echo "[xtuner-local8] ERROR: PROMPT_REPEAT_K must be positive integer, got '${PROMPT_REPEAT_K}'" >&2
  exit 2
fi
default_control_k=$((PROMPT_REPEAT_K / 2))
default_exp_k=$((PROMPT_REPEAT_K - default_control_k))
control_k_raw="${COGRPO_CONTROL_K:-${CO_GRPO_CONTROL_K:-${default_control_k}}}"
exp_k_raw="${COGRPO_EXP_K:-${CO_GRPO_EXP_K:-${default_exp_k}}}"
if ! [[ "${control_k_raw}" =~ ^-?[0-9]+$ ]]; then
  echo "[xtuner-local8] ERROR: COGRPO_CONTROL_K must be integer, got '${control_k_raw}'" >&2
  exit 2
fi
if ! [[ "${exp_k_raw}" =~ ^-?[0-9]+$ ]]; then
  echo "[xtuner-local8] ERROR: COGRPO_EXP_K must be integer, got '${exp_k_raw}'" >&2
  exit 2
fi
control_k_val="${control_k_raw}"
exp_k_val="${exp_k_raw}"
if [ "${control_k_val}" -lt 0 ] || [ "${exp_k_val}" -lt 0 ]; then
  echo "[xtuner-local8] ERROR: COGRPO_CONTROL_K/COGRPO_EXP_K must be >= 0" >&2
  exit 2
fi
dual_stream_enabled=0
effective_control_k=0
effective_exp_k="${PROMPT_REPEAT_K}"
if [ "${control_k_val}" -gt 0 ] || [ "${exp_k_val}" -gt 0 ]; then
  dual_stream_enabled=1
  effective_control_k="${control_k_val}"
  effective_exp_k="${exp_k_val}"
  if [ "${effective_exp_k}" -le 0 ]; then
    effective_exp_k=$((PROMPT_REPEAT_K - effective_control_k))
  fi
  if [ $((effective_control_k + effective_exp_k)) -ne "${PROMPT_REPEAT_K}" ]; then
    echo "[xtuner-local8] ERROR: invalid dual-stream split: control_k=${effective_control_k}, exp_k=${effective_exp_k}, PROMPT_REPEAT_K=${PROMPT_REPEAT_K}" >&2
    echo "[xtuner-local8] Hint: set COGRPO_CONTROL_K + COGRPO_EXP_K == PROMPT_REPEAT_K" >&2
    exit 2
  fi
fi

if [ "${mode}" = "dec" ]; then
  export COGRPO_VERIFIER_LORA_ENABLE="${COGRPO_VERIFIER_LORA_ENABLE:-1}"
  default_verifier_lora_path="${COGRPO_VERIFIER_LORA_PATH_DEFAULT:-/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-verifier_sft_train_55_waitx2_fpfix_plus_v4ckptwait_20260223-02241157/checkpoint-2445}"
  export COGRPO_VERIFIER_LORA_PATH="${COGRPO_VERIFIER_LORA_PATH:-${default_verifier_lora_path}}"
  export COGRPO_VERIFIER_LORA_NAME="${COGRPO_VERIFIER_LORA_NAME:-verifier_lora}"
  export COGRPO_VERIFIER_USE_LORA_ADAPTER_MODEL="${COGRPO_VERIFIER_USE_LORA_ADAPTER_MODEL:-1}"
  export COGRPO_VERIFIER_MODEL="${COGRPO_VERIFIER_MODEL:-${COGRPO_VERIFIER_LORA_NAME}}"
  export COGRPO_VERIFIER_STRICT_MODEL_CHECK="${COGRPO_VERIFIER_STRICT_MODEL_CHECK:-1}"
  export LMDEPLOY_ADAPTERS_JSON="${LMDEPLOY_ADAPTERS_JSON:-{\"${COGRPO_VERIFIER_LORA_NAME}\":\"${COGRPO_VERIFIER_LORA_PATH}\"}}"
  export COGRPO_VERIFIER_LORA_RANK="${COGRPO_VERIFIER_LORA_RANK:-64}"
  export COGRPO_VERIFIER_LORA_ALPHA="${COGRPO_VERIFIER_LORA_ALPHA:-128}"
  export COGRPO_VERIFIER_LORA_DROPOUT="${COGRPO_VERIFIER_LORA_DROPOUT:-0.05}"
  export COGRPO_VERIFIER_LORA_TARGET_MODULES="${COGRPO_VERIFIER_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
  export COGRPO_VERIFIER_LR="${COGRPO_VERIFIER_LR:-1e-5}"
  export COGRPO_VERIFIER_LORA_SYNC_FREQ="${COGRPO_VERIFIER_LORA_SYNC_FREQ:-1}"
  if ! [[ "${COGRPO_VERIFIER_LORA_SYNC_FREQ}" =~ ^[0-9]+$ ]]; then
    echo "[xtuner-local8] ERROR: COGRPO_VERIFIER_LORA_SYNC_FREQ must be integer >= 0, got '${COGRPO_VERIFIER_LORA_SYNC_FREQ}'" >&2
    exit 2
  fi
  verifier_sync_urls_trimmed="$(printf '%s' "${COGRPO_VERIFIER_SERVER_URLS}" | tr -d '[:space:]')"
  verifier_sync_url_dict_trimmed="$(printf '%s' "${COGRPO_VERIFIER_SERVER_URL_DICT}" | tr -d '[:space:]')"
  if is_truthy "${COGRPO_VERIFIER_LORA_ENABLE}" && is_truthy "${COGRPO_STRICT_VERIFIER_SYNC}" && \
     [ "${COGRPO_VERIFIER_LORA_SYNC_FREQ}" -le 0 ]; then
    echo "[xtuner-local8] ERROR: strict verifier sync requires COGRPO_VERIFIER_LORA_SYNC_FREQ > 0 when dec + LoRA is enabled." >&2
    echo "[xtuner-local8] Hint: this run would train verifier LoRA but never sync it to verifier rollout inference." >&2
    exit 2
  fi
  if is_truthy "${COGRPO_VERIFIER_LORA_ENABLE}" && is_truthy "${COGRPO_STRICT_VERIFIER_SYNC}" && \
     [ "${COGRPO_VERIFIER_LORA_SYNC_FREQ}" -gt 0 ] && [ -z "${verifier_sync_urls_trimmed}" ] && [ -z "${verifier_sync_url_dict_trimmed}" ]; then
    echo "[xtuner-local8] ERROR: strict verifier sync requires COGRPO_VERIFIER_SERVER_URLS or COGRPO_VERIFIER_SERVER_URL_DICT when dec + LoRA sync is enabled." >&2
    echo "[xtuner-local8] Hint: provide verifier rollout URLs, or set COGRPO_STRICT_VERIFIER_SYNC=0 for explicit debug-only no-sync runs." >&2
    exit 2
  fi
else
  export COGRPO_VERIFIER_LORA_ENABLE="${COGRPO_VERIFIER_LORA_ENABLE:-0}"
  unset COGRPO_VERIFIER_MODEL
  unset COGRPO_VERIFIER_LORA_NAME
  unset COGRPO_VERIFIER_USE_LORA_ADAPTER_MODEL
  unset COGRPO_VERIFIER_STRICT_MODEL_CHECK
fi

echo "[xtuner-local8] mode=${mode} xtuner=${xtuner_branch}@${xtuner_commit}"
echo "[xtuner-local8] backend=${infer_backend} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[xtuner-local8] model=${model_path}"
echo "[xtuner-local8] dataset=${dataset_path}"
echo "[xtuner-local8] gbs=${GLOBAL_BATCH_SIZE} prk=${PROMPT_REPEAT_K} roll_steps=${ROLLOUT_STEPS} opt_steps=${TRAIN_OPTIMIZER_STEPS}"
echo "[xtuner-local8] actor_update_streams=${ACTOR_UPDATE_STREAMS} curriculum=${USE_CURRICULUM_WEIGHTING} control_w=${CONTROL_GROUP_WEIGHT} curriculum_w=${CURRICULUM_START_WEIGHT}->${CURRICULUM_END_WEIGHT}"
echo "[xtuner-local8] verifier_reward=${COGRPO_VERIFIER_REWARD_MODE} headroom_min=${COGRPO_VERIFIER_REWARD_HEADROOM_MIN} improve_coef=${COGRPO_VERIFIER_REWARD_IMPROVE_COEF}"
echo "[xtuner-local8] strict_dual_stream=${COGRPO_STRICT_DUAL_STREAM} strict_verifier_sync=${COGRPO_STRICT_VERIFIER_SYNC} verifier_update_base=${COGRPO_VERIFIER_UPDATE_BASE}"
if [ "${dual_stream_enabled}" = "1" ]; then
  echo "[xtuner-local8] dual_stream=on control_k=${effective_control_k} exp_k=${effective_exp_k}"
else
  echo "[xtuner-local8] dual_stream=off (all EXP); set COGRPO_CONTROL_K/COGRPO_EXP_K to enable split"
fi
echo "[xtuner-local8] async=${ENABLE_PARTIAL_ROLLOUT} staleness=${STALENESS_THRESHOLD} tail_steps=${TAIL_BATCH_CANDIDATE_STEPS} max_concurrent=${MAX_CONCURRENT} gpu_mem_util=${GPU_MEM_UTIL}"
echo "[xtuner-local8] verifier_parser_mode=${COGRPO_VERIFIER_PARSER_MODE}"
echo "[xtuner-local8] verifier_debug=${COGRPO_VERIFIER_DEBUG} dump=${COGRPO_VERIFIER_DEBUG_DUMP} dump_dir=${COGRPO_VERIFIER_DEBUG_DUMP_DIR}"
if [ "${mode}" = "dec" ]; then
  if [ -n "${COGRPO_VERIFIER_LORA_PATH:-}" ] && [ -d "${COGRPO_VERIFIER_LORA_PATH}" ]; then
    echo "[xtuner-local8] verifier_lora_path=${COGRPO_VERIFIER_LORA_PATH}"
  else
    echo "[xtuner-local8] WARN verifier_lora_path_missing=${COGRPO_VERIFIER_LORA_PATH:-<empty>}"
  fi
  echo "[xtuner-local8] verifier_model=${COGRPO_VERIFIER_MODEL:-<empty>} lora_name=${COGRPO_VERIFIER_LORA_NAME:-<empty>} strict_model_check=${COGRPO_VERIFIER_STRICT_MODEL_CHECK:-0}"
  echo "[xtuner-local8] verifier_lora_modules=${COGRPO_VERIFIER_LORA_TARGET_MODULES}"
  if [ -n "${COGRPO_VERIFIER_SERVER_URL_DICT}" ]; then
    echo "[xtuner-local8] verifier_server_url_dict_set=1"
  else
    echo "[xtuner-local8] verifier_server_urls=${COGRPO_VERIFIER_SERVER_URLS:-<empty>}"
  fi
fi
echo "[xtuner-local8] reward_urls=${REWARD_MODEL_URLS}"
if [ "${REWARD_MODEL_KEY}" = "EMPTY" ]; then
  echo "[xtuner-local8] reward_key=EMPTY"
else
  echo "[xtuner-local8] reward_key_set=1 len=${#REWARD_MODEL_KEY}"
fi

cd "${xtuner_dir}"
bash examples/v1/scripts/run_rl.sh \
  examples/v1/config/rl_qwen25_7B_cogrpo_cf_branch_async.py \
  "${infer_backend}" \
  "${model_path}" \
  "${dataset_path}" \
  "" \
  gpu \
  8
