#!/usr/bin/env bash
set -euo pipefail

# Local (single-node) XTuner CoGRPO v2 by_step + cf_branch (exp-only) in **decoupled** mode.
#
# Requires external verifier rollout server URLs:
# - COGRPO_VERIFIER_SERVER_URLS (comma-separated) or COGRPO_VERIFIER_SERVER_URL_DICT (json dict).
#
# This script only starts the RL training job; you must start verifier servers separately.

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

xtuner_dir="${XTUNER_DIR:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner}"
xtuner_branch_expected="${XTUNER_BRANCH_EXPECTED:-cogrpo-v2-xtuner-control-exp}"

if ! git -C "${xtuner_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[xtuner-local8-dec] ERROR: not a git repo: ${xtuner_dir}" >&2
  exit 1
fi

xtuner_branch="$(git -C "${xtuner_dir}" rev-parse --abbrev-ref HEAD)"
xtuner_commit="$(git -C "${xtuner_dir}" rev-parse --short HEAD)"
if [ "${xtuner_branch}" != "${xtuner_branch_expected}" ]; then
  echo "[xtuner-local8-dec] ERROR: expected xtuner branch ${xtuner_branch_expected}, got ${xtuner_branch}" >&2
  echo "[xtuner-local8-dec] Hint: cd ${xtuner_dir} && git checkout ${xtuner_branch_expected}" >&2
  exit 1
fi

verifier_urls_env="${COGRPO_VERIFIER_SERVER_URLS:-${VERIFIER_SERVER_URLS:-}}"
verifier_url_dict_env="${COGRPO_VERIFIER_SERVER_URL_DICT:-${VERIFIER_SERVER_URL_DICT:-}}"
if [ -z "${verifier_urls_env}" ] && [ -z "${verifier_url_dict_env}" ]; then
  echo "[xtuner-local8-dec] ERROR: missing verifier server URLs." >&2
  echo "[xtuner-local8-dec] Provide COGRPO_VERIFIER_SERVER_URLS or COGRPO_VERIFIER_SERVER_URL_DICT." >&2
  exit 2
fi

model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"
dataset_path="${DATASET_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-dapo-math-17k.jsonl}"
conda_env_path="${CONDA_ENV_PATH:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro}"
lmdeploy_path="${LMDEPLOY_PATH_OVERRIDE:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/lmdeploy_manually_fp8}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi

if [ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ] && [ -f /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
  conda activate "${conda_env_path}"
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "[xtuner-local8-dec] ERROR: ray not found in current env. Please activate your conda env first." >&2
  exit 127
fi

export LMDEPLOY_PATH="${lmdeploy_path}"
if [ ! -d "${LMDEPLOY_PATH}" ]; then
  echo "[xtuner-local8-dec] ERROR: LMDEPLOY_PATH not found: ${LMDEPLOY_PATH}" >&2
  exit 2
fi
export PYTHONPATH="${LMDEPLOY_PATH}:${PYTHONPATH:-}"

export NODE_COUNT=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1

export NUM_WORKERS="${NUM_WORKERS:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export PROMPT_REPEAT_K="${PROMPT_REPEAT_K:-16}"
export ROLLOUT_STEPS="${ROLLOUT_STEPS:-5}"
export TRAIN_OPTIMIZER_STEPS="${TRAIN_OPTIMIZER_STEPS:-1}"
export HF_INTERVAL="${HF_INTERVAL:-0}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
export PACK_MAX_LENGTH="${PACK_MAX_LENGTH:-35840}"

export COGRPO_VERIFIER_SERVER_URLS="${verifier_urls_env}"
export COGRPO_VERIFIER_SERVER_URL_DICT="${verifier_url_dict_env}"

export COGRPO_ENABLE="${COGRPO_ENABLE:-1}"
export COGRPO_TOKEN_CHECK_INTERVAL="${COGRPO_TOKEN_CHECK_INTERVAL:-2048}"
export COGRPO_MIN_STEP_TOKENS="${COGRPO_MIN_STEP_TOKENS:-2048}"
export COGRPO_MAX_INTERVENTIONS="${COGRPO_MAX_INTERVENTIONS:-2}"
export COGRPO_CONFIDENCE_THRESHOLD="${COGRPO_CONFIDENCE_THRESHOLD:-0.0}"
export COGRPO_CF_BRANCH_PROB="${COGRPO_CF_BRANCH_PROB:-1.0}"
export COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE="${COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE:-2}"
export COGRPO_CF_BRANCH_STATE_HASH_MOD="${COGRPO_CF_BRANCH_STATE_HASH_MOD:-1024}"
export COGRPO_CF_BRANCH_K="${COGRPO_CF_BRANCH_K:-2}"
export COGRPO_KEEP_CF_ROLLOUTS="${COGRPO_KEEP_CF_ROLLOUTS:-0}"
export COGRPO_VERIFIER_SKIP_TRUNCATED="${COGRPO_VERIFIER_SKIP_TRUNCATED:-1}"

# Decoupled: enable verifier LoRA and sync it to verifier servers every step by default.
export COGRPO_VERIFIER_LORA_ENABLE="${COGRPO_VERIFIER_LORA_ENABLE:-1}"
export COGRPO_VERIFIER_LORA_RANK="${COGRPO_VERIFIER_LORA_RANK:-8}"
export COGRPO_VERIFIER_LORA_ALPHA="${COGRPO_VERIFIER_LORA_ALPHA:-16}"
export COGRPO_VERIFIER_LORA_DROPOUT="${COGRPO_VERIFIER_LORA_DROPOUT:-0.05}"
export COGRPO_VERIFIER_LR="${COGRPO_VERIFIER_LR:-1e-5}"
export COGRPO_VERIFIER_LORA_SYNC_FREQ="${COGRPO_VERIFIER_LORA_SYNC_FREQ:-1}"
export COGRPO_VERIFIER_LORA_SAVE_FREQ="${COGRPO_VERIFIER_LORA_SAVE_FREQ:-0}"

export COGRPO_VERIFIER_MAX_NEW_TOKENS="${COGRPO_VERIFIER_MAX_NEW_TOKENS:-4096}"
export COGRPO_VERIFIER_MAX_PROMPT_LENGTH="${COGRPO_VERIFIER_MAX_PROMPT_LENGTH:-16384}"
export COGRPO_VERIFIER_MAX_HINT_TOKENS="${COGRPO_VERIFIER_MAX_HINT_TOKENS:-512}"
export COGRPO_ESTIMATED_HINT_TOKENS="${COGRPO_ESTIMATED_HINT_TOKENS:-512}"

# Debug dumps / prints.
export COGRPO_DUMP_TRAJECTORY_EXTRA="${COGRPO_DUMP_TRAJECTORY_EXTRA:-1}"
export COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES="${COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES:-16}"
export COGRPO_PRINT_TRAJECTORY_DEBUG="${COGRPO_PRINT_TRAJECTORY_DEBUG:-1}"
export COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES="${COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES:-4}"
export COGRPO_DEBUG_HINT_CONTEXT="${COGRPO_DEBUG_HINT_CONTEXT:-1}"
export COGRPO_DEBUG_HINT_CONTEXT_TOKENS="${COGRPO_DEBUG_HINT_CONTEXT_TOKENS:-96}"
export XTUNER_PRETTY_TRAJECTORY_JSONL="${XTUNER_PRETTY_TRAJECTORY_JSONL:-0}"

echo "[xtuner-local8-dec] xtuner=${xtuner_branch}@${xtuner_commit} dir=${xtuner_dir}"
echo "[xtuner-local8-dec] model=${model_path}"
echo "[xtuner-local8-dec] dataset=${dataset_path}"
echo "[xtuner-local8-dec] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[xtuner-local8-dec] steps=${ROLLOUT_STEPS} gbs=${GLOBAL_BATCH_SIZE} prk=${PROMPT_REPEAT_K}"
if [ -n "${COGRPO_VERIFIER_SERVER_URL_DICT}" ]; then
  echo "[xtuner-local8-dec] verifier_url_dict_env=SET"
else
  echo "[xtuner-local8-dec] verifier_urls_env=${COGRPO_VERIFIER_SERVER_URLS}"
fi

cd "${xtuner_dir}"
bash examples/v1/scripts/run_rl.sh \
  examples/v1/config/rl_qwen25_7B_cogrpo_cf_branch_async.py \
  lmdeploy \
  "${model_path}" \
  "${dataset_path}" > xtuner_cf_decoupled_debug.log 2>&1

