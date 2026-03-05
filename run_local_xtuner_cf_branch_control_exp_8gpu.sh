#!/usr/bin/env bash
set -euo pipefail

# Local (single-node) XTuner smoke test for:
# - CoGRPO v2 by_step + cf_branch
# - 8 control + 8 exp dual-stream (k=16)
# - optional curriculum weighting
#
# This script is intended to run *inside* an 8-GPU dev machine (no rjob).

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

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
dataset_path="${DATASET_PATH:-${repo_dir}/data/xtuner_dapo_tiny.jsonl}"
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

# Ensure required runtime deps exist.
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "[xtuner-local8] ERROR: python missing runtime dep: torch" >&2
  exit 127
fi
if ! python -c "import cyclopts" >/dev/null 2>&1; then
  echo "[xtuner-local8] ERROR: python missing runtime dep: cyclopts" >&2
  echo "[xtuner-local8] Hint: activate your env, then run: python -m pip install cyclopts" >&2
  echo "[xtuner-local8] Or install xtuner runtime deps: python -m pip install --no-build-isolation -r ${xtuner_dir}/requirements/runtime.txt" >&2
  exit 127
fi
if ! python -c "import mmengine" >/dev/null 2>&1; then
  echo "[xtuner-local8] ERROR: python missing runtime dep: mmengine" >&2
  echo "[xtuner-local8] Hint: python -m pip install --no-build-isolation -r ${xtuner_dir}/requirements/runtime.txt" >&2
  exit 127
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

# Core run knobs (keep small for smoke test; override via env)
export NODE_COUNT=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1

export NUM_WORKERS="${NUM_WORKERS:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export PROMPT_REPEAT_K="${PROMPT_REPEAT_K:-16}"
export ROLLOUT_STEPS="${ROLLOUT_STEPS:-2}"
export TRAIN_OPTIMIZER_STEPS="${TRAIN_OPTIMIZER_STEPS:-1}"
# HF saving can be memory-hungry for FSDP full-model training; disable for local smoke by default.
export HF_INTERVAL="${HF_INTERVAL:-0}"

# DCP checkpoint / auto-resume knobs (optional for local smoke).
export AUTO_RESUME="${AUTO_RESUME:-0}"
export RESUME_FROM="${RESUME_FROM:-}"
export STRICT_LOAD="${STRICT_LOAD:-1}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:--1}"
export CHECKPOINT_MAXKEEP="${CHECKPOINT_MAXKEEP:-2}"
export CHECKPOINT_NO_SAVE_OPTIMIZER="${CHECKPOINT_NO_SAVE_OPTIMIZER:-0}"
export LOAD_OPTIMIZER_STATES="${LOAD_OPTIMIZER_STATES:-1}"
export LOAD_OPTIMIZER_ARGS="${LOAD_OPTIMIZER_ARGS:-1}"

if [ "${CHECKPOINT_NO_SAVE_OPTIMIZER}" = "1" ]; then
  export LOAD_OPTIMIZER_STATES=0
  export LOAD_OPTIMIZER_ARGS=0
fi

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export PACK_MAX_LENGTH="${PACK_MAX_LENGTH:-8192}"

# Dual-stream (8 control + 8 exp)
export COGRPO_CONTROL_K="${COGRPO_CONTROL_K:-8}"
export COGRPO_EXP_K="${COGRPO_EXP_K:-8}"

# Trainer-side CoGRPO advantage mixing
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-co_grpo}"
export ACTOR_UPDATE_STREAMS="${ACTOR_UPDATE_STREAMS:-exp}"   # exp|both

# Curriculum weighting (optional)
export USE_CURRICULUM_WEIGHTING="${USE_CURRICULUM_WEIGHTING:-1}"
export CURRICULUM_START_WEIGHT="${CURRICULUM_START_WEIGHT:-0.3}"
export CURRICULUM_END_WEIGHT="${CURRICULUM_END_WEIGHT:-0.7}"
export CONTROL_GROUP_WEIGHT="${CONTROL_GROUP_WEIGHT:-0.5}"   # used when USE_CURRICULUM_WEIGHTING=0

# CoGRPO rollout (by_step + cf_branch)
export COGRPO_ENABLE="${COGRPO_ENABLE:-1}"
export COGRPO_TOKEN_CHECK_INTERVAL="${COGRPO_TOKEN_CHECK_INTERVAL:-512}"
export COGRPO_MIN_STEP_TOKENS="${COGRPO_MIN_STEP_TOKENS:-512}"
export COGRPO_MAX_INTERVENTIONS="${COGRPO_MAX_INTERVENTIONS:-1}"
export COGRPO_CONFIDENCE_THRESHOLD="${COGRPO_CONFIDENCE_THRESHOLD:-0.0}"

export COGRPO_CF_BRANCH_PROB="${COGRPO_CF_BRANCH_PROB:-1.0}"
export COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE="${COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE:-1}"
export COGRPO_CF_BRANCH_K="${COGRPO_CF_BRANCH_K:-2}"

export COGRPO_KEEP_CF_ROLLOUTS="${COGRPO_KEEP_CF_ROLLOUTS:-0}"
export COGRPO_VERIFIER_SKIP_TRUNCATED="${COGRPO_VERIFIER_SKIP_TRUNCATED:-1}"

# For a pure actor-only smoke test, disable verifier LoRA by default (override if you want to test online verifier training).
export COGRPO_VERIFIER_LORA_ENABLE="${COGRPO_VERIFIER_LORA_ENABLE:-0}"
export COGRPO_VERIFIER_LORA_SAVE_FREQ="${COGRPO_VERIFIER_LORA_SAVE_FREQ:-20}"

# Debug dumps / prints (enabled by default for local debug).
export COGRPO_DUMP_TRAJECTORY_EXTRA="${COGRPO_DUMP_TRAJECTORY_EXTRA:-1}"
export COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES="${COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES:-16}"
export COGRPO_PRINT_TRAJECTORY_DEBUG="${COGRPO_PRINT_TRAJECTORY_DEBUG:-1}"
export COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES="${COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES:-4}"
export COGRPO_DEBUG_HINT_CONTEXT="${COGRPO_DEBUG_HINT_CONTEXT:-1}"
export COGRPO_DEBUG_HINT_CONTEXT_TOKENS="${COGRPO_DEBUG_HINT_CONTEXT_TOKENS:-96}"
export XTUNER_PRETTY_TRAJECTORY_JSONL="${XTUNER_PRETTY_TRAJECTORY_JSONL:-0}"

echo "[xtuner-local8] xtuner=${xtuner_branch}@${xtuner_commit} dir=${xtuner_dir}"
echo "[xtuner-local8] backend=${infer_backend} model=${model_path}"
echo "[xtuner-local8] dataset=${dataset_path}"
echo "[xtuner-local8] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[xtuner-local8] prk=${PROMPT_REPEAT_K} ctrl=${COGRPO_CONTROL_K} exp=${COGRPO_EXP_K} steps=${ROLLOUT_STEPS}"

cd "${xtuner_dir}"
bash examples/v1/scripts/run_rl.sh \
  examples/v1/config/rl_qwen25_7B_cogrpo_cf_branch_async.py \
  "${infer_backend}" \
  "${model_path}" \
  "${dataset_path}" > xtuner_debug.log 2>&1
