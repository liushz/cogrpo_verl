#!/usr/bin/env bash
set -euo pipefail

# Local (single-node) XTuner smoke test for CoGRPO v2 by_step + cf_branch (exp-only),
# with baseline-aligned long output settings (32k response / 35k pack).
#
# Mode: non-decoupled (verifier=actor, verifier LoRA disabled).

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

xtuner_dir="${XTUNER_DIR:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner}"
xtuner_branch_expected="${XTUNER_BRANCH_EXPECTED:-cogrpo-v2-xtuner-control-exp}"

if ! git -C "${xtuner_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[xtuner-local8-exp] ERROR: not a git repo: ${xtuner_dir}" >&2
  exit 1
fi

xtuner_branch="$(git -C "${xtuner_dir}" rev-parse --abbrev-ref HEAD)"
xtuner_commit="$(git -C "${xtuner_dir}" rev-parse --short HEAD)"
if [ "${xtuner_branch}" != "${xtuner_branch_expected}" ]; then
  echo "[xtuner-local8-exp] ERROR: expected xtuner branch ${xtuner_branch_expected}, got ${xtuner_branch}" >&2
  echo "[xtuner-local8-exp] Hint: cd ${xtuner_dir} && git checkout ${xtuner_branch_expected}" >&2
  exit 1
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
  echo "[xtuner-local8-exp] ERROR: ray not found in current env. Please activate your conda env first." >&2
  exit 127
fi

export LMDEPLOY_PATH="${lmdeploy_path}"
if [ ! -d "${LMDEPLOY_PATH}" ]; then
  echo "[xtuner-local8-exp] ERROR: LMDEPLOY_PATH not found: ${LMDEPLOY_PATH}" >&2
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

# Non-decoupled: verifier=actor (disable verifier LoRA).
export COGRPO_VERIFIER_LORA_ENABLE="${COGRPO_VERIFIER_LORA_ENABLE:-0}"

# Debug dumps / prints.
export COGRPO_DUMP_TRAJECTORY_EXTRA="${COGRPO_DUMP_TRAJECTORY_EXTRA:-1}"
export COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES="${COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES:-16}"
export COGRPO_PRINT_TRAJECTORY_DEBUG="${COGRPO_PRINT_TRAJECTORY_DEBUG:-1}"
export COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES="${COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES:-4}"
export COGRPO_DEBUG_HINT_CONTEXT="${COGRPO_DEBUG_HINT_CONTEXT:-1}"
export COGRPO_DEBUG_HINT_CONTEXT_TOKENS="${COGRPO_DEBUG_HINT_CONTEXT_TOKENS:-96}"
export XTUNER_PRETTY_TRAJECTORY_JSONL="${XTUNER_PRETTY_TRAJECTORY_JSONL:-0}"

echo "[xtuner-local8-exp] xtuner=${xtuner_branch}@${xtuner_commit} dir=${xtuner_dir}"
echo "[xtuner-local8-exp] model=${model_path}"
echo "[xtuner-local8-exp] dataset=${dataset_path}"
echo "[xtuner-local8-exp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[xtuner-local8-exp] steps=${ROLLOUT_STEPS} gbs=${GLOBAL_BATCH_SIZE} prk=${PROMPT_REPEAT_K}"

cd "${xtuner_dir}"
bash examples/v1/scripts/run_rl.sh \
  examples/v1/config/rl_qwen25_7B_cogrpo_cf_branch_async.py \
  lmdeploy \
  "${model_path}" \
  "${dataset_path}" > xtuner_cf_exp_only_debug.log 2>&1

