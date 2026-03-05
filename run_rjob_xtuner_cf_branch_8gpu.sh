#!/usr/bin/env bash
set -eo pipefail

if [ -f /etc/profile.d/ssh-init.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/ssh-init.sh || true
fi

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

cluster="llmit_gpu"
gpus=8
nnodes=1

xtuner_dir="/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner"
xtuner_branch_expected="cogrpo-v2-xtuner"

if ! git -C "${xtuner_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[xtuner-cf8dbg] ERROR: not a git repo: ${xtuner_dir}" >&2
  exit 1
fi

xtuner_branch="$(git -C "${xtuner_dir}" rev-parse --abbrev-ref HEAD)"
xtuner_commit="$(git -C "${xtuner_dir}" rev-parse --short HEAD)"
if [ "${xtuner_branch}" != "${xtuner_branch_expected}" ]; then
  echo "[xtuner-cf8dbg] ERROR: expected xtuner branch ${xtuner_branch_expected}, got ${xtuner_branch}" >&2
  echo "[xtuner-cf8dbg] Hint: cd ${xtuner_dir} && git checkout ${xtuner_branch_expected}" >&2
  exit 1
fi

exp_name="${EXP_NAME:-xtcf8dbg-$(date +%m%d%H%M)}"
model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"
dataset_path="${DATASET_PATH:-/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/data/xtuner_dapo_tiny.jsonl}"
conda_env_path="${CONDA_ENV_PATH:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro}"
lmdeploy_path="${LMDEPLOY_PATH_OVERRIDE:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/lmdeploy_manually_fp8}"

global_batch_size="${GLOBAL_BATCH_SIZE:-4}"
prompt_repeat_k="${PROMPT_REPEAT_K:-2}"
rollout_steps="${ROLLOUT_STEPS:-1}"
train_optimizer_steps="${TRAIN_OPTIMIZER_STEPS:-1}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-2048}"
pack_max_length="${PACK_MAX_LENGTH:-8192}"

cogrpo_token_check_interval="${COGRPO_TOKEN_CHECK_INTERVAL:-512}"
cogrpo_min_step_tokens="${COGRPO_MIN_STEP_TOKENS:-512}"
cogrpo_max_interventions="${COGRPO_MAX_INTERVENTIONS:-2}"
cogrpo_confidence_threshold="${COGRPO_CONFIDENCE_THRESHOLD:-0.0}"

cogrpo_cf_branch_prob="${COGRPO_CF_BRANCH_PROB:-1.0}"
cogrpo_cf_branch_max_events_per_sample="${COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE:-1}"
cogrpo_cf_branch_k="${COGRPO_CF_BRANCH_K:-2}"

cogrpo_keep_cf_rollouts="${COGRPO_KEEP_CF_ROLLOUTS:-1}"
cogrpo_verifier_skip_truncated="${COGRPO_VERIFIER_SKIP_TRUNCATED:-1}"

cogrpo_verifier_max_new_tokens="${COGRPO_VERIFIER_MAX_NEW_TOKENS:-1024}"
cogrpo_verifier_max_prompt_length="${COGRPO_VERIFIER_MAX_PROMPT_LENGTH:-8192}"
cogrpo_verifier_max_hint_tokens="${COGRPO_VERIFIER_MAX_HINT_TOKENS:-256}"
cogrpo_estimated_hint_tokens="${COGRPO_ESTIMATED_HINT_TOKENS:-256}"

cogrpo_verifier_lora_enable="${COGRPO_VERIFIER_LORA_ENABLE:-1}"
cogrpo_verifier_lora_rank="${COGRPO_VERIFIER_LORA_RANK:-8}"
cogrpo_verifier_lora_alpha="${COGRPO_VERIFIER_LORA_ALPHA:-16}"
cogrpo_verifier_lora_dropout="${COGRPO_VERIFIER_LORA_DROPOUT:-0.05}"
cogrpo_verifier_lr="${COGRPO_VERIFIER_LR:-1e-5}"
cogrpo_verifier_lora_sync_freq="${COGRPO_VERIFIER_LORA_SYNC_FREQ:-0}"

rjob_name="rjob-${exp_name}"

echo "[xtuner-cf8dbg] xtuner=${xtuner_branch}@${xtuner_commit} dir=${xtuner_dir}"
echo "[xtuner-cf8dbg] name=${rjob_name} model=${model_path}"
echo "[xtuner-cf8dbg] dataset=${dataset_path}"
echo "[xtuner-cf8dbg] rollout_steps=${rollout_steps} global_batch_size=${global_batch_size} prompt_repeat_k=${prompt_repeat_k}"

custom_resource_args=()
if [ "${RJOB_USE_RDMA:-0}" = "1" ]; then
  custom_resource_args=(--custom-resources "rdma/mlnx_shared=${gpus}")
fi

rjob submit \
  --name="${rjob_name}" \
  --gpu="${gpus}" \
  --memory="${RJOB_MEMORY:-512000}" \
  --cpu="${RJOB_CPU:-64}" \
  --charged-group="${cluster}" \
  --private-machine=group \
  --share-host-shm=True \
  --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
  --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
  --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
  --image=registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605 \
  -P "${nnodes}" \
  --host-network=true \
  -e DISTRIBUTED_JOB=true \
  "${custom_resource_args[@]}" \
  -- bash -lc -- "
    unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY &&
    cd ${xtuner_dir} &&
    if ! command -v ray >/dev/null 2>&1; then
      if [ ! -f /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh ]; then
        echo '[xtuner-cf8dbg] ERROR: ray not found and conda.sh not found at /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh' >&2 ;
        exit 127 ;
      fi &&
      source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh &&
      conda activate '${conda_env_path}' ;
    fi &&
    which python &&
    python -V &&
    which ray &&
    ray --version &&
    export LMDEPLOY_PATH='${lmdeploy_path}' &&
    echo \"[xtuner-cf8dbg] LMDEPLOY_PATH=${lmdeploy_path}\" &&
    if [ ! -d \"${lmdeploy_path}\" ]; then
      echo \"[xtuner-cf8dbg] ERROR: LMDEPLOY_PATH not found: ${lmdeploy_path}\" >&2 ;
      exit 2 ;
    fi &&
    export PYTHONPATH=\"${lmdeploy_path}:\${PYTHONPATH:-}\" &&
    if command -v git >/dev/null 2>&1; then
      git -C \"${lmdeploy_path}\" rev-parse --short HEAD 2>/dev/null | sed 's/^/[xtuner-cf8dbg] lmdeploy_git=/' || true ;
    fi &&
    python -c 'import lmdeploy; print(\"[xtuner-cf8dbg] lmdeploy_file=\", lmdeploy.__file__)' &&
    export NODE_COUNT=1 &&
    export NODE_RANK=0 &&
    export MASTER_ADDR=127.0.0.1 &&
    export RAY_MAX_CONCURRENCY=1024 &&
    export GLOBAL_BATCH_SIZE=${global_batch_size} &&
    export PROMPT_REPEAT_K=${prompt_repeat_k} &&
    export ROLLOUT_STEPS=${rollout_steps} &&
    export TRAIN_OPTIMIZER_STEPS=${train_optimizer_steps} &&
    export MAX_PROMPT_LENGTH=${max_prompt_length} &&
    export MAX_RESPONSE_LENGTH=${max_response_length} &&
    export PACK_MAX_LENGTH=${pack_max_length} &&
    export COGRPO_ENABLE=1 &&
    export COGRPO_TOKEN_CHECK_INTERVAL=${cogrpo_token_check_interval} &&
    export COGRPO_MIN_STEP_TOKENS=${cogrpo_min_step_tokens} &&
    export COGRPO_MAX_INTERVENTIONS=${cogrpo_max_interventions} &&
    export COGRPO_CONFIDENCE_THRESHOLD=${cogrpo_confidence_threshold} &&
    export COGRPO_CF_BRANCH_PROB=${cogrpo_cf_branch_prob} &&
    export COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE=${cogrpo_cf_branch_max_events_per_sample} &&
    export COGRPO_CF_BRANCH_K=${cogrpo_cf_branch_k} &&
    export COGRPO_KEEP_CF_ROLLOUTS=${cogrpo_keep_cf_rollouts} &&
    export COGRPO_VERIFIER_SKIP_TRUNCATED=${cogrpo_verifier_skip_truncated} &&
    export COGRPO_VERIFIER_MAX_NEW_TOKENS=${cogrpo_verifier_max_new_tokens} &&
    export COGRPO_VERIFIER_MAX_PROMPT_LENGTH=${cogrpo_verifier_max_prompt_length} &&
    export COGRPO_VERIFIER_MAX_HINT_TOKENS=${cogrpo_verifier_max_hint_tokens} &&
    export COGRPO_ESTIMATED_HINT_TOKENS=${cogrpo_estimated_hint_tokens} &&
    export COGRPO_VERIFIER_LORA_ENABLE=${cogrpo_verifier_lora_enable} &&
    export COGRPO_VERIFIER_LORA_RANK=${cogrpo_verifier_lora_rank} &&
    export COGRPO_VERIFIER_LORA_ALPHA=${cogrpo_verifier_lora_alpha} &&
    export COGRPO_VERIFIER_LORA_DROPOUT=${cogrpo_verifier_lora_dropout} &&
    export COGRPO_VERIFIER_LR=${cogrpo_verifier_lr} &&
    export COGRPO_VERIFIER_LORA_SYNC_FREQ=${cogrpo_verifier_lora_sync_freq} &&
    bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen25_7B_cogrpo_cf_branch_async.py lmdeploy '${model_path}' '${dataset_path}'
  "
