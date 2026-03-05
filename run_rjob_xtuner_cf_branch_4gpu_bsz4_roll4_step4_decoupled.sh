#!/usr/bin/env bash
set -eo pipefail

# 4-GPU rjob: XTuner CoGRPO v2 cf_branch (decoupled-ish debug: enable verifier LoRA training),
# with CompassVerifier reward endpoints aligned with verl.
#
# Target knobs:
# - GLOBAL_BATCH_SIZE=4
# - ROLLOUT_STEPS=4
# - TRAIN_OPTIMIZER_STEPS=4
#
# NOTE:
# This job enables verifier LoRA training paths. Full "decoupled verifier inference"
# (verifier routed to dedicated rollout servers) still requires external verifier servers.

if [ -f /etc/profile.d/ssh-init.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/ssh-init.sh || true
fi

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

cluster="llmit_gpu"
gpus="${GPUS:-4}"
nnodes=1

xtuner_dir="/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner"
xtuner_branch_expected="${XTUNER_BRANCH_EXPECTED:-cogrpo-v2-xtuner-control-exp}"

xtuner_branch="$(git -C "${xtuner_dir}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
xtuner_commit="$(git -C "${xtuner_dir}" rev-parse --short HEAD 2>/dev/null || echo "")"
if [ -z "${xtuner_branch}" ] || [ "${xtuner_branch}" != "${xtuner_branch_expected}" ]; then
  echo "[xtcf4-dec] ERROR: expected xtuner branch ${xtuner_branch_expected}, got ${xtuner_branch:-<none>}" >&2
  exit 1
fi

exp_name="${EXP_NAME:-xtcf4-dec-bsz4-roll4-step4-$(date +%m%d%H%M)}"

model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"
dataset_path="${DATASET_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-dapo-math-17k.jsonl}"
conda_env_path="${CONDA_ENV_PATH:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro}"
lmdeploy_path="${LMDEPLOY_PATH_OVERRIDE:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/lmdeploy_manually_fp8}"

global_batch_size="${GLOBAL_BATCH_SIZE:-4}"
prompt_repeat_k="${PROMPT_REPEAT_K:-16}"
rollout_steps="${ROLLOUT_STEPS:-4}"
train_optimizer_steps="${TRAIN_OPTIMIZER_STEPS:-4}"

max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-32768}"
pack_max_length="${PACK_MAX_LENGTH:-35840}"

hf_interval="${HF_INTERVAL:-0}"

# Reward endpoints (verl-aligned CompassVerifier OpenAI-compatible /v1 endpoints).
reward_model_urls="${REWARD_MODEL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"
reward_model_key="${REWARD_MODEL_KEY:-EMPTY}"
reward_num_actors="${REWARD_NUM_ACTORS:-4}"
reward_timeout_s="${REWARD_TIMEOUT_S:-120}"
reward_retries_per_url="${REWARD_RETRIES_PER_URL:-2}"
compass_debug="${COMPASSVERIFIER_DEBUG:-1}"
compass_debug_max_calls="${COMPASSVERIFIER_DEBUG_MAX_CALLS:-20}"

# CoGRPO knobs
cogrpo_token_check_interval="${COGRPO_TOKEN_CHECK_INTERVAL:-2048}"
cogrpo_min_step_tokens="${COGRPO_MIN_STEP_TOKENS:-2048}"
cogrpo_max_interventions="${COGRPO_MAX_INTERVENTIONS:-2}"
cogrpo_confidence_threshold="${COGRPO_CONFIDENCE_THRESHOLD:-0.0}"

cogrpo_cf_branch_prob="${COGRPO_CF_BRANCH_PROB:-1.0}"
cogrpo_cf_branch_max_events_per_sample="${COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE:-2}"
cogrpo_cf_branch_state_hash_mod="${COGRPO_CF_BRANCH_STATE_HASH_MOD:-1024}"
cogrpo_cf_branch_k="${COGRPO_CF_BRANCH_K:-2}"

cogrpo_keep_cf_rollouts="${COGRPO_KEEP_CF_ROLLOUTS:-0}"
cogrpo_verifier_skip_truncated="${COGRPO_VERIFIER_SKIP_TRUNCATED:-1}"

# Debug dumps / prints
cogrpo_dump_trajectory_extra="${COGRPO_DUMP_TRAJECTORY_EXTRA:-1}"
cogrpo_dump_trajectory_max_samples="${COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES:-16}"
cogrpo_print_traj_debug="${COGRPO_PRINT_TRAJECTORY_DEBUG:-1}"
cogrpo_print_traj_max_samples="${COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES:-4}"
cogrpo_debug_hint_context="${COGRPO_DEBUG_HINT_CONTEXT:-1}"
cogrpo_debug_hint_context_tokens="${COGRPO_DEBUG_HINT_CONTEXT_TOKENS:-96}"

# "Decoupled": enable verifier LoRA training (small rank for 4-GPU debug).
cogrpo_verifier_lora_enable="${COGRPO_VERIFIER_LORA_ENABLE:-1}"
cogrpo_verifier_lora_rank="${COGRPO_VERIFIER_LORA_RANK:-8}"
cogrpo_verifier_lora_alpha="${COGRPO_VERIFIER_LORA_ALPHA:-16}"
cogrpo_verifier_lora_dropout="${COGRPO_VERIFIER_LORA_DROPOUT:-0.05}"
cogrpo_verifier_lr="${COGRPO_VERIFIER_LR:-1e-5}"
cogrpo_verifier_lora_sync_freq="${COGRPO_VERIFIER_LORA_SYNC_FREQ:-0}"

rjob_name="rjob-${exp_name}"

echo "[xtcf4-dec] xtuner=${xtuner_branch}@${xtuner_commit} dir=${xtuner_dir}"
echo "[xtcf4-dec] name=${rjob_name} model=${model_path}"
echo "[xtcf4-dec] dataset=${dataset_path}"
echo "[xtcf4-dec] rollout_steps=${rollout_steps} gbs=${global_batch_size} opt_steps=${train_optimizer_steps} prk=${prompt_repeat_k}"
echo "[xtcf4-dec] reward_urls=${reward_model_urls}"
echo "[xtcf4-dec] COMPASSVERIFIER_DEBUG=${compass_debug} max_calls=${compass_debug_max_calls}"

rjob submit \
  --name="${rjob_name}" \
  --gpu="${gpus}" \
  --memory="${RJOB_MEMORY:-256000}" \
  --cpu="${RJOB_CPU:-32}" \
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
  -- bash -lc -- "
    unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY &&
    cd ${xtuner_dir} &&
    if ! command -v ray >/dev/null 2>&1; then
      source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh &&
      conda activate '${conda_env_path}' ;
    fi &&
    export LMDEPLOY_PATH='${lmdeploy_path}' &&
    export NODE_COUNT=1 &&
    export NODE_RANK=0 &&
    export MASTER_ADDR=127.0.0.1 &&
    export NUM_WORKERS=${gpus} &&
    export RAY_MAX_CONCURRENCY=1024 &&
    export GLOBAL_BATCH_SIZE=${global_batch_size} &&
    export PROMPT_REPEAT_K=${prompt_repeat_k} &&
    export ROLLOUT_STEPS=${rollout_steps} &&
    export TRAIN_OPTIMIZER_STEPS=${train_optimizer_steps} &&
    export MAX_PROMPT_LENGTH=${max_prompt_length} &&
    export MAX_RESPONSE_LENGTH=${max_response_length} &&
    export PACK_MAX_LENGTH=${pack_max_length} &&
    export HF_INTERVAL=${hf_interval} &&
    export REWARD_MODEL_URLS='${reward_model_urls}' &&
    export REWARD_MODEL_KEY='${reward_model_key}' &&
    export REWARD_NUM_ACTORS='${reward_num_actors}' &&
    export REWARD_TIMEOUT_S='${reward_timeout_s}' &&
    export REWARD_RETRIES_PER_URL='${reward_retries_per_url}' &&
    export COMPASSVERIFIER_DEBUG='${compass_debug}' &&
    export COMPASSVERIFIER_DEBUG_MAX_CALLS='${compass_debug_max_calls}' &&
    export COGRPO_ENABLE=1 &&
    export COGRPO_TOKEN_CHECK_INTERVAL=${cogrpo_token_check_interval} &&
    export COGRPO_MIN_STEP_TOKENS=${cogrpo_min_step_tokens} &&
    export COGRPO_MAX_INTERVENTIONS=${cogrpo_max_interventions} &&
    export COGRPO_CONFIDENCE_THRESHOLD=${cogrpo_confidence_threshold} &&
    export COGRPO_CF_BRANCH_PROB=${cogrpo_cf_branch_prob} &&
    export COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE=${cogrpo_cf_branch_max_events_per_sample} &&
    export COGRPO_CF_BRANCH_STATE_HASH_MOD=${cogrpo_cf_branch_state_hash_mod} &&
    export COGRPO_CF_BRANCH_K=${cogrpo_cf_branch_k} &&
    export COGRPO_KEEP_CF_ROLLOUTS=${cogrpo_keep_cf_rollouts} &&
    export COGRPO_VERIFIER_SKIP_TRUNCATED=${cogrpo_verifier_skip_truncated} &&
    export COGRPO_DUMP_TRAJECTORY_EXTRA=${cogrpo_dump_trajectory_extra} &&
    export COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES=${cogrpo_dump_trajectory_max_samples} &&
    export COGRPO_PRINT_TRAJECTORY_DEBUG=${cogrpo_print_traj_debug} &&
    export COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES=${cogrpo_print_traj_max_samples} &&
    export COGRPO_DEBUG_HINT_CONTEXT=${cogrpo_debug_hint_context} &&
    export COGRPO_DEBUG_HINT_CONTEXT_TOKENS=${cogrpo_debug_hint_context_tokens} &&
    export COGRPO_VERIFIER_LORA_ENABLE=${cogrpo_verifier_lora_enable} &&
    export COGRPO_VERIFIER_LORA_RANK=${cogrpo_verifier_lora_rank} &&
    export COGRPO_VERIFIER_LORA_ALPHA=${cogrpo_verifier_lora_alpha} &&
    export COGRPO_VERIFIER_LORA_DROPOUT=${cogrpo_verifier_lora_dropout} &&
    export COGRPO_VERIFIER_LR=${cogrpo_verifier_lr} &&
    export COGRPO_VERIFIER_LORA_SYNC_FREQ=${cogrpo_verifier_lora_sync_freq} &&
    bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen25_7B_cogrpo_cf_branch_async.py lmdeploy '${model_path}' '${dataset_path}' '' gpu ${gpus}
  "
