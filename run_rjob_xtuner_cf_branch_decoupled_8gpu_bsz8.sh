#!/usr/bin/env bash
set -eo pipefail

# 8-GPU rjob launcher for XTuner CoGRPO v2 (cf_branch, exp-only) in **decoupled** mode:
# - verifier inference routed to external rollout servers (LMDeploy/SGLang/vLLM endpoints)
# - verifier LoRA trained online and synced to verifier servers
#
# You must provide verifier rollout server URLs via one of:
# - COGRPO_VERIFIER_SERVER_URLS: comma-separated URLs, aligned with actor rollout ranks
# - COGRPO_VERIFIER_SERVER_URL_DICT: json dict {rank:int -> url:str}
#
# This script keeps output lengths aligned with baseline (32k response / 35k pack) and uses small batch for debug.

if [ -f /etc/profile.d/ssh-init.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/ssh-init.sh || true
fi

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

cluster="llmit_gpu"
gpus="${GPUS:-8}"
nnodes=1

xtuner_dir="/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner"
xtuner_branch_expected="${XTUNER_BRANCH_EXPECTED:-cogrpo-v2-xtuner-control-exp}"

if ! git -C "${xtuner_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[xtuner-cf8-dec] ERROR: not a git repo: ${xtuner_dir}" >&2
  exit 1
fi

xtuner_branch="$(git -C "${xtuner_dir}" rev-parse --abbrev-ref HEAD)"
xtuner_commit="$(git -C "${xtuner_dir}" rev-parse --short HEAD)"
if [ "${xtuner_branch}" != "${xtuner_branch_expected}" ]; then
  echo "[xtuner-cf8-dec] ERROR: expected xtuner branch ${xtuner_branch_expected}, got ${xtuner_branch}" >&2
  echo "[xtuner-cf8-dec] Hint: cd ${xtuner_dir} && git checkout ${xtuner_branch_expected}" >&2
  exit 1
fi

verifier_urls_env="${COGRPO_VERIFIER_SERVER_URLS:-${VERIFIER_SERVER_URLS:-}}"
verifier_url_dict_env="${COGRPO_VERIFIER_SERVER_URL_DICT:-${VERIFIER_SERVER_URL_DICT:-}}"
if [ -z "${verifier_urls_env}" ] && [ -z "${verifier_url_dict_env}" ]; then
  echo "[xtuner-cf8-dec] ERROR: missing verifier server URLs." >&2
  echo "[xtuner-cf8-dec] Provide COGRPO_VERIFIER_SERVER_URLS or COGRPO_VERIFIER_SERVER_URL_DICT." >&2
  exit 2
fi

exp_name="${EXP_NAME:-xtcf8-dec-bsz8-$(date +%m%d%H%M)}"

model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"
dataset_path="${DATASET_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-dapo-math-17k.jsonl}"
conda_env_path="${CONDA_ENV_PATH:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro}"
lmdeploy_path="${LMDEPLOY_PATH_OVERRIDE:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/lmdeploy_manually_fp8}"

global_batch_size="${GLOBAL_BATCH_SIZE:-8}"
prompt_repeat_k="${PROMPT_REPEAT_K:-16}"
rollout_steps="${ROLLOUT_STEPS:-5}"
train_optimizer_steps="${TRAIN_OPTIMIZER_STEPS:-1}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-32768}"
pack_max_length="${PACK_MAX_LENGTH:-35840}"

hf_interval="${HF_INTERVAL:-0}"

# CoGRPO rollout knobs
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

# Verifier inference knobs
cogrpo_verifier_max_new_tokens="${COGRPO_VERIFIER_MAX_NEW_TOKENS:-4096}"
cogrpo_verifier_max_prompt_length="${COGRPO_VERIFIER_MAX_PROMPT_LENGTH:-16384}"
cogrpo_verifier_max_hint_tokens="${COGRPO_VERIFIER_MAX_HINT_TOKENS:-512}"
cogrpo_estimated_hint_tokens="${COGRPO_ESTIMATED_HINT_TOKENS:-512}"

# Verifier LoRA (online)
cogrpo_verifier_lora_enable="${COGRPO_VERIFIER_LORA_ENABLE:-1}"
cogrpo_verifier_lora_rank="${COGRPO_VERIFIER_LORA_RANK:-8}"
cogrpo_verifier_lora_alpha="${COGRPO_VERIFIER_LORA_ALPHA:-16}"
cogrpo_verifier_lora_dropout="${COGRPO_VERIFIER_LORA_DROPOUT:-0.05}"
cogrpo_verifier_lr="${COGRPO_VERIFIER_LR:-1e-5}"
cogrpo_verifier_lora_sync_freq="${COGRPO_VERIFIER_LORA_SYNC_FREQ:-1}"
cogrpo_verifier_lora_save_freq="${COGRPO_VERIFIER_LORA_SAVE_FREQ:-0}"

rjob_name="rjob-${exp_name}"

echo "[xtuner-cf8-dec] xtuner=${xtuner_branch}@${xtuner_commit} dir=${xtuner_dir}"
echo "[xtuner-cf8-dec] name=${rjob_name} model=${model_path}"
echo "[xtuner-cf8-dec] dataset=${dataset_path}"
echo "[xtuner-cf8-dec] rollout_steps=${rollout_steps} gbs=${global_batch_size} prk=${prompt_repeat_k}"
echo "[xtuner-cf8-dec] lengths: prompt=${max_prompt_length} resp=${max_response_length} pack=${pack_max_length}"
echo "[xtuner-cf8-dec] HF_INTERVAL=${hf_interval}"
if [ -n "${verifier_url_dict_env}" ]; then
  echo "[xtuner-cf8-dec] verifier_url_dict_env=SET"
else
  echo "[xtuner-cf8-dec] verifier_urls_env=${verifier_urls_env}"
fi

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
        echo '[xtuner-cf8-dec] ERROR: ray not found and conda.sh not found at /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh' >&2 ;
        exit 127 ;
      fi &&
      source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh &&
      conda activate '${conda_env_path}' ;
    fi &&
    which python &&
    python -V &&
    which ray &&
    ray --version &&
    export LMDEPLOY_PATH=${lmdeploy_path} &&
    echo \"[xtuner-cf8-dec] LMDEPLOY_PATH=${lmdeploy_path}\" &&
    if [ ! -d \"${lmdeploy_path}\" ]; then
      echo \"[xtuner-cf8-dec] ERROR: LMDEPLOY_PATH not found: ${lmdeploy_path}\" >&2 ;
      exit 2 ;
    fi &&
    export PYTHONPATH=\"${lmdeploy_path}:\${PYTHONPATH:-}\" &&
    python -c 'import lmdeploy; print(\"[xtuner-cf8-dec] lmdeploy_file=\", lmdeploy.__file__)' &&
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
    export COGRPO_VERIFIER_SERVER_URLS='${verifier_urls_env}' &&
    export COGRPO_VERIFIER_SERVER_URL_DICT='${verifier_url_dict_env}' &&
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
    export COGRPO_VERIFIER_LORA_SAVE_FREQ=${cogrpo_verifier_lora_save_freq} &&
    export XTUNER_PRETTY_TRAJECTORY_JSONL=${XTUNER_PRETTY_TRAJECTORY_JSONL:-0} &&
    bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen25_7B_cogrpo_cf_branch_async.py lmdeploy '${model_path}' '${dataset_path}'
  "

