#!/usr/bin/env bash
set -eo pipefail

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

if ! git -C "${xtuner_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[xtuner-cf4dbg] ERROR: not a git repo: ${xtuner_dir}" >&2
  exit 1
fi

xtuner_branch="$(git -C "${xtuner_dir}" rev-parse --abbrev-ref HEAD)"
xtuner_commit="$(git -C "${xtuner_dir}" rev-parse --short HEAD)"
if [ "${xtuner_branch}" != "${xtuner_branch_expected}" ]; then
  echo "[xtuner-cf4dbg] ERROR: expected xtuner branch ${xtuner_branch_expected}, got ${xtuner_branch}" >&2
  echo "[xtuner-cf4dbg] Hint: cd ${xtuner_dir} && git checkout ${xtuner_branch_expected}" >&2
  exit 1
fi

exp_name="${EXP_NAME:-xtcf4dbg-lmdeployfp8-$(date +%m%d%H%M)}"
model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"
dataset_path="${DATASET_PATH:-/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/data/xtuner_dapo_tiny.jsonl}"
conda_env_path="${CONDA_ENV_PATH:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro}"
lmdeploy_path="${LMDEPLOY_PATH_OVERRIDE:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/lmdeploy_manually_fp8}"

# Ray placement-group resources (must satisfy xtuner's `AcceleratorResourcesConfig` sanity checks).
# With rjob `--cpu=32` and `GPUS=4`, set a conservative default to avoid:
#   "Not enough available CPUs in Ray cluster ... less than needed_cpu"
cpus_per_worker="${CPUS_PER_WORKER:-4}"
cpu_mem_per_worker_gb="${CPU_MEM_PER_WORKER_GB:-16}"

global_batch_size="${GLOBAL_BATCH_SIZE:-4}"
prompt_repeat_k="${PROMPT_REPEAT_K:-16}"
rollout_steps="${ROLLOUT_STEPS:-1}"
train_optimizer_steps="${TRAIN_OPTIMIZER_STEPS:-1}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-32768}"
pack_max_length="${PACK_MAX_LENGTH:-35840}"

# HuggingFace checkpoint saving can trigger large CPU RAM spikes for FSDP full-model training.
# For 4-GPU smoke tests, disable by default (set HF_INTERVAL>0 to enable; final step always saves when enabled).
hf_interval="${HF_INTERVAL:-0}"

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

# Optional: dump CoGRPO extra_info (hints/intervention positions) into trajectory jsonl for debugging.
cogrpo_dump_trajectory_extra="${COGRPO_DUMP_TRAJECTORY_EXTRA:-0}"
cogrpo_dump_trajectory_max_samples="${COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES:-0}"

cogrpo_verifier_max_new_tokens="${COGRPO_VERIFIER_MAX_NEW_TOKENS:-512}"
cogrpo_verifier_max_prompt_length="${COGRPO_VERIFIER_MAX_PROMPT_LENGTH:-8192}"
cogrpo_verifier_max_hint_tokens="${COGRPO_VERIFIER_MAX_HINT_TOKENS:-256}"
cogrpo_estimated_hint_tokens="${COGRPO_ESTIMATED_HINT_TOKENS:-256}"

cogrpo_verifier_lora_enable="${COGRPO_VERIFIER_LORA_ENABLE:-0}"
cogrpo_verifier_lora_rank="${COGRPO_VERIFIER_LORA_RANK:-8}"
cogrpo_verifier_lora_alpha="${COGRPO_VERIFIER_LORA_ALPHA:-16}"
cogrpo_verifier_lora_dropout="${COGRPO_VERIFIER_LORA_DROPOUT:-0.05}"
cogrpo_verifier_lr="${COGRPO_VERIFIER_LR:-1e-5}"
cogrpo_verifier_lora_sync_freq="${COGRPO_VERIFIER_LORA_SYNC_FREQ:-0}"

rjob_name="rjob-${exp_name}"

echo "[xtuner-cf4dbg] xtuner=${xtuner_branch}@${xtuner_commit} dir=${xtuner_dir}"
echo "[xtuner-cf4dbg] name=${rjob_name} model=${model_path}"
echo "[xtuner-cf4dbg] dataset=${dataset_path}"
echo "[xtuner-cf4dbg] rollout_steps=${rollout_steps} gbs=${global_batch_size} prk=${prompt_repeat_k}"
echo "[xtuner-cf4dbg] lengths: prompt=${max_prompt_length} resp=${max_response_length} pack=${pack_max_length}"
echo "[xtuner-cf4dbg] HF_INTERVAL=${hf_interval}"

custom_resource_args=()
if [ "${RJOB_USE_RDMA:-0}" = "1" ]; then
  custom_resource_args=(--custom-resources "rdma/mlnx_shared=${gpus}")
fi

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
  "${custom_resource_args[@]}" \
  -- bash -lc -- "
    unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY &&
    cd ${xtuner_dir} &&
    if ! command -v ray >/dev/null 2>&1; then
      if [ ! -f /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh ]; then
        echo '[xtuner-cf4dbg] ERROR: ray not found and conda.sh not found at /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh' >&2 ;
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
    echo \"[xtuner-cf4dbg] LMDEPLOY_PATH=${lmdeploy_path}\" &&
    if [ ! -d \"${lmdeploy_path}\" ]; then
      echo \"[xtuner-cf4dbg] ERROR: LMDEPLOY_PATH not found: ${lmdeploy_path}\" >&2 ;
      exit 2 ;
    fi &&
    if command -v git >/dev/null 2>&1; then
      git -C \"${lmdeploy_path}\" rev-parse --short HEAD 2>/dev/null | sed 's/^/[xtuner-cf4dbg] lmdeploy_git=/' || true ;
    fi &&
    python -c 'import lmdeploy; print(\"[xtuner-cf4dbg] lmdeploy_file=\", lmdeploy.__file__)' &&
    export NODE_COUNT=1 &&
    export NODE_RANK=0 &&
    export MASTER_ADDR=127.0.0.1 &&
    export XTUNER_RAY_TEMP_DIR=\${XTUNER_RAY_TEMP_DIR:-/tmp/ray_xtcf4_\$(date +%m%d%H%M%S)} &&
    echo \"[xtuner-cf4dbg] XTUNER_RAY_TEMP_DIR=\${XTUNER_RAY_TEMP_DIR}\" &&
    export NUM_WORKERS=${gpus} &&
    export RAY_MAX_CONCURRENCY=1024 &&
    export CPUS_PER_WORKER=${cpus_per_worker} &&
    export CPU_MEM_PER_WORKER_GB=${cpu_mem_per_worker_gb} &&
    export GLOBAL_BATCH_SIZE=${global_batch_size} &&
    export PROMPT_REPEAT_K=${prompt_repeat_k} &&
    export ROLLOUT_STEPS=${rollout_steps} &&
    export TRAIN_OPTIMIZER_STEPS=${train_optimizer_steps} &&
    export MAX_PROMPT_LENGTH=${max_prompt_length} &&
    export MAX_RESPONSE_LENGTH=${max_response_length} &&
    export PACK_MAX_LENGTH=${pack_max_length} &&
    export HF_INTERVAL=${hf_interval} &&
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
    # Pass ACCELERATOR_PER_NODE explicitly to support 4-GPU debug jobs where CUDA_VISIBLE_DEVICES may be empty.
    bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen25_7B_cogrpo_cf_branch_async.py lmdeploy '${model_path}' '${dataset_path}' '' gpu ${gpus}
  "
