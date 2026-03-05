#!/usr/bin/env bash
set -eo pipefail

# 4-GPU rjob: debug Verifier "High no-decision rate" under long-context (32k).
# Only shrinks global train batch-size + rollout.n to keep it schedulable.
#
# Usage:
#   bash repos/repro/run_rjob_fast_debug_nodecision_hf181_4gpu.sh

if [ -f /etc/profile.d/ssh-init.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/ssh-init.sh || true
fi

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

cluster="llmit_gpu"
gpus=4
nnodes=1

exp_name="${EXP_NAME:-cogrpo4dbg-hf181-32k-bs4-r4-$(date +%m%d%H%M)}"
model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/ckpt/cold_start_full_qwen2d5-7b/20260223230703_mixed_full/20260224105827/hf-181}"
work_dir="${WORK_DIR:-/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train}"
dataset_path="${DATASET_PATH:-/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/data/debug_dapo_mini64.parquet}"

rjob_name="rjob-${exp_name}"

echo "[rjob4dbg][nodecision] name=${rjob_name}"

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
  --custom-resources rdma/mlnx_shared="${gpus}" \
  -- bash -lc -- "
    unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY &&
    cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro &&
    export EXP_NAME='${exp_name}' &&
    export WORK_DIR='${work_dir}' &&
    export MODEL_PATH='${model_path}' &&
    export SRC_PARQUET='${dataset_path}' &&
    export MINI_PARQUET='${dataset_path}' &&
    export MINI_ROWS=64 &&
    export N_GPUS_PER_NODE=${gpus} &&
    # ====== core debug knobs ======
    export TRAIN_BSZ=4 &&
    export MICRO_BSZ=1 &&
    export ROLLOUT_N=4 &&
    export STEPS=1 &&
    export DUAL_ROLLOUT_DUMP_FREQ=1 &&
    export ROLLOUT_DUMP_FREQ=1 &&
    export GPU_MEM_UTIL=0.8 &&
    export VERIFIER_CREDIT_ASSIGNMENT=cf_branch &&
    export VERL_VERIFIER_DEBUG_LOG_OVERRIDE=1 &&
    export VERIFIER_LORA_PATH='' &&
    export VERIFIER_LORA_RANK=0 &&
    export RESPONSE_N=32 &&
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS=65536 &&
    export MAX_PROMPT_LENGTH=2048 &&
    export TOKEN_CHECK_INTERVAL=2048 &&
    export MIN_STEP_TOKENS=2048 &&
    export MAX_INTERVENTIONS=2 &&
    export VERIFIER_MAX_NEW_TOKENS=4096 &&
    export CONFIDENCE_THRESHOLD=0.0 &&
    export ACTOR_UPDATE_STREAMS=both &&
    export CF_BRANCH_MAX_EVENTS=2 &&
    export CF_BRANCH_K=2 &&
    bash run_fast_debug_cogrpo_8gpu_mini_dapo.sh
  "
