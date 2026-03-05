#!/usr/bin/env bash
set -eo pipefail

# 8-GPU rjob fast debug runner:
# - Uses the same launcher as online CoGRPO (`scripts/run_multinodes_cgrpo_v2.sh`)
# - Uses a tiny parquet subset to avoid 1.79M-row token filtering
# - Intended to quickly catch startup crashes (vLLM weight sync, verifier parsing)
#
# Example:
#   bash run_rjob_fast_debug_cogrpo_8gpu.sh
#
# Env overrides:
#   EXP_NAME=...
#   MODEL_PATH=...
#   WORK_DIR=...

if [ -f /etc/profile.d/ssh-init.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/ssh-init.sh || true
fi

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

cluster="llmit_gpu"
gpus=8
nnodes=1

exp_name="${EXP_NAME:-cogrpo8dbg-$(date +%m%d%H%M)}"
model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"
work_dir="${WORK_DIR:-/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train}"

# Use local mini parquet (already in repo, writable mount).
dataset_path="${DATASET_PATH:-/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/data/debug_dapo_mini64.parquet}"

rjob_name="rjob-${exp_name}"

echo "[rjob8dbg] name=${rjob_name} model=${model_path}"
echo "[rjob8dbg] dataset=${dataset_path} work_dir=${work_dir}"

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
  --custom-resources rdma/mlnx_shared="${gpus}" \
  -- bash -lc -- "
    unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY &&
    cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro &&
    export WORK_DIR='${work_dir}' &&
    export MODEL_PATH='${model_path}' &&
    export SRC_PARQUET='${dataset_path}' &&
    export MINI_PARQUET='${dataset_path}' &&
    export MINI_ROWS=64 &&
    export RESPONSE_N=4 &&
    export TRAIN_BSZ=4 &&
    export MICRO_BSZ=1 &&
    export ROLLOUT_N=2 &&
    export STEPS=1 &&
    export MAX_PROMPT_LENGTH=2048 &&
    export TOKEN_CHECK_INTERVAL=512 &&
    export MIN_STEP_TOKENS=512 &&
    export MAX_INTERVENTIONS=2 &&
    export VERIFIER_MAX_NEW_TOKENS=2048 &&
    export CONFIDENCE_THRESHOLD=0.0 &&
    export CF_BRANCH_MAX_EVENTS=1 &&
    export CF_BRANCH_K=2 &&
    bash run_fast_debug_cogrpo_8gpu_mini_dapo.sh
  "
