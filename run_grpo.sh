#!/usr/bin/env bash
set -eo pipefail

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

# 参数设置
model_name="/mnt/shared-storage-user/opencompass-shared/liuhongwei/interns1/repro_exp/s1_model/interns1-8b-hf-1951"
dataset_name="passrate_math_merged"
exp_name="cogrpo-verifier-8b"


cluster="${CLUSTER:-llmit_gpu}"

nnodes=4
n_gpus_per_node=8
gen_tp=2
gpu_memory_utilization=0.8

# 获取日期作为 rjob 名称
datetime=$(date +%m%d)

# 设置 HuggingFace 离线模式和缓存路径
hf_cache_dir="/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub"

echo "rjob-grpo-$exp_name-$datetime"
rjob delete "rjob-grpo-$exp_name-$datetime" || true
rjob submit \
    --name="rjob-grpo-$exp_name-$datetime" \
    --gpu=$n_gpus_per_node \
    --memory=1200000 \
    --cpu=128 \
    --charged-group=$cluster \
    --private-machine=group \
    --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
    --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
    --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
    --image=registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605 \
    -P $nnodes \
    --host-network=true \
    -e DISTRIBUTED_JOB=true \
    --custom-resources rdma/mlnx_shared=$n_gpus_per_node \
    -- bash -c "
        # rjob runtime may source /etc/profile.d scripts that assume unset vars are OK (e.g. ZSH_VERSION).
        # Disable `nounset`/`errexit` defensively to avoid spurious startup failures from cluster wrappers.
        set +u
        set +e
        set -o pipefail
        export HF_HOME=$hf_cache_dir &&
        export HF_HUB_CACHE=$hf_cache_dir &&
        export HUGGINGFACE_HUB_CACHE=$hf_cache_dir &&
        export HF_DATASETS_CACHE=/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/hf_datasets_cache &&
        export HF_DATASETS_OFFLINE=1 &&
        export TRANSFORMERS_OFFLINE=1 &&
        export HF_EVALUATE_OFFLINE=1 &&
        export HF_HUB_OFFLINE=1 &&
        echo 'HuggingFace offline mode configured' &&
        chmod +x /mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/run_multinodes_grpo.sh &&
        /mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/run_multinodes_grpo.sh '$model_name' $nnodes $n_gpus_per_node $gen_tp $gpu_memory_utilization '$dataset_name'
    "
