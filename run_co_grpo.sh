#!/bin/bash

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

# ========== 基础参数设置 ==========
model_name="/mnt/shared-storage-user/opencompass-shared/liuhongwei/interns1/repro_exp/s1_model/interns1-8b-hf-1951"
dataset_name="passrate_math_merged"
exp_name="grpo-verifier-8b-headroom"
verl_debug=1

# ========== 训练模式 ==========
# full: 正常训练 actor + verifier
# verifier_lora_only: freeze actor（不更新），只更新 verifier LoRA
co_grpo_mode=${CO_GRPO_MODE:-"full"}

# ========== Resume 配置 ==========
# 原地续跑（推荐）：填已有实验的 default_local_dir（包含 global_step_* 与 latest_checkpointed_iteration.txt）
# - 置空则从头跑新实验
resume_dir=""
# 分叉续跑（可选）：指定某个 global_step_* 目录；与 resume_dir 二选一（resume_dir 优先）
resume_path=""

# ========== 集群配置（Mini版本 - 较小资源） ==========
cluster="llmit_gpu"
# cluster="opencompass_gpu"


nnodes=2
n_gpus_per_node=8
gen_tp=1
gpu_memory_utilization=0.8
# NOTE: vLLM(v1) 启动会做 profile_run 以初始化 KV cache；gpu_memory_utilization 过高时可能直接触发 CUDA 报错（尤其启用 LoRA）。
# cpu 为 16x总卡数 内存按比例计算（参考 run_co_grpo.sh: 1200000 MB for 64 GPUs）

# cpu=$((16 * nnodes * n_gpus_per_node))
# memory=$((120000 * nnodes * n_gpus_per_node))  # 按比例计算，约 128 GB per GPU


# 获取日期作为 rjob 名称
datetime=$(date +%m%d)


# 设置 HuggingFace 离线模式和缓存路径
hf_cache_dir="/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub"

echo "rjob-$exp_name-$datetime"
rjob delete "rjob-$exp_name-$datetime"
rjob submit \
    --name="rjob-$exp_name-$datetime" \
    --gpu=$n_gpus_per_node \
    --memory=1200000 \
    --cpu=128 \
    --charged-group=$cluster \
    --private-machine=group \
    --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
    --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
    --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
    --image=registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605 \
    -P $nnodes \
    --host-network=true \
    -e DISTRIBUTED_JOB=true \
    --custom-resources rdma/mlnx_shared=$n_gpus_per_node \
    -- bash -c "
        export HF_HOME=$hf_cache_dir &&
        export HF_HUB_CACHE=$hf_cache_dir &&
        export HUGGINGFACE_HUB_CACHE=$hf_cache_dir &&
        export HF_DATASETS_CACHE=/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/hf_datasets_cache &&
        export HF_DATASETS_OFFLINE=1 &&
        export TRANSFORMERS_OFFLINE=1 &&
        export HF_EVALUATE_OFFLINE=1 &&
        export HF_HUB_OFFLINE=1 &&
        export VERL_DEBUG=$verl_debug &&
        export CO_GRPO_MODE='${co_grpo_mode}' &&
        export RESUME_DIR='${resume_dir}' &&
        export RESUME_PATH='${resume_path}' &&
        echo 'HuggingFace offline mode configured' &&
        echo 'VERL_DEBUG set to $verl_debug - full responses will be logged' &&
        chmod +x /mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/run_multinodes_cgrpo.sh &&
        /mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/run_multinodes_cgrpo.sh '$model_name' $nnodes $n_gpus_per_node $gen_tp $gpu_memory_utilization '$dataset_name' '$verl_debug'
    "
