#!/bin/bash

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

# ========== 基础参数设置 ==========
model_name="/mnt/shared-storage-user/opencompass-shared/liuhongwei/interns1/repro_exp/s1_model/interns1-8b-hf-1951"
dataset_name="passrate_math_merged"
exp_name="cgrpo-verifier-8b-mini"
verl_debug=1

# ========== 集群配置（Mini版本 - 较小资源） ==========
# cluster="llmit_gpu"
cluster="opencompass_gpu"


nnodes=2
n_gpus_per_node=8
gen_tp=2
gpu_memory_utilization=0.7
# cpu 为 16x总卡数 内存按比例计算（参考 run_co_grpo.sh: 1200000 MB for 64 GPUs）

# cpu=$((16 * nnodes * n_gpus_per_node))
# memory=$((120000 * nnodes * n_gpus_per_node))  # 按比例计算，约 128 GB per GPU


# ========== Co-GRPO Mini特有参数 ==========
# Verifier干预模式：
#   verifier_intervention_mode=by_step    # 或 by_response
#
# 干预正则化（防止过度干预）：
#   intervention_penalty_freq_coef=0.1   # 频率惩罚系数
#   intervention_penalty_len_coef=0.01   # 长度惩罚系数
#
# Curriculum Learning（动态权重）：
#   use_curriculum_weighting=True
#   curriculum_start_weight=0.3          # 早期30% control, 70% exp
#   curriculum_end_weight=0.7            # 后期70% control, 30% exp
#
# by_step模式参数：
#   max_interventions=5                  # 最多干预次数（mini版本）
#   token_check_interval=5               # token检查间隔
#   entropy_threshold=0.5                # 熵阈值


# 获取日期作为 rjob 名称
datetime=$(date +%m%d)

# 设置 HuggingFace 离线模式和缓存路径
hf_cache_dir="/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub"

echo "rjob-$exp_name-$datetime"
rjob delete "rjob-$exp_name-$datetime"
rjob submit \
    --name="rjob-$exp_name-$datetime" \
    --gpu=8 \
    --memory=1200000 \
    --cpu=128 \
    --charged-group=$cluster \
    --private-machine=group \
    --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
    --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
    --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
    --image=registry.h.pjlab.org.cn/ailab-opencompass/liuhongwei-workspace:20250826195256 \
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
        echo 'HuggingFace offline mode configured' &&
        echo 'VERL_DEBUG set to $verl_debug - full responses will be logged' &&
        chmod +x /mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/run_multinodes_cgrpo_mini.sh &&
        /mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/run_multinodes_cgrpo_mini.sh '$model_name' $nnodes $n_gpus_per_node $gen_tp $gpu_memory_utilization '$dataset_name' '$verl_debug'
    "
