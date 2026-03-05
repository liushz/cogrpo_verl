#!/bin/bash

###############################################################################
# LLaMA-Factory LoRA 训练启动脚本 - Interns1 Verifier 训练
# 数据格式: OpenAI messages 格式 (无需转换)
###############################################################################

cd /mnt/shared-storage-user/liuhongwei/main_works/scripts

############################### 训练配置 ###############################

# 模型配置 (确保路径正确)
model_path="/mnt/shared-storage-user/opencompass-shared/liuhongwei/interns1/repro_exp/s1_model/interns1-8b-hf-1951"

# 数据路径 (OpenAI messages 格式，无需转换)
# dataset_path="/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/scripts/lora_cold_data/verifier_data_output/formated_to_train/verifier_sft_train_data_final_fix_prompt.jsonl"
# dataset_path="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/cache/verifier_data_output/verifier_sft_train_data.jsonl"
# dataset_path="/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/scripts/lora_cold_data/outputs/train/v4go_full02020312_wait_combo02022309_plusall_20260204/train_37_allwait.jsonl"

datetime=$(date +%m%d%H)
# dataset_name="verifier_sft_train_55_waitx2"
# dataset_name="verifier_sft_train_37_waitx2"
dataset_name="verifier_sft_train_37_wait"


dataset_path=""

# 输出目录
export WORK_DIR="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-${dataset_name}-${datetime}"

# 训练超参
global_batch_size=32
epochs=3
lr=5e-5
seq_len=20480
hf_interval=100

# 集群配置
cluster=llmit_gpu
# cluster=opencompass_gpu

gpu_num=8
num_workers=1

# 模型配置/命名
true_model_name="verifier-interns1-lora-cold"

# LoRA 配置
lora_rank=64
lora_alpha=128
lora_target=all
lora_dropout=0.05

# 计算每设备批量大小和梯度累积
# global_batch = per_device × gpu × num_workers × gradient_accumulation
# per_device_train_batch_size=$((global_batch_size / gpu_num / num_workers))
per_device_train_batch_size=2
gradient_accumulation_steps=4

# 数据集名称 (需要在 dataset_info.json 中注册)


############################### 提交任务 ###############################

# Shorten job name to fit within 63 char limit for K8s labels (job_name + task_name must be <= 63)
job_name="verifier-s1-cold-e${epochs}-b${global_batch_size}-${datetime}"

echo "========================================="
echo "任务名称: $job_name"
echo "模型路径: $model_path"
echo "数据路径: $dataset_path"
echo "全局批量大小: $global_batch_size"
echo "每设备批量大小: $per_device_train_batch_size"
echo "学习率: $lr"
echo "轮次: $epochs"
echo "序列长度: $seq_len"
echo "========================================="

# RDMA 选项 (多节点时启用)
if [ $num_workers -gt 1 ]; then
    echo "多节点训练，请求RDMA资源"
    rdma_option="--custom-resources rdma/mlnx_shared=$gpu_num"
else
    echo "单节点训练"
    rdma_option=""
fi

# 提交任务
# rjob delete $job_name
rjob submit \
    --name="$job_name" \
    --gpu=8 \
    --memory=1024000 \
    --cpu=128 \
    --charged-group=$cluster \
    --private-machine=group \
    --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
    --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
    --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --image=registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605 \
    -P $num_workers \
    --host-network=true \
    -e DISTRIBUTED_JOB=true \
    $rdma_option \
    -- bash -c "
        # 设置环境变量
        export HF_HOME=/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub &&
        export HF_HUB_CACHE=/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub &&
        export HUGGINGFACE_HUB_CACHE=/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub &&
        export HF_DATASETS_OFFLINE=1 &&
        export TRANSFORMERS_OFFLINE=1 &&
        export HF_EVALUATE_OFFLINE=1 &&
        export HF_HUB_OFFLINE=1 &&
        echo 'HuggingFace offline mode configured' &&

        # 分布式训练超时配置
        export TORCH_DISTRIBUTED_STORE_TIMEOUT=3600 &&
        export NCCL_TIMEOUT=3600 &&
        export NCCL_IB_TIMEOUT=22 &&
        export NCCL_DEBUG=INFO &&
        export NCCL_ASYNC_ERROR_HANDLING=1 &&
        # 从 JOB_ID 提取数字部分计算 MASTER_PORT，确保十进制解释
        job_id_str="\${JOB_ID:-0}" &&
        job_num=\$(echo "\$job_id_str" | grep -o '[0-9]' | tr -d '\n' | tail -c 4 | sed 's/^0*//') &&
        job_num=\${job_num:-0} &&
        export MASTER_PORT=\$((29500 + job_num % 1000)) &&
        echo 'MASTER_PORT=' \$MASTER_PORT &&
        echo 'Distributed training timeout configured: 3600s' &&

        # 映射分布式环境变量
        export NNODES=\${NODE_COUNT:-1} &&
        export NPROC_PER_NODE=\${PROC_PER_NODE:-$gpu_num} &&
        if [ "\$NNODES" -eq 1 ]; then
            export MASTER_ADDR=127.0.0.1
        else
            export MASTER_ADDR=\${MASTER_ADDR:-\$(hostname -i | awk '{print \$1}')}
        fi &&
        echo 'NNODES=' \$NNODES 'NPROC_PER_NODE=' \$NPROC_PER_NODE &&
        echo 'MASTER_ADDR=' \$MASTER_ADDR &&

        # 激活 conda 环境
        source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh &&
        conda activate /mnt/shared-storage-user/liuhongwei/miniconda3/envs/lamf &&
        export PATH=/mnt/shared-storage-user/liuhongwei/miniconda3/envs/lamf/bin:\$PATH &&
        echo 'Python: '\$(which python) &&
        python --version &&
        python -c 'import llamafactory; print(f\"llamafactory version: {llamafactory.__version__}\")' &&

        # 检查数据集是否已注册
        if python /mnt/shared-storage-user/liuhongwei/main_works/scripts/update_dataset_info.py --show $dataset_name >/dev/null 2>&1; then
            echo 'Dataset $dataset_name already registered, skipping...'
        else
            echo 'Registering dataset $dataset_name...' &&
            python /mnt/shared-storage-user/liuhongwei/main_works/scripts/update_dataset_info.py \\
                --add $dataset_name \\
                --file $dataset_path \\
                --format openai || echo 'Dataset registration warning'
        fi &&

        # 创建输出目录
        mkdir -p $WORK_DIR &&

        # 创建 LLaMA-Factory 配置文件
        config_file='$WORK_DIR/interns1_lora_config.yaml' &&

        cat > \$config_file << EOFCONFIG
### model
model_name_or_path: $model_path
trust_remote_code: true
use_fast_tokenizer: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: $lora_rank
lora_alpha: $lora_alpha
lora_target: $lora_target
lora_dropout: $lora_dropout

### dataset
dataset: $dataset_name
template: qwen3
cutoff_len: $seq_len
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 4
group_by_length: true

### output
output_dir: $WORK_DIR
logging_steps: 10
save_steps: $hf_interval
plot_loss: true
overwrite_output_dir: true
save_only_model: true
save_total_limit: 5
report_to: none

### train
per_device_train_batch_size: $per_device_train_batch_size
gradient_accumulation_steps: $gradient_accumulation_steps
learning_rate: $lr
num_train_epochs: $epochs
lr_scheduler_type: cosine
warmup_ratio: 0.1
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
bf16: true
ddp_timeout: 180000000
optim: adamw_torch_fused
resume_from_checkpoint: null

### eval
# eval_dataset: $dataset_name
# val_size: 0.01
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: $hf_interval
EOFCONFIG

        # 保存配置信息
        touch $WORK_DIR/train_config.txt &&
        echo 'Training config:' > $WORK_DIR/train_config.txt &&
        echo 'model_path: $model_path' >> $WORK_DIR/train_config.txt &&
        echo 'dataset: $dataset_name' >> $WORK_DIR/train_config.txt &&
        echo 'dataset_file: $dataset_path' >> $WORK_DIR/train_config.txt &&
        echo 'lora_rank: $lora_rank' >> $WORK_DIR/train_config.txt &&
        echo 'lora_alpha: $lora_alpha' >> $WORK_DIR/train_config.txt &&
        echo 'lora_target: $lora_target' >> $WORK_DIR/train_config.txt &&
        echo 'per_device_batch: $per_device_train_batch_size' >> $WORK_DIR/train_config.txt &&
        echo 'gradient_accum: $gradient_accumulation_steps' >> $WORK_DIR/train_config.txt &&
        echo 'global_batch: $global_batch_size' >> $WORK_DIR/train_config.txt &&
        echo 'learning_rate: $lr' >> $WORK_DIR/train_config.txt &&
        echo 'epochs: $epochs' >> $WORK_DIR/train_config.txt &&
        echo 'seq_len: $seq_len' >> $WORK_DIR/train_config.txt &&

        # 切换到 LLaMA-Factory 目录
        cd /mnt/shared-storage-user/liuhongwei/main_works/repos/LLaMA-Factory &&

        # 运行训练
        echo 'Starting LLaMA-Factory LoRA training...' &&
        /mnt/shared-storage-user/liuhongwei/miniconda3/envs/lamf/bin/python -m llamafactory.cli train \$config_file 2>&1 | tee $WORK_DIR/train_log.txt &&

        echo 'Training completed. Results saved to: $WORK_DIR'
    "
