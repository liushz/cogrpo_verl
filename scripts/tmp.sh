#!/bin/bash
set -euo pipefail

###############################################################################
# 单机(8卡)节点内直接运行的 LoRA 训练脚本（不含 rjob submit）
#
# 说明:
#   - 直接在已分配的 8-GPU 节点/容器内执行即可启动训练
#   - 逻辑对齐 scripts/launch_verifier_interns1_lora_llamafactory_v4plusckptwait.sh
#
# 用法:
#   # 从头训练
#   bash scripts/tmp.sh fpstartfix train_55_waitx2
#   bash scripts/tmp.sh drop_nonfp train_37_waitx2
#
#   # 续训（从最后一个 checkpoint 继续）
#   bash scripts/tmp.sh fpstartfix train_55_waitx2 \
#     --resume-last /path/to/prev_output_dir \
#     --new-epoch 5
#
# 参数:
#   --resume-last DIR   : 从 DIR 中的最后一个 checkpoint 续训
#   --new-epoch N       : 续训后的总 epoch 数（如已训 3 epoch，要训到 5 epoch 则填 5）
#
# 可选环境变量:
#   WORK_DIR_OVERRIDE=/path/to/workdir
#   NPROC_PER_NODE=8          # 默认使用 torch.cuda.device_count()
#   MASTER_PORT=29500         # 默认由 llamafactory 自动找可用端口
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
###############################################################################

# 解析位置参数
variant="${1:-fpstartfix}"
train_set="${2:-train_55_waitx2}"
shift 2 2>/dev/null || true

# 解析可选参数
resume_last_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume-last)
      resume_last_dir="$2"
      shift 2
      ;;
    --new-epoch)
      new_epoch="$2"
      shift 2
      ;;
    *)
      echo "❌ Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

data_dir_fpstartfix="/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/scripts/lora_cold_data/outputs/train/v4go_full02020312_wait_combo02022309_plusall_plus_v4ckptwait_fpstartfix_20260223"
data_dir_drop_nonfp="/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/scripts/lora_cold_data/outputs/train/v4go_full02020312_wait_combo02022309_plusall_plus_v4ckptwait_drop_nonfp_fpstartfix_20260223"

case "${variant}" in
  fpstartfix|all)
    dataset_dir="${data_dir_fpstartfix}"
    vtag="fpfix"
    ;;
  drop_nonfp|drop)
    dataset_dir="${data_dir_drop_nonfp}"
    vtag="drop"
    ;;
  *)
    echo "❌ Unknown VARIANT: ${variant}" >&2
    exit 2
    ;;
esac

case "${train_set}" in
  train_37_allwait|train_37_waitx2|train_55_waitx2) ;;
  *)
    echo "❌ Unknown TRAIN_SET: ${train_set}" >&2
    exit 2
    ;;
esac

set_label="${train_set#train_}" # 37_allwait / 37_waitx2 / 55_waitx2
datetime="$(date +%m%d%H%M)"

# RL 起点模型（见 mds/MERGE.md）
model_path="/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170"

# 数据 (OpenAI messages 格式)
dataset_name="verifier_sft_train_${set_label}_${vtag}_plus_v4ckptwait_20260223"
dataset_path="${dataset_dir}/${train_set}.jsonl"

if [[ ! -f "${dataset_path}" ]]; then
  echo "❌ dataset_path not found: ${dataset_path}" >&2
  exit 1
fi

# 输出目录
work_root="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp"
work_dir_default="${work_root}/verifier-lora-cold-${dataset_name}-${datetime}"
export WORK_DIR="${WORK_DIR_OVERRIDE:-${work_dir_default}}"
mkdir -p "${WORK_DIR}"

############################### 续训配置 ###############################

resume_checkpoint=""
if [[ -n "${resume_last_dir:-}" ]]; then
  if [[ ! -d "${resume_last_dir}" ]]; then
    echo "❌ resume_last_dir not found: ${resume_last_dir}" >&2
    exit 1
  fi
  # 查找最后一个 checkpoint（按 step 编号最大的）
  last_ckpt="$(find "${resume_last_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -t- -k2 -n | tail -1)"
  if [[ -n "${last_ckpt}" ]]; then
    resume_checkpoint="${last_ckpt}"
    echo "📌 Found checkpoint to resume: ${resume_checkpoint}"
  else
    echo "❌ No checkpoint-* found in ${resume_last_dir}" >&2
    exit 1
  fi
  # 续训时复用原输出目录
  export WORK_DIR="${resume_last_dir}"
  # 续训时不要覆盖输出目录
  overwrite_output_dir="false"
fi

############################### 训练配置 ###############################

global_batch_size=32
epochs=5

# 覆盖 epoch 数（续训场景）
if [[ -n "${new_epoch:-}" ]]; then
  epochs="${new_epoch}"
  echo "📌 Overriding epochs to: ${epochs}"
fi
lr=5e-5
seq_len=20480
hf_interval=200

per_device_train_batch_size=2
gradient_accumulation_steps=4

lora_rank=64
lora_alpha=128
lora_target=all
lora_dropout=0.05

############################### 环境变量 ###############################

export HF_HOME=/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub
export HF_HUB_CACHE=/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub
export HUGGINGFACE_HUB_CACHE=/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1

export TORCH_DISTRIBUTED_STORE_TIMEOUT=3600
export NCCL_TIMEOUT=3600
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# 单机默认分布式参数（llamafactory 会自动 torchrun 多进程）
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# MASTER_PORT 可不设：llamafactory 会自动找可用端口

############################### Conda / 依赖 ###############################

source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liuhongwei/miniconda3/envs/lamf
export PATH=/mnt/shared-storage-user/liuhongwei/miniconda3/envs/lamf/bin:$PATH

echo "Python: $(which python)"
python --version
python -c 'import llamafactory; print(f"llamafactory version: {llamafactory.__version__}")'

visible_gpu_count="$(
python - <<'PY'
import torch

print(torch.cuda.device_count())
PY
)"
if [[ "${visible_gpu_count}" -lt 1 ]]; then
  echo "❌ torch.cuda.device_count() == 0. Are you on a GPU node?" >&2
  exit 3
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-${visible_gpu_count}}"
if [[ "${NPROC_PER_NODE}" -gt "${visible_gpu_count}" ]]; then
  echo "❌ NPROC_PER_NODE=${NPROC_PER_NODE} > visible_gpu_count=${visible_gpu_count}." >&2
  exit 3
fi

echo "variant=${variant} train_set=${train_set}"
echo "model_path=${model_path}"
echo "dataset_name=${dataset_name}"
echo "dataset_path=${dataset_path}"
echo "WORK_DIR=${WORK_DIR}"
echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT:-<auto>}"

############################### dataset_info 注册 ###############################

llamafactory_repo="/mnt/shared-storage-user/liuhongwei/main_works/repos/LLaMA-Factory"
dataset_info_json="${llamafactory_repo}/data/dataset_info.json"
if [[ ! -f "${dataset_info_json}" ]]; then
  echo "❌ dataset_info_json not found: ${dataset_info_json}" >&2
  exit 4
fi

python - "${dataset_info_json}" "${dataset_name}" "${dataset_path}" <<'PY'
import fcntl
import json
import sys

p, name, file_path = sys.argv[1:4]
cfg = {
    "file_name": file_path,
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    # NOTE: LLaMA-Factory DatasetAttr.join() 会把 tags 里缺失字段覆盖成 None，
    # 因此需要把 role/content/user/assistant/system/observation/function 都显式写全。
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
        "observation_tag": "observation",
        "function_tag": "function_call",
    },
}

with open(p, "r+", encoding="utf-8") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    d = json.load(f)
    existed = name in d
    d[name] = cfg
    f.seek(0)
    f.truncate()
    json.dump(d, f, ensure_ascii=False, indent=2)
    f.flush()
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

print("dataset_info:", ("exists" if existed else "add"), name)
PY

############################### 写 config + 开始训练 ###############################

config_file="${WORK_DIR}/interns1_lora_config.yaml"
cat > "${config_file}" <<EOFCONFIG
### model
model_name_or_path: ${model_path}
trust_remote_code: true
use_fast_tokenizer: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: ${lora_rank}
lora_alpha: ${lora_alpha}
lora_target: ${lora_target}
lora_dropout: ${lora_dropout}

### dataset
dataset: ${dataset_name}
template: qwen3
cutoff_len: ${seq_len}
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 4
group_by_length: true

### output
output_dir: ${WORK_DIR}
logging_steps: 10
save_steps: ${hf_interval}
plot_loss: true
overwrite_output_dir: ${overwrite_output_dir:-true}
save_only_model: true
save_total_limit: 5
report_to: none

### train
per_device_train_batch_size: ${per_device_train_batch_size}
gradient_accumulation_steps: ${gradient_accumulation_steps}
learning_rate: ${lr}
num_train_epochs: ${epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
bf16: true
ddp_timeout: 180000000
optim: adamw_torch_fused
resume_from_checkpoint: ${resume_checkpoint:-null}

### eval
# eval_dataset: ${dataset_name}
# val_size: 0.01
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: ${hf_interval}
EOFCONFIG

cat > "${WORK_DIR}/train_config.txt" <<EOFINFO
model_path: ${model_path}
dataset: ${dataset_name}
dataset_file: ${dataset_path}
lora_rank: ${lora_rank}
lora_alpha: ${lora_alpha}
lora_target: ${lora_target}
per_device_batch: ${per_device_train_batch_size}
gradient_accum: ${gradient_accumulation_steps}
global_batch: ${global_batch_size}
learning_rate: ${lr}
epochs: ${epochs}
seq_len: ${seq_len}
resume_from_checkpoint: ${resume_checkpoint:-null}
NNODES: ${NNODES}
NODE_RANK: ${NODE_RANK}
NPROC_PER_NODE: ${NPROC_PER_NODE}
MASTER_ADDR: ${MASTER_ADDR}
MASTER_PORT: ${MASTER_PORT:-<auto>}
EOFINFO

cd "${llamafactory_repo}"
echo "Starting LLaMA-Factory LoRA training..."
python -m llamafactory.cli train "${config_file}" 2>&1 | tee "${WORK_DIR}/train_log.txt"

echo "Training completed. Results saved to: ${WORK_DIR}"
