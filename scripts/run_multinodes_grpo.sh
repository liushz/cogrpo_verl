set -exo pipefail

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
conda_env_path="${VERL_CONDA_ENV_PATH:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro}"
conda activate "${conda_env_path}"


export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export SWANLAB_MODE="local"
export SWANLAB_LOG_DIR=/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/swanlab
export VERL_AUTO_PADDING=1
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

export REWARD_MODEL_URLS="${REWARD_MODEL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"
export REWARD_MODEL_KEY="${REWARD_MODEL_KEY:-EMPTY}"

# Bypass any HTTP proxy for internal reward endpoints.
reward_proxy_hosts="$(echo "${REWARD_MODEL_URLS}" | tr ',' '\n' | sed -E 's#^[a-zA-Z]+://##; s#/.*$##; s#:.*$##' | paste -sd',' -)"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${reward_proxy_hosts},localhost,127.0.0.1"
export no_proxy="${NO_PROXY}"

model=$1

timestamp=$(date +"%Y%m%d_%H%M%S")

# grpo related
adv_estimator=grpo
use_kl_loss=True
kl_coef=0.0
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28
norm_adv_by_std_in_grpo=True

# data related
response_n=32
train_batch_size=16
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * $response_n))
max_model_len=$((max_prompt_length + max_response_length))

nnodes=$2
n_gpus_per_node=$3

use_dynamic_bsz=True
actor_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))
ref_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))

offload=False

sp=1

gen_tp=$4
gpu_memory_utilization=$5
num_generation_per_prompt=16

train_file_name=$6

temperature=1.0
top_p=1.0
top_k=-1

project_name=grpo
exp_name=$(basename "$model")_${train_file_name}_${response_n}k_${timestamp}
save_dir=/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/checkpoints/${project_name}/${exp_name}

# 检查必需参数
if [ $# -lt 6 ]; then
    echo "错误: 脚本需要 6 个参数:"
    echo "  $0 <model> <nnodes> <n_gpus_per_node> <gen_tp> <gpu_memory_utilization> <train_file_name>"
    exit 1
fi

# 检查必需的环境变量
if [ -z "${NODE_RANK}" ]; then
    echo "警告: NODE_RANK 未设置，默认为 0"
    export NODE_RANK=0
fi

if [ "${NODE_RANK}" != "0" ] && [ -z "${MASTER_ADDR}" ]; then
    echo "错误: Worker 节点需要设置 MASTER_ADDR 环境变量"
    exit 1
fi

if [ "${NODE_RANK:-0}" == "0" ]
then
ray stop --force >/dev/null 2>&1 || true
ray start --head --port=8266 --num-gpus="${n_gpus_per_node}" &
# 等待 Ray 启动
sleep 3

TARGET_GPU=$((nnodes * n_gpus_per_node))
CHECK_INTERVAL=10
get_ray_gpu() {
    # Ray CLI output format varies across versions; use Ray API for robustness.
    python3 - <<'PY' 2>/dev/null || echo 0
import ray
ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
print(int(ray.cluster_resources().get("GPU", 0)))
PY
}
while true; do
    GPU_COUNT=$(get_ray_gpu)
    
    echo "$(date +'%Y-%m-%d %H:%M:%S') - Current total number of GPUs in the Ray cluster: ${GPU_COUNT}"
    
    if [ "${GPU_COUNT}" -eq "${TARGET_GPU}" ]; then
        echo "✅ The total number of GPUs in the Ray cluster has reached ${TARGET_GPU}, exit waiting"
        break 
    else
        echo "⌛ The total number of GPUs in the Ray cluster has not reached the target (${TARGET_GPU}), ${CHECK_INTERVAL} seconds later will check again..."
        sleep "${CHECK_INTERVAL}"  
    fi
done

# 检查数据文件是否存在
if [ ! -f "data/${train_file_name}.parquet" ]; then
    echo "错误: 训练文件不存在: data/${train_file_name}.parquet"
    exit 1
fi
if [ ! -f "data/aime-2024.parquet" ]; then
    echo "警告: 验证文件不存在: data/aime-2024.parquet"
fi

python3 -m verl.trainer.main_ppo \
    data.train_files=$(realpath data/${train_file_name}.parquet) \
    data.val_files=$(realpath data/aime-2024.parquet) \
    data.prompt_key="prompt" \
    data.train_batch_size="${train_batch_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="${model}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.checkpoint.save_contents=["hf_model"] \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_batch_size}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=30 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${offload}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}" \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    +critic.model.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ref_ppo_max_token_len}" \
    actor_rollout_ref.rollout.n="${num_generation_per_prompt}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${actor_ppo_max_token_len}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=40 \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async=True \
    "+reward_model.reward_model_urls='${REWARD_MODEL_URLS}'" \
    "+reward_model.reward_model_key='${REWARD_MODEL_KEY}'" \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.norm_adv_by_std_in_grpo="${norm_adv_by_std_in_grpo}" \
    trainer.val_before_train=True \
    trainer.total_epochs=10 \
    trainer.total_training_steps=1500 \
    trainer.resume_mode=disable \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${save_dir}" \
    trainer.rollout_data_dir="${save_dir}/rollout_data" \
    +trainer.rollout_dump_freq=20 \
    trainer.validation_data_dir="${save_dir}/validation_data" \
    trainer.logger=["console","swanlab"] \
    trainer.nnodes="${nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.log_val_generations=100 \
    trainer.save_freq=20 \
    trainer.test_freq=20

else
# init worker
sleep 10
ray stop --force >/dev/null 2>&1 || true
ray start --address=$MASTER_ADDR:8266 --num-gpus="${n_gpus_per_node}" &
# 等待 Ray 启动
sleep 3
# 保持 worker 节点运行
sleep infinity
fi
