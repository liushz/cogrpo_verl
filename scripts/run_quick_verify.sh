#!/bin/bash
set -exo pipefail

# DEBUG MARKER: Quick verify script started
echo "==== QUICK VERIFY SCRIPT STARTED ====" >&2
echo "==== n_gpus_per_node=$n_gpus_per_node ====" >&2

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro

export PYTHONPATH=$PYTHONPATH:$(pwd)
export SWANLAB_MODE="local"
export SWANLAB_LOG_DIR=/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/swanlab
export VERL_AUTO_PADDING=1
export VERL_LOG_DIR=/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/logs
export REWARD_MODEL_URLS="http://100.99.37.1:21002/v1,http://100.99.37.1:21000/v1,http://100.99.37.1:21003/v1,http://100.99.37.1:21001/v1"
export REWARD_MODEL_KEY="EMPTY"
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
export HYDRA_FULL_ERROR=1

model=$1
# Default to 1 node, 2 GPUs for quick verification
nnodes=${2:-1}
n_gpus_per_node=${3:-2}

timestamp=$(date +"%Y%m%d_%H%M%S")

# co-grpo related
adv_estimator=co_grpo
use_kl_loss=True
kl_coef=0.0
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28
norm_adv_by_std_in_grpo=True
control_group_weight=0.5

# verifier related
verifier_lora_rank=64
verifier_lora_alpha=128
verifier_lora_dropout=0.05
verifier_lr=1e-5
verifier_loss_weight=1.0
verifier_lora_path="/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/xtuner_train/verifier-interns1-lora-cold-010819/checkpoint-960"

# verifier intervention mode: "by_response" or "by_step"
verifier_intervention_mode=by_step

# by_step mode parameters
token_check_interval=5
entropy_threshold=0.5
use_entropy_filter=True
max_interventions=5
confidence_threshold=0.7

# intervention penalty
intervention_penalty_freq_coef=0.1
intervention_penalty_len_coef=0.01

# curriculum learning
use_curriculum_weighting=True
curriculum_start_weight=0.3
curriculum_end_weight=0.7

# QUICK VERIFICATION: Smaller batch sizes and shorter sequences
response_n=8  # Reduced from 32 to 8 for faster verification
train_batch_size=4  # Reduced from 16 to 4
max_prompt_length=512  # Reduced from 2048 to 512
max_response_length=$((1024 * $response_n))
max_model_len=$((max_prompt_length + max_response_length))

use_dynamic_bsz=True
actor_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))
ref_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))

offload=False
sp=1

gen_tp=${4:-1}
gpu_memory_utilization=${5:-0.8}
num_generation_per_prompt=4  # Reduced from 8 to 4 for faster verification

train_file_name=${6:-passrate_math_merged}

temperature=1.0
top_p=1.0
top_k=-1

project_name=co_grpo_quick_verify
intervention_mode_short=${verifier_intervention_mode//by_/}
curriculum_short="cur${curriculum_start_weight//./_}to${curriculum_end_weight//./_}"
penalty_short="pen${intervention_penalty_freq_coef//./_}"
exp_name=$(basename "$model")_${train_file_name}_${response_n}k_${intervention_mode_short}_${curriculum_short}_${penalty_short}_${timestamp}
save_dir=/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/checkpoints/${project_name}/${exp_name}
resume_path=""
resume_mode=auto

LOG_FILE=/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/logs/verl_log_rank0_${timestamp}.txt

# Force NODE_RANK to 0 for single-node training
# This overrides any environment variable set by rjob
NODE_RANK=0
echo "=== Forcing NODE_RANK=$NODE_RANK for single-node training ===" >&2
echo "=== n_gpus_per_node=$n_gpus_per_node, total_training_steps=3 ===" >&2

if [ $NODE_RANK == 0 ]
then
ray start --head --port=8266 &

echo "=== Log will be saved to: $LOG_FILE ===" >&2
echo "=== About to run training with total_training_steps=3, n_gpus_per_node=$n_gpus_per_node ===" >&2

TARGET_GPU=$((nnodes * n_gpus_per_node))
CHECK_INTERVAL=10
get_ray_gpu() {
    ray status 2>/dev/null | grep -oP '/\K[\d.]+(?=\s+GPU)' | awk '{print int($1)}' || echo 0
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
    actor_rollout_ref.actor.checkpoint.save_contents=[hf_model] \
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
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${offload}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}" \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    # GRPO doesn't need critic, disable it
    # critic.model.path="${model}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    # Temporarily disable LoRA to ensure stable training
    # +actor_rollout_ref.rollout.lora_kwargs.enable_lora=True \
    # +actor_rollout_ref.rollout.lora_kwargs.max_loras=2 \
    # +actor_rollout_ref.rollout.lora_kwargs.max_lora_rank="${verifier_lora_rank}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=40 \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async=True \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.norm_adv_by_std_in_grpo="${norm_adv_by_std_in_grpo}" \
    algorithm.control_group_weight="${control_group_weight}" \
    algorithm.verifier_intervention_mode="${verifier_intervention_mode}" \
    algorithm.token_check_interval="${token_check_interval}" \
    algorithm.entropy_threshold="${entropy_threshold}" \
    algorithm.use_entropy_filter="${use_entropy_filter}" \
    +algorithm.max_interventions="${max_interventions}" \
    +algorithm.confidence_threshold="${confidence_threshold}" \
    +algorithm.intervention_penalty.freq_coef="${intervention_penalty_freq_coef}" \
    +algorithm.intervention_penalty.len_coef="${intervention_penalty_len_coef}" \
    +algorithm.use_curriculum_weighting="${use_curriculum_weighting}" \
    +algorithm.curriculum_start_weight="${curriculum_start_weight}" \
    +algorithm.curriculum_end_weight="${curriculum_end_weight}" \
    verifier.lora_rank="${verifier_lora_rank}" \
    verifier.lora_alpha="${verifier_lora_alpha}" \
    verifier.lora_dropout="${verifier_lora_dropout}" \
    verifier.optim.lr="${verifier_lr}" \
    verifier.loss_weight="${verifier_loss_weight}" \
    +verifier.lora_path="${verifier_lora_path}" \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    trainer.resume_mode="${resume_mode}" \
    trainer.resume_from_path="${resume_path}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${save_dir}" \
    trainer.rollout_data_dir="${save_dir}/rollout_data" \
    +trainer.rollout_dump_freq=1 \
    trainer.validation_data_dir="${save_dir}/validation_data" \
    trainer.logger=["console","tensorboard","swanlab"] \
    trainer.nnodes="${nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.log_val_generations=10 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    2>&1 | tee "$LOG_FILE"

else
# init worker
sleep 10
ray start --address=$MASTER_ADDR:8266 &
bash -lc -- "sleep infinity"
fi

