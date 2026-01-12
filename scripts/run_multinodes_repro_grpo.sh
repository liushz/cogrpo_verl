set -exo pipefail

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro

export PYTHONPATH=$PYTHONPATH:$(pwd)
export SWANLAB_MODE="local"
export SWANLAB_LOG_DIR=/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/swanlab
export VERL_AUTO_PADDING=1
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

model=$1

timestamp=$(date +"%Y%m%d_%H%M%S")

# grpo related
adv_estimator=repro_grpo
use_kl_loss=True
kl_coef=0.0
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28
norm_adv_by_std_in_grpo=True
alpha=0.05

# data related
response_n=32
train_batch_size=256
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
num_generation_per_prompt=8

train_file_name=$6

temperature=1.0
top_p=1.0
top_k=-1

project_name=repro_grpo
exp_name=$(basename "$model")_${train_file_name}_${response_n}k_${alpha//./_}_${timestamp}
save_dir=/mnt/shared-storage-user/opencompass-shared/liuhongwei/CompassVerifier/verl_train/checkpoints/${project_name}/${exp_name}
resume_path=/mnt/shared-storage-user/opencompass-shared/liuhongwei/CompassVerifier/verl_train/checkpoints/repro_grpo/interns1-mini-8b-hf-1951_passrate_math_merged_16k_0_05_20251031_180215/global_step_200

if [ $NODE_RANK == 0 ]
then
ray start --head --port=8266 &

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
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${offload}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}" \
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
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=40 \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async=True \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.norm_adv_by_std_in_grpo="${norm_adv_by_std_in_grpo}" \
    +algorithm.repro_alpha="${alpha}" \
    trainer.val_before_train=True \
    trainer.total_epochs=10 \
    trainer.total_training_steps=500 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${resume_path}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${save_dir}" \
    trainer.rollout_data_dir="${save_dir}/rollout_data" \
    +trainer.rollout_dump_freq=10 \
    trainer.validation_data_dir="${save_dir}/validation_data" \
    trainer.logger=["console","tensorboard","swanlab"] \
    trainer.nnodes="${nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.log_val_generations=100 \
    trainer.save_freq=20 \
    trainer.test_freq=20

else
# init worker
sleep 10
ray start --address=$MASTER_ADDR:8266 &
bash -lc -- "sleep infinity"
fi