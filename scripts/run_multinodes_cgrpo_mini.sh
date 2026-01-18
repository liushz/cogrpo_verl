set -exo pipefail

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro

# CRITICAL: Uninstall any existing verl package to ensure local code is used
echo "=== Uninstalling existing verl package ==="
pip uninstall -y verl 2>/dev/null || echo "No existing verl package found"

# CRITICAL: Clear Python bytecode cache to ensure latest code is used
echo "=== Clearing Python bytecode cache ==="
find verl -name "*.pyc" -delete 2>/dev/null || true
find verl -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "Python cache cleared"

# IMPORTANT: Put current directory at the BEGINNING of PYTHONPATH
# This ensures our local code is loaded instead of the Docker image's code
export PYTHONPATH=$(pwd):$PYTHONPATH

# Verify which verl module will be loaded
echo "=== Verifying PYTHONPATH and verl module location ==="
echo "PYTHONPATH=$PYTHONPATH"
python3 /mnt/shared-storage-user/liuhongwei/main_works/repos/repro/verify_code_version.py
echo "=== End verification ==="

export SWANLAB_MODE="local"
export SWANLAB_LOG_DIR=/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/swanlab
export VERL_AUTO_PADDING=1
export VERL_LOG_DIR=/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/logs

# Apply vLLM Qwen3 fix at runtime (patch vLLM to handle HF weight names)
python3 -c "
import sys
from pathlib import Path

# Try to patch vLLM Qwen3Attention to add load_weights method
try:
    import vllm.model_executor.models.qwen3 as qwen3_module
    from vllm.model_executor.models.utils import default_weight_loader

    Qwen3Attention = qwen3_module.Qwen3Attention

    # Check if load_weights already exists
    if not hasattr(Qwen3Attention, 'load_weights'):
        print('Patching vLLM Qwen3Attention to add load_weights method...')

        def load_weights(self, weights):
            '''Load weights mapping HF q_proj/k_proj/v_proj to vLLM qkv_proj'''
            from collections.abc import Iterable
            stacked_params_mapping = [
                ('qkv_proj', 'q_proj', 'q'),
                ('qkv_proj', 'k_proj', 'k'),
                ('qkv_proj', 'v_proj', 'v'),
            ]
            params_dict = dict(self.named_parameters(remove_duplicate=False))
            loaded_params = set()

            for name, loaded_weight in weights:
                if 'rotary_emb.inv_freq' in name:
                    continue
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    vllm_name = name.replace(weight_name, param_name)
                    if vllm_name not in params_dict:
                        continue
                    param = params_dict[vllm_name]
                    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(vllm_name)
                    break
            return loaded_params

        # Add the method to the class
        Qwen3Attention.load_weights = load_weights
        print('vLLM Qwen3Attention patch applied successfully!')
    else:
        print('vLLM Qwen3Attention already has load_weights method')
except Exception as e:
    print(f'Warning: Failed to patch vLLM: {e}', file=sys.stderr)
"
# export VERL_LOGGING_LEVEL=DEBUG  # too many logs
export REWARD_MODEL_URLS="http://100.99.37.1:21002/v1,http://100.99.37.1:21000/v1,http://100.99.37.1:21003/v1,http://100.99.37.1:21001/v1"
export REWARD_MODEL_KEY="EMPTY"
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
export HYDRA_FULL_ERROR=1

model=$1
debug_quick=${7:-0}

timestamp=$(date +"%Y%m%d_%H%M%S")
# Use timestamped log prefix; actual files will append _rank{rank}.txt
export VERL_LOG_PREFIX="verl_log_${timestamp}"

# co-grpo related
adv_estimator=co_grpo
use_kl_loss=True
kl_coef=0.0
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28
norm_adv_by_std_in_grpo=True
control_group_weight=0.5

# lora_rank=64
# lora_alpha=128
# lora_target=all
# lora_dropout=0.05

# verifier related
verifier_lora_rank=64
verifier_lora_alpha=128
verifier_lora_dropout=0.05
verifier_lr=1e-5
verifier_loss_weight=1.0
verifier_lora_path="/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/xtuner_train/verifier-interns1-lora-cold-011316/checkpoint-486"  # Path to pretrained verifier LoRA, empty to train from scratch
verifier_max_new_tokens=$((1024 * 2))
verifier_temperature=1.0
verifier_logprobs=1
verifier_max_hint_tokens=128

# verifier intervention mode: "by_response" or "by_step"
verifier_intervention_mode=by_step

# by_step mode parameters

token_check_interval=1024  # Check verifier every N tokens
min_step_tokens=1024  # Minimum tokens before honoring stop boundaries

entropy_threshold=0.5
use_entropy_filter=True
max_interventions=5
confidence_threshold=0.7
# Minimum tokens before honoring stop boundaries (defaults to token_check_interval if not set)
 
# intervention penalty (prevent over-intervention)
intervention_penalty_freq_coef=0.01
intervention_penalty_len_coef=0.001

# curriculum learning (dynamic control/exp weight)
use_curriculum_weighting=True
curriculum_start_weight=0.7  # Early: 70% control, 30% exp (rely on policy)
curriculum_end_weight=0.3    # Late: 30% control, 70% exp (rely on verifier)

# data related (Mini version - smaller batch sizes)
response_n=32  # keep 32k response to avoid truncation, use 8k to debug
train_batch_size=8  # Must satisfy: (train_batch_size * n) % (n_gpus) == 0
	                      # With n=4 and 16 GPUs: 4*4=16, 16%16=0 ✓
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * $response_n))
max_model_len=$((max_prompt_length + max_response_length))
verifier_max_prompt_length="${max_response_length}"
# vLLM scheduler safety:
# max_num_batched_tokens is the total token budget for one scheduling batch in vLLM.
# Setting it too large (e.g. 2*(prompt+response) here) can cause vLLM to pack too much work and OOM.
# Keep it close to max_model_len to bound peak KV/cache usage.
rollout_max_num_batched_tokens="${max_model_len}"

nnodes=$2
n_gpus_per_node=$3

# Quick debug mode: keep 32k response, reduce overall runtime.
trainer_total_epochs=10
trainer_total_training_steps=500
trainer_rollout_dump_freq=10
trainer_log_val_generations=100
trainer_save_freq=20
trainer_test_freq=20
trainer_verifier_lora_sync_freq=10

if [ "${debug_quick}" = "1" ]; then
    echo "=== debug_quick=1: enable fast debug knobs (keep response_n=${response_n}k) ==="
    train_batch_size=2
    verifier_max_new_tokens=512
    verifier_max_hint_tokens=128
    max_interventions=1
    # Keep prompts smaller for verifier and ensure by_step path is exercised early.
    token_check_interval=1024
    min_step_tokens=1024
    confidence_threshold=0.0
    use_entropy_filter=False
    entropy_threshold=0.0

    trainer_total_epochs=1
    trainer_total_training_steps=2
    trainer_rollout_dump_freq=1
    trainer_log_val_generations=0
    trainer_save_freq=0
    trainer_test_freq=0
    trainer_verifier_lora_sync_freq=1
fi

# Disable dynamic bsz and explicitly set micro batch size per GPU
# Note: micro=2 was used before and caused OOM/NCCL timeout at ~150GB
#       micro=1 would be safer (~75GB) but slower
#       micro=2 with offload=True might work, but monitor carefully
use_dynamic_bsz=False
ppo_micro_batch_size_per_gpu=1  # Each GPU processes 2 samples
ref_log_prob_micro_batch_size_per_gpu=1  # Ref model micro batch size
rollout_log_prob_micro_batch_size_per_gpu=1  # Rollout log_prob micro batch size
# actor_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))
# ref_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))
actor_ppo_max_token_len=$((ppo_micro_batch_size_per_gpu * (max_prompt_length + max_response_length)))
ref_ppo_max_token_len=$((ref_log_prob_micro_batch_size_per_gpu * (max_prompt_length + max_response_length)))

offload=True

sp=1

gen_tp=$4
gpu_memory_utilization=$5
num_generation_per_prompt=8  # Must satisfy: (train_batch_size * n) % n_gpus == 0
                             # With train_batch_size=4, n=4: 4*4=16, 16%16=0 ✓

train_file_name=$6

temperature=1.0
top_p=1.0
top_k=-1

project_name=co_grpo_mini
# Build experiment name with key configs
intervention_mode_short=${verifier_intervention_mode//by_/}  # by_step -> step, by_response -> response
curriculum_short="cur${curriculum_start_weight//./_}to${curriculum_end_weight//./_}"
penalty_short="pen${intervention_penalty_freq_coef//./_}"
exp_name=$(basename "$model")_${train_file_name}_${response_n}k_${intervention_mode_short}_${curriculum_short}_${penalty_short}_${timestamp}
save_dir=/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/checkpoints/${project_name}/${exp_name}
resume_path=""

if [ "${resume_path}" == "" ]; then
    resume_mode="auto"
else
    resume_mode="resume_path"
fi


LOG_FILE=/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/logs/${VERL_LOG_PREFIX}_rank0.txt

if [ $NODE_RANK == 0 ]
then
ray start --head --port=8266 &

echo "=== Log will be saved to: $LOG_FILE ==="
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-DEBUG}
export VERL_EXPERIMENT_ID="${exp_name}"

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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ref_log_prob_micro_batch_size_per_gpu}" \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.response_length="${max_response_length}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ref_ppo_max_token_len}" \
    actor_rollout_ref.rollout.n="${num_generation_per_prompt}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${rollout_max_num_batched_tokens}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    +actor_rollout_ref.rollout.stop_token_ids=[151645] \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${rollout_log_prob_micro_batch_size_per_gpu}" \
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
    algorithm.control_group_weight="${control_group_weight}" \
    algorithm.verifier_intervention_mode="${verifier_intervention_mode}" \
    algorithm.token_check_interval="${token_check_interval}" \
    algorithm.entropy_threshold="${entropy_threshold}" \
    algorithm.use_entropy_filter="${use_entropy_filter}" \
    +algorithm.min_step_tokens="${min_step_tokens}" \
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
    +verifier.max_new_tokens="${verifier_max_new_tokens}" \
    +verifier.max_prompt_length="${verifier_max_prompt_length}" \
    +verifier.temperature="${verifier_temperature}" \
    +verifier.logprobs="${verifier_logprobs}" \
    +verifier.max_hint_tokens="${verifier_max_hint_tokens}" \
    trainer.val_before_train=False \
    trainer.total_epochs="${trainer_total_epochs}" \
    trainer.total_training_steps="${trainer_total_training_steps}" \
    trainer.resume_mode="${resume_mode}" \
    trainer.resume_from_path="${resume_path}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${save_dir}" \
    trainer.rollout_data_dir="${save_dir}/rollout_data" \
    +trainer.rollout_dump_freq="${trainer_rollout_dump_freq}" \
    +trainer.verifier_lora_sync_freq="${trainer_verifier_lora_sync_freq}" \
    trainer.validation_data_dir="${save_dir}/validation_data" \
    trainer.logger=["console","tensorboard","swanlab"] \
    trainer.nnodes="${nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.log_val_generations="${trainer_log_val_generations}" \
    trainer.save_freq="${trainer_save_freq}" \
    trainer.test_freq="${trainer_test_freq}" \
    2>&1 | tee "$LOG_FILE"

    # === Memory safety knobs === \
    # vLLM swap must be under engine_kwargs.vllm to take effect \
    # +actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=64 \
    # Verifier offload to CPU (requires Hydra + key to create subtree) \
    # +verifier.fsdp_config.param_offload=True \
    # +verifier.fsdp_config.optimizer_offload=True \

else
# init worker
sleep 10
ray start --address=$MASTER_ADDR:8266 &
bash -lc -- "sleep infinity"
fi
