set -exo pipefail

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro

# CRITICAL: Uninstall any existing verl package to ensure local code is used
# Set VERL_CLEAN=0 to skip cleanup (default: 1).
VERL_CLEAN=${VERL_CLEAN:-1}
if [ "${VERL_CLEAN}" = "1" ]; then
    echo "=== Uninstalling existing verl package ==="
    pip uninstall -y verl 2>/dev/null || echo "No existing verl package found"

    # CRITICAL: Clear Python bytecode cache to ensure latest code is used
    echo "=== Clearing Python bytecode cache ==="
    find verl -name "*.pyc" -delete 2>/dev/null || true
    find verl -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "Python cache cleared"
else
    echo "=== VERL_CLEAN=0: skip verl uninstall and pycache cleanup ==="
fi

# IMPORTANT: Put current directory at the BEGINNING of PYTHONPATH
# This ensures our local code is loaded instead of the Docker image's code
export PYTHONPATH=$(pwd):$PYTHONPATH

# Verify which verl module will be loaded
echo "=== Verifying PYTHONPATH and verl module location ==="
echo "PYTHONPATH=$PYTHONPATH"
# python3 /mnt/shared-storage-user/liuhongwei/main_works/repos/repro/verify_code_version.py
echo "=== End verification ==="

work_dir=/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train
# work_dir=/mnt/shared-storage-user/liuhongwei/main_works/temp_debug

export SWANLAB_MODE="local"
export SWANLAB_LOG_DIR=${work_dir}/swanlab
export VERL_AUTO_PADDING=1
export VERL_LOG_DIR=${work_dir}/logs
# Ray: record object ref creation sites for `ray memory --group-by=STACK_TRACE`.
export RAY_record_ref_creation_sites=${RAY_record_ref_creation_sites:-1}
# Torch NCCL: enable FlightRecorder so timeouts show the call-site (helps diagnose rare hangs).
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}

# torch.distributed collective timeout (seconds). Keep default 30min unless overridden.
export VERL_NCCL_TIMEOUT_SEC=${VERL_NCCL_TIMEOUT_SEC:-1800}

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
export REWARD_MODEL_URLS="http://100.101.166.1:22005/v1,http://100.101.166.1:22004/v1,http://100.101.166.1:22003/v1,http://100.101.166.1:22002/v1"
export REWARD_MODEL_KEY="EMPTY"
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
export HYDRA_FULL_ERROR=1

model=$1

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

# parallel control/exp rollout (dual vLLM engines)
parallel_control_exp=True

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
# verifier_lora_path="/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/xtuner_train/verifier-interns1-lora-cold-011316/checkpoint-486"  # Path to pretrained verifier LoRA, empty to train from scratch
verifier_lora_path="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-interns1-lora-cold-012422"  # Path to pretrained verifier LoRA, empty to train from scratch

# training mode
# - CO_GRPO_MODE=full (default): update actor + verifier
# - CO_GRPO_MODE=verifier_lora_only: freeze actor; only update verifier LoRA
co_grpo_mode=${CO_GRPO_MODE:-"full"}
freeze_actor=False
if [ "${co_grpo_mode}" = "verifier_lora_only" ]; then
    freeze_actor=True
fi


verifier_max_new_tokens=$((1024 * 2))
verifier_temperature=1.0
verifier_logprobs=1
verifier_max_hint_tokens=512

# verifier intervention mode: "by_response" or "by_step"
verifier_intervention_mode=by_step

# vLLM KV/prefix cache optimization.
# In by_step mode we call vLLM generate() repeatedly; prefix caching can be faster but can also be unstable
# (e.g. "Failed to reset prefix cache because some blocks are not freed yet"). Default to safer off.
rollout_enable_kv_cache_optimization=${rollout_enable_kv_cache_optimization:-""}
if [ -z "${rollout_enable_kv_cache_optimization}" ]; then
    if [ "${verifier_intervention_mode}" = "by_step" ]; then
        rollout_enable_kv_cache_optimization=False
    else
        rollout_enable_kv_cache_optimization=True
    fi
fi

# by_step mode parameters

token_check_interval=4096  # Check verifier every N tokens
min_step_tokens=4096  # Minimum tokens before honoring stop boundaries

entropy_threshold=0.5
use_entropy_filter=True
max_interventions=5
confidence_threshold=0.7
# Minimum tokens before honoring stop boundaries (defaults to token_check_interval if not set)
 
# intervention penalty (weak; applied only on non-improving samples in trainer)
intervention_penalty_freq_coef=0.005
intervention_penalty_len_coef=0.0


# curriculum learning (dynamic control/exp weight)
use_curriculum_weighting=True
curriculum_start_weight=0.7  # Early: 70% control, 30% exp (rely on policy)
curriculum_end_weight=0.5    # Late: 50% control, 50% exp 

# verifier reward shaping (soft boost for positive gap)
verifier_reward_improve_coef=1.0
verifier_reward_mode=${verifier_reward_mode:-headroom}  # gap | headroom
verifier_reward_headroom_min=${verifier_reward_headroom_min:-0.05}
verifier_reward_tie_no_intervention_weight=1.0

# data related (Mini version - smaller batch sizes)
response_n=32  # keep 32k response to avoid truncation, use 8k to debug
train_batch_size=64  # Must satisfy: (train_batch_size * n) % (n_gpus) == 0
	                      # With n=4 and 16 GPUs: 4*4=16, 16%16=0 ✓
max_prompt_length=$((1024))
max_response_length=$((1024 * $response_n))
max_model_len=$((max_prompt_length + max_response_length))

model_max_len=${MODEL_MAX_LEN:-40960}
estimated_hint_tokens=${estimated_hint_tokens:-256}
hint_headroom=$((max_interventions * estimated_hint_tokens))
max_model_len=$((max_prompt_length + max_response_length + hint_headroom))
if [ "${max_model_len}" -gt "${model_max_len}" ]; then
    max_model_len=${model_max_len}
fi
# Verifier prompt length budget (affects both verifier inference + verifier PPO training).
# Setting this too large (e.g. 32k) can OOM during verifier backward because prompts include long partial student responses.
# You can override at runtime: `verifier_max_prompt_length=16384 bash ...`
verifier_max_prompt_length=${verifier_max_prompt_length:-$((1024 * 16))}

# Verifier PPO micro-batching overrides (independent from actor settings).
# Default to safer settings to avoid dynamic packing multiple long verifier prompts into one micro-batch.
# You can override at runtime: `verifier_use_dynamic_bsz=True verifier_ppo_max_token_len_per_gpu=32768 bash ...`
verifier_use_dynamic_bsz=${verifier_use_dynamic_bsz:-False}
verifier_ppo_micro_batch_size_per_gpu=${verifier_ppo_micro_batch_size_per_gpu:-1}
verifier_ppo_max_token_len_per_gpu=${verifier_ppo_max_token_len_per_gpu:-$((verifier_max_prompt_length + verifier_max_new_tokens))}

nnodes=$2
n_gpus_per_node=$3

# Default to split GPUs evenly between control and exp rollout engines.
control_rollout_gpus_per_node=4
exp_rollout_gpus_per_node=4

# Quick debug mode: keep 32k response, reduce overall runtime.
trainer_total_epochs=${trainer_total_epochs:-10}
trainer_total_training_steps=${trainer_total_training_steps:-1000}
trainer_rollout_dump_freq=${trainer_rollout_dump_freq:-20}
trainer_dual_rollout_dump_freq=${trainer_dual_rollout_dump_freq:-10}
trainer_log_val_generations=${trainer_log_val_generations:-100}
trainer_save_freq=${trainer_save_freq:-20}
trainer_test_freq=${trainer_test_freq:-20}
trainer_verifier_lora_sync_freq=${trainer_verifier_lora_sync_freq:-1}
# Save verifier LoRA checkpoints alongside base checkpoints.
# Override at runtime: `trainer_verifier_lora_save_freq=20 bash ...`
trainer_verifier_lora_save_freq=${trainer_verifier_lora_save_freq:-$trainer_save_freq}
# Control rollout weight sync frequency (for parallel_control_exp=True).
# Exporting full FSDP actor state_dict is huge; syncing every step can cause Ray object spilling.
# Override at runtime: `trainer_control_rollout_sync_freq=10 bash ...`
trainer_control_rollout_sync_freq=${trainer_control_rollout_sync_freq:-5}

# Reward computation (Ray task scheduling / object store pressure).
# Override at runtime: `reward_async=false bash ...`
reward_async=${reward_async:-true}

# Validation control:
# - DO_VAL=false (default): disable validation runs (still requires a valid `data.val_files` for dataset init)
# - DO_VAL=true: enable periodic validation by `trainer_test_freq`
do_val=${DO_VAL:-false}
do_val_norm=$(echo "${do_val}" | tr '[:upper:]' '[:lower:]')
if [ "${do_val_norm}" != "true" ]; then
    trainer_test_freq=0
    trainer_log_val_generations=0
fi

# Enable dynamic batch size (automatic micro batch sizing based on token length)
# This allows more efficient GPU utilization and reduces OOM risk
offload=False
# Default: enable dynamic batch size. Allow runtime override:
# `use_dynamic_bsz=false bash run_multinodes_cgrpo.sh ...`
use_dynamic_bsz=${use_dynamic_bsz:-True}
use_dynamic_bsz_norm=$(echo "${use_dynamic_bsz}" | tr '[:upper:]' '[:lower:]')

if [ "${use_dynamic_bsz_norm}" = "true" ]; then
    echo "=== Using dynamic batch size (automatic micro batch sizing) ==="

    # With dynamic_bsz=True, max_token_len_per_gpu is used to determine micro batch size
    # Formula: micro_batch_size = floor(max_token_len / (prompt_len + response_len))

    # Micro batch sizes are still needed for ref/rollout log_prob computation (fallback if dynamic fails)
    ppo_micro_batch_size_per_gpu=1  # Will be overridden by dynamic_bsz
    ref_log_prob_micro_batch_size_per_gpu=1
    rollout_log_prob_micro_batch_size_per_gpu=1
else
    echo "=== Using fixed micro batch size ==="

    ppo_micro_batch_size_per_gpu=2 # Each GPU processes 2 samples
    ref_log_prob_micro_batch_size_per_gpu=2  # Ref model micro batch size
    rollout_log_prob_micro_batch_size_per_gpu=2  # Rollout log_prob micro batch size
fi

actor_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))
ref_ppo_max_token_len=$((2 * max_prompt_length + 2 * max_response_length))
# NOTE: vLLM(v1) 初始化时会用 max_num_batched_tokens 做 profile_run；设太大在启用 LoRA 时可能触发 CUDA illegal memory access。
# 如需提速，建议从 2x 开始逐步上调并观察是否稳定（同时配合 gpu_memory_utilization）。
rollout_max_num_batched_tokens=${rollout_max_num_batched_tokens:-$((2 * max_prompt_length + 2 * max_response_length))}



sp=1

gen_tp=$4
gpu_memory_utilization=$5
num_generation_per_prompt=8  # Must satisfy: (train_batch_size * n) % n_gpus == 0
                             # With train_batch_size=4, n=4: 4*4=16, 16%16=0 ✓

train_file_name=$6
val_file_name=${val_file_name:-aime-2024}

temperature=1.0
top_p=1.0
top_k=-1

project_name=co_grpo_mini
# Build experiment name with key configs
intervention_mode_short=${verifier_intervention_mode//by_/}  # by_step -> step, by_response -> response
curriculum_short="cur${curriculum_start_weight//./_}to${curriculum_end_weight//./_}"
penalty_short="pen${intervention_penalty_freq_coef//./_}"
bsz_suffix="dynamic"
if [ "${use_dynamic_bsz_norm}" != "true" ]; then
    bsz_suffix="fixed"
fi
# Resume knobs:
# - RESUME_DIR: resume in-place from an existing run directory (default_local_dir)
# - RESUME_PATH: resume from a specific global_step_* folder (can branch into a new save_dir)
resume_dir=${RESUME_DIR:-""}
resume_path=${RESUME_PATH:-""}

if [ -n "${resume_dir}" ]; then
    save_dir="${resume_dir}"
    exp_name=$(basename "${save_dir}")
else
    exp_name=$(basename "$model")_${train_file_name}_${response_n}k_${intervention_mode_short}_${curriculum_short}_${penalty_short}_${bsz_suffix}_${timestamp}
    save_dir=${work_dir}/checkpoints/${project_name}/${exp_name}
fi

if [ "${resume_path}" == "" ]; then
    resume_mode="auto"
else
    resume_mode="resume_path"
fi


LOG_FILE=${work_dir}/logs/${VERL_LOG_PREFIX}_rank0.txt

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
    data.val_files=$(realpath data/${val_file_name}.parquet) \
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
	actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz_norm}" \
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
	actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz_norm}" \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ref_log_prob_micro_batch_size_per_gpu}" \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.enable_kv_cache_optimization="${rollout_enable_kv_cache_optimization}" \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
	actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
	actor_rollout_ref.rollout.response_length="${max_response_length}" \
	actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz_norm}" \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${rollout_log_prob_micro_batch_size_per_gpu}" \
	actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ref_ppo_max_token_len}" \
    actor_rollout_ref.rollout.n="${num_generation_per_prompt}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${rollout_max_num_batched_tokens}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    +actor_rollout_ref.rollout.stop_token_ids=[151645] \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=40 \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async=${reward_async} \
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
    +algorithm.verifier_reward_weighting.improve_coef="${verifier_reward_improve_coef}" \
    +algorithm.verifier_reward_weighting.tie_no_intervention_weight="${verifier_reward_tie_no_intervention_weight}" \
    +algorithm.verifier_reward_mode="${verifier_reward_mode}" \
    +algorithm.verifier_reward_headroom_min="${verifier_reward_headroom_min}" \
    verifier.lora_rank="${verifier_lora_rank}" \
    verifier.lora_alpha="${verifier_lora_alpha}" \
    verifier.lora_dropout="${verifier_lora_dropout}" \
    verifier.optim.lr="${verifier_lr}" \
    verifier.loss_weight="${verifier_loss_weight}" \
    verifier.freeze_actor="${freeze_actor}" \
    +verifier.lora_path="${verifier_lora_path}" \
    +verifier.max_new_tokens="${verifier_max_new_tokens}" \
    +verifier.max_prompt_length="${verifier_max_prompt_length}" \
    +verifier.temperature="${verifier_temperature}" \
    +verifier.logprobs="${verifier_logprobs}" \
    +verifier.max_hint_tokens="${verifier_max_hint_tokens}" \
    +verifier.use_dynamic_bsz="${verifier_use_dynamic_bsz}" \
    +verifier.ppo_micro_batch_size_per_gpu="${verifier_ppo_micro_batch_size_per_gpu}" \
    +verifier.ppo_max_token_len_per_gpu="${verifier_ppo_max_token_len_per_gpu}" \
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
    +trainer.dual_rollout_dump_freq="${trainer_dual_rollout_dump_freq}" \
    +trainer.verifier_lora_sync_freq="${trainer_verifier_lora_sync_freq}" \
    +trainer.verifier_lora_save_freq="${trainer_verifier_lora_save_freq}" \
    +trainer.control_rollout_sync_freq="${trainer_control_rollout_sync_freq}" \
    trainer.logger=["console","swanlab"] \
    trainer.parallel_control_exp="${parallel_control_exp}" \
    trainer.control_rollout_gpus_per_node="${control_rollout_gpus_per_node}" \
    trainer.exp_rollout_gpus_per_node="${exp_rollout_gpus_per_node}" \
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
