#!/usr/bin/env bash
# NOTE: avoid nounset (-u) here; some cluster images install DEBUG traps that
# reference unset vars (e.g. ZSH_VERSION), which can crash startup.
set -eo pipefail

model="$1"
nnodes="$2"
n_gpus_per_node="$3"
gen_tp="$4"
gpu_memory_utilization="$5"
train_file_name="$6"
verl_debug="${7:-0}"

# Positional args from run_co_grpo.sh (single entry ownership)
exp_name="${8:-}"
work_dir="${9:-/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train}"
preset="${10:-full}"
co_grpo_mode="${11:-}"
resume_dir="${12:-}"
resume_path="${13:-}"
verifier_lora_path="${14:-}"

response_n="${15:-16}"
train_batch_size="${16:-32}"
token_check_interval="${17:-2048}"
min_step_tokens="${18:-2048}"
max_interventions="${19:-5}"
verifier_max_hint_tokens="${20:-512}"
estimated_hint_tokens="${21:-512}"

control_group_weight="${22:-0.5}"
use_curriculum_weighting="${23:-True}"
curriculum_start_weight="${24:-0.3}"
curriculum_end_weight="${25:-0.5}"
intervention_penalty_freq_coef="${26:-0.01}"
intervention_penalty_len_coef="${27:-0.0}"
verifier_reward_mode="${28:-headroom}"
verifier_reward_improve_coef="${29:-1.0}"
verifier_reward_headroom_min="${30:-0.05}"

trainer_rollout_dump_freq="${31:-10}"
trainer_dual_rollout_dump_freq="${32:-5}"
use_dynamic_bsz="${33:-False}"
micro_bsz_per_gpu="${34:-2}"
verifier_reward_tie_no_intervention_weight="${35:-1.0}"

verifier_intervention_mode="${36:-by_step}"
confidence_threshold="${37:-0.7}"
entropy_threshold="${38:-0.5}"
use_entropy_filter="${39:-True}"

verifier_lora_rank="${40:-64}"
verifier_lora_alpha="${41:-128}"
verifier_lora_dropout="${42:-0.05}"
verifier_lr="${43:-1e-5}"
verifier_loss_weight="${44:-1.0}"
verifier_max_new_tokens="${45:-4096}"
verifier_logprobs="${46:-1}"
verifier_temperature="${47:-1.0}"

trainer_total_epochs="${48:-10}"
trainer_total_training_steps="${49:-1000}"
trainer_save_freq="${50:-20}"
trainer_test_freq="${51:--1}"
trainer_log_val_generations="${52:-False}"
trainer_control_rollout_sync_freq="${53:-5}"
trainer_verifier_lora_sync_freq="${54:-1}"
trainer_verifier_lora_save_freq="${55:-20}"

max_prompt_length="${56:-1024}"

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro

export PYTHONPATH="$(pwd)"

export SWANLAB_MODE="local"
export VERL_AUTO_PADDING="1"
export RAY_record_ref_creation_sites="1"
export VERL_NCCL_TIMEOUT_SEC="1800"
export VERL_DEBUG="${verl_debug}"
export REWARD_MODEL_URLS="${REWARD_MODEL_URLS:-http://100.101.166.1:22005/v1,http://100.101.166.1:22004/v1,http://100.101.166.1:22003/v1,http://100.101.166.1:22002/v1}"
export REWARD_MODEL_KEY="${REWARD_MODEL_KEY:-EMPTY}"

# Bypass any HTTP proxy for internal reward endpoints.
reward_proxy_hosts="$(echo "${REWARD_MODEL_URLS}" | tr ',' '\n' | sed -E 's#^[a-zA-Z]+://##; s#/.*$##; s#:.*$##' | paste -sd',' -)"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${reward_proxy_hosts},localhost,127.0.0.1"
export no_proxy="${NO_PROXY}"

# Ray config
ray_port="${RAY_PORT:-8266}"

# -------------------------
# Fixed defaults in this launcher
# -------------------------
max_response_length=0 # computed below

# Max model length (cap to model context length)
model_max_len_cap=40960

# Algorithm knobs
adv_estimator="co_grpo"
kl_coef=0.001
norm_adv_by_std_in_grpo=True
verifier_max_prompt_length="$((1024 * 16))"

# Trainer
parallel_control_exp=True

if [ "${preset}" = "mini" ]; then
  if [ -z "${co_grpo_mode}" ]; then
    co_grpo_mode="verifier_lora_only"
  fi
else
  if [ -z "${co_grpo_mode}" ]; then
    co_grpo_mode="full"
  fi
fi

freeze_actor=False
if [ "${co_grpo_mode}" = "verifier_lora_only" ]; then
  freeze_actor=True
fi

# -------------------------
# Derived lengths / budgets
# -------------------------
max_response_length="$((1024 * response_n))"

# by_step: disable kv cache optimization (it can hurt by-step scheduling behavior)
rollout_enable_kv_cache_optimization=True
if [ "${verifier_intervention_mode}" = "by_step" ]; then
  rollout_enable_kv_cache_optimization=False
fi

# Reserve headroom for hints so EXP isn't truncated at 33k just because of hints.
# Actor-generated tokens are still capped by max_response_length inside the rollout worker.
hint_headroom="$((max_interventions * estimated_hint_tokens))"
max_model_len="$((max_prompt_length + max_response_length + hint_headroom))"

# Verifier uses the same vLLM engine. Ensure engine max_model_len is large enough
# for (verifier prompt budget) + (verifier max_new_tokens), otherwise verifier output
# gets truncated before reaching the final decision line.
min_model_len_for_verifier="$((verifier_max_prompt_length + verifier_max_new_tokens))"
if [ "${max_model_len}" -lt "${min_model_len_for_verifier}" ]; then
  max_model_len="${min_model_len_for_verifier}"
fi

if [ "${max_model_len}" -gt "${model_max_len_cap}" ]; then
  max_model_len="${model_max_len_cap}"
fi

echo "[launcher] prompt=${max_prompt_length} response=${max_response_length} max_interventions=${max_interventions} estimated_hint_tokens=${estimated_hint_tokens}"
echo "[launcher] verifier_max_prompt_length=${verifier_max_prompt_length} verifier_max_new_tokens=${verifier_max_new_tokens} verifier_max_hint_tokens=${verifier_max_hint_tokens}"
echo "[launcher] max_model_len=${max_model_len} (cap=${model_max_len_cap})"

# Sanity: verifier prompt budget must fit after reserving output tokens.
if [ "$((max_model_len - verifier_max_new_tokens))" -lt "${verifier_max_prompt_length}" ]; then
  echo "[launcher][WARN] verifier prompt budget may be clamped: max_model_len(${max_model_len}) - verifier_max_new_tokens(${verifier_max_new_tokens}) < verifier_max_prompt_length(${verifier_max_prompt_length})" >&2
fi

# rollout micro-batching knobs
use_dynamic_bsz="${use_dynamic_bsz:-False}"
use_dynamic_bsz_norm="$(echo "${use_dynamic_bsz}" | tr '[:upper:]' '[:lower:]')"
micro_bsz_per_gpu="${micro_bsz_per_gpu:-2}"
ppo_micro_batch_size_per_gpu="${micro_bsz_per_gpu}"
ref_log_prob_micro_batch_size_per_gpu="${micro_bsz_per_gpu}"
rollout_log_prob_micro_batch_size_per_gpu="${micro_bsz_per_gpu}"

actor_ppo_max_token_len="$((2 * max_prompt_length + 2 * max_response_length))"
ref_ppo_max_token_len="$((2 * max_prompt_length + 2 * max_response_length))"
rollout_max_num_batched_tokens="$((2 * max_prompt_length + 2 * max_response_length))"

control_rollout_gpus_per_node="$((n_gpus_per_node / 2))"
exp_rollout_gpus_per_node="$((n_gpus_per_node - control_rollout_gpus_per_node))"

timestamp="$(date +%Y%m%d_%H%M%S)"
if [ -z "${exp_name}" ]; then
  exp_name="$(basename "${model}")_${train_file_name}_${response_n}k_${preset}_${timestamp}"
fi

export SWANLAB_LOG_DIR="${work_dir}/swanlab"
export VERL_LOG_DIR="${work_dir}/logs"

LOG_FILE="${work_dir}/logs/verl_log_${exp_name}_rank0.txt"

save_dir="${work_dir}/checkpoints/co_grpo_v2/${exp_name}"
if [ -n "${resume_dir}" ]; then
  save_dir="${resume_dir}"
fi

resume_mode="auto"
if [ -n "${resume_path}" ]; then
  resume_mode="resume_path"
fi

reward_async=True
temperature=1.0
top_p=1.0
top_k=-1
sp=1
offload=False
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28
num_generation_per_prompt=8

if [ -z "${NODE_RANK:-}" ]; then NODE_RANK=0; fi
if [ -z "${MASTER_ADDR:-}" ]; then MASTER_ADDR="127.0.0.1"; fi

detect_num_gpus() {
  local detected
  detected="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d '[:space:]')"
  if [ -z "${detected}" ] || [ "${detected}" -le 0 ] 2>/dev/null; then
    detected="${n_gpus_per_node}"
  fi
  echo "${detected}"
}
ray_num_gpus="$(detect_num_gpus)"

if [ "${NODE_RANK}" = "0" ]; then
  ray start --head --port="${ray_port}" --num-gpus="${ray_num_gpus}" &
  echo "=== Log will be saved to: ${LOG_FILE} ==="
  export VERL_LOGGING_LEVEL="INFO"
  export VERL_EXPERIMENT_ID="${exp_name}"

  TARGET_GPU="$((nnodes * n_gpus_per_node))"
  CHECK_INTERVAL=10
  get_ray_gpu() {
    # With `set -o pipefail`, `ray status` may be non-zero while ray boots;
    # avoid emitting duplicate lines like "0\n0".
    { ray status 2>/dev/null || true; } | awk '
      /GPU[s]?[[:space:]]*$/ {
        split($1, a, "/");
        if (a[2] != "") { print int(a[2]); found=1; exit }
      }
      END { if (!found) print 0 }
    ' | head -n 1 | tr -d "[:space:]"
  }
  while true; do
    GPU_COUNT="$(get_ray_gpu)"
    echo "$(date +'%Y-%m-%d %H:%M:%S') - Current total number of GPUs in the Ray cluster: ${GPU_COUNT}"
    if [ "${GPU_COUNT}" -eq "${TARGET_GPU}" ]; then
      break
    fi
    sleep "${CHECK_INTERVAL}"
  done

  python3 -m verl.trainer.main_ppo \
    data.train_files="$(realpath "data/${train_file_name}.parquet")" \
    data.val_files="$(realpath "data/aime-2024.parquet")" \
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
    actor_rollout_ref.rollout.temperature="${temperature}" \
    actor_rollout_ref.rollout.top_p="${top_p}" \
    actor_rollout_ref.rollout.top_k="${top_k}" \
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
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async="${reward_async}" \
    "+reward_model.reward_model_urls='${REWARD_MODEL_URLS}'" \
    "+reward_model.reward_model_key='${REWARD_MODEL_KEY}'" \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.norm_adv_by_std_in_grpo="${norm_adv_by_std_in_grpo}" \
    algorithm.control_group_weight="${control_group_weight}" \
    +algorithm.use_curriculum_weighting="${use_curriculum_weighting}" \
    +algorithm.curriculum_start_weight="${curriculum_start_weight}" \
    +algorithm.curriculum_end_weight="${curriculum_end_weight}" \
    algorithm.verifier_intervention_mode="${verifier_intervention_mode}" \
    algorithm.token_check_interval="${token_check_interval}" \
    algorithm.entropy_threshold="${entropy_threshold}" \
    algorithm.use_entropy_filter="${use_entropy_filter}" \
    +algorithm.min_step_tokens="${min_step_tokens}" \
    +algorithm.max_interventions="${max_interventions}" \
    +algorithm.confidence_threshold="${confidence_threshold}" \
    +algorithm.intervention_penalty.freq_coef="${intervention_penalty_freq_coef}" \
    +algorithm.intervention_penalty.len_coef="${intervention_penalty_len_coef}" \
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
    trainer.val_before_train=False \
    trainer.total_epochs="${trainer_total_epochs}" \
    trainer.total_training_steps="${trainer_total_training_steps}" \
    trainer.resume_mode="${resume_mode}" \
    trainer.resume_from_path="${resume_path}" \
    trainer.project_name="co_grpo_v2" \
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
    2>&1 | tee "${LOG_FILE}"
else
  sleep 10
  ray start --address="${MASTER_ADDR}:${ray_port}" --num-gpus="${ray_num_gpus}" &
  bash -lc -- "sleep infinity"
fi
