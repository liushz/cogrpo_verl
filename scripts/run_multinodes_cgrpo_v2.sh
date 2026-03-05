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
intervention_penalty_freq_coef="${26:-0.0}"
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
confidence_threshold="${37:-0.0}"

verifier_lora_rank="${38:-64}"
verifier_lora_alpha="${39:-128}"
verifier_lora_dropout="${40:-0.05}"
verifier_lr="${41:-1e-5}"
verifier_loss_weight="${42:-1.0}"
verifier_max_new_tokens="${43:-4096}"
verifier_logprobs="${44:-1}"
verifier_temperature="${45:-1.0}"

trainer_total_epochs="${46:-10}"
trainer_total_training_steps="${47:-1000}"
trainer_save_freq="${48:-20}"
trainer_test_freq="${49:--1}"
trainer_log_val_generations="${50:-False}"
trainer_control_rollout_sync_freq="${51:-5}"
trainer_verifier_lora_sync_freq="${52:-1}"
trainer_verifier_lora_save_freq="${53:-20}"

max_prompt_length="${54:-1024}"
verifier_credit_assignment="${55:-global_gap}"
cf_branch_prob="${56:-1.0}"
cf_branch_max_events_per_sample="${57:-0}"
cf_branch_state_hash_mod="${58:-1024}"
cf_branch_k="${CF_BRANCH_K:-1}"
if ! [[ "${cf_branch_k}" =~ ^[0-9]+$ ]]; then
  echo "[launcher][ERR] CF_BRANCH_K must be an integer (got ${cf_branch_k})." >&2
  exit 2
fi
if [ "${cf_branch_k}" -lt 1 ]; then
  echo "[launcher][ERR] CF_BRANCH_K must be >=1 (got ${cf_branch_k})." >&2
  exit 2
fi

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
conda_env_path="${VERL_CONDA_ENV_PATH:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro}"
conda activate "${conda_env_path}"

export PYTHONPATH="$(pwd)"

export SWANLAB_MODE="local"
export VERL_AUTO_PADDING="1"
export RAY_record_ref_creation_sites="1"
# Timeout for torch.distributed collectives (defaults to PyTorch historical 30min).
# NOTE: Large multi-node GRPO/CoGRPO runs with long generations can exceed 30min
# in a single step if one DP rank becomes a straggler. Allow overriding via env.
export VERL_NCCL_TIMEOUT_SEC="${VERL_NCCL_TIMEOUT_SEC:-1800}"
export VERL_DEBUG="${verl_debug}"
# Verifier debug logs are extremely verbose (prompt/response previews). Keep them
# off by default even if the base image sets VERL_VERIFIER_DEBUG_LOG.
# To enable: VERL_VERIFIER_DEBUG_LOG_OVERRIDE=1
export VERL_VERIFIER_DEBUG_LOG="${VERL_VERIFIER_DEBUG_LOG_OVERRIDE:-0}"
export REWARD_MODEL_URLS="${REWARD_MODEL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"
export REWARD_MODEL_KEY="${REWARD_MODEL_KEY:-EMPTY}"

# NOTE: vLLM's CuMemAllocator asserts if PYTORCH_CUDA_ALLOC_CONF contains
# `expandable_segments:true` (incompatible with vLLM memory pool):
#   AssertionError: Expandable segments are not compatible with memory pool.
# Some base images or user shells set it globally; strip it to keep rollout stable.
if [ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && echo "${PYTORCH_CUDA_ALLOC_CONF}" | grep -qi "expandable_segments:true"; then
  echo "[launcher][WARN] PYTORCH_CUDA_ALLOC_CONF includes expandable_segments:true which is incompatible with vLLM; removing it." >&2
  export PYTORCH_CUDA_ALLOC_CONF="$(echo "${PYTORCH_CUDA_ALLOC_CONF}" | sed -E 's/(^|,)[[:space:]]*expandable_segments:[Tt]rue([[:space:]]*,|$)/\\1\\2/g; s/,,/,/g; s/^,|,$//g')"
  if [ -z "${PYTORCH_CUDA_ALLOC_CONF}" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF
  fi
fi

# HuggingFace datasets cache must be writable (datasets uses a FileLock even for local parquet).
# Some environments set HF_HOME to a shared read-only mount, which makes default
# "$HF_HOME/datasets" unusable and causes:
#   OSError: [Errno 30] Read-only file system: '.../datasets/... .lock'
#
# Default to a per-job local path; override via HF_DATASETS_CACHE if desired.
default_hf_datasets_cache="/tmp/hf_datasets_cache/${JOB_ID:-${exp_name:-unknown_job}}"
hf_datasets_cache="${HF_DATASETS_CACHE:-${default_hf_datasets_cache}}"
mkdir -p "${hf_datasets_cache}" 2>/dev/null || true
if ! touch "${hf_datasets_cache}/.writable_check" 2>/dev/null; then
  echo "[launcher][WARN] HF_DATASETS_CACHE is not writable: ${hf_datasets_cache} -> fallback to ${default_hf_datasets_cache}" >&2
  hf_datasets_cache="${default_hf_datasets_cache}"
  mkdir -p "${hf_datasets_cache}" 2>/dev/null || true
fi
rm -f "${hf_datasets_cache}/.writable_check" 2>/dev/null || true
export HF_DATASETS_CACHE="${hf_datasets_cache}"

# vLLM stability knobs
# - Default to V0 engine: V1 path can be fragile for long-context by_step +
#   LoRA (profile_run dummy forward may crash; some runs observe output-count
#   mismatch). Can be overridden by setting VLLM_USE_V1=1 in env.
# NOTE: Some base images export VLLM_USE_V1=1 globally. We want the launcher
# default to be stable, so we *override* VLLM_USE_V1 by default.
# To try V1, explicitly set: VERL_VLLM_USE_V1=1
export VLLM_USE_V1="${VERL_VLLM_USE_V1:-0}"
echo "[launcher] VLLM_USE_V1=${VLLM_USE_V1} (override with VERL_VLLM_USE_V1=1)"
# - Verifier LoRA is hot-loaded/reloaded at runtime.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-1}"

# Bypass any HTTP proxy for internal reward endpoints.
reward_proxy_hosts="$(echo "${REWARD_MODEL_URLS}" | tr ',' '\n' | sed -E 's#^[a-zA-Z]+://##; s#/.*$##; s#:.*$##' | paste -sd',' -)"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${reward_proxy_hosts},localhost,127.0.0.1"
export no_proxy="${NO_PROXY}"

# Ray config
ray_port="${RAY_PORT:-8266}"

# Ray object store (Plasma) memory.
# Long-context rollouts can produce multi-GB DataProto objects. With Ray defaults,
# object store memory is often capped (~200GB), which can cause heavy spill-to-disk
# and massive I/O stalls. Allow overriding the object store size explicitly.
#
# Usage:
#   export VERL_RAY_OBJECT_STORE_MEMORY_GB=450
ray_object_store_args=()
if [ -n "${VERL_RAY_OBJECT_STORE_MEMORY_GB:-}" ]; then
  if ! [[ "${VERL_RAY_OBJECT_STORE_MEMORY_GB}" =~ ^[0-9]+$ ]]; then
    echo "[launcher][ERR] VERL_RAY_OBJECT_STORE_MEMORY_GB must be an integer (got ${VERL_RAY_OBJECT_STORE_MEMORY_GB})." >&2
    exit 2
  fi
  if [ "${VERL_RAY_OBJECT_STORE_MEMORY_GB}" -le 0 ]; then
    echo "[launcher][ERR] VERL_RAY_OBJECT_STORE_MEMORY_GB must be > 0 (got ${VERL_RAY_OBJECT_STORE_MEMORY_GB})." >&2
    exit 2
  fi
  ray_object_store_bytes="$((VERL_RAY_OBJECT_STORE_MEMORY_GB * 1024 * 1024 * 1024))"
  ray_object_store_args+=(--object-store-memory="${ray_object_store_bytes}")
  echo "[launcher] Ray object store memory: ${VERL_RAY_OBJECT_STORE_MEMORY_GB} GB (${ray_object_store_bytes} bytes)"
fi

# -------------------------
# Fixed defaults in this launcher
# -------------------------
max_response_length=0 # computed below

# Max model length (cap to model context length)
model_max_len_cap=40960

# Algorithm knobs
adv_estimator="${ADV_ESTIMATOR:-co_grpo}"
adv_estimator_norm="$(echo "${adv_estimator}" | tr '[:upper:]' '[:lower:]')"
case "${adv_estimator_norm}" in
  co_grpo|cogrpo|co-grpo) adv_estimator="co_grpo" ;;
  grpo) adv_estimator="grpo" ;;
  *) adv_estimator="${adv_estimator_norm}" ;;
esac
kl_coef=0.001
norm_adv_by_std_in_grpo=True
verifier_max_prompt_length="$((1024 * 16))"

# Trainer
parallel_control_exp="${PARALLEL_CONTROL_EXP:-True}"
parallel_control_exp_norm="$(echo "${parallel_control_exp}" | tr '[:upper:]' '[:lower:]')"
case "${parallel_control_exp_norm}" in
  1|true) parallel_control_exp=True ;;
  0|false) parallel_control_exp=False ;;
  *)
    echo "[launcher][ERR] Invalid PARALLEL_CONTROL_EXP=${PARALLEL_CONTROL_EXP} (expected True/False/1/0)" >&2
    exit 2
    ;;
esac

# cf-branch credit assignment runs EXP-only (no global control stream).
verifier_credit_assignment_norm="$(echo "${verifier_credit_assignment}" | tr '[:upper:]' '[:lower:]')"
if [ "${verifier_credit_assignment_norm}" = "cf_branch" ] || [ "${verifier_credit_assignment_norm}" = "cf" ]; then
  parallel_control_exp=False
  if [[ "${max_interventions}" =~ ^[0-9]+$ ]] && [[ "${cf_branch_max_events_per_sample}" =~ ^[0-9]+$ ]]; then
    if [ "${cf_branch_max_events_per_sample}" -gt 0 ] && [ "${max_interventions}" -gt "${cf_branch_max_events_per_sample}" ]; then
      echo "[launcher][WARN] max_interventions(${max_interventions}) > cf_branch_max_events_per_sample(${cf_branch_max_events_per_sample}); events beyond max-events will not have counterfactual baselines and will be dropped from verifier cf-branch reward (cf_missing_ratio increases)." >&2
    fi
  fi
fi

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

# Verifier update mode:
# - LoRA verifier (verifier_lora_path non-empty): update LoRA only.
# - No LoRA (same-base ablation): update the shared base model on verifier trajectories.
verifier_update_base="${VERIFIER_UPDATE_BASE:-}"
if [ -z "${verifier_update_base}" ]; then
  if [ -z "${verifier_lora_path}" ]; then
    verifier_update_base=True
  else
    verifier_update_base=False
  fi
else
  verifier_update_base_norm="$(echo "${verifier_update_base}" | tr '[:upper:]' '[:lower:]')"
  case "${verifier_update_base_norm}" in
    1|true) verifier_update_base=True ;;
    0|false) verifier_update_base=False ;;
    *)
      echo "[launcher][ERR] Invalid VERIFIER_UPDATE_BASE=${VERIFIER_UPDATE_BASE} (expected True/False/1/0)" >&2
      exit 2
      ;;
  esac
fi

# -------------------------
# Derived lengths / budgets
# -------------------------
max_response_length="$((1024 * response_n))"

# by_step: disable kv cache optimization (it can hurt by-step scheduling behavior)
# Allow explicit override for debugging:
#   ROLLOUT_ENABLE_KV_CACHE_OPTIMIZATION=True|False|1|0
rollout_enable_kv_cache_optimization="${ROLLOUT_ENABLE_KV_CACHE_OPTIMIZATION:-}"
if [ -n "${rollout_enable_kv_cache_optimization}" ]; then
  rollout_enable_kv_cache_optimization="$(echo "${rollout_enable_kv_cache_optimization}" | tr '[:upper:]' '[:lower:]')"
  case "${rollout_enable_kv_cache_optimization}" in
    1|true) rollout_enable_kv_cache_optimization=True ;;
    0|false) rollout_enable_kv_cache_optimization=False ;;
    *)
      echo "[launcher][ERR] Invalid ROLLOUT_ENABLE_KV_CACHE_OPTIMIZATION=${ROLLOUT_ENABLE_KV_CACHE_OPTIMIZATION} (expected True/False/1/0)" >&2
      exit 2
      ;;
  esac
else
  rollout_enable_kv_cache_optimization=True
  if [ "${verifier_intervention_mode}" = "by_step" ]; then
    rollout_enable_kv_cache_optimization=False
  fi
fi

# Reserve headroom for hints so EXP isn't truncated at 33k just because of hints.
# Actor-generated tokens are still capped by max_response_length inside the rollout worker.
#
# NOTE (stability): When max_interventions=0, prompt+response can exactly hit max_model_len
# (e.g., 2048+32768=34816). We've observed rare vLLM CUDA illegal memory access crashes in
# this exact-boundary case under long-context GRPO sampling. Keep a small base headroom so
# max_model_len stays strictly larger than prompt+response even without interventions.
hint_headroom="$((max_interventions * estimated_hint_tokens))"
base_headroom=0
if [ "${hint_headroom}" -le 0 ]; then
  base_headroom=1024
fi
max_model_len="$((max_prompt_length + max_response_length + hint_headroom + base_headroom))"

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

echo "[launcher] prompt=${max_prompt_length} response=${max_response_length} max_interventions=${max_interventions} estimated_hint_tokens=${estimated_hint_tokens} base_headroom=${base_headroom}"
echo "[launcher] verifier_max_prompt_length=${verifier_max_prompt_length} verifier_max_new_tokens=${verifier_max_new_tokens} verifier_max_hint_tokens=${verifier_max_hint_tokens}"
echo "[launcher] max_model_len=${max_model_len} (cap=${model_max_len_cap})"
echo "[launcher] verifier_update_base=${verifier_update_base} verifier_lora_path=${verifier_lora_path:-<empty>}"

# -------------------------
# Dataset paths (allow passing absolute parquet paths)
# -------------------------
resolve_parquet_path() {
  local spec="$1"
  if [ -z "${spec}" ]; then
    return 1
  fi

  # 1) Direct file path.
  if [ -f "${spec}" ]; then
    realpath "${spec}"
    return 0
  fi

  # 2) If user passed a path without extension.
  if [[ "${spec}" == */* ]] && [ -f "${spec}.parquet" ]; then
    realpath "${spec}.parquet"
    return 0
  fi

  # 3) Local repo dataset shortcut: data/<name>.parquet
  if [ -f "data/${spec}.parquet" ]; then
    realpath "data/${spec}.parquet"
    return 0
  fi

  echo "[launcher][ERR] Cannot resolve parquet path from '${spec}'." >&2
  echo "[launcher][HINT] Pass either an absolute parquet path, or a dataset name under ./data/ (without .parquet)." >&2
  return 2
}

train_file_path="$(resolve_parquet_path "${train_file_name}")"
val_file_spec="${VERL_VAL_FILE:-data/aime-2024.parquet}"
val_file_path="$(resolve_parquet_path "${val_file_spec}")"
echo "[launcher] train_file=${train_file_path}"
echo "[launcher] val_file=${val_file_path}"

# Sanity: verifier prompt budget must fit after reserving output tokens.
if [ "$((max_model_len - verifier_max_new_tokens))" -lt "${verifier_max_prompt_length}" ]; then
  echo "[launcher][WARN] verifier prompt budget may be clamped: max_model_len(${max_model_len}) - verifier_max_new_tokens(${verifier_max_new_tokens}) < verifier_max_prompt_length(${verifier_max_prompt_length})" >&2
fi

# Parallelism knobs (must be defined before FSDP batch-size sanity checks).
sp=1
num_generation_per_prompt="${ROLLOUT_N:-8}"
if ! [[ "${num_generation_per_prompt}" =~ ^[0-9]+$ ]]; then
  echo "[launcher][ERR] ROLLOUT_N must be an integer (got ${num_generation_per_prompt})." >&2
  exit 2
fi
if [ "${num_generation_per_prompt}" -lt 1 ]; then
  echo "[launcher][ERR] ROLLOUT_N must be >=1 (got ${num_generation_per_prompt})." >&2
  exit 2
fi

# rollout micro-batching knobs
use_dynamic_bsz="${use_dynamic_bsz:-False}"
use_dynamic_bsz_norm="$(echo "${use_dynamic_bsz}" | tr '[:upper:]' '[:lower:]')"
micro_bsz_per_gpu="${micro_bsz_per_gpu:-2}"
ppo_micro_batch_size_per_gpu="${micro_bsz_per_gpu}"
ref_log_prob_micro_batch_size_per_gpu="${micro_bsz_per_gpu}"
rollout_log_prob_micro_batch_size_per_gpu="${micro_bsz_per_gpu}"

if ! [[ "${train_batch_size}" =~ ^[0-9]+$ ]] || ! [[ "${ppo_micro_batch_size_per_gpu}" =~ ^[0-9]+$ ]]; then
  echo "[launcher][ERR] train_batch_size and micro_bsz_per_gpu must be integers (got train_batch_size=${train_batch_size}, micro_bsz_per_gpu=${ppo_micro_batch_size_per_gpu})" >&2
  exit 2
fi
if [ "${ppo_micro_batch_size_per_gpu}" -lt 1 ]; then
  echo "[launcher][ERR] micro_bsz_per_gpu must be >=1 (got ${ppo_micro_batch_size_per_gpu})" >&2
  exit 2
fi
# Match FSDP normalization in `verl/workers/fsdp_workers.py`:
#   normalized_ppo_mini_batch_size = ppo_mini_batch_size * rollout.n // dp_size
# This normalized value must be divisible by `ppo_micro_batch_size_per_gpu`.
world_size="$((nnodes * n_gpus_per_node))"
if [ "${sp}" -lt 1 ]; then
  echo "[launcher][ERR] ulysses_sequence_parallel_size (sp) must be >=1 (got ${sp})" >&2
  exit 2
fi
if [ "$((world_size % sp))" -ne 0 ]; then
  echo "[launcher][ERR] world_size(${world_size}) must be divisible by sp(${sp}) for FSDP normalization." >&2
  exit 2
fi
dp_size="$((world_size / sp))"
mini_times_n="$((train_batch_size * num_generation_per_prompt))"
if [ "$((mini_times_n % dp_size))" -ne 0 ]; then
  echo "[launcher][ERR] ppo_mini_batch_size(${train_batch_size}) * rollout.n(${num_generation_per_prompt}) must be divisible by dp_size(${dp_size})." >&2
  echo "[launcher][HINT] Adjust --train-bsz so that train_batch_size * num_generation_per_prompt / dp_size is an integer." >&2
  exit 2
fi
normalized_mini="$((mini_times_n / dp_size))"
if [ "${normalized_mini}" -lt 1 ]; then
  echo "[launcher][ERR] normalized ppo_mini_batch_size is < 1 (got ${normalized_mini}). Increase train_batch_size." >&2
  exit 2
fi
if [ "$((normalized_mini % ppo_micro_batch_size_per_gpu))" -ne 0 ]; then
  echo "[launcher][ERR] normalized ppo_mini_batch_size(${normalized_mini}) must be divisible by micro_bsz_per_gpu(${ppo_micro_batch_size_per_gpu})." >&2
  echo "[launcher][INFO] normalized_mini = train_batch_size(${train_batch_size}) * rollout.n(${num_generation_per_prompt}) / dp_size(${dp_size})" >&2
  exit 2
fi

actor_ppo_max_token_len="$((2 * max_prompt_length + 2 * max_response_length))"
ref_ppo_max_token_len="$((2 * max_prompt_length + 2 * max_response_length))"
rollout_max_num_batched_tokens_default="$((2 * max_prompt_length + 2 * max_response_length))"
# vLLM chunked prefill requires `max_num_batched_tokens >= max_model_len`, otherwise
# vLLMRolloutSPMD raises at startup. This can happen when we enlarge max_model_len
# to satisfy verifier prompt/new-token budgets (e.g. by_step verifier).
if [ "${rollout_max_num_batched_tokens_default}" -lt "${max_model_len}" ]; then
  rollout_max_num_batched_tokens_default="${max_model_len}"
fi
# vLLM scheduler limits concurrency by total tokens across active sequences. In by_step mode
# sequences can quickly grow to many thousands of tokens; the default (2*(prompt+response))
# may underutilize GPUs by forcing micro-batching. Allow explicit override.
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${rollout_max_num_batched_tokens_default}}"
if ! [[ "${rollout_max_num_batched_tokens}" =~ ^[0-9]+$ ]]; then
  echo "[launcher][ERR] ROLLOUT_MAX_NUM_BATCHED_TOKENS must be an integer (got ${rollout_max_num_batched_tokens})." >&2
  exit 2
fi
if [ "${rollout_max_num_batched_tokens}" -lt 1 ]; then
  echo "[launcher][ERR] ROLLOUT_MAX_NUM_BATCHED_TOKENS must be >=1 (got ${rollout_max_num_batched_tokens})." >&2
  exit 2
fi

# NOTE: With very large `max_num_batched_tokens` (e.g. default 2*(prompt+response)
# for 32k rollouts), some long-context runs have observed
# "CUDA error: an illegal memory access was encountered" in vLLM (including
# non-LoRA runs). To keep experiments stable by default, cap the total batched
# tokens to a "1 long sequence + some headroom" budget, unless the user explicitly
# overrides ROLLOUT_MAX_NUM_BATCHED_TOKENS.
if [ -z "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-}" ]; then
  stability_batched_tokens_cap="$((max_model_len + 8192))"
  if [ "${rollout_max_num_batched_tokens}" -gt "${stability_batched_tokens_cap}" ]; then
    echo "[launcher][WARN] Capping max_num_batched_tokens ${rollout_max_num_batched_tokens} -> ${stability_batched_tokens_cap} for vLLM stability (set ROLLOUT_MAX_NUM_BATCHED_TOKENS to override)."
    rollout_max_num_batched_tokens="${stability_batched_tokens_cap}"
  fi
fi

# vLLM scheduler limits concurrency by both tokens and sequences.
# Default 512 is aggressive for long-context (32k) multi-node runs and can
# amplify KV-cache preemption (RECOMPUTE) and instability. Use a safer default.
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-256}"
if ! [[ "${rollout_max_num_seqs}" =~ ^[0-9]+$ ]]; then
  echo "[launcher][ERR] ROLLOUT_MAX_NUM_SEQS must be an integer (got ${rollout_max_num_seqs})." >&2
  exit 2
fi
if [ "${rollout_max_num_seqs}" -lt 1 ]; then
  echo "[launcher][ERR] ROLLOUT_MAX_NUM_SEQS must be >=1 (got ${rollout_max_num_seqs})." >&2
  exit 2
fi

control_rollout_gpus_per_node="$((n_gpus_per_node / 2))"
exp_rollout_gpus_per_node="$((n_gpus_per_node - control_rollout_gpus_per_node))"
if [ "${parallel_control_exp}" = "True" ]; then
  if [ -n "${CONTROL_ROLLOUT_GPUS_PER_NODE:-}" ] && [ -n "${EXP_ROLLOUT_GPUS_PER_NODE:-}" ]; then
    control_rollout_gpus_per_node="${CONTROL_ROLLOUT_GPUS_PER_NODE}"
    exp_rollout_gpus_per_node="${EXP_ROLLOUT_GPUS_PER_NODE}"
  elif [ -n "${CONTROL_ROLLOUT_GPUS_PER_NODE:-}" ]; then
    control_rollout_gpus_per_node="${CONTROL_ROLLOUT_GPUS_PER_NODE}"
    exp_rollout_gpus_per_node="$((n_gpus_per_node - control_rollout_gpus_per_node))"
  elif [ -n "${EXP_ROLLOUT_GPUS_PER_NODE:-}" ]; then
    exp_rollout_gpus_per_node="${EXP_ROLLOUT_GPUS_PER_NODE}"
    control_rollout_gpus_per_node="$((n_gpus_per_node - exp_rollout_gpus_per_node))"
  fi

  if [ "${control_rollout_gpus_per_node}" -lt 1 ] || [ "${exp_rollout_gpus_per_node}" -lt 1 ]; then
    echo "[launcher][ERR] control/exp rollout GPUs must both be >=1 when PARALLEL_CONTROL_EXP=True (got control=${control_rollout_gpus_per_node}, exp=${exp_rollout_gpus_per_node})" >&2
    exit 2
  fi
  if [ "$((control_rollout_gpus_per_node + exp_rollout_gpus_per_node))" -ne "${n_gpus_per_node}" ]; then
    echo "[launcher][ERR] control_rollout_gpus_per_node + exp_rollout_gpus_per_node must equal n_gpus_per_node (${n_gpus_per_node}) when PARALLEL_CONTROL_EXP=True (got control=${control_rollout_gpus_per_node}, exp=${exp_rollout_gpus_per_node})" >&2
    exit 2
  fi
else
  # Not used when PARALLEL_CONTROL_EXP=False; keep config readable.
  control_rollout_gpus_per_node="null"
  exp_rollout_gpus_per_node="null"
fi

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

# NOTE: `reward_model.launch_reward_fn_async=True` runs reward computation in a Ray
# remote task (`compute_reward_async`). For long-context training (e.g. 32k) with
# large effective batch sizes (train_bsz * rollout_n), this forces Ray to
# serialize multi-GB `DataProto` objects into the Ray object store and often
# triggers TB-scale spilling, causing heavy I/O and instability.
#
# Default: keep reward computation on the driver process (reward_async=False).
# To override (not recommended for 32k multi-node): export VERL_REWARD_ASYNC=1.
reward_async="${VERL_REWARD_ASYNC:-False}"
reward_async_norm="$(echo "${reward_async}" | tr '[:upper:]' '[:lower:]')"
case "${reward_async_norm}" in
  1|true) reward_async=True ;;
  0|false|"") reward_async=False ;;
  *)
    echo "[launcher][ERR] Invalid VERL_REWARD_ASYNC=${VERL_REWARD_ASYNC} (expected True/False/1/0)" >&2
    exit 2
    ;;
esac
temperature=1.0
top_p=1.0
top_k=-1
offload=False
use_kl_loss="${USE_KL_LOSS:-False}"
use_kl_loss_norm="$(echo "${use_kl_loss}" | tr '[:upper:]' '[:lower:]')"
case "${use_kl_loss_norm}" in
  1|true) use_kl_loss=True ;;
  0|false|"") use_kl_loss=False ;;
  *)
    echo "[launcher][ERR] Invalid USE_KL_LOSS=${USE_KL_LOSS} (expected True/False/1/0)" >&2
    exit 2
    ;;
esac
kl_loss_coef="${KL_LOSS_COEF:-}"
if [ -z "${kl_loss_coef}" ]; then
  if [ "${use_kl_loss}" = "True" ]; then
    kl_loss_coef="0.001"
  else
    kl_loss_coef="0.0"
  fi
fi
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c="${CLIP_RATIO_C:-10.0}"
# Actor update mode for Co-GRPO:
# - exp  : update actor on EXP-only (historical behavior)
# - both : update actor on (control + exp) streams (8+8 when rollout_n=8)
actor_update_streams="${ACTOR_UPDATE_STREAMS:-exp}"

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
  ray start --head --port="${ray_port}" --num-gpus="${ray_num_gpus}" "${ray_object_store_args[@]}" &
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
    "data.train_files=${train_file_path}" \
    "data.val_files=${val_file_path}" \
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
    actor_rollout_ref.actor.clip_ratio_c="${clip_ratio_c}" \
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
    actor_rollout_ref.rollout.max_num_seqs="${rollout_max_num_seqs}" \
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
    +algorithm.min_step_tokens="${min_step_tokens}" \
    +algorithm.max_interventions="${max_interventions}" \
    +algorithm.confidence_threshold="${confidence_threshold}" \
    +algorithm.actor_update_streams="${actor_update_streams}" \
    +algorithm.verifier_credit_assignment="${verifier_credit_assignment}" \
    +algorithm.cf_branch_prob="${cf_branch_prob}" \
    +algorithm.cf_branch_max_events_per_sample="${cf_branch_max_events_per_sample}" \
    +algorithm.cf_branch_state_hash_mod="${cf_branch_state_hash_mod}" \
    +algorithm.cf_branch_k="${cf_branch_k}" \
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
    +verifier.update_base="${verifier_update_base}" \
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
  ray start --address="${MASTER_ADDR}:${ray_port}" --num-gpus="${ray_num_gpus}" "${ray_object_store_args[@]}" &
  bash -lc -- "sleep infinity"
fi
