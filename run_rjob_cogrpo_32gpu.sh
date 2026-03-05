#!/usr/bin/env bash
set -eo pipefail

# Ensure the local shell inherits kubebrain environment variables so the `rjob`
# CLI works even when invoked outside kubebrain-managed shells.
if [ -f /etc/profile.d/ssh-init.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/ssh-init.sh || true
fi

# Avoid proxy interference when calling rjob API endpoints.
unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

# 32-GPU rjob launcher for Co-GRPO v2 (cf_branch credit assignment).
#
# This is intentionally similar to `run_dev_debug_cogrpo_8gpu.sh`, but runs on
# rjob with multi-node settings.
#
# Example:
#   bash run_rjob_cogrpo_32gpu.sh --steps 100 --response-n 32 --rollout-n 8

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

_sanitize_csv_urls() {
  local raw="$1"
  raw="${raw#<}"
  raw="${raw%>}"
  local out=""
  local part=""
  IFS=',' read -r -a _parts <<<"${raw}"
  for part in "${_parts[@]}"; do
    part="$(echo "${part}" | sed -e 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    part="${part#<}"
    part="${part%>}"
    [[ -n "${part}" ]] || continue
    if [[ -n "${out}" ]]; then
      out+=","
    fi
    out+="${part}"
  done
  echo "${out}"
}

# -------------------------
# Model / data
# -------------------------
# Default to current Co-GRPO experiments: cispo cold-start 7B actor + verifier LoRA.
model_path="/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170"
dataset_name="/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-dapo-math-17k.parquet"
val_dataset_path="/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-aime-2024.parquet"
verifier_lora_path="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-verifier_sft_train_55_waitx2_fpfix_plus_v4ckptwait_20260223-02241157/checkpoint-2445"

# -------------------------
# Cluster / job identity
# -------------------------
cluster="llmit_gpu"
nnodes=4
n_gpus_per_node=8
gen_tp=1
gpu_memory_utilization=0.85
verl_debug=1

work_dir="/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train"
exp_name=""
# Resume support (used to continue from an existing checkpoint).
# - resume_dir: override save_dir (checkpoint folder root)
# - resume_path: explicit global_step_* folder to load (takes precedence)
resume_dir=""
resume_path=""

# vLLM engine selection (v0/v1). v1 can behave differently (sometimes more
# stable, sometimes more fragile depending on features). Default to v0 unless
# user explicitly requests v1.
verl_vllm_use_v1="${VERL_VLLM_USE_V1:-0}"

# -------------------------
# Distributed stability knobs
# -------------------------
# Timeout for torch.distributed collectives inside FSDP workers.
# Multi-node long-context rollouts can exceed 30min per step; use a larger timeout.
nccl_timeout_sec="${VERL_NCCL_TIMEOUT_SEC:-36000}"
# Enable NCCL flight recorder for debugging collective timeouts (0 disables).
torch_nccl_trace_buffer_size="${TORCH_NCCL_TRACE_BUFFER_SIZE:-0}"

# -------------------------
# Rollout / trainer knobs
# -------------------------
response_n=32
train_batch_size=128
micro_bsz_per_gpu=2

rollout_n=8
token_check_interval=2048
min_step_tokens=2048
max_interventions=2
actor_update_streams="both"

verifier_max_hint_tokens=512
estimated_hint_tokens=512
verifier_max_new_tokens=4096
confidence_threshold=0.0

trainer_total_epochs=10
trainer_total_training_steps=1000
trainer_save_freq=20
trainer_test_freq=-1
trainer_log_val_generations="False"
# NOTE: Prefer dual-rollout dumps (control/exp JSON with rewards) and disable the
# extra decoded-text rollout dump to save disk + reduce overhead.
trainer_rollout_dump_freq=0
trainer_dual_rollout_dump_freq=5
trainer_control_rollout_sync_freq=5
trainer_verifier_lora_sync_freq=1
trainer_verifier_lora_save_freq=20
max_prompt_length=2048

# Ray object store memory (GB). Helps avoid Ray spill for 32k rollouts.
# Leave empty to keep Ray defaults.
ray_object_store_memory_gb="${VERL_RAY_OBJECT_STORE_MEMORY_GB:-}"

# -------------------------
# Credit assignment (cf_branch)
# -------------------------
verifier_credit_assignment="cf_branch"
# Note: cf-branch sampling is controlled by cf_branch_prob. For non-cf credit
# assignment modes (e.g. global_gap), we will default to cf_branch_prob=0.0
# unless the user explicitly overrides it.
cf_branch_prob="1.0"
cf_branch_max_events_per_sample="2"
cf_branch_state_hash_mod="1024"
cf_branch_k="2"

# Advantage estimator
# - co_grpo: Co-GRPO (dual-stream / verifier interventions)
# - grpo   : plain GRPO baseline (no verifier interventions)
adv_estimator="co_grpo"

# -------------------------
# Intervention penalty (cost)
# -------------------------
# NOTE: For cold-start verifier, a non-zero penalty can quickly collapse
# intervention rate before cf-branch signal stabilizes. Default to 0.0.
intervention_penalty_freq_coef="0.0"
intervention_penalty_len_coef="0.0"

# Verifier LoRA training hyperparams (set lora_rank=0 to disable LoRA updates).
verifier_lora_rank=64
verifier_lora_alpha=128
verifier_lora_dropout=0.05
verifier_lr=1e-5
verifier_loss_weight=1.0

# -------------------------
# vLLM concurrency overrides (optional)
# -------------------------
rollout_enable_kv_cache_optimization="" # empty => keep launcher default (by_step defaults to False)
# Default 512 is aggressive for 32k long-context multi-node runs (high KV-cache
# preemption + instability). Keep a safer default; override via --max-num-seqs.
rollout_max_num_seqs="256"
# Keep empty by default so the launcher can pick a safe value (and cap it for
# LoRA-enabled long-context runs).
rollout_max_num_batched_tokens=""

# -------------------------
# Reward endpoints
# -------------------------
export REWARD_MODEL_URLS="${REWARD_MODEL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"
export REWARD_MODEL_URLS="$(_sanitize_csv_urls "${REWARD_MODEL_URLS}")"
if [ -z "${REWARD_MODEL_URLS}" ]; then
  echo "[rjob32][ERR] REWARD_MODEL_URLS is empty after sanitation." >&2
  exit 2
fi
export REWARD_MODEL_KEY="${REWARD_MODEL_KEY:-EMPTY}"

# -------------------------
# rjob resource requests
# -------------------------
rjob_cpu="${RJOB_CPU:-128}"
rjob_memory="${RJOB_MEMORY:-1200000}"
# Mount host /dev/shm into the container to give Ray a large shared-memory
# object store. Without this, /dev/shm is often a tiny tmpfs (e.g. 7GB),
# which causes Ray to spill massive rollout DataProto objects to disk.
rjob_share_host_shm="${RJOB_SHARE_HOST_SHM:-True}"

cf_branch_prob_user_set=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --exp-name) exp_name="$2"; shift 2 ;;
    --model-path) model_path="$2"; shift 2 ;;
    --dataset) dataset_name="$2"; shift 2 ;;
    --val-dataset) val_dataset_path="$2"; shift 2 ;;
    --verifier-lora-path) verifier_lora_path="$2"; shift 2 ;;
    --no-verifier-lora) verifier_lora_path=""; verifier_lora_rank=0; shift 1 ;;
    --verifier-lora-rank) verifier_lora_rank="$2"; shift 2 ;;
    --nnodes) nnodes="$2"; shift 2 ;;
    --gpus-per-node) n_gpus_per_node="$2"; shift 2 ;;
    --tp) gen_tp="$2"; shift 2 ;;
    --gpu-mem) gpu_memory_utilization="$2"; shift 2 ;;
    --steps) trainer_total_training_steps="$2"; shift 2 ;;
    --train-bsz) train_batch_size="$2"; shift 2 ;;
    --micro-bsz) micro_bsz_per_gpu="$2"; shift 2 ;;
    --response-n) response_n="$2"; shift 2 ;;
    --rollout-n) rollout_n="$2"; shift 2 ;;
    --resume-dir) resume_dir="$2"; shift 2 ;;
    --resume-path) resume_path="$2"; shift 2 ;;
    --token-check-interval) token_check_interval="$2"; shift 2 ;;
    --min-step-tokens) min_step_tokens="$2"; shift 2 ;;
    --max-interventions) max_interventions="$2"; shift 2 ;;
    --actor-update-streams) actor_update_streams="$2"; shift 2 ;;
    --verifier-max-new-tokens) verifier_max_new_tokens="$2"; shift 2 ;;
    --confidence-threshold) confidence_threshold="$2"; shift 2 ;;
    --cf-branch-k) cf_branch_k="$2"; shift 2 ;;
    --cf-branch-prob) cf_branch_prob="$2"; cf_branch_prob_user_set=1; shift 2 ;;
    --cf-branch-max-events) cf_branch_max_events_per_sample="$2"; shift 2 ;;
    --verifier-credit-assignment) verifier_credit_assignment="$2"; shift 2 ;;
    --adv-estimator) adv_estimator="$2"; shift 2 ;;
    --intervention-penalty-freq-coef) intervention_penalty_freq_coef="$2"; shift 2 ;;
    --intervention-penalty-len-coef) intervention_penalty_len_coef="$2"; shift 2 ;;
    --max-num-seqs) rollout_max_num_seqs="$2"; shift 2 ;;
    --max-num-batched-tokens) rollout_max_num_batched_tokens="$2"; shift 2 ;;
    --dump-freq) trainer_rollout_dump_freq="$2"; shift 2 ;;
    --dual-dump-freq) trainer_dual_rollout_dump_freq="$2"; shift 2 ;;
    --max-prompt-length) max_prompt_length="$2"; shift 2 ;;
    --nccl-timeout) nccl_timeout_sec="$2"; shift 2 ;;
    --nccl-trace-buffer-size) torch_nccl_trace_buffer_size="$2"; shift 2 ;;
    --ray-object-store-gb) ray_object_store_memory_gb="$2"; shift 2 ;;
    --prefix-cache) rollout_enable_kv_cache_optimization="True"; shift 1 ;;
    --no-prefix-cache) rollout_enable_kv_cache_optimization="False"; shift 1 ;;
    --vllm-v1) verl_vllm_use_v1="1"; shift 1 ;;
    --vllm-v0) verl_vllm_use_v1="0"; shift 1 ;;
    --rjob-cpu) rjob_cpu="$2"; shift 2 ;;
    --rjob-memory) rjob_memory="$2"; shift 2 ;;
    --share-host-shm) rjob_share_host_shm="True"; shift 1 ;;
    --no-share-host-shm) rjob_share_host_shm="False"; shift 1 ;;
    -h|--help)
      echo "Usage: bash run_rjob_cogrpo_32gpu.sh [options]"
      echo "  --exp-name <name>"
      echo "  --model-path <path>             (default: ${model_path})"
      echo "  --dataset <name|path>           (default: ${dataset_name})"
      echo "  --val-dataset <path>            (default: ${val_dataset_path})"
      echo "  --verifier-lora-path <path>     (default: ${verifier_lora_path})"
      echo "  --no-verifier-lora              (disable verifier LoRA: set lora_rank=0)"
      echo "  --verifier-lora-rank <int>      (default: ${verifier_lora_rank})"
      echo "  --nnodes <int>                  (default: 4)"
      echo "  --gpus-per-node <int>           (default: 8)"
      echo "  --tp <int>                      (default: 1)"
      echo "  --gpu-mem <float>               (default: 0.8)"
      echo "  --steps <int>                   (default: 1000)"
      echo "  --train-bsz <int>               (default: 128)"
      echo "  --micro-bsz <int>               (default: 2)"
      echo "  --response-n <int>              (default: 32 => 32k)"
      echo "  --rollout-n <int>               (default: 8)"
      echo "  --resume-dir <path>             (override checkpoint save_dir)"
      echo "  --resume-path <path>            (explicit global_step_* path to load)"
      echo "  --token-check-interval <int>    (default: 2048)"
      echo "  --min-step-tokens <int>         (default: 2048)"
      echo "  --max-interventions <int>       (default: 2)"
      echo "  --actor-update-streams <mode>   (default: both; exp|both|control)"
      echo "  --verifier-max-new-tokens <int> (default: 4096)"
      echo "  --confidence-threshold <float>  (default: 0.0; 0=off)"
      echo "  --verifier-credit-assignment <name> (default: cf_branch; global_gap|cf_branch|cf)"
      echo "  --cf-branch-k <int>             (default: 2)"
      echo "  --cf-branch-prob <float>        (default: 1.0; set 0.0 to disable cf sampling)"
      echo "  --cf-branch-max-events <int>    (default: 2)"
      echo "  --adv-estimator <name>          (default: co_grpo; grpo|co_grpo)"
      echo "  --intervention-penalty-freq-coef <float> (default: 0.0)"
      echo "  --intervention-penalty-len-coef <float>  (default: 0.0)"
      echo "  --max-num-seqs <int>            (default: 512)"
      echo "  --max-num-batched-tokens <int>  (default: <empty>; launcher picks + caps for LoRA)"
      echo "  --dump-freq <int>               (default: 0; disable decoded rollout dump)"
      echo "  --dual-dump-freq <int>          (default: 5; dumps dual_rollout_data JSON)"
      echo "  --max-prompt-length <int>       (default: 2048)"
      echo "  --nccl-timeout <int>            (default: 36000; seconds)"
      echo "  --nccl-trace-buffer-size <int>  (default: 0; 0 disables flight recorder)"
      echo "  --ray-object-store-gb <int>     (default: ${ray_object_store_memory_gb:-<empty>}; override Ray object store memory)"
      echo "  --prefix-cache                  (force enable vLLM prefix caching)"
      echo "  --no-prefix-cache               (force disable vLLM prefix caching)"
      echo "  --vllm-v1                       (set VERL_VLLM_USE_V1=1)"
      echo "  --vllm-v0                       (set VERL_VLLM_USE_V1=0; default)"
      echo "  --rjob-cpu <int>                (default: ${rjob_cpu})"
      echo "  --rjob-memory <int>             (default: ${rjob_memory})"
      echo "  --share-host-shm                (default: True; reduces Ray spill for 32k)"
      echo "  --no-share-host-shm             (disable host /dev/shm mount)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

verifier_credit_assignment_norm="$(echo "${verifier_credit_assignment}" | tr '[:upper:]' '[:lower:]')"
case "${verifier_credit_assignment_norm}" in
  global_gap|cf_branch|cf) verifier_credit_assignment="${verifier_credit_assignment_norm}" ;;
  *)
    echo "[rjob32][ERR] --verifier-credit-assignment must be one of: global_gap|cf_branch|cf (got ${verifier_credit_assignment})." >&2
    exit 2
    ;;
esac

# Default to disabling cf sampling when not using cf-based credit assignment.
if [ "${cf_branch_prob_user_set}" -eq 0 ] && [ "${verifier_credit_assignment}" != "cf_branch" ] && [ "${verifier_credit_assignment}" != "cf" ]; then
  cf_branch_prob="0.0"
fi

actor_update_streams_norm="$(echo "${actor_update_streams}" | tr '[:upper:]' '[:lower:]')"
case "${actor_update_streams_norm}" in
  exp|control|both) actor_update_streams="${actor_update_streams_norm}" ;;
  *)
    echo "[rjob32][ERR] --actor-update-streams must be one of: exp|control|both (got ${actor_update_streams})." >&2
    exit 2
    ;;
esac

adv_estimator_norm="$(echo "${adv_estimator}" | tr '[:upper:]' '[:lower:]')"
case "${adv_estimator_norm}" in
  grpo|co_grpo) adv_estimator="${adv_estimator_norm}" ;;
  *)
    echo "[rjob32][ERR] --adv-estimator must be one of: grpo|co_grpo (got ${adv_estimator})." >&2
    exit 2
    ;;
esac

# Plain GRPO baseline: disable verifier LoRA by default to avoid initializing
# unused PEFT / runtime-LoRA machinery (and to keep settings aligned with
# "no verifier intervention" baselines).
if [ "${adv_estimator}" = "grpo" ] || [ "${max_interventions}" = "0" ]; then
  verifier_lora_path=""
  verifier_lora_rank=0
fi

# If user clears verifier_lora_path manually, disable LoRA updates by default.
if [ -z "${verifier_lora_path}" ]; then
  verifier_lora_rank=0
fi

# Default object store size for long-context rjob runs when host /dev/shm is mounted.
if [ -z "${ray_object_store_memory_gb}" ] && [ "${rjob_share_host_shm}" = "True" ]; then
  ray_object_store_memory_gb=450
fi
if [ -n "${ray_object_store_memory_gb}" ]; then
  if ! [[ "${ray_object_store_memory_gb}" =~ ^[0-9]+$ ]]; then
    echo "[rjob32][ERR] --ray-object-store-gb must be an integer (got ${ray_object_store_memory_gb})." >&2
    exit 2
  fi
  if [ "${ray_object_store_memory_gb}" -le 0 ]; then
    echo "[rjob32][ERR] --ray-object-store-gb must be >0 (got ${ray_object_store_memory_gb})." >&2
    exit 2
  fi
fi

if [ -z "${exp_name}" ]; then
  exp_name="cogrpo32-$(date +%m%d%H%M)"
fi

launcher_script="/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/run_multinodes_cgrpo_v2.sh"
hf_cache_dir="/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub"
rjob_name="rjob_${exp_name//-/_}"

echo "[rjob32] name=${rjob_name} cluster=${cluster} nnodes=${nnodes} gpu_per_node=${n_gpus_per_node} tp=${gen_tp} gpu_mem=${gpu_memory_utilization}"
echo "[rjob32] exp_name=${exp_name} dataset=${dataset_name} response_n=${response_n} rollout_n=${rollout_n}"
echo "[rjob32] val_dataset=${val_dataset_path}"
echo "[rjob32] train_bsz=${train_batch_size} micro_bsz=${micro_bsz_per_gpu} steps=${trainer_total_training_steps}"
echo "[rjob32] adv_estimator=${adv_estimator}"
echo "[rjob32] actor_update_streams=${actor_update_streams}"
echo "[rjob32] verifier_credit_assignment=${verifier_credit_assignment} cf_branch_prob=${cf_branch_prob}"
echo "[rjob32] cf_branch_k=${cf_branch_k} cf_max_events=${cf_branch_max_events_per_sample}"
if [ -n "${resume_dir}" ] || [ -n "${resume_path}" ]; then
  echo "[rjob32] resume_dir=${resume_dir:-<empty>} resume_path=${resume_path:-<empty>}"
fi
echo "[rjob32] vllm_max_num_seqs=${rollout_max_num_seqs} vllm_max_num_batched_tokens=${rollout_max_num_batched_tokens:-<empty>}"
echo "[rjob32] VERL_VLLM_USE_V1=${verl_vllm_use_v1}"
echo "[rjob32] intervention_penalty_freq_coef=${intervention_penalty_freq_coef} intervention_penalty_len_coef=${intervention_penalty_len_coef}"
echo "[rjob32] verifier_lora_path=${verifier_lora_path:-<empty>} verifier_lora_rank=${verifier_lora_rank}"
echo "[rjob32] REWARD_MODEL_URLS=${REWARD_MODEL_URLS}"
echo "[rjob32] nccl_timeout_sec=${nccl_timeout_sec} TORCH_NCCL_TRACE_BUFFER_SIZE=${torch_nccl_trace_buffer_size}"
if [ -n "${ray_object_store_memory_gb}" ]; then
  echo "[rjob32] VERL_RAY_OBJECT_STORE_MEMORY_GB=${ray_object_store_memory_gb}"
fi

rjob submit \
  --name="${rjob_name}" \
  --gpu="${n_gpus_per_node}" \
  --memory="${rjob_memory}" \
  --cpu="${rjob_cpu}" \
  --charged-group="${cluster}" \
  --private-machine=group \
  --share-host-shm="${rjob_share_host_shm}" \
  --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
  --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
  --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
  --image=registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605 \
  -P "${nnodes}" \
  --host-network=true \
  -e DISTRIBUTED_JOB=true \
  --custom-resources rdma/mlnx_shared="${n_gpus_per_node}" \
  -- bash -c "
    unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY &&
    export HF_HOME='${hf_cache_dir}' &&
    export HF_HUB_CACHE='${hf_cache_dir}' &&
    export HUGGINGFACE_HUB_CACHE='${hf_cache_dir}' &&
    export HF_DATASETS_CACHE='/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/hf_datasets_cache' &&
    export HF_DATASETS_OFFLINE=1 &&
    export TRANSFORMERS_OFFLINE=1 &&
    export HF_EVALUATE_OFFLINE=1 &&
    export HF_HUB_OFFLINE=1 &&
    export VERL_NCCL_TIMEOUT_SEC='${nccl_timeout_sec}' &&
    export TORCH_NCCL_TRACE_BUFFER_SIZE='${torch_nccl_trace_buffer_size}' &&
    export REWARD_MODEL_URLS='${REWARD_MODEL_URLS}' &&
    export REWARD_MODEL_KEY='${REWARD_MODEL_KEY}' &&
    export PARALLEL_CONTROL_EXP='False' &&
    export ADV_ESTIMATOR='${adv_estimator}' &&
    export ACTOR_UPDATE_STREAMS='${actor_update_streams}' &&
    export USE_KL_LOSS='False' &&
    export KL_LOSS_COEF='0.0' &&
    export CLIP_RATIO_C='10.0' &&
    export VERL_VAL_FILE='${val_dataset_path}' &&
    export VERL_VLLM_USE_V1='${verl_vllm_use_v1}' &&
    export VERL_RAY_OBJECT_STORE_MEMORY_GB='${ray_object_store_memory_gb}' &&
    export ROLLOUT_N='${rollout_n}' &&
    export CF_BRANCH_K='${cf_branch_k}' &&
    export ROLLOUT_ENABLE_KV_CACHE_OPTIMIZATION='${rollout_enable_kv_cache_optimization}' &&
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS='${rollout_max_num_batched_tokens}' &&
    export ROLLOUT_MAX_NUM_SEQS='${rollout_max_num_seqs}' &&
    chmod +x '${launcher_script}' &&
    '${launcher_script}' \
      '${model_path}' \
      '${nnodes}' \
      '${n_gpus_per_node}' \
      '${gen_tp}' \
      '${gpu_memory_utilization}' \
      '${dataset_name}' \
      '${verl_debug}' \
      '${exp_name}' \
      '${work_dir}' \
      full full '${resume_dir}' '${resume_path}' \
      '${verifier_lora_path}' \
      '${response_n}' \
      '${train_batch_size}' \
      '${token_check_interval}' '${min_step_tokens}' '${max_interventions}' '${verifier_max_hint_tokens}' '${estimated_hint_tokens}' \
      0.5 True 0.3 0.5 \
      '${intervention_penalty_freq_coef}' '${intervention_penalty_len_coef}' \
      headroom 1.0 0.05 \
      '${trainer_rollout_dump_freq}' '${trainer_dual_rollout_dump_freq}' \
      False '${micro_bsz_per_gpu}' \
      1.0 \
      by_step '${confidence_threshold}' \
      '${verifier_lora_rank}' '${verifier_lora_alpha}' '${verifier_lora_dropout}' '${verifier_lr}' '${verifier_loss_weight}' \
      '${verifier_max_new_tokens}' \
      1 1.0 \
      '${trainer_total_epochs}' '${trainer_total_training_steps}' \
      '${trainer_save_freq}' '${trainer_test_freq}' '${trainer_log_val_generations}' \
      '${trainer_control_rollout_sync_freq}' '${trainer_verifier_lora_sync_freq}' '${trainer_verifier_lora_save_freq}' \
      '${max_prompt_length}' \
      '${verifier_credit_assignment}' '${cf_branch_prob}' '${cf_branch_max_events_per_sample}' '${cf_branch_state_hash_mod}'
  "
