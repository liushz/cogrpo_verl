#!/usr/bin/env bash
set -eo pipefail

# 8-GPU rjob launcher for XTuner CoGRPO v2 (cf_branch, exp-only smoke).
#
# Goal: verify the XTuner pipeline can run with the *real* long-output settings
# (32k response / 35k pack), while using a small training batch size (GBS=8).
#
# Notes:
# - Keeps output length knobs aligned with the baseline (do NOT shorten 32k).
# - Keeps it lightweight via small `ROLLOUT_STEPS` / `TRAIN_OPTIMIZER_STEPS`.
# - Uses lmdeploy fp8 repo copy under your own directory by default.

if [ -f /etc/profile.d/ssh-init.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/ssh-init.sh || true
fi

# Avoid proxy interference when calling rjob API endpoints.
unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

is_truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

cluster="llmit_gpu"
gpus_per_node="${GPUS_PER_NODE:-${GPUS:-8}}"
nnodes="${NNODES:-1}"
if ! [[ "${gpus_per_node}" =~ ^[0-9]+$ ]] || [ "${gpus_per_node}" -le 0 ]; then
  echo "[xtuner-cf8-bsz8] ERROR: GPUS_PER_NODE/GPUS must be positive integer, got '${gpus_per_node}'" >&2
  exit 2
fi
if ! [[ "${nnodes}" =~ ^[0-9]+$ ]] || [ "${nnodes}" -le 0 ]; then
  echo "[xtuner-cf8-bsz8] ERROR: NNODES must be positive integer, got '${nnodes}'" >&2
  exit 2
fi
total_gpus=$((gpus_per_node * nnodes))

xtuner_dir="/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner"
xtuner_branch_expected="${XTUNER_BRANCH_EXPECTED:-cogrpo-v2-xtuner-control-exp}"

if ! git -C "${xtuner_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[xtuner-cf8-bsz8] ERROR: not a git repo: ${xtuner_dir}" >&2
  exit 1
fi

xtuner_branch="$(git -C "${xtuner_dir}" rev-parse --abbrev-ref HEAD)"
xtuner_commit="$(git -C "${xtuner_dir}" rev-parse --short HEAD)"
if [ "${xtuner_branch}" != "${xtuner_branch_expected}" ]; then
  echo "[xtuner-cf8-bsz8] ERROR: expected xtuner branch ${xtuner_branch_expected}, got ${xtuner_branch}" >&2
  echo "[xtuner-cf8-bsz8] Hint: cd ${xtuner_dir} && git checkout ${xtuner_branch_expected}" >&2
  exit 1
fi

exp_name="${EXP_NAME:-xtcf8-bsz8-$(date +%m%d%H%M)}"

model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"
dataset_path="${DATASET_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-dapo-math-17k.jsonl}"
conda_env_path="${CONDA_ENV_PATH:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro}"
lmdeploy_path="${LMDEPLOY_PATH_OVERRIDE:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/lmdeploy_manually_fp8}"

# Keep output length knobs aligned with baseline (32k output).
global_batch_size="${GLOBAL_BATCH_SIZE:-8}"
prompt_repeat_k="${PROMPT_REPEAT_K:-16}"
rollout_steps="${ROLLOUT_STEPS:-5}"
train_optimizer_steps="${TRAIN_OPTIMIZER_STEPS:-1}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-32768}"
pack_max_length="${PACK_MAX_LENGTH:-35840}"
enable_partial_rollout="${ENABLE_PARTIAL_ROLLOUT:-1}"
staleness_threshold="${STALENESS_THRESHOLD:-0.0}"
tail_batch_candidate_steps="${TAIL_BATCH_CANDIDATE_STEPS:-4}"
tail_batch_trigger_size="${TAIL_BATCH_TRIGGER_SIZE:-0}"
max_concurrent="${MAX_CONCURRENT:-8}"
gpu_mem_util="${GPU_MEM_UTIL:-0.88}"

# HuggingFace checkpoint saving can trigger large CPU RAM spikes for FSDP full-model training.
# Disable by default for debug runs; set HF_INTERVAL>0 to enable.
hf_interval="${HF_INTERVAL:-0}"

# DCP checkpoint / auto-resume knobs (to make restarts "verl-like").
#
# Notes:
# - Checkpoints are stored under WORK_DIR/<timestamp>/checkpoints/ckpt-step-N/.
# - AUTO_RESUME=1 will automatically pick the latest checkpoint recorded in WORK_DIR/.xtuner_grpo.
auto_resume="${AUTO_RESUME:-1}"
checkpoint_interval="${CHECKPOINT_INTERVAL:-20}"
checkpoint_maxkeep="${CHECKPOINT_MAXKEEP:-2}"
checkpoint_no_save_optimizer="${CHECKPOINT_NO_SAVE_OPTIMIZER:-0}"
strict_load="${STRICT_LOAD:-1}"
resume_from="${RESUME_FROM:-}"
load_optimizer_states="${LOAD_OPTIMIZER_STATES:-1}"
load_optimizer_args="${LOAD_OPTIMIZER_ARGS:-1}"

# For short debug runs (e.g. rollout_steps=5), ensure we still emit at least one checkpoint
# by clamping interval to (rollout_steps - 1).
if [ "${checkpoint_interval}" -gt 0 ] && [ "${rollout_steps}" -gt 1 ] && [ "${checkpoint_interval}" -ge "${rollout_steps}" ]; then
  checkpoint_interval="$((rollout_steps - 1))"
fi

if [ "${checkpoint_no_save_optimizer}" = "1" ]; then
  # Avoid resume crashes when optimizer snapshots are intentionally skipped.
  load_optimizer_states=0
  load_optimizer_args=0
fi

# CoGRPO rollout knobs (by_step + cf_branch).
#
# Keep aligned with verl defaults:
# - check interval / min step tokens: 2048
# - max interventions: 2
cogrpo_token_check_interval="${COGRPO_TOKEN_CHECK_INTERVAL:-2048}"
cogrpo_min_step_tokens="${COGRPO_MIN_STEP_TOKENS:-2048}"
cogrpo_max_interventions="${COGRPO_MAX_INTERVENTIONS:-2}"
cogrpo_confidence_threshold="${COGRPO_CONFIDENCE_THRESHOLD:-0.0}"

cogrpo_cf_branch_prob="${COGRPO_CF_BRANCH_PROB:-1.0}"
cogrpo_cf_branch_max_events_per_sample="${COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE:-2}"
cogrpo_cf_branch_state_hash_mod="${COGRPO_CF_BRANCH_STATE_HASH_MOD:-1024}"
cogrpo_cf_branch_k="${COGRPO_CF_BRANCH_K:-2}"
if ! [[ "${prompt_repeat_k}" =~ ^[0-9]+$ ]] || [ "${prompt_repeat_k}" -le 0 ]; then
  echo "[xtuner-cf8-bsz8] ERROR: PROMPT_REPEAT_K must be positive integer, got '${prompt_repeat_k}'" >&2
  exit 2
fi
default_control_k=$((prompt_repeat_k / 2))
default_exp_k=$((prompt_repeat_k - default_control_k))
cogrpo_control_k="${COGRPO_CONTROL_K:-${CO_GRPO_CONTROL_K:-${default_control_k}}}"
cogrpo_exp_k="${COGRPO_EXP_K:-${CO_GRPO_EXP_K:-${default_exp_k}}}"
actor_update_streams="${ACTOR_UPDATE_STREAMS:-both}"
use_curriculum_weighting="${USE_CURRICULUM_WEIGHTING:-1}"
control_group_weight="${CONTROL_GROUP_WEIGHT:-0.5}"
curriculum_start_weight="${CURRICULUM_START_WEIGHT:-0.3}"
curriculum_end_weight="${CURRICULUM_END_WEIGHT:-0.5}"
cogrpo_verifier_reward_mode="${COGRPO_VERIFIER_REWARD_MODE:-headroom}"
cogrpo_verifier_reward_headroom_min="${COGRPO_VERIFIER_REWARD_HEADROOM_MIN:-0.05}"
cogrpo_verifier_reward_improve_coef="${COGRPO_VERIFIER_REWARD_IMPROVE_COEF:-1.0}"
cogrpo_strict_dual_stream="${COGRPO_STRICT_DUAL_STREAM:-1}"
cogrpo_verifier_update_base="${COGRPO_VERIFIER_UPDATE_BASE:-0}"
cogrpo_strict_verifier_sync="${COGRPO_STRICT_VERIFIER_SYNC:-1}"

# Reduce Ray object-store pressure by default (verl-aligned behavior).
cogrpo_keep_cf_rollouts="${COGRPO_KEEP_CF_ROLLOUTS:-0}"
cogrpo_verifier_skip_truncated="${COGRPO_VERIFIER_SKIP_TRUNCATED:-1}"

# Optional: dump CoGRPO extra_info (hints/intervention positions) into trajectory jsonl for debugging.
cogrpo_dump_trajectory_extra="${COGRPO_DUMP_TRAJECTORY_EXTRA:-1}"
cogrpo_dump_trajectory_max_samples="${COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES:-16}"
cogrpo_dump_verl_compat_fields="${COGRPO_DUMP_VERL_COMPAT_FIELDS:-1}"
# Print high-signal debug lines to rank_0.log (safe; bounded by max samples).
cogrpo_print_traj_debug="${COGRPO_PRINT_TRAJECTORY_DEBUG:-1}"
cogrpo_print_traj_max_samples="${COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES:-4}"
# Dump hint insertion context snippets into trajectory extra_info (for post-mortem).
cogrpo_debug_hint_context="${COGRPO_DEBUG_HINT_CONTEXT:-1}"
cogrpo_debug_hint_context_tokens="${COGRPO_DEBUG_HINT_CONTEXT_TOKENS:-96}"


# Verifier inference knobs (non-decoupled verifier=actor by default; LoRA disabled for smoke).
cogrpo_verifier_max_new_tokens="${COGRPO_VERIFIER_MAX_NEW_TOKENS:-4096}"
cogrpo_verifier_max_prompt_length="${COGRPO_VERIFIER_MAX_PROMPT_LENGTH:-16384}"
cogrpo_verifier_max_hint_tokens="${COGRPO_VERIFIER_MAX_HINT_TOKENS:-512}"
cogrpo_estimated_hint_tokens="${COGRPO_ESTIMATED_HINT_TOKENS:-512}"

cogrpo_verifier_lora_enable="${COGRPO_VERIFIER_LORA_ENABLE:-0}"
cogrpo_verifier_lora_save_freq="${COGRPO_VERIFIER_LORA_SAVE_FREQ:-20}"
cogrpo_verifier_lora_path="${COGRPO_VERIFIER_LORA_PATH:-}"
cogrpo_verifier_lora_rank="${COGRPO_VERIFIER_LORA_RANK:-64}"
cogrpo_verifier_lora_alpha="${COGRPO_VERIFIER_LORA_ALPHA:-128}"
cogrpo_verifier_lora_dropout="${COGRPO_VERIFIER_LORA_DROPOUT:-0.05}"
cogrpo_verifier_lr="${COGRPO_VERIFIER_LR:-1e-5}"
cogrpo_verifier_lora_sync_freq="${COGRPO_VERIFIER_LORA_SYNC_FREQ:-1}"
cogrpo_verifier_model="${COGRPO_VERIFIER_MODEL:-}"
cogrpo_verifier_server_url="${COGRPO_VERIFIER_SERVER_URL:-}"
cogrpo_verifier_server_urls="${COGRPO_VERIFIER_SERVER_URLS:-${VERIFIER_SERVER_URLS:-}}"
cogrpo_verifier_server_url_dict="${COGRPO_VERIFIER_SERVER_URL_DICT:-${VERIFIER_SERVER_URL_DICT:-}}"
cogrpo_wait_verifier_ready="${COGRPO_WAIT_VERIFIER_READY:-1}"
cogrpo_verifier_wait_timeout_s="${COGRPO_VERIFIER_WAIT_TIMEOUT_S:-7200}"
cogrpo_verifier_wait_interval_s="${COGRPO_VERIFIER_WAIT_INTERVAL_S:-10}"
cogrpo_verifier_wait_check_update="${COGRPO_VERIFIER_WAIT_CHECK_UPDATE_ENDPOINT:-auto}"
if [ -z "${cogrpo_verifier_server_urls}" ] && [ -n "${cogrpo_verifier_server_url}" ]; then
  cogrpo_verifier_server_urls="${cogrpo_verifier_server_url}"
fi

# Verifier debug / parser mode (optional; enabled by watcher when needed).
cogrpo_verifier_debug="${COGRPO_VERIFIER_DEBUG:-0}"
cogrpo_verifier_debug_max_calls="${COGRPO_VERIFIER_DEBUG_MAX_CALLS:-200}"
cogrpo_verifier_debug_text_chars="${COGRPO_VERIFIER_DEBUG_TEXT_CHARS:-640}"
cogrpo_verifier_debug_dump="${COGRPO_VERIFIER_DEBUG_DUMP:-0}"
cogrpo_verifier_debug_dump_max_calls="${COGRPO_VERIFIER_DEBUG_DUMP_MAX_CALLS:-200}"
cogrpo_verifier_debug_dump_dir="${COGRPO_VERIFIER_DEBUG_DUMP_DIR:-/tmp/xtuner_verifier_debug}"
cogrpo_verifier_debug_dump_full_text="${COGRPO_VERIFIER_DEBUG_DUMP_FULL_TEXT:-0}"
cogrpo_verifier_parser_mode="${COGRPO_VERIFIER_PARSER_MODE:-auto}"
cogrpo_verl_hint_injection_py="${COGRPO_VERL_HINT_INJECTION_PY:-${REPRO_ROOT:-/mnt/shared-storage-user/liuhongwei/main_works/repos/repro}/verl/workers/rollout/vllm_rollout/verifier_hint_injection.py}"
cogrpo_verl_rollout_spmd_py="${COGRPO_VERL_ROLLOUT_SPMD_PY:-${REPRO_ROOT:-/mnt/shared-storage-user/liuhongwei/main_works/repos/repro}/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py}"

# Reward endpoints / key (keep aligned with local debug defaults unless user overrides).
reward_model_urls="${REWARD_MODEL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"
reward_model_key="${REWARD_MODEL_KEY:-EMPTY}"

if ! [[ "${cogrpo_control_k}" =~ ^[0-9]+$ ]] || ! [[ "${cogrpo_exp_k}" =~ ^[0-9]+$ ]]; then
  echo "[xtuner-cf8-bsz8] ERROR: COGRPO_CONTROL_K/COGRPO_EXP_K must be integer >= 0, got control='${cogrpo_control_k}', exp='${cogrpo_exp_k}'" >&2
  exit 2
fi
if [ $((cogrpo_control_k + cogrpo_exp_k)) -ne "${prompt_repeat_k}" ]; then
  echo "[xtuner-cf8-bsz8] ERROR: invalid stream split control_k=${cogrpo_control_k}, exp_k=${cogrpo_exp_k}, PROMPT_REPEAT_K=${prompt_repeat_k}" >&2
  exit 2
fi
case "${actor_update_streams}" in
  both|control|exp) ;;
  *)
    echo "[xtuner-cf8-bsz8] ERROR: ACTOR_UPDATE_STREAMS must be one of {both,control,exp}, got '${actor_update_streams}'" >&2
    exit 2
    ;;
esac
if ! [[ "${cogrpo_verifier_lora_sync_freq}" =~ ^[0-9]+$ ]]; then
  echo "[xtuner-cf8-bsz8] ERROR: COGRPO_VERIFIER_LORA_SYNC_FREQ must be integer >= 0, got '${cogrpo_verifier_lora_sync_freq}'" >&2
  exit 2
fi
if ! [[ "${cogrpo_verifier_wait_timeout_s}" =~ ^[0-9]+$ ]]; then
  echo "[xtuner-cf8-bsz8] ERROR: COGRPO_VERIFIER_WAIT_TIMEOUT_S must be integer >= 0, got '${cogrpo_verifier_wait_timeout_s}'" >&2
  exit 2
fi
if ! [[ "${cogrpo_verifier_wait_interval_s}" =~ ^[0-9]+$ ]] || [ "${cogrpo_verifier_wait_interval_s}" -le 0 ]; then
  echo "[xtuner-cf8-bsz8] ERROR: COGRPO_VERIFIER_WAIT_INTERVAL_S must be positive integer, got '${cogrpo_verifier_wait_interval_s}'" >&2
  exit 2
fi
verifier_sync_urls_trimmed="$(printf '%s' "${cogrpo_verifier_server_urls}" | tr -d '[:space:]')"
verifier_sync_url_dict_trimmed="$(printf '%s' "${cogrpo_verifier_server_url_dict}" | tr -d '[:space:]')"
if is_truthy "${cogrpo_verifier_lora_enable}" && is_truthy "${cogrpo_strict_verifier_sync}" && \
   [ "${cogrpo_verifier_lora_sync_freq}" -le 0 ]; then
  echo "[xtuner-cf8-bsz8] ERROR: strict verifier sync requires COGRPO_VERIFIER_LORA_SYNC_FREQ > 0 when LoRA is enabled." >&2
  echo "[xtuner-cf8-bsz8] Hint: this run would train verifier LoRA but never sync it to verifier rollout inference." >&2
  exit 2
fi
if is_truthy "${cogrpo_verifier_lora_enable}" && is_truthy "${cogrpo_strict_verifier_sync}" && \
   [ "${cogrpo_verifier_lora_sync_freq}" -gt 0 ] && [ -z "${verifier_sync_urls_trimmed}" ] && [ -z "${verifier_sync_url_dict_trimmed}" ]; then
  echo "[xtuner-cf8-bsz8] ERROR: strict verifier sync requires COGRPO_VERIFIER_SERVER_URLS or COGRPO_VERIFIER_SERVER_URL_DICT when LoRA sync is enabled." >&2
  echo "[xtuner-cf8-bsz8] Hint: provide verifier rollout URLs, or set COGRPO_STRICT_VERIFIER_SYNC=0 for explicit debug-only no-sync runs." >&2
  exit 2
fi

rjob_name="rjob-${exp_name}"

echo "[xtuner-cf8-bsz8] xtuner=${xtuner_branch}@${xtuner_commit} dir=${xtuner_dir}"
echo "[xtuner-cf8-bsz8] name=${rjob_name} model=${model_path}"
echo "[xtuner-cf8-bsz8] dataset=${dataset_path}"
echo "[xtuner-cf8-bsz8] nnodes=${nnodes} gpus_per_node=${gpus_per_node} total_gpus=${total_gpus}"
echo "[xtuner-cf8-bsz8] rollout_steps=${rollout_steps} gbs=${global_batch_size} prk=${prompt_repeat_k}"
echo "[xtuner-cf8-bsz8] lengths: prompt=${max_prompt_length} resp=${max_response_length} pack=${pack_max_length}"
echo "[xtuner-cf8-bsz8] async=${enable_partial_rollout} staleness=${staleness_threshold} tail_steps=${tail_batch_candidate_steps} tail_trigger=${tail_batch_trigger_size} max_concurrent=${max_concurrent} gpu_mem_util=${gpu_mem_util}"
echo "[xtuner-cf8-bsz8] streams: control_k=${cogrpo_control_k} exp_k=${cogrpo_exp_k} actor_update=${actor_update_streams}"
echo "[xtuner-cf8-bsz8] curriculum: enable=${use_curriculum_weighting} control_w=${control_group_weight} start=${curriculum_start_weight} end=${curriculum_end_weight}"
echo "[xtuner-cf8-bsz8] verifier_reward=${cogrpo_verifier_reward_mode} headroom_min=${cogrpo_verifier_reward_headroom_min} improve_coef=${cogrpo_verifier_reward_improve_coef}"
echo "[xtuner-cf8-bsz8] strict_dual_stream=${cogrpo_strict_dual_stream} strict_verifier_sync=${cogrpo_strict_verifier_sync} verifier_update_base=${cogrpo_verifier_update_base}"
echo "[xtuner-cf8-bsz8] wait_verifier_ready=${cogrpo_wait_verifier_ready} timeout_s=${cogrpo_verifier_wait_timeout_s} interval_s=${cogrpo_verifier_wait_interval_s} check_update=${cogrpo_verifier_wait_check_update}"
if [ -n "${cogrpo_verifier_server_url_dict}" ]; then
  echo "[xtuner-cf8-bsz8] verifier_server_url_dict_set=1"
else
  echo "[xtuner-cf8-bsz8] verifier_server_urls=${cogrpo_verifier_server_urls:-<empty>}"
fi
echo "[xtuner-cf8-bsz8] resume: AUTO_RESUME=${auto_resume} RESUME_FROM=${resume_from:-<auto>} CKPT_INTERVAL=${checkpoint_interval} CKPT_MAXKEEP=${checkpoint_maxkeep} CKPT_NO_OPT=${checkpoint_no_save_optimizer}"
echo "[xtuner-cf8-bsz8] HF_INTERVAL=${hf_interval}"

custom_resource_args=()
if [ "${RJOB_USE_RDMA:-0}" = "1" ]; then
  custom_resource_args=(--custom-resources "rdma/mlnx_shared=${gpus_per_node}")
fi

rjob submit \
  --name="${rjob_name}" \
  --gpu="${gpus_per_node}" \
  --memory="${RJOB_MEMORY:-512000}" \
  --cpu="${RJOB_CPU:-64}" \
  --charged-group="${cluster}" \
  --private-machine=group \
  --share-host-shm=True \
  --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
  --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
  --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
  --image=registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605 \
  -P "${nnodes}" \
  --host-network=true \
  -e DISTRIBUTED_JOB=true \
  "${custom_resource_args[@]}" \
  -- bash -lc -- "
    unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY &&
    cd ${xtuner_dir} &&
    if ! command -v ray >/dev/null 2>&1; then
      if [ ! -f /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh ]; then
        echo '[xtuner-cf8-bsz8] ERROR: ray not found and conda.sh not found at /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh' >&2 ;
        exit 127 ;
      fi &&
      source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh &&
      conda activate '${conda_env_path}' ;
    fi &&
    which python &&
    python -V &&
    which ray &&
    ray --version &&
    export LMDEPLOY_PATH=${lmdeploy_path} &&
    echo \"[xtuner-cf8-bsz8] LMDEPLOY_PATH=${lmdeploy_path}\" &&
    if [ ! -d \"${lmdeploy_path}\" ]; then
      echo \"[xtuner-cf8-bsz8] ERROR: LMDEPLOY_PATH not found: ${lmdeploy_path}\" >&2 ;
      exit 2 ;
    fi &&
    export PYTHONPATH=\"${lmdeploy_path}:\${PYTHONPATH:-}\" &&
    if command -v git >/dev/null 2>&1; then
      git -C \"${lmdeploy_path}\" rev-parse --short HEAD 2>/dev/null | sed 's/^/[xtuner-cf8-bsz8] lmdeploy_git=/' || true ;
    fi &&
    python -c 'import lmdeploy; print(\"[xtuner-cf8-bsz8] lmdeploy_file=\", lmdeploy.__file__)' &&
    export NODE_COUNT=\${NODE_COUNT:-${nnodes}} &&
    export NODE_RANK=\${NODE_RANK:-0} &&
    export MASTER_ADDR=\${MASTER_ADDR:-127.0.0.1} &&
    echo \"[xtuner-cf8-bsz8] dist: NODE_COUNT=\${NODE_COUNT} NODE_RANK=\${NODE_RANK} MASTER_ADDR=\${MASTER_ADDR}\" &&
    export NUM_WORKERS=${total_gpus} &&
    export RAY_MAX_CONCURRENCY=1024 &&
    export XTUNER_RAY_TEMP_DIR=\${XTUNER_RAY_TEMP_DIR:-/tmp/ray_xtuner_\$(date +%m%d%H%M%S)} &&
    export REWARD_MODEL_URLS='${reward_model_urls}' &&
    export REWARD_MODEL_KEY='${reward_model_key}' &&
    export GLOBAL_BATCH_SIZE=${global_batch_size} &&
    export PROMPT_REPEAT_K=${prompt_repeat_k} &&
    export ROLLOUT_STEPS=${rollout_steps} &&
    export TRAIN_OPTIMIZER_STEPS=${train_optimizer_steps} &&
    export ENABLE_PARTIAL_ROLLOUT=${enable_partial_rollout} &&
    export STALENESS_THRESHOLD='${staleness_threshold}' &&
    export TAIL_BATCH_CANDIDATE_STEPS=${tail_batch_candidate_steps} &&
    export TAIL_BATCH_TRIGGER_SIZE=${tail_batch_trigger_size} &&
    export MAX_CONCURRENT=${max_concurrent} &&
    export GPU_MEM_UTIL='${gpu_mem_util}' &&
    export MAX_PROMPT_LENGTH=${max_prompt_length} &&
    export MAX_RESPONSE_LENGTH=${max_response_length} &&
    export PACK_MAX_LENGTH=${pack_max_length} &&
    export HF_INTERVAL=${hf_interval} &&
    export AUTO_RESUME=${auto_resume} &&
    export RESUME_FROM='${resume_from}' &&
    export STRICT_LOAD=${strict_load} &&
    export CHECKPOINT_INTERVAL=${checkpoint_interval} &&
    export CHECKPOINT_MAXKEEP=${checkpoint_maxkeep} &&
    export CHECKPOINT_NO_SAVE_OPTIMIZER=${checkpoint_no_save_optimizer} &&
    export LOAD_OPTIMIZER_STATES=${load_optimizer_states} &&
    export LOAD_OPTIMIZER_ARGS=${load_optimizer_args} &&
    export COGRPO_ENABLE=1 &&
    export COGRPO_CONTROL_K=${cogrpo_control_k} &&
    export COGRPO_EXP_K=${cogrpo_exp_k} &&
    export ACTOR_UPDATE_STREAMS='${actor_update_streams}' &&
    export USE_CURRICULUM_WEIGHTING='${use_curriculum_weighting}' &&
    export CONTROL_GROUP_WEIGHT='${control_group_weight}' &&
    export CURRICULUM_START_WEIGHT='${curriculum_start_weight}' &&
    export CURRICULUM_END_WEIGHT='${curriculum_end_weight}' &&
    export COGRPO_TOKEN_CHECK_INTERVAL=${cogrpo_token_check_interval} &&
    export COGRPO_MIN_STEP_TOKENS=${cogrpo_min_step_tokens} &&
    export COGRPO_MAX_INTERVENTIONS=${cogrpo_max_interventions} &&
    export COGRPO_CONFIDENCE_THRESHOLD=${cogrpo_confidence_threshold} &&
    export COGRPO_CF_BRANCH_PROB=${cogrpo_cf_branch_prob} &&
    export COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE=${cogrpo_cf_branch_max_events_per_sample} &&
    export COGRPO_CF_BRANCH_STATE_HASH_MOD=${cogrpo_cf_branch_state_hash_mod} &&
    export COGRPO_CF_BRANCH_K=${cogrpo_cf_branch_k} &&
    export COGRPO_VERIFIER_REWARD_MODE='${cogrpo_verifier_reward_mode}' &&
    export COGRPO_VERIFIER_REWARD_HEADROOM_MIN='${cogrpo_verifier_reward_headroom_min}' &&
    export COGRPO_VERIFIER_REWARD_IMPROVE_COEF='${cogrpo_verifier_reward_improve_coef}' &&
    export COGRPO_STRICT_DUAL_STREAM='${cogrpo_strict_dual_stream}' &&
    export COGRPO_KEEP_CF_ROLLOUTS=${cogrpo_keep_cf_rollouts} &&
    export COGRPO_VERIFIER_SKIP_TRUNCATED=${cogrpo_verifier_skip_truncated} &&
    export COGRPO_DUMP_TRAJECTORY_EXTRA=${cogrpo_dump_trajectory_extra} &&
    export COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES=${cogrpo_dump_trajectory_max_samples} &&
    export COGRPO_DUMP_VERL_COMPAT_FIELDS=${cogrpo_dump_verl_compat_fields} &&
    export COGRPO_PRINT_TRAJECTORY_DEBUG=${cogrpo_print_traj_debug} &&
    export COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES=${cogrpo_print_traj_max_samples} &&
    export COGRPO_DEBUG_HINT_CONTEXT=${cogrpo_debug_hint_context} &&
    export COGRPO_DEBUG_HINT_CONTEXT_TOKENS=${cogrpo_debug_hint_context_tokens} &&
    export COGRPO_VERIFIER_MAX_NEW_TOKENS=${cogrpo_verifier_max_new_tokens} &&
    export COGRPO_VERIFIER_MAX_PROMPT_LENGTH=${cogrpo_verifier_max_prompt_length} &&
    export COGRPO_VERIFIER_MAX_HINT_TOKENS=${cogrpo_verifier_max_hint_tokens} &&
    export COGRPO_ESTIMATED_HINT_TOKENS=${cogrpo_estimated_hint_tokens} &&
    export COGRPO_VERIFIER_MODEL='${cogrpo_verifier_model}' &&
    export COGRPO_VERIFIER_SERVER_URL='${cogrpo_verifier_server_url}' &&
    export COGRPO_VERIFIER_SERVER_URLS='${cogrpo_verifier_server_urls}' &&
    export COGRPO_VERIFIER_SERVER_URL_DICT='${cogrpo_verifier_server_url_dict}' &&
    export COGRPO_WAIT_VERIFIER_READY='${cogrpo_wait_verifier_ready}' &&
    export COGRPO_VERIFIER_WAIT_TIMEOUT_S='${cogrpo_verifier_wait_timeout_s}' &&
    export COGRPO_VERIFIER_WAIT_INTERVAL_S='${cogrpo_verifier_wait_interval_s}' &&
    export COGRPO_VERIFIER_WAIT_CHECK_UPDATE_ENDPOINT='${cogrpo_verifier_wait_check_update}' &&
    export COGRPO_VERIFIER_UPDATE_BASE='${cogrpo_verifier_update_base}' &&
    export COGRPO_STRICT_VERIFIER_SYNC='${cogrpo_strict_verifier_sync}' &&
    export COGRPO_VERIFIER_LORA_ENABLE=${cogrpo_verifier_lora_enable} &&
    export COGRPO_VERIFIER_LORA_PATH='${cogrpo_verifier_lora_path}' &&
    export COGRPO_VERIFIER_LORA_RANK='${cogrpo_verifier_lora_rank}' &&
    export COGRPO_VERIFIER_LORA_ALPHA='${cogrpo_verifier_lora_alpha}' &&
    export COGRPO_VERIFIER_LORA_DROPOUT='${cogrpo_verifier_lora_dropout}' &&
    export COGRPO_VERIFIER_LR='${cogrpo_verifier_lr}' &&
    export COGRPO_VERIFIER_LORA_SYNC_FREQ='${cogrpo_verifier_lora_sync_freq}' &&
    export COGRPO_VERIFIER_LORA_SAVE_FREQ=${cogrpo_verifier_lora_save_freq} &&
    export COGRPO_VERIFIER_DEBUG='${cogrpo_verifier_debug}' &&
    export COGRPO_VERIFIER_DEBUG_MAX_CALLS='${cogrpo_verifier_debug_max_calls}' &&
    export COGRPO_VERIFIER_DEBUG_TEXT_CHARS='${cogrpo_verifier_debug_text_chars}' &&
    export COGRPO_VERIFIER_DEBUG_DUMP='${cogrpo_verifier_debug_dump}' &&
    export COGRPO_VERIFIER_DEBUG_DUMP_MAX_CALLS='${cogrpo_verifier_debug_dump_max_calls}' &&
    export COGRPO_VERIFIER_DEBUG_DUMP_DIR='${cogrpo_verifier_debug_dump_dir}' &&
    export COGRPO_VERIFIER_DEBUG_DUMP_FULL_TEXT='${cogrpo_verifier_debug_dump_full_text}' &&
    export COGRPO_VERIFIER_PARSER_MODE='${cogrpo_verifier_parser_mode}' &&
    export COGRPO_VERL_HINT_INJECTION_PY='${cogrpo_verl_hint_injection_py}' &&
    export COGRPO_VERL_ROLLOUT_SPMD_PY='${cogrpo_verl_rollout_spmd_py}' &&
    export XTUNER_PRETTY_TRAJECTORY_JSONL=${XTUNER_PRETTY_TRAJECTORY_JSONL:-0} &&
    if [ \"\${COGRPO_WAIT_VERIFIER_READY:-0}\" = \"1\" ]; then
      python /mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/wait_verifier_ready.py ;
    fi &&
    bash examples/v1/scripts/run_rl.sh examples/v1/config/rl_qwen25_7B_cogrpo_cf_branch_async.py lmdeploy '${model_path}' '${dataset_path}'
  "
