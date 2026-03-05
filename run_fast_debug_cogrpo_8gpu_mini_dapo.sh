#!/usr/bin/env bash
set -eo pipefail

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

# Fast single-node 8-GPU debug runner for CoGRPO (+ Verifier by_step).
# - Builds a tiny parquet subset from the huge dapo dataset (no 1.79M-row scan)
# - Runs 1-2 training steps to quickly surface:
#   - Verifier "<GO>/<WAIT>" no-decision issues
#   - Verifier LoRA load / base-only behavior
#   - vLLM weight-sync / FSDP->vLLM key-mismatch issues
#
# Usage (defaults are safe + fast):
#   bash repos/repro/run_fast_debug_cogrpo_8gpu_mini_dapo.sh
#
# To align closer to online long-context settings, override (slower):
#   bash repos/repro/run_fast_debug_cogrpo_8gpu_mini_dapo.sh --response-n 32 --max-prompt-length 2048

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"
verifier_lora_path="${VERIFIER_LORA_PATH:-}"
work_dir="${WORK_DIR:-/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train}"

# Allow running on fewer GPUs for quicker scheduling/debugging (default: 8).
n_gpus_per_node="${N_GPUS_PER_NODE:-8}"
if ! [[ "${n_gpus_per_node}" =~ ^[0-9]+$ ]] || [ "${n_gpus_per_node}" -lt 1 ]; then
  echo "[fastdbg][ERR] N_GPUS_PER_NODE must be a positive integer (got ${n_gpus_per_node})" >&2
  exit 2
fi

# Huge training parquet (1.79M rows repeated). We'll take only a tiny head().
src_parquet="${SRC_PARQUET:-/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-dapo-math-17k.parquet}"
mini_parquet="${MINI_PARQUET:-data/debug_dapo_mini64.parquet}"
mini_rows="${MINI_ROWS:-64}"

# Keep defaults fast; override if you want exact online budgets.
response_n="${RESPONSE_N:-4}"               # max_response_length = 4k tokens
# dp_size=8 on single node; keep train_bsz * rollout_n == 8 by default (fast).
train_batch_size="${TRAIN_BSZ:-4}"         # must satisfy: train_bsz * rollout_n % dp_size == 0
micro_bsz_per_gpu="${MICRO_BSZ:-1}"
rollout_n="${ROLLOUT_N:-2}"
gpu_memory_utilization="${GPU_MEM_UTIL:-0.85}"
trainer_total_training_steps="${STEPS:-1}"

# Verifier by_step knobs
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
token_check_interval="${TOKEN_CHECK_INTERVAL:-512}"
min_step_tokens="${MIN_STEP_TOKENS:-512}"
max_interventions="${MAX_INTERVENTIONS:-2}"
verifier_max_hint_tokens="${VERIFIER_MAX_HINT_TOKENS:-512}"
estimated_hint_tokens="${ESTIMATED_HINT_TOKENS:-512}"
verifier_max_new_tokens="${VERIFIER_MAX_NEW_TOKENS:-2048}"
confidence_threshold="${CONFIDENCE_THRESHOLD:-0.0}"

# Verifier training knobs (match online defaults unless overridden).
verifier_lora_rank="${VERIFIER_LORA_RANK:-64}"
verifier_lora_alpha="${VERIFIER_LORA_ALPHA:-128}"
verifier_lora_dropout="${VERIFIER_LORA_DROPOUT:-0.05}"
verifier_lr="${VERIFIER_LR:-1e-5}"
verifier_loss_weight="${VERIFIER_LOSS_WEIGHT:-1.0}"

# Dumping is expensive; keep off for debug.
trainer_rollout_dump_freq="${ROLLOUT_DUMP_FREQ:-0}"
trainer_dual_rollout_dump_freq="${DUAL_ROLLOUT_DUMP_FREQ:-0}"

# Credit assignment (default: global_gap; enable cf_branch with --cf-branch).
verifier_credit_assignment="${VERIFIER_CREDIT_ASSIGNMENT:-global_gap}"
cf_branch_prob="${CF_BRANCH_PROB:-1.0}"
cf_branch_max_events_per_sample="${CF_BRANCH_MAX_EVENTS:-0}"
cf_branch_state_hash_mod="${CF_BRANCH_HASH_MOD:-1024}"
cf_branch_k="${CF_BRANCH_K:-2}"

exp_name="${EXP_NAME:-localdbg-cogrpo-$(date +%m%d%H%M%S)}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-path) model_path="$2"; shift 2 ;;
    --verifier-lora-path) verifier_lora_path="$2"; shift 2 ;;
    --no-verifier-lora) verifier_lora_path=""; shift 1 ;;
    --work-dir) work_dir="$2"; shift 2 ;;
    --src-parquet) src_parquet="$2"; shift 2 ;;
    --mini-parquet) mini_parquet="$2"; shift 2 ;;
    --mini-rows) mini_rows="$2"; shift 2 ;;
    --response-n) response_n="$2"; shift 2 ;;
    --train-bsz) train_batch_size="$2"; shift 2 ;;
    --micro-bsz) micro_bsz_per_gpu="$2"; shift 2 ;;
    --rollout-n) rollout_n="$2"; shift 2 ;;
    --steps) trainer_total_training_steps="$2"; shift 2 ;;
    --gpu-mem) gpu_memory_utilization="$2"; shift 2 ;;
    --max-prompt-length) max_prompt_length="$2"; shift 2 ;;
    --token-check-interval) token_check_interval="$2"; shift 2 ;;
    --min-step-tokens) min_step_tokens="$2"; shift 2 ;;
    --max-interventions) max_interventions="$2"; shift 2 ;;
    --verifier-max-hint-tokens) verifier_max_hint_tokens="$2"; shift 2 ;;
    --estimated-hint-tokens) estimated_hint_tokens="$2"; shift 2 ;;
    --verifier-max-new-tokens) verifier_max_new_tokens="$2"; shift 2 ;;
    --confidence-threshold) confidence_threshold="$2"; shift 2 ;;
    --dump-freq) trainer_rollout_dump_freq="$2"; shift 2 ;;
    --dual-dump-freq) trainer_dual_rollout_dump_freq="$2"; shift 2 ;;
    --verifier-credit-assignment) verifier_credit_assignment="$2"; shift 2 ;;
    --cf-branch)
      verifier_credit_assignment="cf_branch"
      shift 1
      ;;
    --cf-branch-k) cf_branch_k="$2"; shift 2 ;;
    --cf-branch-prob) cf_branch_prob="$2"; shift 2 ;;
    --cf-branch-max-events) cf_branch_max_events_per_sample="$2"; shift 2 ;;
    --cf-branch-hash-mod) cf_branch_state_hash_mod="$2"; shift 2 ;;
    --exp-name) exp_name="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: bash repos/repro/run_fast_debug_cogrpo_8gpu_mini_dapo.sh [options]
  --model-path <path>
  --verifier-lora-path <path> | --no-verifier-lora
  --src-parquet <path>         (huge parquet; default dapo17k repeated)
  --mini-parquet <path>        (output tiny parquet; default: data/debug_dapo_mini64.parquet)
  --mini-rows <int>            (default: 64)
  --response-n <int>           (default: 4 => 4k max response tokens)
  --train-bsz <int>            (default: 4)
  --rollout-n <int>            (default: 2)
  --steps <int>                (default: 1)
  --cf-branch                  (enable counterfactual credit assignment)
  --cf-branch-k <int>          (default: 2)
  --max-interventions <int>    (default: 2)
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if ! [[ "${mini_rows}" =~ ^[0-9]+$ ]] || [ "${mini_rows}" -le 0 ]; then
  echo "[fastdbg][ERR] mini_rows must be a positive int (got ${mini_rows})" >&2
  exit 2
fi

# Ensure mini dataset isn't smaller than train_batch_size (StatefulDataLoader drop_last=True).
if [[ "${train_batch_size}" =~ ^[0-9]+$ ]] && [ "${mini_rows}" -lt "${train_batch_size}" ]; then
  echo "[fastdbg][WARN] mini_rows(${mini_rows}) < train_batch_size(${train_batch_size}); bumping mini_rows to ${train_batch_size}." >&2
  mini_rows="${train_batch_size}"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  # Default to a dense [0..n_gpus_per_node-1] list.
  if [ "${n_gpus_per_node}" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES="0"
  else
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((n_gpus_per_node - 1)))"
  fi
fi
export NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_PORT="${RAY_PORT:-8266}"

# Make verifier output parsing visible.
export VERL_VERIFIER_DEBUG_LOG_OVERRIDE="${VERL_VERIFIER_DEBUG_LOG_OVERRIDE:-1}"

# cf-branch sampling count is read from env (see launcher).
export CF_BRANCH_K="${cf_branch_k}"

# HuggingFace cache dirs:
# Some clusters mount shared HF caches as read-only; `datasets` needs a writable
# lock/cache directory even for local parquet.
if [ -d "/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub" ]; then
  export HF_HOME="${HF_HOME:-/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
fi
default_hf_datasets_cache="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/hf_datasets_cache"
hf_datasets_cache="${HF_DATASETS_CACHE:-${default_hf_datasets_cache}}"
mkdir -p "${hf_datasets_cache}" 2>/dev/null || true
if ! touch "${hf_datasets_cache}/.writable_check" 2>/dev/null; then
  echo "[fastdbg][WARN] HF_DATASETS_CACHE is not writable: ${hf_datasets_cache} -> fallback to ${default_hf_datasets_cache}" >&2
  hf_datasets_cache="${default_hf_datasets_cache}"
  mkdir -p "${hf_datasets_cache}" 2>/dev/null || true
fi
rm -f "${hf_datasets_cache}/.writable_check" 2>/dev/null || true
export HF_DATASETS_CACHE="${hf_datasets_cache}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro

python scripts/make_parquet_mini.py --src "${src_parquet}" --out "${mini_parquet}" --n "${mini_rows}"

echo "[fastdbg] exp_name=${exp_name}"
echo "[fastdbg] model_path=${model_path}"
echo "[fastdbg] verifier_lora_path=${verifier_lora_path:-<empty>}"
echo "[fastdbg] mini_parquet=${mini_parquet} (rows=${mini_rows})"
echo "[fastdbg] response_n=${response_n} train_bsz=${train_batch_size} rollout_n=${rollout_n} steps=${trainer_total_training_steps}"
echo "[fastdbg] max_prompt_length=${max_prompt_length} token_check_interval=${token_check_interval} min_step_tokens=${min_step_tokens} max_interventions=${max_interventions} verifier_max_new_tokens=${verifier_max_new_tokens}"
echo "[fastdbg] verifier_credit_assignment=${verifier_credit_assignment} cf_branch_k=${CF_BRANCH_K}"
echo "[fastdbg] verifier_lora_rank=${verifier_lora_rank} verifier_lora_alpha=${verifier_lora_alpha} verifier_lora_dropout=${verifier_lora_dropout} verifier_lr=${verifier_lr} verifier_loss_weight=${verifier_loss_weight}"

ray stop -f || true

PARALLEL_CONTROL_EXP="False" \
ROLLOUT_N="${rollout_n}" \
  bash scripts/run_multinodes_cgrpo_v2.sh \
  "${model_path}" \
  1 "${n_gpus_per_node}" 1 "${gpu_memory_utilization}" \
  "${mini_parquet}" \
  1 \
  "${exp_name}" \
  "${work_dir}" \
  full full "" "" \
  "${verifier_lora_path}" \
  "${response_n}" \
  "${train_batch_size}" \
  "${token_check_interval}" "${min_step_tokens}" "${max_interventions}" "${verifier_max_hint_tokens}" "${estimated_hint_tokens}" \
  0.5 True 0.3 0.5 \
  0.0 0.0 \
  headroom 1.0 0.05 \
  "${trainer_rollout_dump_freq}" "${trainer_dual_rollout_dump_freq}" \
  False "${micro_bsz_per_gpu}" \
  1.0 \
  by_step "${confidence_threshold}" \
  "${verifier_lora_rank}" "${verifier_lora_alpha}" "${verifier_lora_dropout}" "${verifier_lr}" "${verifier_loss_weight}" \
  "${verifier_max_new_tokens}" \
  1 1.0 \
  1 "${trainer_total_training_steps}" \
  999999 -1 False \
  999999 1 999999 \
  "${max_prompt_length}" \
  "${verifier_credit_assignment}" "${cf_branch_prob}" "${cf_branch_max_events_per_sample}" "${cf_branch_state_hash_mod}"
