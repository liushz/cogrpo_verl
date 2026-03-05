#!/usr/bin/env bash
set -eo pipefail

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

# 8-GPU devbox debug launcher (no rjob)
# - Keeps 16k rollout response length (response_n=16)
# - Verifier input budget is fixed to 16k inside launcher (verifier.max_prompt_length=16384)
# - Verifier output budget defaults to 4k (verifier.max_new_tokens=4096)
# - Hint is capped at 512 tokens (verifier.max_hint_tokens=512)
# - Reduce batch sizes to avoid OOM

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

model_path="/mnt/shared-storage-user/opencompass-shared/liuhongwei/interns1/repro_exp/s1_model/interns1-8b-hf-1951"
dataset_name="passrate_math_merged"
verifier_lora_path="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-verifier_sft_train_55_waitx2-020418/checkpoint-915"

work_dir="/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train"

response_n=32
train_batch_size=32
micro_bsz_per_gpu=2
gpu_memory_utilization=0.8
trainer_total_training_steps=200

max_prompt_length=1024

rollout_n=8
token_check_interval=2048
min_step_tokens=2048
max_interventions=5
verifier_max_hint_tokens=512
estimated_hint_tokens=512

verifier_max_new_tokens=4096
confidence_threshold=0.0

trainer_rollout_dump_freq=10
trainer_dual_rollout_dump_freq=2

parallel_control_exp="False"
# control_rollout_gpus_per_node="2"
# exp_rollout_gpus_per_node="6"

# Default: do not override launcher. In by_step mode the launcher defaults this to False.
rollout_enable_kv_cache_optimization="" # set to True/False to override launcher default
rollout_max_num_batched_tokens="" # empty => use launcher default
rollout_max_num_seqs="512" # vLLM max_num_seqs (default in ppo_trainer.yaml)
exp_name=""

# Credit assignment
verifier_credit_assignment="global_gap"
cf_branch_prob="1.0"
cf_branch_max_events_per_sample="0"
cf_branch_state_hash_mod="1024"
cf_branch_k="1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --exp-name) exp_name="$2"; shift 2 ;;
    --train-bsz) train_batch_size="$2"; shift 2 ;;
    --micro-bsz) micro_bsz_per_gpu="$2"; shift 2 ;;
    --gpu-mem) gpu_memory_utilization="$2"; shift 2 ;;
    --steps) trainer_total_training_steps="$2"; shift 2 ;;
    --verifier-max-new-tokens) verifier_max_new_tokens="$2"; shift 2 ;;
    --max-prompt-length) max_prompt_length="$2"; shift 2 ;;
    --response-n) response_n="$2"; shift 2 ;;
    --rollout-n) rollout_n="$2"; shift 2 ;;
    --confidence-threshold) confidence_threshold="$2"; shift 2 ;;
    --token-check-interval) token_check_interval="$2"; shift 2 ;;
    --min-step-tokens) min_step_tokens="$2"; shift 2 ;;
    --max-interventions) max_interventions="$2"; shift 2 ;;
    --verifier-max-hint-tokens) verifier_max_hint_tokens="$2"; shift 2 ;;
    --estimated-hint-tokens) estimated_hint_tokens="$2"; shift 2 ;;
    --dump-freq) trainer_rollout_dump_freq="$2"; shift 2 ;;
    --dual-dump-freq) trainer_dual_rollout_dump_freq="$2"; shift 2 ;;
    --parallel-control-exp) parallel_control_exp="True"; shift 1 ;;
    --no-parallel-control-exp) parallel_control_exp="False"; shift 1 ;;
    --control-gpus) control_rollout_gpus_per_node="$2"; shift 2 ;;
    --exp-gpus) exp_rollout_gpus_per_node="$2"; shift 2 ;;
    --prefix-cache) rollout_enable_kv_cache_optimization="True"; shift 1 ;;
    --no-prefix-cache) rollout_enable_kv_cache_optimization="False"; shift 1 ;;
    --max-num-batched-tokens) rollout_max_num_batched_tokens="$2"; shift 2 ;;
    --max-num-seqs) rollout_max_num_seqs="$2"; shift 2 ;;
    --cf-branch) verifier_credit_assignment="cf_branch"; parallel_control_exp="False"; shift 1 ;;
    --verifier-credit-assignment) verifier_credit_assignment="$2"; shift 2 ;;
    --cf-branch-prob) cf_branch_prob="$2"; shift 2 ;;
    --cf-branch-max-events) cf_branch_max_events_per_sample="$2"; shift 2 ;;
    --cf-branch-hash-mod) cf_branch_state_hash_mod="$2"; shift 2 ;;
    --cf-branch-k) cf_branch_k="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash run_dev_debug_cogrpo_8gpu.sh [options]"
      echo "  --exp-name <name>"
      echo "  --train-bsz <int>               (default: 8)"
      echo "  --micro-bsz <int>               (default: 2)"
      echo "  --gpu-mem <float>               (default: 0.8)"
      echo "  --steps <int>                   (default: 200)"
      echo "  --response-n <int>              (default: 16 => 16k)"
      echo "  --rollout-n <int>               (default: 8)"
      echo "  --confidence-threshold <float>  (default: 0.0; 0=off)"
      echo "  --cf-branch                     (enable cf-branch credit; EXP-only; disables parallel control/exp)"
      echo "  --verifier-credit-assignment <global_gap|cf_branch>"
      echo "  --cf-branch-prob <float>         (default: 1.0)"
      echo "  --cf-branch-max-events <int>     (default: 0 => auto=max_interventions in trainer)"
      echo "  --cf-branch-hash-mod <int>       (default: 1024)"
      echo "  --cf-branch-k <int>             (default: 1; per-event counterfactual samples)"
      echo "  --verifier-max-new-tokens <int> (default: 4096)"
      echo "  --token-check-interval <int>    (default: 2048)"
      echo "  --min-step-tokens <int>         (default: 2048)"
      echo "  --max-interventions <int>       (default: 5)"
      echo "  --verifier-max-hint-tokens <int> (default: 512)"
      echo "  --estimated-hint-tokens <int>   (default: 512)"
      echo "  --dump-freq <int>               (default: 10; set large to reduce IO)"
      echo "  --dual-dump-freq <int>          (default: 5; set large to reduce IO)"
      echo "  --parallel-control-exp          (default)"
      echo "  --no-parallel-control-exp"
      echo "  --control-gpus <int>            (only when parallel enabled; default: half)"
      echo "  --exp-gpus <int>                (only when parallel enabled; default: remaining)"
      echo "  --max-prompt-length <int>       (default: 1024)"
      echo "  --prefix-cache                  (force enable vLLM prefix caching; even in by_step)"
      echo "  --no-prefix-cache               (force disable vLLM prefix caching)"
      echo "  --max-num-batched-tokens <int>  (override vLLM max_num_batched_tokens; helps concurrency)"
      echo "  --max-num-seqs <int>            (override vLLM max_num_seqs; helps concurrency)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "${exp_name}" ]; then
  exp_name="devdbg-base-$(date +%m%d%H%M)"
fi

verifier_credit_assignment_norm="$(echo "${verifier_credit_assignment}" | tr '[:upper:]' '[:lower:]')"
if [ "${verifier_credit_assignment_norm}" = "cf_branch" ] || [ "${verifier_credit_assignment_norm}" = "cf" ]; then
  parallel_control_exp="False"
fi

if ! [[ "${train_batch_size}" =~ ^[0-9]+$ ]] || ! [[ "${micro_bsz_per_gpu}" =~ ^[0-9]+$ ]]; then
  echo "[devdbg][ERR] --train-bsz and --micro-bsz must be integers (got train_batch_size=${train_batch_size}, micro_bsz_per_gpu=${micro_bsz_per_gpu})" >&2
  exit 2
fi
if [ "${micro_bsz_per_gpu}" -lt 1 ]; then
  echo "[devdbg][ERR] --micro-bsz must be >=1 (got ${micro_bsz_per_gpu})" >&2
  exit 2
fi
if [ "$((train_batch_size % micro_bsz_per_gpu))" -ne 0 ]; then
  echo "[devdbg][ERR] train_batch_size (${train_batch_size}) must be divisible by micro_bsz_per_gpu (${micro_bsz_per_gpu}) to satisfy FSDP normalization asserts." >&2
  echo "[devdbg][HINT] Try: --train-bsz $((train_batch_size / micro_bsz_per_gpu * micro_bsz_per_gpu))  (nearest lower divisible) or set --micro-bsz 1." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_PORT="${RAY_PORT:-8266}"

# HF offline/cache (align with rjob launcher)
export HF_HOME="/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1

# Reward endpoints (optional)
export REWARD_MODEL_URLS="${REWARD_MODEL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"
export REWARD_MODEL_KEY="${REWARD_MODEL_KEY:-EMPTY}"

# Bypass any cluster HTTP proxy for internal reward endpoints.
reward_proxy_hosts="$(echo "${REWARD_MODEL_URLS}" | tr ',' '\n' | sed -E 's#^[a-zA-Z]+://##; s#/.*$##; s#:.*$##' | paste -sd',' -)"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${reward_proxy_hosts},localhost,127.0.0.1"
export no_proxy="${NO_PROXY}"

source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro
export PYTHONPATH="$(pwd)"

ray stop -f || true

control_gpus_log="${control_rollout_gpus_per_node:-<auto>}"
exp_gpus_log="${exp_rollout_gpus_per_node:-<auto>}"
if [ "${parallel_control_exp}" != "True" ]; then
  control_gpus_log="(ignored)"
  exp_gpus_log="all"
fi

echo "[devdbg] exp_name=${exp_name} response_n=${response_n} rollout_n=${rollout_n} train_batch_size=${train_batch_size} micro_bsz_per_gpu=${micro_bsz_per_gpu} verifier_max_new_tokens=${verifier_max_new_tokens} confidence_threshold=${confidence_threshold} token_check_interval=${token_check_interval} min_step_tokens=${min_step_tokens} max_interventions=${max_interventions} dump_freq=${trainer_rollout_dump_freq} dual_dump_freq=${trainer_dual_rollout_dump_freq} verifier_credit_assignment=${verifier_credit_assignment} cf_prob=${cf_branch_prob} cf_max_events=${cf_branch_max_events_per_sample} cf_hash_mod=${cf_branch_state_hash_mod} cf_k=${cf_branch_k} parallel_control_exp=${parallel_control_exp} control_gpus=${control_gpus_log} exp_gpus=${exp_gpus_log} prefix_cache=${rollout_enable_kv_cache_optimization:-<auto>}"

# Positional interface (kept consistent with scripts/run_multinodes_cgrpo_v2.sh)
PARALLEL_CONTROL_EXP="${parallel_control_exp}" \
CONTROL_ROLLOUT_GPUS_PER_NODE="${control_rollout_gpus_per_node}" \
EXP_ROLLOUT_GPUS_PER_NODE="${exp_rollout_gpus_per_node}" \
ROLLOUT_N="${rollout_n}" \
CF_BRANCH_K="${cf_branch_k}" \
ROLLOUT_ENABLE_KV_CACHE_OPTIMIZATION="${rollout_enable_kv_cache_optimization}" \
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${rollout_max_num_batched_tokens}" \
ROLLOUT_MAX_NUM_SEQS="${rollout_max_num_seqs}" \
  bash scripts/run_multinodes_cgrpo_v2.sh \
  "${model_path}" \
  1 8 1 "${gpu_memory_utilization}" \
  "${dataset_name}" \
  1 \
  "${exp_name}" \
  "${work_dir}" \
  full full "" "" \
  "${verifier_lora_path}" \
  "${response_n}" \
  "${train_batch_size}" \
  "${token_check_interval}" "${min_step_tokens}" "${max_interventions}" "${verifier_max_hint_tokens}" "${estimated_hint_tokens}" \
  0.5 True 0.3 0.5 \
  0.01 0.0 \
  headroom 1.0 0.05 \
  "${trainer_rollout_dump_freq}" "${trainer_dual_rollout_dump_freq}" \
  False "${micro_bsz_per_gpu}" \
  1.0 \
  by_step "${confidence_threshold}" \
  64 128 0.05 1e-5 1.0 \
  "${verifier_max_new_tokens}" \
  1 1.0 \
  1 "${trainer_total_training_steps}" \
  20 -1 False \
  5 1 20 \
  "${max_prompt_length}" \
  "${verifier_credit_assignment}" "${cf_branch_prob}" "${cf_branch_max_events_per_sample}" "${cf_branch_state_hash_mod}"
