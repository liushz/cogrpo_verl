#!/bin/bash
set -eo pipefail

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

# =========================
# Basic
# =========================
model_name="/mnt/shared-storage-user/opencompass-shared/liuhongwei/interns1/repro_exp/s1_model/interns1-8b-hf-1951"
dataset_name="passrate_math_merged"
verifier_lora_path="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-verifier_sft_train_55_waitx2-020418/checkpoint-915"

# =========================
# Cluster / run identity
# =========================
cluster="llmit_gpu"
nnodes=2                # 16 GPUs total
n_gpus_per_node=8
gen_tp=1
gpu_memory_utilization=0.8
verl_debug=1

# =========================
# rjob resource requests
# =========================
# NOTE: Allow overriding via env/flags for quick scheduling experiments.
# Default stays at the original value (128).
rjob_cpu="${RJOB_CPU:-128}"
rjob_memory="${RJOB_MEMORY:-1200000}"

preset="full"
co_grpo_mode="full"
resume_dir=""
resume_path=""
work_dir="/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train"

ablation_tag="base"
exp_name=""

# =========================
# Fixed test setup (requested)
# =========================
response_n=16
train_batch_size=32
max_prompt_length=1024

# =========================
# Commonly tuned Co-GRPO knobs
# =========================
verifier_intervention_mode="by_step"
token_check_interval=2048
min_step_tokens=2048
max_interventions=5
confidence_threshold=0.0

verifier_max_hint_tokens=512
estimated_hint_tokens=512
control_group_weight=0.5
use_curriculum_weighting="True"
curriculum_start_weight=0.3
curriculum_end_weight=0.5

# Disable intervention penalty by default (cost-based discouragement).
# For cold-start verifier, this penalty tends to quickly collapse intervention rate
# before we get stable positive cf-branch signal. You can still override via flags:
#   --intervention-penalty-freq-coef / --intervention-penalty-len-coef
intervention_penalty_freq_coef=0.0
intervention_penalty_len_coef=0.0

verifier_reward_mode="headroom"
verifier_reward_improve_coef=1.0
verifier_reward_tie_no_intervention_weight=1.0
verifier_reward_headroom_min=0.05

# Verifier optimization knobs
verifier_lora_rank=64
verifier_lora_alpha=128
verifier_lora_dropout=0.05
verifier_lr=1e-5
verifier_loss_weight=1.0
verifier_max_new_tokens=4096
verifier_logprobs=1
verifier_temperature=1.0

# Batch / trainer knobs
use_dynamic_bsz="False"
micro_bsz_per_gpu=2

trainer_total_epochs=10
trainer_total_training_steps=1000
trainer_save_freq=20
trainer_test_freq=-1
trainer_log_val_generations="False"
trainer_rollout_dump_freq=10
trainer_dual_rollout_dump_freq=1
trainer_control_rollout_sync_freq=5
trainer_verifier_lora_sync_freq=1
trainer_verifier_lora_save_freq=20

while [ "$#" -gt 0 ]; do
    case "$1" in
        --ablation-tag) ablation_tag="$2"; shift 2 ;;
        --exp-name) exp_name="$2"; shift 2 ;;
        --token-check-interval) token_check_interval="$2"; shift 2 ;;
        --min-step-tokens) min_step_tokens="$2"; shift 2 ;;
        --max-interventions) max_interventions="$2"; shift 2 ;;
        --confidence-threshold|--confidence_threshold) confidence_threshold="$2"; shift 2 ;;
        --verifier-max-hint-tokens) verifier_max_hint_tokens="$2"; shift 2 ;;
        --estimated-hint-tokens) estimated_hint_tokens="$2"; shift 2 ;;
        --control-group-weight) control_group_weight="$2"; shift 2 ;;
        --curriculum-start-weight) curriculum_start_weight="$2"; shift 2 ;;
        --curriculum-end-weight) curriculum_end_weight="$2"; shift 2 ;;
        --intervention-penalty-freq-coef) intervention_penalty_freq_coef="$2"; shift 2 ;;
        --intervention-penalty-len-coef) intervention_penalty_len_coef="$2"; shift 2 ;;
        --verifier-reward-mode) verifier_reward_mode="$2"; shift 2 ;;
        --verifier-improve-coef) verifier_reward_improve_coef="$2"; shift 2 ;;
        --verifier-headroom-min) verifier_reward_headroom_min="$2"; shift 2 ;;
        --response-n) response_n="$2"; shift 2 ;;
        --train-batch-size) train_batch_size="$2"; shift 2 ;;
        --rjob-cpu) rjob_cpu="$2"; shift 2 ;;
        --rjob-memory) rjob_memory="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run_co_grpo.sh [options]"
            echo "  --ablation-tag <tag>"
            echo "  --exp-name <name>"
            echo "  --token-check-interval <int>"
            echo "  --min-step-tokens <int>"
            echo "  --max-interventions <int>"
            echo "  --confidence-threshold <float>"
            echo "  --verifier-max-hint-tokens <int>"
            echo "  --estimated-hint-tokens <int>"
            echo "  --control-group-weight <float>"
            echo "  --curriculum-start-weight <float>"
            echo "  --curriculum-end-weight <float>"
            echo "  --intervention-penalty-freq-coef <float>"
            echo "  --intervention-penalty-len-coef <float>"
            echo "  --verifier-reward-mode <headroom|gap>"
            echo "  --verifier-improve-coef <float>"
            echo "  --verifier-headroom-min <float>"
            echo "  --response-n <int>"
            echo "  --train-batch-size <int>"
            echo "  --rjob-cpu <int>"
            echo "  --rjob-memory <int>"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "${exp_name}" ]; then
    exp_name="grpo-verifier-8b-v2-${ablation_tag}_$(date +%m%d%H%M)"
fi

hf_cache_dir="/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub"
rjob_name="rjob_${exp_name//-/_}"
launcher_script="/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/run_multinodes_cgrpo_v2.sh"

echo "[run_co_grpo] name=${rjob_name} nnodes=${nnodes} gpu_per_node=${n_gpus_per_node}"
echo "[run_co_grpo] exp=${exp_name} mode=${co_grpo_mode} preset=${preset}"
echo "[run_co_grpo] response_n=${response_n} train_batch_size=${train_batch_size}"
echo "[run_co_grpo] token_check_interval=${token_check_interval} min_step_tokens=${min_step_tokens} max_interventions=${max_interventions}"
echo "[run_co_grpo] confidence_threshold=${confidence_threshold}"
echo "[run_co_grpo] rjob_cpu=${rjob_cpu} rjob_memory=${rjob_memory}"

# rjob delete "${rjob_name}" || true
rjob submit \
    --name="${rjob_name}" \
    --gpu="${n_gpus_per_node}" \
    --memory="${rjob_memory}" \
    --cpu="${rjob_cpu}" \
    --charged-group="${cluster}" \
    --private-machine=group \
    --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
    --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
    --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
    --image=registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605 \
    -P "${nnodes}" \
    --host-network=true \
    -e DISTRIBUTED_JOB=true \
    --custom-resources rdma/mlnx_shared="${n_gpus_per_node}" \
    -- bash -c "
        export HF_HOME='${hf_cache_dir}' &&
        export HF_HUB_CACHE='${hf_cache_dir}' &&
        export HUGGINGFACE_HUB_CACHE='${hf_cache_dir}' &&
        export HF_DATASETS_CACHE='/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/hf_datasets_cache' &&
        export HF_DATASETS_OFFLINE=1 &&
        export TRANSFORMERS_OFFLINE=1 &&
        export HF_EVALUATE_OFFLINE=1 &&
        export HF_HUB_OFFLINE=1 &&
        chmod +x '${launcher_script}' &&
        '${launcher_script}' \
          '${model_name}' \
          '${nnodes}' \
          '${n_gpus_per_node}' \
          '${gen_tp}' \
          '${gpu_memory_utilization}' \
          '${dataset_name}' \
          '${verl_debug}' \
          '${exp_name}' \
          '${work_dir}' \
          '${preset}' \
          '${co_grpo_mode}' \
          '${resume_dir}' \
          '${resume_path}' \
          '${verifier_lora_path}' \
          '${response_n}' \
          '${train_batch_size}' \
          '${token_check_interval}' \
          '${min_step_tokens}' \
          '${max_interventions}' \
          '${verifier_max_hint_tokens}' \
          '${estimated_hint_tokens}' \
          '${control_group_weight}' \
          '${use_curriculum_weighting}' \
          '${curriculum_start_weight}' \
          '${curriculum_end_weight}' \
          '${intervention_penalty_freq_coef}' \
          '${intervention_penalty_len_coef}' \
          '${verifier_reward_mode}' \
          '${verifier_reward_improve_coef}' \
          '${verifier_reward_headroom_min}' \
          '${trainer_rollout_dump_freq}' \
          '${trainer_dual_rollout_dump_freq}' \
          '${use_dynamic_bsz}' \
          '${micro_bsz_per_gpu}' \
          '${verifier_reward_tie_no_intervention_weight}' \
          '${verifier_intervention_mode}' \
          '${confidence_threshold}' \
          '${verifier_lora_rank}' \
          '${verifier_lora_alpha}' \
          '${verifier_lora_dropout}' \
          '${verifier_lr}' \
          '${verifier_loss_weight}' \
          '${verifier_max_new_tokens}' \
          '${verifier_logprobs}' \
          '${verifier_temperature}' \
          '${trainer_total_epochs}' \
          '${trainer_total_training_steps}' \
          '${trainer_save_freq}' \
          '${trainer_test_freq}' \
          '${trainer_log_val_generations}' \
          '${trainer_control_rollout_sync_freq}' \
          '${trainer_verifier_lora_sync_freq}' \
          '${trainer_verifier_lora_save_freq}' \
          '${max_prompt_length}'
    "
