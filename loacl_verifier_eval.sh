#!/usr/bin/env bash
set -eEuo pipefail

# 64k LMDeploy backend, mode=both, repeat-aware CompassVerifier tail-only eval.
# NOTE: Requires cluster quota; if quota is exceeded, --watch will exit early with a fatal message.

JSONL="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/data/test/aime2025-I.jsonl"
PROMPT_KEY="question"

# Align with old OpenCompass config: use the policy actor checkpoint.
BASE_MODEL="/mnt/shared-storage-user/opencompass-shared/liuhongwei/interns1/repro_exp/s1_model/interns1-8b-hf-1951"

EVAL_URLS="http://100.101.166.1:22005/v1,http://100.101.166.1:22004/v1,http://100.101.166.1:22003/v1,http://100.101.166.1:22002/v1"

LMDEPLOY_IMAGE="registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605"
LMDEPLOY_API_EXTRA_CLI="--backend pytorch --session-len 65536 --max-batch-size 128"

COMMON_ARGS=(
  --jsonl "${JSONL}"
  --prompt-key "${PROMPT_KEY}"
  --base-model "${BASE_MODEL}"
  --shards 4
  --repeat 4
  --mode both
  --infer-engine lmdeploy
  --align-opencompass-64k
  --degeneration-guard
  --lmdeploy-use-api-server
  --lmdeploy-api-extra-cli "${LMDEPLOY_API_EXTRA_CLI}"
  --image "${LMDEPLOY_IMAGE}"
  --conda-env repro
  --private-machine group
  --eval
  --eval-urls "${EVAL_URLS}"
  --eval-workers 32    
  --eval-batch-size 64
  --batch-size 2
)

WATCH_ARGS=(--watch --watch-interval 60)

# Verifier LoRAs:
VERIFIER_55="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-verifier_sft_train_55_waitx2-020418/checkpoint-915"
VERIFIER_37="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-verifier_sft_train_37_waitx2-020511/checkpoint-1521"

bash repos/repro/run_rjob_eval_cogrpo_verifier.sh \
  --out-dir /mnt/shared-storage-user/liuhongwei/main_works/temp_debug/eval/aime2025-I_lmdeploy64k_rjob/55_waitx2 \
  --verifier-lora "${VERIFIER_55}" \
  "${COMMON_ARGS[@]}" \
  "${WATCH_ARGS[@]}" &
pid_55=$!

bash repos/repro/run_rjob_eval_cogrpo_verifier.sh \
  --out-dir /mnt/shared-storage-user/liuhongwei/main_works/temp_debug/eval/aime2025-I_lmdeploy64k_rjob/37_waitx2 \
  --verifier-lora "${VERIFIER_37}" \
  "${COMMON_ARGS[@]}" \
  "${WATCH_ARGS[@]}" &
pid_37=$!

wait "${pid_55}"
wait "${pid_37}"
