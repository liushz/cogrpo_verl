#!/usr/bin/env bash
set -euo pipefail

# Convert old "pretty" trajectory dumps to rollout_w_answer jsonl for verifier cold-start.
#
# Example:
#   bash scripts/verifier_data_gen/run_step0_convert_trajectory_rollouts.sh \
#     /mnt/shared-storage-gpfs2/llmit1/user/lvchengqi/ckpt/xtuner_v1/interns1_1_delivery \
#     ./outputs/verifier_coldstart_from_trajectory \
#     math
#
# Output:
#   <OUTPUT_DIR>/rollout_w_answer/converted_from_trajectory.jsonl

cd "$(dirname "$0")/../.."

SOURCE_DIR="${1:?SOURCE_DIR required}"
OUTPUT_DIR="${2:?OUTPUT_DIR required}"
DATA_SOURCE="${3:-math}"

python3 scripts/verifier_data_gen/step0_convert_trajectory_rollouts.py \
  --source_dir "${SOURCE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --data_source "${DATA_SOURCE}" \
  --skip_rollout_subdir \
  --dedup_per_action \
  --require_finish_reason_stop \
  --min_output_chars 20

echo
echo "[OK] Converted rollouts written under: ${OUTPUT_DIR}/rollout_w_answer"

