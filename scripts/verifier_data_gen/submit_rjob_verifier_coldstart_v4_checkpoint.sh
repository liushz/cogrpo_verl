#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Submit rjob(s) for Verifier cold-start data pipeline v4 (checkpoint-only).

Defaults:
  - Oracle: last 8 gpt-oss-120b URLs from ~/main_works/scripts/server.log
  - Actor:  all hf-170 URLs from server.log (filters unreachable)
  - Reward: last 4 cv_32b URLs from server.log
  - Smoke input: rollout_w_answer_cap2_sample_bad5 (few incorrect cases)
  - Full  input: rollout_w_answer_cap2 (≈4.7w lines)

Usage:
  bash scripts/verifier_data_gen/submit_rjob_verifier_coldstart_v4_checkpoint.sh --mode smoke
  bash scripts/verifier_data_gen/submit_rjob_verifier_coldstart_v4_checkpoint.sh --mode full
  bash scripts/verifier_data_gen/submit_rjob_verifier_coldstart_v4_checkpoint.sh --mode both

Options:
  --mode {smoke|full|both}   (default: smoke)
  --resume-full OUT_DIR      Resume an existing full run (needs step1/1.5 already done in OUT_DIR)
  --server-log PATH          (default: ~/main_works/scripts/server.log)
  --dry-run                  Print rjob command(s) only

Env overrides (optional):
  CLUSTER, IMAGE, PRIVATE_MACHINE
  CPU_SMOKE, MEM_SMOKE_MB
  CPU_FULL,  MEM_FULL_MB
  PIPELINE_SH, OUTPUT_ROOT
  SMOKE_ROLLOUT_DIR, FULL_ROLLOUT_DIR, TOKENIZER_PATH
EOF
}

mode="smoke"
server_log="/mnt/shared-storage-user/liuhongwei/main_works/scripts/server.log"
dry_run=0
resume_full_out_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --mode) mode="$2"; shift 2 ;;
    --resume-full) resume_full_out_dir="$2"; shift 2 ;;
    --server-log) server_log="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

PIPELINE_SH="${PIPELINE_SH:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/scripts/lora_cold_data/run_verifier_data_pipeline_v4_checkpoint.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/scripts/lora_cold_data/outputs}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"

SMOKE_ROLLOUT_DIR="${SMOKE_ROLLOUT_DIR:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/scripts/lora_cold_data/raw_rollout_pool_math_20260216/rollout_w_answer_cap2_sample_bad5}"
FULL_ROLLOUT_DIR="${FULL_ROLLOUT_DIR:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/scripts/lora_cold_data/raw_rollout_pool_math_20260216/rollout_w_answer_cap2}"

CLUSTER="${CLUSTER:-llmit_gpu}"
IMAGE="${IMAGE:-registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605}"
PRIVATE_MACHINE="${PRIVATE_MACHINE:-group}"

CPU_SMOKE="${CPU_SMOKE:-8}"
MEM_SMOKE_MB="${MEM_SMOKE_MB:-32768}"
CPU_FULL="${CPU_FULL:-64}"
MEM_FULL_MB="${MEM_FULL_MB:-262144}"

if [[ ! -f "${server_log}" ]]; then
  echo "❌ server.log not found: ${server_log}" >&2
  exit 1
fi
if [[ ! -x "${PIPELINE_SH}" ]]; then
  echo "❌ pipeline script not executable: ${PIPELINE_SH}" >&2
  exit 1
fi

_uniq_preserve_order() {
  awk '!seen[$0]++'
}

_filter_pingable_urls() {
  local urls="$1"
  local ok=()
  local u
  for u in ${urls}; do
    local code
    code="$(curl -sS --connect-timeout 1 --max-time 2 -o /dev/null -w '%{http_code}' "${u}/models" || echo "000")"
    if [[ "${code}" == "200" ]]; then
      ok+=("${u}")
    else
      echo "WARN: drop unreachable url (HTTP ${code}): ${u}" >&2
    fi
  done
  printf "%s\n" "${ok[@]}"
}

_to_csv() {
  paste -sd, -
}

_url_host() {
  sed -E 's#^https?://([^/:]+).*#\1#'
}

echo "=== Extract model URLs from: ${server_log}"
oracle_urls_raw="$(awk '/gpt-oss-120b/ {print prev} {prev=$0}' "${server_log}" | tail -n 8 | _uniq_preserve_order)"
verifier_urls_raw="$(awk '/cv_32b/ {print prev} {prev=$0}' "${server_log}" | tail -n 4 | _uniq_preserve_order)"
actor_urls_raw="$(awk '/cispo-cold-start-model\/hf-170/ {print prev} {prev=$0}' "${server_log}" | _uniq_preserve_order)"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy || true
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="${NO_PROXY}"

echo "=== Filter pingable URLs (need NO_PROXY)"
oracle_urls_ok="$(_filter_pingable_urls "${oracle_urls_raw}")"
verifier_urls_ok="$(_filter_pingable_urls "${verifier_urls_raw}")"
actor_urls_ok="$(_filter_pingable_urls "${actor_urls_raw}")"

if [[ -z "${oracle_urls_ok}" ]]; then
  echo "❌ No reachable oracle URLs found." >&2
  exit 1
fi
if [[ -z "${verifier_urls_ok}" ]]; then
  echo "❌ No reachable verifier URLs found." >&2
  exit 1
fi
if [[ -z "${actor_urls_ok}" ]]; then
  echo "❌ No reachable actor URLs found." >&2
  exit 1
fi

oracle_urls_csv="$(printf "%s\n" ${oracle_urls_ok} | _to_csv)"
verifier_urls_csv="$(printf "%s\n" ${verifier_urls_ok} | _to_csv)"
actor_urls_csv="$(printf "%s\n" ${actor_urls_ok} | _to_csv)"

no_proxy_hosts="$(
  {
    printf "%s\n" ${oracle_urls_ok}
    printf "%s\n" ${verifier_urls_ok}
    printf "%s\n" ${actor_urls_ok}
  } | _url_host | sort -u | _to_csv
)"
export NO_PROXY="${no_proxy_hosts},localhost,127.0.0.1"
export no_proxy="${NO_PROXY}"

first_actor_url="$(printf "%s\n" ${actor_urls_ok} | head -n 1)"
first_verifier_url="$(printf "%s\n" ${verifier_urls_ok} | head -n 1)"

actor_model_id="$(
  curl -sS --connect-timeout 2 --max-time 4 "${first_actor_url}/models" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id","").strip())'
)"
reward_model_id="$(
  curl -sS --connect-timeout 2 --max-time 4 "${first_verifier_url}/models" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id","").strip())'
)"

if [[ -z "${actor_model_id}" ]]; then
  echo "❌ Failed to detect actor model id from: ${first_actor_url}/models" >&2
  exit 1
fi
if [[ -z "${reward_model_id}" ]]; then
  echo "❌ Failed to detect reward model id from: ${first_verifier_url}/models" >&2
  exit 1
fi

echo "Oracle URLs:   ${oracle_urls_csv}"
echo "Actor URLs:    ${actor_urls_csv}"
echo "Verifier URLs: ${verifier_urls_csv}"
echo "Actor model:   ${actor_model_id}"
echo "Reward model:  ${reward_model_id}"
echo "NO_PROXY:      ${NO_PROXY}"

_submit_one() {
  local name="$1"
  local cpu="$2"
  local mem_mb="$3"
  local rollout_dir="$4"
  local out_dir="$5"
  local extra_env_block="$6"

  local folder
  folder="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # Note: most logic runs from mounted GPFS paths, so --folder is only to keep upload small.
  local job_cmd
  job_cmd="$(cat <<EOF
set -euo pipefail
export ORACLE_MODEL_URLS='${oracle_urls_csv}'
export INTERN_S1_MINI_MODEL_URL='${actor_urls_csv}'
export VERIFIER_MODEL_URL='${verifier_urls_csv}'
export ACTOR_MODEL='${actor_model_id}'
export REWARD_MODEL='${reward_model_id}'
export NO_PROXY='${NO_PROXY}'
export no_proxy='${NO_PROXY}'
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy || true

${extra_env_block}

echo "[run] rollout_dir=${rollout_dir}"
echo "[run] out_dir=${out_dir}"
echo "[run] tokenizer=${TOKENIZER_PATH}"
bash '${PIPELINE_SH}' '${rollout_dir}' '${out_dir}' '${TOKENIZER_PATH}'
EOF
)"

  local submit=(rjob submit
    --name="${name}"
    --folder="${folder}"
    --gpu=0
    --cpu="${cpu}"
    --memory="${mem_mb}"
    --charged-group="${CLUSTER}"
    --private-machine="${PRIVATE_MACHINE}"
    --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit
    --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei
    --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared
    --image="${IMAGE}"
    --host-network=true
    -- bash -c "${job_cmd}"
  )

  if [[ "${dry_run}" == "1" ]]; then
    printf '%q ' "${submit[@]}"
    echo
  else
    "${submit[@]}"
  fi
}

ts="$(date +%m%d%H%M%S)"

if [[ "${mode}" == "smoke" || "${mode}" == "both" ]]; then
  smoke_out="${OUTPUT_ROOT}/rjob_smoke_v4_ckpt_${ts}"
  smoke_env="$(cat <<'EOF'
export TEST_MODE_MAX_CASES=3
export NUM_CHECKPOINTS=2
export MAX_INTERVENTIONS=1
export K_BASE=1
export K_HINT=1
export ORACLE_CANDIDATES_PER_CHECKPOINT=1
export MAX_NEW_TOKENS=256
export ROLLOUT_WORKERS=4
export ITEM_WORKERS=1
export MAX_WORKERS=8
export ORACLE_MAX_COMPLETION_TOKENS=512
export ORACLE_TIMEOUT=60
EOF
)"
  _submit_one "rjob-verifier-coldstart-smoke-${ts}" "${CPU_SMOKE}" "${MEM_SMOKE_MB}" "${SMOKE_ROLLOUT_DIR}" "${smoke_out}" "${smoke_env}"
fi

if [[ "${mode}" == "full" || "${mode}" == "both" ]]; then
  full_out="${OUTPUT_ROOT}/rjob_full_v4_ckpt_math_cap2_${ts}"
  full_env="$(cat <<'EOF'
export TEST_MODE_MAX_CASES=0
export K_BASE=4
export K_HINT=4
export TAU=0.1
export TOKEN_CHECK_INTERVAL=4096
export NUM_CHECKPOINTS=8
export MAX_INTERVENTIONS=5
export ORACLE_CANDIDATES_PER_CHECKPOINT=2
export MAX_NEW_TOKENS=32768
export ROLLOUT_WORKERS=32
export ITEM_WORKERS=8
export MAX_WORKERS=32
EOF
)"
  _submit_one "rjob-verifier-coldstart-full-${ts}" "${CPU_FULL}" "${MEM_FULL_MB}" "${FULL_ROLLOUT_DIR}" "${full_out}" "${full_env}"
fi

if [[ -n "${resume_full_out_dir}" ]]; then
  # Resume step2 using existing intermediate outputs. You MUST stop the old job first if it is still running,
  # otherwise concurrent writers may corrupt jsonl outputs.
  resume_env="$(cat <<'EOF'
export RESUME_FROM_STEP2=1
export TEST_MODE_MAX_CASES=0
export K_BASE=4
export K_HINT=4
export TAU=0.1
export TOKEN_CHECK_INTERVAL=4096
export NUM_CHECKPOINTS=8
export MAX_INTERVENTIONS=5
export ORACLE_CANDIDATES_PER_CHECKPOINT=2
export MAX_NEW_TOKENS=32768
export ROLLOUT_WORKERS=128
export ITEM_WORKERS=32
export MAX_WORKERS=32
EOF
)"
  _submit_one "rjob-verifier-coldstart-resume-${ts}" "${CPU_FULL}" "${MEM_FULL_MB}" "${FULL_ROLLOUT_DIR}" "${resume_full_out_dir}" "${resume_env}"
fi
