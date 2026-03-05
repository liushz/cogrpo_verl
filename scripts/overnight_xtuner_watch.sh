#!/usr/bin/env bash
set -euo pipefail

if [ -f /etc/profile.d/ssh-init.sh ]; then
  source /etc/profile.d/ssh-init.sh || true
fi

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

REPO_ROOT="/mnt/shared-storage-user/liuhongwei/main_works/repos/repro"
cd "${REPO_ROOT}"

mkdir -p /mnt/shared-storage-user/liuhongwei/main_works/temp_debug/logs
RUN_TAG="$(date +%m%d_%H%M%S)"
LOG_FILE="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/logs/overnight_xtuner_watch_${RUN_TAG}.log"
STATE_FILE="/tmp/overnight_xtuner_watch_${RUN_TAG}.state"

EIGHT_JOBS=(
  "rjob-xtcf8-verify-0304005515-16691838"
  "rjob-xtcf8-verify2-0304010622-83262313"
)
THIRTYTWO_JOB=""
THIRTYTWO_RETRY=0
EIGHT_RETRY=0

VERIFIER_LORA_PATH="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-verifier_sft_train_55_waitx2_fpfix_plus_v4ckptwait_20260223-02241157/checkpoint-2445"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

save_state() {
  {
    echo "EIGHT_JOBS=${EIGHT_JOBS[*]}"
    echo "THIRTYTWO_JOB=${THIRTYTWO_JOB}"
    echo "THIRTYTWO_RETRY=${THIRTYTWO_RETRY}"
    echo "EIGHT_RETRY=${EIGHT_RETRY}"
  } > "${STATE_FILE}"
}

parse_submitted_job_name() {
  sed -n 's/.*created rjob_name: //p' | tail -n 1
}

submit_8_job() {
  local exp="xtcf8-auto-${RUN_TAG}-$(date +%H%M%S)"
  log "submit 8gpu xtuner validation exp=${exp}"
  local out
  out=$(
    EXP_NAME="${exp}" \
    RJOB_MEMORY=320000 \
    RJOB_CPU=32 \
    COGRPO_VERIFIER_LORA_ENABLE=1 \
    COGRPO_VERIFIER_LORA_PATH="${VERIFIER_LORA_PATH}" \
    COGRPO_VERIFIER_LORA_RANK=64 \
    COGRPO_VERIFIER_LORA_ALPHA=128 \
    COGRPO_VERIFIER_LORA_DROPOUT=0.05 \
    COGRPO_VERIFIER_LR=1e-5 \
    COGRPO_VERIFIER_LORA_SYNC_FREQ=1 \
    COGRPO_VERIFIER_DEBUG=1 \
    COGRPO_VERIFIER_DEBUG_MAX_CALLS=300 \
    COGRPO_VERIFIER_DEBUG_TEXT_CHARS=640 \
    COGRPO_VERIFIER_DEBUG_DUMP=1 \
    COGRPO_VERIFIER_DEBUG_DUMP_MAX_CALLS=300 \
    COGRPO_VERIFIER_PARSER_MODE=auto \
    REWARD_MODEL_KEY=EMPTY \
    bash run_rjob_xtuner_cf_branch_8gpu_bsz8.sh 2>&1
  )
  echo "${out}" >> "${LOG_FILE}"
  local job
  job="$(echo "${out}" | parse_submitted_job_name)"
  if [ -z "${job}" ]; then
    log "submit 8gpu failed: cannot parse job name"
    return 1
  fi
  log "submitted 8gpu job=${job}"
  EIGHT_JOBS+=("${job}")
  return 0
}

submit_32_job() {
  local exp="cogrpo32-base-lora-${RUN_TAG}"
  log "submit 32gpu bsz128 base+lora exp=${exp}"
  local out
  out=$(
    REWARD_MODEL_KEY=EMPTY \
    bash run_rjob_cogrpo_32gpu.sh \
      --exp-name "${exp}" \
      --train-bsz 128 \
      --micro-bsz 2 \
      --response-n 32 \
      --rollout-n 8 \
      --verifier-lora-path "${VERIFIER_LORA_PATH}" \
      --steps 1000 2>&1
  )
  echo "${out}" >> "${LOG_FILE}"
  local job
  job="$(echo "${out}" | parse_submitted_job_name)"
  if [ -z "${job}" ]; then
    log "submit 32gpu failed: cannot parse job name"
    return 1
  fi
  THIRTYTWO_JOB="${job}"
  log "submitted 32gpu job=${job}"
  return 0
}

get_status() {
  local job="$1"
  local line
  line="$(rjob get "${job}" 2>/dev/null | sed -n '1p' || true)"
  if [ -z "${line}" ]; then
    echo "UNKNOWN"
    return
  fi
  echo "${line}" | sed -n 's/.*): //p' | awk '{print $1}'
}

collect_8_metrics() {
  local job="$1"
  local logs
  logs="$(rjob logs job "${job}" -n 4000 2>/dev/null || true)"
  local dec ins wait lora_ok lora_zero err
  dec=$(echo "${logs}" | grep -c "\[CoGRPO\]\[VerifierDebug\] decision " || true)
  ins=$(echo "${logs}" | grep -c "\[CoGRPO\]\[VerifierDebug\] inserted_hint" || true)
  wait=$(echo "${logs}" | grep -c "decision .* tag=WAIT" || true)
  lora_ok=$(echo "${logs}" | grep -Ec "Loaded verifier LoRA \(scanned=.*loaded=[1-9]" || true)
  lora_zero=$(echo "${logs}" | grep -c "Loaded 0 verifier LoRA tensors" || true)
  err=$(echo "${logs}" | grep -Ec "Traceback|RayTaskError|RuntimeError|AssertionError|\[ERROR\]" || true)
  echo "${dec} ${ins} ${wait} ${lora_ok} ${lora_zero} ${err}"
}

log "watchdog start tag=${RUN_TAG}"
log "state_file=${STATE_FILE}"
log "initial 8gpu jobs: ${EIGHT_JOBS[*]}"

while true; do
  save_state

  if [ "${#EIGHT_JOBS[@]}" -gt 0 ]; then
    mapfile -t EIGHT_JOBS < <(printf '%s\n' "${EIGHT_JOBS[@]}" | awk 'NF && !seen[$0]++')
  fi

  eight_ok=0
  alive_jobs=()

  for job in "${EIGHT_JOBS[@]:-}"; do
    [ -n "${job}" ] || continue
    st="$(get_status "${job}")"
    log "8gpu job=${job} status=${st}"

    case "${st}" in
      Running|Inqueue|STARTING|Starting)
        alive_jobs+=("${job}")
        ;;
      Succeeded)
        read -r dec ins wait lora_ok lora_zero err < <(collect_8_metrics "${job}")
        log "8gpu job=${job} metrics decision=${dec} inserted=${ins} wait=${wait} lora_ok=${lora_ok} lora_zero=${lora_zero} err=${err}"
        if [ "${err}" -eq 0 ] && [ "${lora_ok}" -ge 1 ] && [ "${lora_zero}" -eq 0 ] && { [ "${ins}" -ge 1 ] || [ "${wait}" -ge 1 ]; }; then
          eight_ok=1
          alive_jobs+=("${job}")
          log "8gpu validation PASSED on job=${job}"
        else
          log "8gpu validation FAILED on job=${job}; will resubmit"
        fi
        ;;
      Failed|Stopped|CreatedFailed)
        log "8gpu job=${job} ended with status=${st}; will resubmit"
        ;;
      *)
        alive_jobs+=("${job}")
        ;;
    esac
  done

  EIGHT_JOBS=("${alive_jobs[@]:-}")

  if [ "${eight_ok}" -ne 1 ]; then
    has_active=0
    for job in "${EIGHT_JOBS[@]:-}"; do
      st="$(get_status "${job}")"
      if [[ "${st}" =~ ^(Running|Inqueue|STARTING|Starting)$ ]]; then
        has_active=1
        break
      fi
    done
    if [ "${has_active}" -eq 0 ]; then
      EIGHT_RETRY=$((EIGHT_RETRY + 1))
      log "no active 8gpu job, retry=${EIGHT_RETRY}, submit new one"
      submit_8_job || true
    fi
    sleep 120
    continue
  fi

  if [ -z "${THIRTYTWO_JOB}" ]; then
    THIRTYTWO_RETRY=$((THIRTYTWO_RETRY + 1))
    submit_32_job || true
    sleep 120
    continue
  fi

  st32="$(get_status "${THIRTYTWO_JOB}")"
  log "32gpu job=${THIRTYTWO_JOB} status=${st32}"
  case "${st32}" in
    Running)
      log "32gpu job is RUNNING. watchdog done."
      save_state
      exit 0
      ;;
    Failed|Stopped|CreatedFailed)
      THIRTYTWO_RETRY=$((THIRTYTWO_RETRY + 1))
      log "32gpu job failed status=${st32}, retry=${THIRTYTWO_RETRY}"
      THIRTYTWO_JOB=""
      ;;
    *)
      ;;
  esac

  sleep 120
done
