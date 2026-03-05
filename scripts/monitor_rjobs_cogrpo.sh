#!/usr/bin/env bash
set -euo pipefail

# Monitor rjob trainings and auto-clone on early failures.
#
# Usage:
#   bash scripts/monitor_rjobs_cogrpo.sh <job1> <job2> ...
#
# Env:
#   SLEEP_SEC=600              # poll interval
#   TARGET_STEP=10             # mark "stable" when reaching this step
#   AUTO_CLONE_ON_FAIL=1       # auto clone+stop failed jobs
#   MAX_RETRIES=3              # max clone retries per job
#   STATE_DIR=/path/to/state   # persistent state

# Ensure kubebrain env for `rjob` even inside tmux/no-login shells.
if [ -f /etc/profile.d/ssh-init.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/ssh-init.sh || true
fi

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

sleep_sec="${SLEEP_SEC:-600}"
target_step="${TARGET_STEP:-10}"
auto_clone_on_fail="${AUTO_CLONE_ON_FAIL:-1}"
max_retries="${MAX_RETRIES:-3}"
state_dir="${STATE_DIR:-/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/rjob_monitor_state}"

mkdir -p "${state_dir}"

if [ "$#" -lt 1 ]; then
  echo "[monitor][ERR] Provide at least one rjob name." >&2
  exit 2
fi

strip_ansi() {
  # Remove common ANSI escape sequences.
  sed -r 's/\x1B\[[0-9;]*[A-Za-z]//g'
}

get_retry_count() {
  local key="$1"
  local f="${state_dir}/${key}.retries"
  if [ -f "${f}" ]; then
    cat "${f}"
  else
    echo 0
  fi
}

set_retry_count() {
  local key="$1"
  local v="$2"
  echo "${v}" >"${state_dir}/${key}.retries"
}

mark_done() {
  local key="$1"
  touch "${state_dir}/${key}.done"
}

is_done() {
  local key="$1"
  [ -f "${state_dir}/${key}.done" ]
}

handled_fail_key() {
  local job="$1"
  echo "${state_dir}/${job}.last_fail_sig"
}

extract_failed_replicas() {
  # Input: rjob get output.
  sed -nE 's/.*\|- replica ([^:]+):[[:space:]]*FAILED.*/\1/p'
}

extract_step_from_logs() {
  # Best-effort parsing of tqdm progress:
  # "Training Progress:  10%|...| 10/1000 [...]"
  #
  # Input: rjob logs output (tail).
  local last
  last="$(grep -a "Training Progress" | tail -n 1 || true)"
  if [ -z "${last}" ]; then
    return 0
  fi
  echo "${last}" | sed -nE 's/.*Training Progress:[^0-9]*([0-9]+)\/.*/\1/p' | head -n 1
}

clone_job() {
  local job="$1"
  local retries="$2"
  local desired_showname="${job}-retry$((retries + 1))"
  echo "[monitor][ACTION] rjob clone ${job} (showname=${desired_showname})" >&2

  # NOTE: `rjob clone --name` behaves like a "showname" (annotation) setter on some clusters.
  # The actual Kubernetes metadata.name may be auto-normalized / auto-generated, so we must
  # resolve the real rjob name via `rjob list --name <showname>`.
  local clone_out showname list_out new_job
  clone_out="$(rjob clone "${job}" --name "${desired_showname}" --stop-original true 2>&1)" || {
    echo "${clone_out}" >&2
    return 1
  }
  echo "${clone_out}" >&2

  showname="$(printf '%s\n' "${clone_out}" | sed -nE 's/.*new job is ([^[:space:]]+).*/\1/p' | tail -n 1)"
  if [ -z "${showname}" ]; then
    showname="${desired_showname}"
  fi

  # Wait for job creation to become visible.
  new_job=""
  for _ in $(seq 1 60); do
    list_out="$(rjob list --name "${showname}" 2>/dev/null || true)"
    new_job="$(printf '%s\n' "${list_out}" | awk '$3 == \"[INFO]\" && $4 == \"rjob\" {print $5; exit}')"
    if [ -n "${new_job}" ] && [ "${new_job}" != "${job}" ]; then
      break
    fi
    sleep 2
  done
  if [ -z "${new_job}" ]; then
    echo "[monitor][ERR] clone succeeded but cannot resolve new job name for showname=${showname}" >&2
    return 1
  fi

  echo "${new_job}"
}

jobs=("$@")

while true; do
  now="$(date '+%F %T')"
  echo "========== ${now} =========="

  next_jobs=()
  for job in "${jobs[@]}"; do
    if is_done "${job}"; then
      echo "[monitor] ${job}: done (reached step>=${target_step})"
      next_jobs+=("${job}")
      continue
    fi

    out="$(rjob get "${job}" 2>&1 || true)"
    echo "${out}"

    failed_replicas="$(printf '%s\n' "${out}" | extract_failed_replicas || true)"
    if [ -n "${failed_replicas}" ]; then
      fail_sig="$(printf '%s\n' "${failed_replicas}" | tr '\n' ',' | sed 's/,$//')"
      last_sig=""
      if [ -f "$(handled_fail_key "${job}")" ]; then
        last_sig="$(cat "$(handled_fail_key "${job}")" 2>/dev/null || true)"
      fi

      # Only react on new failure transitions/signatures (avoid spamming logs every poll).
      if [ "${fail_sig}" != "${last_sig}" ]; then
        echo "${fail_sig}" >"$(handled_fail_key "${job}")"
        echo "[monitor][WARN] ${job}: replica FAILED (${fail_sig}) -> collecting logs"
        echo "----- logs (job=${job}, tail=200) -----"
        rjob logs job "${job}" --tail-lines 200 2>&1 | strip_ansi || true
        while read -r replica; do
          [ -z "${replica}" ] && continue
          echo "----- logs (replica=${replica}, tail=200) -----"
          rjob logs replica "${replica}" --tail-lines 200 2>&1 | strip_ansi || true
        done <<<"${failed_replicas}"

        if [ "${auto_clone_on_fail}" = "1" ]; then
          retries="$(get_retry_count "${job}")"
          if [ "${retries}" -lt "${max_retries}" ]; then
            if new_job="$(clone_job "${job}" "${retries}")"; then
              set_retry_count "${job}" "$((retries + 1))"
              next_jobs+=("${new_job}")
              continue
            fi
          fi
        fi
      fi

      next_jobs+=("${job}")
      continue
    else
      # Clear failure signature when job is healthy again.
      rm -f "$(handled_fail_key "${job}")" 2>/dev/null || true
    fi

    # If job is Running, try to parse step from tail logs.
    job_status_line="$(printf '%s\n' "${out}" | head -n 1)"
    if printf '%s\n' "${job_status_line}" | grep -q ": Running\\|: RUNNING"; then
      logs_out="$(rjob logs job "${job}" --tail-lines 200 2>&1 | strip_ansi || true)"
      step="$(printf '%s\n' "${logs_out}" | extract_step_from_logs || true)"
      if [ -n "${step}" ]; then
        echo "[monitor] ${job}: parsed_step=${step}"
        if [ "${step}" -ge "${target_step}" ] 2>/dev/null; then
          echo "[monitor][OK] ${job}: reached step>=${target_step}; marking done"
          mark_done "${job}"
        fi
      fi
    fi

    next_jobs+=("${job}")
  done

  jobs=("${next_jobs[@]}")
  echo "[monitor] sleep ${sleep_sec}s..."
  sleep "${sleep_sec}"
done
