#!/usr/bin/env bash
set -eo pipefail

if [ -f /etc/profile.d/ssh-init.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/ssh-init.sh || true
fi

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

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

_latest_run_group_id() {
  local out_root="$1"
  local ds_dir="${out_root}/datasets"
  if [[ ! -d "${ds_dir}" ]]; then
    return 1
  fi
  ls -1 "${ds_dir}" 2>/dev/null | sort | tail -n 1
}

cluster="${CLUSTER:-llmit_gpu}"
gpus="${GPUS:-8}"
nnodes="${NNODES:-1}"
cpu="${RJOB_CPU:-64}"
memory="${RJOB_MEMORY:-512000}"
use_rdma="${USE_RDMA:-1}"
use_private_machine="${PRIVATE_MACHINE:-1}"

image="${IMAGE:-registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605}"
job_prefix="${JOB_PREFIX:-rjob-offline-align-aime2024}"
job_name="${RJOB_NAME:-${job_prefix}-$(date +%m%d%H%M)}"

out_root="${OUT_ROOT:-/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/offline_eval_verifier}"
run_group_id="${RUN_GROUP_ID:-}"
if [[ -z "${run_group_id}" ]]; then
  if latest="$(_latest_run_group_id "${out_root}")"; then
    run_group_id="${latest}"
  else
    run_group_id="$(date +%Y%m%d_%H%M%S)_$RANDOM"
  fi
fi

eval_urls="${EVAL_URLS:-http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1}"
eval_urls="$(_sanitize_csv_urls "${eval_urls}")"
if [[ -z "${eval_urls}" ]]; then
  echo "[rjob][ERR] EVAL_URLS is empty after sanitation." >&2
  exit 2
fi

echo "[rjob] name=${job_name}"
echo "[rjob] run_group_id=${run_group_id}"
echo "[rjob] out_root=${out_root}"
echo "[rjob] datasets=${DATASETS:-aime2024} mode=${MODE:-both} control_variant=${CONTROL_VARIANT:-both}"
echo "[rjob] metric_mode=${EVAL_METRIC_MODE:-both} repeat(aime2024)=${AIME2024_REPEAT:-32}"
echo "[rjob] debug_n=${DEBUG_N:-0} monitor_interval=${PROGRESS_INTERVAL:-30}"
echo "[rjob] actor_model=${ACTOR_MODEL:-<default>}"
echo "[rjob] verifier_model=${VERIFIER_MODEL:-<default>}"
echo "[rjob] verifier_lora=${VERIFIER_LORA-<default>}"
echo "[rjob] ckpt_root=${CKPT_ROOT:-<default>}"
echo "[rjob] actor_transport=${ACTOR_TRANSPORT:-openai} actor_api_base=${ACTOR_API_BASE:-<default>}"
echo "[rjob] actor_api_model=${ACTOR_API_MODEL:-<auto>}"
echo "[rjob] actor_api_mode=${ACTOR_API_MODE:-auto} actor_server_pool=${ACTOR_SERVER_POOL:-auto} actor_api_port_base=${ACTOR_API_PORT_BASE:-0}"
echo "[rjob] token_check(chunk/exact)=${CHUNK_TOKEN_CHECK_INTERVAL:-4096}/${EXACT_TOKEN_CHECK_INTERVAL:-65536} resume_from_tmp=${RESUME_FROM_TMP:-0}"
echo "[rjob] use_rdma=${use_rdma}"
echo "[rjob] private_machine=${use_private_machine}"

rdma_args=()
if [[ "${use_rdma}" == "1" ]]; then
  rdma_args=(--custom-resources "rdma/mlnx_shared=${gpus}")
fi

private_args=()
if [[ "${use_private_machine}" == "1" ]]; then
  private_args=(--private-machine=group)
fi

gpu_ids_str=""
if [[ "${gpus}" =~ ^[0-9]+$ ]] && [[ "${gpus}" -gt 0 ]]; then
  gpu_ids_str="$(seq 0 $((gpus - 1)) | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
fi
if [[ -z "${gpu_ids_str}" ]]; then
  gpu_ids_str="0"
fi

rjob submit \
  --name="${job_name}" \
  --gpu="${gpus}" \
  --memory="${memory}" \
  --cpu="${cpu}" \
  --charged-group="${cluster}" \
  "${private_args[@]}" \
  --share-host-shm=True \
  --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
  --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
  --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
  --image="${image}" \
  -P "${nnodes}" \
  --host-network=true \
  -e DISTRIBUTED_JOB=true \
  "${rdma_args[@]}" \
  -- bash -lc -- "
    unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY
    cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro
    export EVAL_URLS='${eval_urls}'
    export RUN_GROUP_ID='${run_group_id}'
    export OUT_ROOT='${out_root}'
    export DATASETS='${DATASETS:-aime2024}'
    export MODE='${MODE:-both}'
    export CONTROL_VARIANT='${CONTROL_VARIANT:-both}'
    export EVAL_METRIC_MODE='${EVAL_METRIC_MODE:-both}'
    export OC_ROOT='${OC_ROOT:-}'
    export OC_REPO_ROOT='${OC_REPO_ROOT:-}'
    export OC_PRED_ABBR='${OC_PRED_ABBR:-}'
    export OC_RES_ABBR='${OC_RES_ABBR:-}'
    export OC_EVAL_CONFIG='${OC_EVAL_CONFIG:-}'
    export OC_LLM_JUDGE='${OC_LLM_JUDGE:-auto}'
    export ACTOR_MODEL='${ACTOR_MODEL:-}'
    export VERIFIER_MODEL='${VERIFIER_MODEL:-}'
    export VERIFIER_LORA='${VERIFIER_LORA-}'
    export CKPT_ROOT='${CKPT_ROOT:-}'
    export GPQA_REPEAT='${GPQA_REPEAT:-}'
    export AIME_REPEAT='${AIME_REPEAT:-}'
    export AIME2024_REPEAT='${AIME2024_REPEAT:-32}'
    export AIME2025_REPEAT='${AIME2025_REPEAT:-}'
    export DEBUG_N='${DEBUG_N:-0}'
    export MONITOR_PROGRESS='${MONITOR_PROGRESS:-1}'
    export PROGRESS_INTERVAL='${PROGRESS_INTERVAL:-30}'
    export INTERVENTION_AUDIT_N='${INTERVENTION_AUDIT_N:-20}'
    export GPU_IDS='${GPU_IDS:-${gpu_ids_str}}'
    export SHARDS='${SHARDS:-${gpus}}'
    export BACKEND='${BACKEND:-lmdeploy}'
    export CONDA_ENV='${CONDA_ENV:-oc}'
    export USE_VERIFIER_SYSTEM_PROMPT='${USE_VERIFIER_SYSTEM_PROMPT:-1}'
    export ACTOR_TRANSPORT='${ACTOR_TRANSPORT:-openai}'
    export ACTOR_API_BASE='${ACTOR_API_BASE:-}'
    export ACTOR_API_MODEL='${ACTOR_API_MODEL:-}'
    export ACTOR_API_MODE='${ACTOR_API_MODE:-auto}'
    export ACTOR_API_TIMEOUT='${ACTOR_API_TIMEOUT:-600}'
    export START_ACTOR_API_SERVER='${START_ACTOR_API_SERVER:-0}'
    export ACTOR_API_PORT='${ACTOR_API_PORT:-0}'
    export ACTOR_API_TP='${ACTOR_API_TP:-1}'
    export ACTOR_API_WORKER_NUM='${ACTOR_API_WORKER_NUM:-1}'
    export ACTOR_API_EXTRA_CLI='${ACTOR_API_EXTRA_CLI:-}'
    export ACTOR_SERVER_POOL='${ACTOR_SERVER_POOL:-auto}'
    export ACTOR_API_PORT_BASE='${ACTOR_API_PORT_BASE:-0}'
    export ACTOR_DISABLE_THINKING='${ACTOR_DISABLE_THINKING:-0}'
    export CHUNK_TOKEN_CHECK_INTERVAL='${CHUNK_TOKEN_CHECK_INTERVAL:-4096}'
    export EXACT_TOKEN_CHECK_INTERVAL='${EXACT_TOKEN_CHECK_INTERVAL:-65536}'
    export RESUME_FROM_TMP='${RESUME_FROM_TMP:-0}'
    export DO_CV_EVAL='${DO_CV_EVAL:-}'
    export MAX_SEQ_LEN='${MAX_SEQ_LEN:-65536}'
    export MAX_OUT_LEN='${MAX_OUT_LEN:-65536}'
    export TEMPERATURE='${TEMPERATURE:-0.7}'
    export TOP_P='${TOP_P:-1.0}'
    export TOP_K='${TOP_K:-40}'
    export DEGENERATION_GUARD='${DEGENERATION_GUARD:-0}'
    export MAX_SAMPLE_SECONDS='${MAX_SAMPLE_SECONDS:-0}'
    bash run_offline_eval_verifier_intervention_8gpu.sh
  "
