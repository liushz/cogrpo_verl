#!/usr/bin/env bash
# Offline eval for OC alignment + verifier intervention attribution.
#
# Design:
# - Build eval datasets from OC predictions (dedup to unique questions).
# - Run one or more arms per dataset:
#     control_oc_exact  : control, single-shot generation (TOKEN_CHECK_INTERVAL ~= MAX_OUT_LEN)
#     control_chunked   : control, chunked generation
#     exp_chunked       : exp, chunked generation with verifier intervention
# - Produce two metric families:
#     1) OpenCompass-style cascade metrics (oc)
#     2) CompassVerifier metrics (cv)
# - Produce generation diagnostics and intervention audit samples.
set -eEuo pipefail

_die() { echo "ERROR: $*" >&2; exit 1; }

_sanitize_eval_urls() {
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

_is_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

_is_writable_dir() {
  local p="$1"
  [[ -d "${p}" ]] || mkdir -p "${p}" 2>/dev/null || return 1
  [[ -w "${p}" ]]
}

_is_readable_dir() {
  local p="$1"
  [[ -d "${p}" && -r "${p}" ]]
}

_ensure_writable_env_dir() {
  local var_name="$1"
  local fallback="$2"
  local cur="${!var_name:-}"
  local chosen="${cur}"
  if [[ -z "${chosen}" ]]; then
    chosen="${fallback}"
  fi
  if ! _is_writable_dir "${chosen}"; then
    chosen="${fallback}"
    _is_writable_dir "${chosen}" || _die "Failed to create writable cache dir for ${var_name}: ${chosen}"
  fi
  export "${var_name}=${chosen}"
}

_ensure_readable_or_local_env_dir() {
  local var_name="$1"
  local local_fallback="$2"
  local preferred_readonly="${3:-}"
  local cur="${!var_name:-}"
  local chosen=""

  if [[ -n "${cur}" ]] && _is_readable_dir "${cur}"; then
    chosen="${cur}"
  elif [[ -n "${preferred_readonly}" ]] && _is_readable_dir "${preferred_readonly}"; then
    chosen="${preferred_readonly}"
  else
    chosen="${local_fallback}"
    _is_writable_dir "${chosen}" || _die "Failed to create local fallback dir for ${var_name}: ${chosen}"
  fi
  export "${var_name}=${chosen}"
}

_setup_no_proxy() {
  local urls_csv="$1"
  local hosts=""
  hosts="$("${PYTHON_BIN}" - "${urls_csv}" <<'PY'
import re, sys
urls = sys.argv[1] if len(sys.argv) > 1 else ""
hosts = []
for u in (urls or "").split(","):
    u = (u or "").strip().strip("<>").strip()
    if not u:
        continue
    u = re.sub(r"^https?://", "", u)
    u = u.split("/", 1)[0]
    h = u.split(":", 1)[0].strip()
    if h:
        hosts.append(h)
print(",".join(sorted(set(hosts))))
PY
)"
  local fixed="localhost,127.0.0.1,::1,.svc,.pjlab.org.cn"
  if [[ -n "${hosts}" ]]; then
    export NO_PROXY="${hosts},${fixed}${NO_PROXY:+,${NO_PROXY}}"
  else
    export NO_PROXY="${fixed}${NO_PROXY:+,${NO_PROXY}}"
  fi
  export no_proxy="${NO_PROXY}"
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
  echo "[env] NO_PROXY=${NO_PROXY}"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------
# OpenCompass reference
# --------------------------
OC_ROOT="${OC_ROOT:-/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/data/oc_eval}"
OC_PRED_ABBR="${OC_PRED_ABBR:-qwen2_5_7b_baseline_s20}"  # used to extract prompts/gold
OC_RES_ABBR="${OC_RES_ABBR:-${OC_PRED_ABBR}}"           # used as baseline in results/
OC_REPO_ROOT="${OC_REPO_ROOT:-/mnt/shared-storage-user/opencompass-shared/qa-llm-cicd/opencompass-main2/opencompass}"
OC_EVAL_CONFIG="${OC_EVAL_CONFIG:-}"
OC_LLM_JUDGE="${OC_LLM_JUDGE:-auto}"                    # auto|on|off
OC_JUDGE_BATCH_SIZE="${OC_JUDGE_BATCH_SIZE:-64}"        # force judge batching for faster OC-aligned scoring
OC_JUDGE_QPS="${OC_JUDGE_QPS:-0}"                       # 0 means keep OC config
OC_JUDGE_MAX_WORKERS="${OC_JUDGE_MAX_WORKERS:-0}"       # 0 means keep OC config
OC_JUDGE_REPLICA_WORKERS="${OC_JUDGE_REPLICA_WORKERS:-1}"  # parallel over repeat replicas
OC_REQUIRE_JUDGE_PROMPT="${OC_REQUIRE_JUDGE_PROMPT:-1}"    # 1: require judge prompt from OC evaluator cfg
DO_OC_PRED_DIFF="${DO_OC_PRED_DIFF:-1}"
OC_DIFF_TOP_N="${OC_DIFF_TOP_N:-20}"

_is_usable_oc_repo() {
  local p="$1"
  [[ -d "${p}" ]] && [[ -d "${p}/configs" ]] && [[ -f "${p}/__init__.py" ]]
}

_resolve_oc_repo_root() {
  local p
  for p in \
    "${OC_REPO_ROOT}" \
    "/mnt/shared-storage-user/opencompass-shared/qa-llm-cicd/opencompass-main2/opencompass" \
    "/mnt/shared-storage-user/auto-eval-pipeline/opencompass@f1e50d4/opencompass" \
    "/mnt/shared-storage-user/auto-eval-pipeline/opencompass@f1e50d4.bak20260226-new/opencompass" \
    "/mnt/shared-storage-user/opencompass-shared/liushudong/opencompass/opencompass"
  do
    if _is_usable_oc_repo "${p}"; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

if ! OC_REPO_ROOT="$(_resolve_oc_repo_root)"; then
  _die "Missing usable OC_REPO_ROOT. Tried current/default and common shared OpenCompass paths."
fi
echo "[oc] oc_repo_root=${OC_REPO_ROOT}"

[[ -d "${OC_ROOT}/predictions/${OC_PRED_ABBR}" ]] || _die "Missing: ${OC_ROOT}/predictions/${OC_PRED_ABBR}"
[[ -d "${OC_ROOT}/results/${OC_RES_ABBR}" ]] || _die "Missing: ${OC_ROOT}/results/${OC_RES_ABBR}"

# --------------------------
# Output layout
# --------------------------
OUT_ROOT="${OUT_ROOT:-/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/data/oc_eval/local_verifier_eval}"
RUN_GROUP_ID="${RUN_GROUP_ID:-$(date +%Y%m%d_%H%M%S)_$RANDOM}"
DATASET_DIR="${DATASET_DIR:-${OUT_ROOT}/datasets/${RUN_GROUP_ID}}"
mkdir -p "${DATASET_DIR}"

# Keep HF/evaluate dynamic module/cache under a writable runtime path.
OC_RUNTIME_CACHE_ROOT="${OC_RUNTIME_CACHE_ROOT:-${OUT_ROOT}/_oc_runtime_cache/${RUN_GROUP_ID}}"
_ensure_writable_env_dir OC_RUNTIME_CACHE_ROOT "${OUT_ROOT}/_oc_runtime_cache/${RUN_GROUP_ID}"
_ensure_writable_env_dir XDG_CACHE_HOME "${OC_RUNTIME_CACHE_ROOT}/xdg_cache"
_ensure_writable_env_dir HF_HOME "${OC_RUNTIME_CACHE_ROOT}/hf_home"
_ensure_writable_env_dir HF_MODULES_CACHE "${OC_RUNTIME_CACHE_ROOT}/hf_modules"
_ensure_writable_env_dir EVALUATE_CACHE "${OC_RUNTIME_CACHE_ROOT}/evaluate"
_ensure_writable_env_dir HF_DATASETS_CACHE "${OC_RUNTIME_CACHE_ROOT}/datasets"
# Prefer shared readonly HF hub cache for tokenizer/model lookup (needed by OC LLM judge),
# while keeping module/datasets/evaluate caches writable in runtime cache root.
HF_HUB_CACHE_PREFERRED="${HF_HUB_CACHE_PREFERRED:-/mnt/shared-storage-user/large-model-center-share-weights/hf_hub}"
_ensure_readable_or_local_env_dir HF_HUB_CACHE "${HF_HOME}/hub" "${HF_HUB_CACHE_PREFERRED}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
TIKTOKEN_CACHE_PREFERRED="${TIKTOKEN_CACHE_PREFERRED:-/mnt/shared-storage-user/auto-eval-pipeline/opencompass/llmeval/share_tiktoken}"
if [[ -d "${TIKTOKEN_CACHE_PREFERRED}" && -r "${TIKTOKEN_CACHE_PREFERRED}" ]]; then
  export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${TIKTOKEN_CACHE_PREFERRED}}"
else
  _ensure_writable_env_dir TIKTOKEN_CACHE_DIR "${OC_RUNTIME_CACHE_ROOT}/tiktoken"
fi
echo "[cache] OC_RUNTIME_CACHE_ROOT=${OC_RUNTIME_CACHE_ROOT}"
echo "[cache] HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "[cache] HF_MODULES_CACHE=${HF_MODULES_CACHE}"
echo "[cache] TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR}"

# --------------------------
# Eval runner knobs
# --------------------------
export BACKEND="${BACKEND:-lmdeploy}"
REQUIRE_IMAGE_ENV="${REQUIRE_IMAGE_ENV:-0}"            # 1: disallow user conda override; require image env
case "${REQUIRE_IMAGE_ENV}" in
  0|1) ;;
  *) _die "Invalid REQUIRE_IMAGE_ENV=${REQUIRE_IMAGE_ENV} (expected: 0|1)" ;;
esac

_default_conda_env="oc"
if [[ "${REQUIRE_IMAGE_ENV}" == "1" ]]; then
  _default_conda_env="none"
fi
export CONDA_ENV="${CONDA_ENV:-${_default_conda_env}}"
CONDA_SH="${CONDA_SH:-/mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh}"
if [[ "${REQUIRE_IMAGE_ENV}" == "1" && "${CONDA_ENV}" != "none" ]]; then
  _die "REQUIRE_IMAGE_ENV=1 forbids CONDA_ENV=${CONDA_ENV} (would override image env). Set CONDA_ENV=none."
fi
if [[ "${CONDA_ENV}" != "none" ]]; then
  [[ -f "${CONDA_SH}" ]] || _die "conda.sh not found at ${CONDA_SH} (set CONDA_ENV=none to skip activation)"
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
OC_SUMMARY_PYTHON="${OC_SUMMARY_PYTHON:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/oc/bin/python}"
if [[ ! -x "${OC_SUMMARY_PYTHON}" ]]; then
  OC_SUMMARY_PYTHON="${PYTHON_BIN}"
fi
echo "[env] conda_env=${CONDA_ENV} python=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "[env] oc_summary_python=${OC_SUMMARY_PYTHON}"

if [[ "${REQUIRE_IMAGE_ENV}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/assert_oc_image_env.py" --require-image-env >/dev/null
  echo "[env] image_env_guard=ok (no user conda override detected)"
fi

export USE_VERIFIER_SYSTEM_PROMPT="${USE_VERIFIER_SYSTEM_PROMPT:-1}"
export ONLINE_MATH_PROMPT="${ONLINE_MATH_PROMPT:-0}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
export MAX_OUT_LEN="${MAX_OUT_LEN:-65536}"
export LMDEPLOY_SESSION_LEN="${LMDEPLOY_SESSION_LEN:-65536}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:-40}"
_USER_ACTOR_API_MODE="${ACTOR_API_MODE:-}"
_USER_ACTOR_DISABLE_THINKING="${ACTOR_DISABLE_THINKING:-}"
_USER_ACTOR_OC_REQUEST_EXACT="${ACTOR_OC_REQUEST_EXACT:-}"
export ACTOR_TRANSPORT="${ACTOR_TRANSPORT:-openai}"     # openai|pipeline
export ACTOR_API_BASE="${ACTOR_API_BASE:-}"
export ACTOR_API_KEY="${ACTOR_API_KEY:-}"
export ACTOR_API_TIMEOUT="${ACTOR_API_TIMEOUT:-600}"
export ACTOR_OPENAI_CLIENT="${ACTOR_OPENAI_CLIENT:-oc_sdk}"
export ACTOR_OC_REPO_PARENT="${ACTOR_OC_REPO_PARENT:-/mnt/shared-storage-user/auto-eval-pipeline/opencompass@f1e50d4}"
export ACTOR_API_MODE="${ACTOR_API_MODE:-auto}"        # auto|completions|chat
export START_ACTOR_API_SERVER="${START_ACTOR_API_SERVER:-0}"
export ACTOR_API_PORT="${ACTOR_API_PORT:-0}"
export ACTOR_API_TP="${ACTOR_API_TP:-1}"
export ACTOR_API_WORKER_NUM="${ACTOR_API_WORKER_NUM:-1}"
export ACTOR_API_EXTRA_CLI="${ACTOR_API_EXTRA_CLI:-}"
export ACTOR_SERVER_POOL="${ACTOR_SERVER_POOL:-auto}"  # auto|0|1
export ACTOR_API_PORT_BASE="${ACTOR_API_PORT_BASE:-0}"
export ACTOR_DISABLE_THINKING="${ACTOR_DISABLE_THINKING:-0}"
export ACTOR_OC_REQUEST_EXACT="${ACTOR_OC_REQUEST_EXACT:-0}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-}"
OC_ALIGN_STRICT="${OC_ALIGN_STRICT:-1}"                # 1: fail fast when OC alignment preconditions are broken
case "${OC_ALIGN_STRICT}" in
  0|1) ;;
  *) _die "Invalid OC_ALIGN_STRICT=${OC_ALIGN_STRICT} (expected: 0|1)" ;;
esac

if [[ "${OC_ALIGN_STRICT}" == "1" ]]; then
  # OpenCompass OpenAISDK in this setup is chat-style and does not pass extra_body.enable_thinking.
  if [[ -z "${ACTOR_API_KEY}" ]]; then
    export ACTOR_API_KEY="sk-admin"
  fi
  if [[ "${_USER_ACTOR_API_MODE}" == "" || "${ACTOR_API_MODE}" == "auto" ]]; then
    export ACTOR_API_MODE="chat"
  fi
  if [[ "${_USER_ACTOR_DISABLE_THINKING}" == "" ]]; then
    export ACTOR_DISABLE_THINKING=1
  fi
  if [[ "${_USER_ACTOR_OC_REQUEST_EXACT}" == "" ]]; then
    export ACTOR_OC_REQUEST_EXACT=1
  fi
  if [[ "${START_ACTOR_API_SERVER}" == "1" && -z "${ACTOR_API_EXTRA_CLI}" ]]; then
    export ACTOR_API_EXTRA_CLI="--backend pytorch --session-len ${MAX_SEQ_LEN} --max-batch-size 1024"
  fi
  if [[ -z "${TOKENIZER_PATH}" ]]; then
    _die "OC_ALIGN_STRICT=1 requires TOKENIZER_PATH for auditable tokenizer parity (expected OC tokenizer_path)."
  fi
  [[ -d "${TOKENIZER_PATH}" ]] || _die "TOKENIZER_PATH not found: ${TOKENIZER_PATH}"
fi

if [[ "${ACTOR_TRANSPORT}" == "openai" && "${START_ACTOR_API_SERVER}" != "1" && -z "${ACTOR_API_BASE}" ]]; then
  echo "[auto] ACTOR_TRANSPORT=openai but ACTOR_API_BASE is empty; auto enable START_ACTOR_API_SERVER=1"
  export START_ACTOR_API_SERVER=1
fi
case "${ACTOR_API_MODE}" in
  auto|completions|chat) ;;
  *) _die "Invalid ACTOR_API_MODE=${ACTOR_API_MODE} (expected: auto|completions|chat)" ;;
esac
if [[ "${OC_ALIGN_STRICT}" == "1" ]]; then
  [[ "${ACTOR_API_MODE}" == "chat" ]] || _die "OC_ALIGN_STRICT=1 requires ACTOR_API_MODE=chat."
  [[ "${ACTOR_DISABLE_THINKING}" == "1" ]] || _die "OC_ALIGN_STRICT=1 requires ACTOR_DISABLE_THINKING=1."
fi
case "${ACTOR_SERVER_POOL}" in
  auto|0|1) ;;
  *) _die "Invalid ACTOR_SERVER_POOL=${ACTOR_SERVER_POOL} (expected: auto|0|1)" ;;
esac
if [[ "${ACTOR_TRANSPORT}" == "openai" ]]; then
  if [[ "${START_ACTOR_API_SERVER}" == "1" ]]; then
    # Local auto-start server often reports model id as a path/basename (not OC abbr).
    # Keep empty by default to auto-pick the only served model.
    export ACTOR_API_MODEL="${ACTOR_API_MODEL:-}"
  else
    # External gateway defaults to OC model abbr for strict alignment.
    export ACTOR_API_MODEL="${ACTOR_API_MODEL:-${OC_RES_ABBR}}"
  fi
else
  export ACTOR_API_MODEL="${ACTOR_API_MODEL:-}"
fi

if [[ -n "${EVAL_URLS:-}" ]]; then
  export EVAL_URLS="$(_sanitize_eval_urls "${EVAL_URLS}")"
fi
EVAL_URLS="${EVAL_URLS:-}"
if [[ "${EVAL_URLS}" == "" ]]; then
  export EVAL_URLS="http://100.96.129.1:21000/v1,http://100.96.129.1:21001/v1,http://100.99.155.1:21002/v1,http://100.99.155.1:21003/v1"
fi

_no_proxy_inputs="${EVAL_URLS}"
if [[ -n "${ACTOR_API_BASE}" ]]; then
  _no_proxy_inputs="${_no_proxy_inputs},${ACTOR_API_BASE}"
fi
_setup_no_proxy "${_no_proxy_inputs}"

# Metric families to produce:
# - oc   : OpenCompass-style cascade evaluator metrics
# - cv   : CompassVerifier metrics
# - both : produce both
EVAL_METRIC_MODE="${EVAL_METRIC_MODE:-both}"            # oc|cv|both
case "${EVAL_METRIC_MODE}" in
  oc|cv|both) ;;
  *) _die "Invalid EVAL_METRIC_MODE=${EVAL_METRIC_MODE} (expected: oc|cv|both)" ;;
esac
if [[ "${OC_ALIGN_STRICT}" == "1" && ( "${EVAL_METRIC_MODE}" == "oc" || "${EVAL_METRIC_MODE}" == "both" ) ]]; then
  [[ "${ACTOR_TRANSPORT}" == "openai" ]] || _die "OC_ALIGN_STRICT=1 requires ACTOR_TRANSPORT=openai for OC-aligned runs."
fi

# Arm control:
# - MODE controls top-level scenario:
#   control -> run control arm(s) only
#   exp     -> run exp arm only
#   both    -> run exp + control arm(s)
MODE="${MODE:-both}"                                     # control|exp|both
case "${MODE}" in
  control|exp|both) ;;
  *) _die "Invalid MODE=${MODE} (expected: control|exp|both)" ;;
esac

# Which control arms to run when MODE includes control:
#   oc_exact | chunked | both
CONTROL_VARIANT="${CONTROL_VARIANT:-both}"
case "${CONTROL_VARIANT}" in
  oc_exact|chunked|both) ;;
  *) _die "Invalid CONTROL_VARIANT=${CONTROL_VARIANT} (expected: oc_exact|chunked|both)" ;;
esac

# Resume safety:
# - default RESUME_FROM_TMP=0 to avoid parameter-mix contamination.
RESUME_FROM_TMP="${RESUME_FROM_TMP:-0}"
export RESUME_FROM_TMP

# Degeneration/timeout guards
export DEGENERATION_GUARD="${DEGENERATION_GUARD:-1}"
export MAX_SAMPLE_SECONDS="${MAX_SAMPLE_SECONDS:-600}"

# Token-check intervals by arm
CHUNK_TOKEN_CHECK_INTERVAL="${CHUNK_TOKEN_CHECK_INTERVAL:-4096}"
EXACT_TOKEN_CHECK_INTERVAL="${EXACT_TOKEN_CHECK_INTERVAL:-${MAX_OUT_LEN}}"
_is_int "${CHUNK_TOKEN_CHECK_INTERVAL}" || _die "CHUNK_TOKEN_CHECK_INTERVAL must be int"
_is_int "${EXACT_TOKEN_CHECK_INTERVAL}" || _die "EXACT_TOKEN_CHECK_INTERVAL must be int"

# Whether to call cv scorer inside run_local runner.
_USER_DO_CV_EVAL="${DO_CV_EVAL:-}"
_default_do_cv=1
if [[ "${EVAL_METRIC_MODE}" == "oc" ]]; then
  _default_do_cv=0
fi
DO_CV_EVAL_RUN="${_USER_DO_CV_EVAL:-${_default_do_cv}}"
if [[ "${DO_CV_EVAL_RUN}" == "1" && -z "${EVAL_URLS}" ]]; then
  _die "EVAL_URLS required when DO_CV_EVAL_RUN=1"
fi

# --------------------------
# Script dependencies
# --------------------------
RUNNER_SH="${REPO_ROOT}/run_local_eval_cogrpo_gs20_vllm_8gpu.sh"
BUILD_DS_PY="${REPO_ROOT}/scripts/build_oc_dataset_jsonl.py"
SUM_CV_PYK="${REPO_ROOT}/scripts/summarize_passk_cv_scored_jsonl.py"
SUM_DIAG_PY="${REPO_ROOT}/scripts/summarize_generation_diagnostics.py"
SUM_OC_PY="${REPO_ROOT}/scripts/summarize_oc_cascade_metrics.py"
DIFF_OC_PY="${REPO_ROOT}/scripts/compare_offline_run_with_oc_predictions.py"
REPORT_PY="${REPO_ROOT}/scripts/report_offline_alignment.py"

[[ -f "${RUNNER_SH}" ]] || _die "Missing runner: ${RUNNER_SH}"
[[ -f "${BUILD_DS_PY}" ]] || _die "Missing script: ${BUILD_DS_PY}"
[[ -f "${SUM_CV_PYK}" ]] || _die "Missing script: ${SUM_CV_PYK}"
[[ -f "${SUM_DIAG_PY}" ]] || _die "Missing script: ${SUM_DIAG_PY}"
[[ -f "${SUM_OC_PY}" ]] || _die "Missing script: ${SUM_OC_PY}"
[[ -f "${DIFF_OC_PY}" ]] || _die "Missing script: ${DIFF_OC_PY}"
[[ -f "${REPORT_PY}" ]] || _die "Missing script: ${REPORT_PY}"

# --------------------------
# Dataset selection
# --------------------------
DATASETS="${DATASETS:-GPQA_diamond,aime2024,aime2025}"
DATASETS="${DATASETS// /,}"
read -r -a DATASET_ARR <<<"${DATASETS//,/ }"
[[ "${#DATASET_ARR[@]}" -gt 0 ]] || _die "Empty DATASETS (supported: GPQA_diamond,aime2024,aime2025)"

AIME2024_PATH_OVERRIDE="${AIME2024_PATH_OVERRIDE:-}"
if [[ -n "${AIME2024_PATH_OVERRIDE}" ]]; then
  [[ -f "${AIME2024_PATH_OVERRIDE}" ]] || _die "AIME2024_PATH_OVERRIDE not found: ${AIME2024_PATH_OVERRIDE}"
fi

_want_ds() {
  local want="$1"
  local x
  for x in "${DATASET_ARR[@]}"; do
    [[ "${x}" == "${want}" ]] && return 0
  done
  return 1
}

for _ds in "${DATASET_ARR[@]}"; do
  case "${_ds}" in
    GPQA_diamond|aime2024|aime2025) ;;
    *) _die "Unsupported DATASETS entry: ${_ds} (supported: GPQA_diamond,aime2024,aime2025)" ;;
  esac
done

# --------------------------
# Build datasets from OpenCompass prompts/gold
# --------------------------
echo "[oc] building datasets into: ${DATASET_DIR}"
if _want_ds "GPQA_diamond"; then
  "${PYTHON_BIN}" "${BUILD_DS_PY}" \
    --oc-root "${OC_ROOT}" --pred-abbr "${OC_PRED_ABBR}" --dataset GPQA_diamond \
    --out-jsonl "${DATASET_DIR}/GPQA_diamond.jsonl" >/dev/null
fi
if _want_ds "aime2024"; then
  if [[ -n "${AIME2024_PATH_OVERRIDE}" ]]; then
    cp -f "${AIME2024_PATH_OVERRIDE}" "${DATASET_DIR}/aime2024.jsonl"
    echo "[oc] aime2024 override -> ${DATASET_DIR}/aime2024.jsonl (src=${AIME2024_PATH_OVERRIDE})"
  else
    "${PYTHON_BIN}" "${BUILD_DS_PY}" \
      --oc-root "${OC_ROOT}" --pred-abbr "${OC_PRED_ABBR}" --dataset aime2024 \
      --out-jsonl "${DATASET_DIR}/aime2024.jsonl" >/dev/null
  fi
fi
if _want_ds "aime2025"; then
  "${PYTHON_BIN}" "${BUILD_DS_PY}" \
    --oc-root "${OC_ROOT}" --pred-abbr "${OC_PRED_ABBR}" --dataset aime2025 \
    --out-jsonl "${DATASET_DIR}/aime2025.jsonl" >/dev/null
fi

# --------------------------
# Repeat setup
# --------------------------
_repeat_gpqa="${GPQA_REPEAT:-8}"
_repeat_aime2024="${AIME2024_REPEAT:-${AIME_REPEAT:-32}}"
_repeat_aime2025="${AIME2025_REPEAT:-${AIME_REPEAT:-32}}"
_is_int "${_repeat_gpqa}" || _die "GPQA_REPEAT must be int, got: ${_repeat_gpqa}"
_is_int "${_repeat_aime2024}" || _die "AIME2024_REPEAT/AIME_REPEAT must be int, got: ${_repeat_aime2024}"
_is_int "${_repeat_aime2025}" || _die "AIME2025_REPEAT/AIME_REPEAT must be int, got: ${_repeat_aime2025}"
_is_int "${OC_JUDGE_BATCH_SIZE}" || _die "OC_JUDGE_BATCH_SIZE must be int, got: ${OC_JUDGE_BATCH_SIZE}"
_is_int "${OC_JUDGE_QPS}" || _die "OC_JUDGE_QPS must be int, got: ${OC_JUDGE_QPS}"
_is_int "${OC_JUDGE_MAX_WORKERS}" || _die "OC_JUDGE_MAX_WORKERS must be int, got: ${OC_JUDGE_MAX_WORKERS}"
_is_int "${OC_JUDGE_REPLICA_WORKERS}" || _die "OC_JUDGE_REPLICA_WORKERS must be int, got: ${OC_JUDGE_REPLICA_WORKERS}"
case "${OC_REQUIRE_JUDGE_PROMPT}" in
  0|1) ;;
  *) _die "OC_REQUIRE_JUDGE_PROMPT must be 0|1, got: ${OC_REQUIRE_JUDGE_PROMPT}" ;;
esac
echo "[repeat] GPQA_diamond=${_repeat_gpqa} aime2024=${_repeat_aime2024} aime2025=${_repeat_aime2025}"

# --------------------------
# Arm setup
# --------------------------
declare -a ARMS=()
if [[ "${MODE}" == "control" || "${MODE}" == "both" ]]; then
  case "${CONTROL_VARIANT}" in
    oc_exact) ARMS+=("control_oc_exact") ;;
    chunked) ARMS+=("control_chunked") ;;
    both) ARMS+=("control_oc_exact" "control_chunked") ;;
  esac
fi
if [[ "${MODE}" == "exp" || "${MODE}" == "both" ]]; then
  ARMS+=("exp_chunked")
fi
[[ "${#ARMS[@]}" -gt 0 ]] || _die "No arms selected (MODE=${MODE}, CONTROL_VARIANT=${CONTROL_VARIANT})"
echo "[arms] ${ARMS[*]}"

_arm_mode() {
  case "$1" in
    control_oc_exact|control_chunked) echo "control" ;;
    exp_chunked) echo "exp" ;;
    *) return 1 ;;
  esac
}

_arm_side() {
  case "$1" in
    control_oc_exact|control_chunked) echo "control" ;;
    exp_chunked) echo "exp" ;;
    *) return 1 ;;
  esac
}

_arm_token_interval() {
  case "$1" in
    control_oc_exact) echo "${EXACT_TOKEN_CHECK_INTERVAL}" ;;
    control_chunked|exp_chunked) echo "${CHUNK_TOKEN_CHECK_INTERVAL}" ;;
    *) return 1 ;;
  esac
}

_assert_run_actor_transport() {
  local merged_jsonl="$1"
  local expect_transport="$2"
  [[ -f "${merged_jsonl}" ]] || _die "Missing merged.jsonl for transport audit: ${merged_jsonl}"
  "${PYTHON_BIN}" - "${merged_jsonl}" "${expect_transport}" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
expect = sys.argv[2]
line = ""
with path.open("r", encoding="utf-8") as f:
    for raw in f:
        raw = raw.strip()
        if raw:
            line = raw
            break
if not line:
    raise SystemExit(f"empty merged jsonl: {path}")
obj = json.loads(line)
actual = str(obj.get("actor_transport") or "").strip()
if not actual:
    raise SystemExit(
        f"missing actor_transport in first record: {path}. "
        "This usually means old runner output (not OC-aligned)."
    )
if actual != expect:
    raise SystemExit(
        f"actor_transport mismatch in {path}: expect={expect}, actual={actual}"
    )
print(f"[align] actor_transport={actual} ok ({path})")
PY
}

# --------------------------
# Run + summarize
# --------------------------
declare -A RUN_DIRS=()   # key: dataset:arm

INTERVENTION_AUDIT_N="${INTERVENTION_AUDIT_N:-20}"
_is_int "${INTERVENTION_AUDIT_N}" || _die "INTERVENTION_AUDIT_N must be int"

_run_one() {
  local dataset="$1"
  local jsonl="$2"
  local repeat="$3"
  local arm="$4"
  local mode_arm side_arm interval_arm
  mode_arm="$(_arm_mode "${arm}")"
  side_arm="$(_arm_side "${arm}")"
  interval_arm="$(_arm_token_interval "${arm}")"

  [[ -f "${jsonl}" ]] || _die "Missing dataset jsonl: ${jsonl}"

  local tag="${TAG_PREFIX:-offline_verifier}_${dataset}/${arm}"
  local run_id="${RUN_GROUP_ID}_${dataset}_${arm}"

  echo ""
  echo "============================================================"
  echo "[run] dataset=${dataset} arm=${arm} mode=${mode_arm} repeat=${repeat}"
  echo "[run] out_root=${OUT_ROOT} tag=${tag} run_id=${run_id}"
  echo "[run] actor_transport=${ACTOR_TRANSPORT} actor_api_model=${ACTOR_API_MODEL:-<auto>} actor_api_mode=${ACTOR_API_MODE} actor_openai_client=${ACTOR_OPENAI_CLIENT} actor_disable_thinking=${ACTOR_DISABLE_THINKING} actor_oc_request_exact=${ACTOR_OC_REQUEST_EXACT} actor_server_pool=${ACTOR_SERVER_POOL} token_check_interval=${interval_arm} tokenizer_path=${TOKENIZER_PATH:-<base-model>} resume_from_tmp=${RESUME_FROM_TMP} do_cv=${DO_CV_EVAL_RUN}"

  OUT_ROOT="${OUT_ROOT}" TAG="${tag}" RUN_ID="${run_id}" \
  DATA_JSONL="${jsonl}" PROMPT_KEY="question" REPEAT="${repeat}" \
  MODE="${mode_arm}" TOKEN_CHECK_INTERVAL="${interval_arm}" \
  DO_CV_EVAL="${DO_CV_EVAL_RUN}" \
  ACTOR_TRANSPORT="${ACTOR_TRANSPORT}" ACTOR_API_BASE="${ACTOR_API_BASE}" ACTOR_API_KEY="${ACTOR_API_KEY}" \
  ACTOR_API_MODEL="${ACTOR_API_MODEL}" \
  ACTOR_API_MODE="${ACTOR_API_MODE}" \
  ACTOR_API_TIMEOUT="${ACTOR_API_TIMEOUT}" ACTOR_OPENAI_CLIENT="${ACTOR_OPENAI_CLIENT}" ACTOR_OC_REPO_PARENT="${ACTOR_OC_REPO_PARENT}" \
  START_ACTOR_API_SERVER="${START_ACTOR_API_SERVER}" \
  ACTOR_API_PORT="${ACTOR_API_PORT}" ACTOR_API_TP="${ACTOR_API_TP}" \
  ACTOR_API_WORKER_NUM="${ACTOR_API_WORKER_NUM}" \
  ACTOR_API_EXTRA_CLI="${ACTOR_API_EXTRA_CLI}" \
  ACTOR_SERVER_POOL="${ACTOR_SERVER_POOL}" ACTOR_API_PORT_BASE="${ACTOR_API_PORT_BASE}" \
  ACTOR_DISABLE_THINKING="${ACTOR_DISABLE_THINKING}" \
  ACTOR_OC_REQUEST_EXACT="${ACTOR_OC_REQUEST_EXACT}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  RESUME_FROM_TMP="${RESUME_FROM_TMP}" \
  bash "${RUNNER_SH}"

  local run_dir="${OUT_ROOT}/${tag}/${run_id}"
  [[ -d "${run_dir}" ]] || _die "Expected run_dir missing: ${run_dir}"
  RUN_DIRS["${dataset}:${arm}"]="${run_dir}"
  if [[ "${OC_ALIGN_STRICT}" == "1" && ( "${EVAL_METRIC_MODE}" == "oc" || "${EVAL_METRIC_MODE}" == "both" ) ]]; then
    _assert_run_actor_transport "${run_dir}/merged.jsonl" "${ACTOR_TRANSPORT}"
  fi

  # -------- Diagnostics --------
  local diag_json="${run_dir}/diag.${arm}.json"
  local audit_jsonl="${run_dir}/exp_audit.${arm}.jsonl"
  if [[ "${mode_arm}" == "exp" ]]; then
    "${PYTHON_BIN}" "${SUM_DIAG_PY}" \
      --in-jsonl "${run_dir}/merged.jsonl" \
      --mode "${mode_arm}" \
      --audit-sample-n "${INTERVENTION_AUDIT_N}" \
      --audit-out "${audit_jsonl}" \
      --out-json "${diag_json}" >/dev/null
  else
    "${PYTHON_BIN}" "${SUM_DIAG_PY}" \
      --in-jsonl "${run_dir}/merged.jsonl" \
      --mode "${mode_arm}" \
      --out-json "${diag_json}" >/dev/null
  fi

  # -------- CV metrics --------
  if [[ "${EVAL_METRIC_MODE}" == "cv" || "${EVAL_METRIC_MODE}" == "both" ]]; then
    if [[ "${DO_CV_EVAL_RUN}" == "1" ]]; then
      local cv_json="${run_dir}/cv_passk.${arm}.json"
      local cv_mode="${mode_arm}"
      "${PYTHON_BIN}" "${SUM_CV_PYK}" \
        --in-jsonl "${run_dir}/merged.cv.jsonl" \
        --mode "${cv_mode}" \
        --out-json "${cv_json}" >/dev/null
    else
      echo "[warn] skip cv summary for ${dataset}/${arm}: DO_CV_EVAL_RUN=${DO_CV_EVAL_RUN}"
    fi
  fi

  # -------- OC metrics --------
  if [[ "${EVAL_METRIC_MODE}" == "oc" || "${EVAL_METRIC_MODE}" == "both" ]]; then
    local oc_json="${run_dir}/oc_metrics.${arm}.json"
    local oc_args=(
      --in-jsonl "${run_dir}/merged.jsonl"
      --dataset "${dataset}"
      --side "${side_arm}"
      --repeat "${repeat}"
      --prompt-key "question"
      --answer-key "answer"
      --oc-repo-root "${OC_REPO_ROOT}"
      --oc-root "${OC_ROOT}"
      --oc-model-abbr "${OC_RES_ABBR}"
      --llm-judge "${OC_LLM_JUDGE}"
      --llm-judge-out-dir "${run_dir}/oc_cascade_eval"
      --require-oc-judge-prompt "${OC_REQUIRE_JUDGE_PROMPT}"
      --replica-workers "${OC_JUDGE_REPLICA_WORKERS}"
      --out-json "${oc_json}"
    )
    if [[ "${OC_JUDGE_BATCH_SIZE}" -gt 0 ]]; then
      oc_args+=(--judge-batch-size "${OC_JUDGE_BATCH_SIZE}")
    fi
    if [[ "${OC_JUDGE_QPS}" -gt 0 ]]; then
      oc_args+=(--judge-query-per-second "${OC_JUDGE_QPS}")
    fi
    if [[ "${OC_JUDGE_MAX_WORKERS}" -gt 0 ]]; then
      oc_args+=(--judge-max-workers "${OC_JUDGE_MAX_WORKERS}")
    fi
    if [[ -n "${OC_EVAL_CONFIG}" ]]; then
      oc_args+=(--oc-eval-config "${OC_EVAL_CONFIG}")
    fi
    "${OC_SUMMARY_PYTHON}" "${SUM_OC_PY}" "${oc_args[@]}" >/dev/null

    if [[ "${DO_OC_PRED_DIFF}" == "1" && -n "${OC_PRED_ABBR}" ]]; then
      local oc_diff_json="${run_dir}/oc_pred_diff.${arm}.json"
      local oc_diff_jsonl="${run_dir}/oc_pred_diff_top.${arm}.jsonl"
      if ! "${PYTHON_BIN}" "${DIFF_OC_PY}" \
        --in-jsonl "${run_dir}/merged.jsonl" \
        --dataset "${dataset}" \
        --side "${side_arm}" \
        --oc-root "${OC_ROOT}" \
        --oc-pred-abbr "${OC_PRED_ABBR}" \
        --prompt-key "question" \
        --answer-key "answer" \
        --top-diff-n "${OC_DIFF_TOP_N}" \
        --out-json "${oc_diff_json}" \
        --out-jsonl "${oc_diff_jsonl}" >/dev/null
      then
        echo "[warn] skip oc prediction diff for ${dataset}/${arm}" >&2
      fi
    fi
  fi
}

if _want_ds "GPQA_diamond"; then
  for _arm in "${ARMS[@]}"; do
    _run_one "GPQA_diamond" "${DATASET_DIR}/GPQA_diamond.jsonl" "${_repeat_gpqa}" "${_arm}"
  done
fi
if _want_ds "aime2024"; then
  for _arm in "${ARMS[@]}"; do
    _run_one "aime2024" "${DATASET_DIR}/aime2024.jsonl" "${_repeat_aime2024}" "${_arm}"
  done
fi
if _want_ds "aime2025"; then
  for _arm in "${ARMS[@]}"; do
    _run_one "aime2025" "${DATASET_DIR}/aime2025.jsonl" "${_repeat_aime2025}" "${_arm}"
  done
fi

# --------------------------
# Final report
# --------------------------
REPORT_MD="${OUT_ROOT}/report_offline_alignment_${RUN_GROUP_ID}.md"
REPORT_JSON="${OUT_ROOT}/report_offline_alignment_${RUN_GROUP_ID}.json"

declare -a REPORT_RUN_ARGS=()
for _ds in "${DATASET_ARR[@]}"; do
  for _arm in "${ARMS[@]}"; do
    _k="${_ds}:${_arm}"
    if [[ -n "${RUN_DIRS[$_k]:-}" ]]; then
      REPORT_RUN_ARGS+=("--run" "${_ds}:${_arm}=${RUN_DIRS[$_k]}")
    fi
  done
done

[[ "${#REPORT_RUN_ARGS[@]}" -gt 0 ]] || _die "No runs completed; nothing to report."

"${OC_SUMMARY_PYTHON}" "${REPORT_PY}" \
  --oc-root "${OC_ROOT}" \
  --oc-model-abbr "${OC_RES_ABBR}" \
  "${REPORT_RUN_ARGS[@]}" \
  --out-md "${REPORT_MD}" \
  --out-json "${REPORT_JSON}" >/dev/null

echo ""
echo "[done] report_md=${REPORT_MD}"
echo "[done] report_json=${REPORT_JSON}"
echo "[done] dataset_dir=${DATASET_DIR}"
echo "[done] runs:"
for _k in "${!RUN_DIRS[@]}"; do
  echo "  ${_k}=${RUN_DIRS[${_k}]}"
done
