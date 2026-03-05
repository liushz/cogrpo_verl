#!/usr/bin/env bash
# Submit N parallel 1-GPU rjob eval tasks for Co-GRPO actor(base) + verifier LoRA.
#
# This script:
# 1) Splits an input .jsonl into N shards on shared storage.
# 2) Submits N independent rjob tasks (1 GPU each). Tasks may queue independently.
# 3) Writes per-shard output jsonl files; you can merge after all jobs finish.
#
# Example (your current case):
# bash repos/repro/run_rjob_eval_cogrpo_verifier.sh \
#   --jsonl /mnt/shared-storage-user/liuhongwei/main_works/temp_debug/data/test/aime2025-I.jsonl \
#   --prompt-key question \
#   --base-model /mnt/shared-storage-user/opencompass-shared/liuhongwei/interns1/repro_exp/s1_model/interns1-8b-hf-1951 \
#   --verifier-lora /mnt/shared-storage-user/liuhongwei/main_works/temp_debug/lora_verifier_ckp/verifier-lora-cold-verifier_sft_train_55_waitx2-020418/checkpoint-915 \
#   --shards 4 \
#   --repeat 4 \
#   --out-dir /mnt/shared-storage-user/liuhongwei/main_works/temp_debug/eval/aime2025-I_rjob-0205-14 \
#   --eval \
#   --eval-urls http://100.101.166.1:22005/v1,http://100.101.166.1:22004/v1,http://100.101.166.1:22003/v1,http://100.101.166.1:22002/v1 \
#   --resume
#
# Common overrides:
#   cluster=opencompass_gpu image=... bash ... --max-response-tokens 32768 --token-check-interval 4096
set -eEo pipefail

# Avoid proxy interference when calling rjob API endpoints.
unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY

usage() {
  cat <<'EOF'
Usage:
  bash repos/repro/run_rjob_eval_cogrpo_verifier.sh --jsonl PATH --base-model PATH (--verifier-lora PATH | --verifier-model PATH) --shards N --out-dir DIR [options]

Required:
  --jsonl PATH              Input JSONL with prompt field (e.g. key "question")
  --base-model PATH         HF model dir for actor base (config.json must exist)
  --verifier-lora PATH      PEFT adapter dir (adapter_config.json must exist)
  --verifier-model PATH     Full HF verifier model dir (config.json must exist; only supported with --infer-engine lmdeploy)
  --shards N                Number of shards/jobs to submit (each job uses 1 GPU)
  --out-dir DIR             Parent output directory; a run folder "DATASET.TIMESTAMP/" will be created inside

	Options:
	  --prompt-key KEY          JSON field for prompt (default: question)
	  --resume                  Resume an existing run dir: skip completed shard inference and/or eval
	  --run-dir DIR             Explicit run dir to resume (e.g. OUT_DIR/DATASET.TIMESTAMP). If omitted with --resume, picks latest matching run.
	  --infer-engine ENGINE     Inference engine for actor/verifier pipeline: vllm|lmdeploy (default: vllm)
	  --eval                    Enable CompassVerifier tail-only evaluation and write cv_score into outputs
	  --eval-backend BACKEND    Where to run CompassVerifier eval: local|rjob (default: local)
	  --eval-urls URLS           Comma-separated CompassVerifier model URLs; required if --eval is set
	  --eval-workers N          CompassVerifier eval concurrency (passed to eval script; default: 0=auto)
	  --eval-batch-size N       CompassVerifier eval in-flight batch size (default: 64)
	  --workers N               Alias of --eval-workers (kept for convenience)
	  --64k                     Preset: set --vllm-max-model-len to 65536 (keeps other defaults unless explicitly set)
	  --align-opencompass-64k   Preset: align with older OpenCompass lmdeploy 64k-style eval (sets --64k + max tokens + system prompt)
	  --use-verifier-system-prompt  Add training-style system prompt for base actor generation (matches meta_template begin prompt)
	  --base-temperature FLOAT   Actor/base-model temperature (passed to eval script --temperature; default: 0.8)
	  --repeat N                 Repeat each sample N times (default: 1). Adds origin_info._orig_line_idx/_repeat_id and reports micro/macro averages.
	  --mode MODE               control|exp|both (default: both)
	  --job-prefix NAME         rjob name prefix (default: rjob-eval-cogrpo-verifier)
	  --cluster NAME            charged-group (default: llmit_gpu)
  --image IMAGE             container image (default: repo training image)
  --private-machine VALUE   rjob private-machine value; use "none" to omit (default: group)
  --cpu N                   CPUs per job (default: 16)
  --memory MB               Memory per job in MB (default: 180000)
  --conda-env NAME          conda env name inside container (default: oc_vllm)
  --conda-sh PATH           conda.sh path to source (optional; auto-detect if empty)
  --dry-run                 Print rjob commands only, do not submit
  --replace                 Delete existing jobs with same name before submit (off by default)
  --debug                   Enable bash xtrace and keep verbose logs
  --rdma                    Require RDMA resource (off by default; avoids unschedulable single-GPU jobs)
  --watch                   Poll job status until all finish (default interval: 300s)
	--watch-interval SECONDS  Poll interval seconds (default: 300)
	--merge-after-watch       After --watch completes, merge shard outputs into a single JSONL (default: on)
	--no-merge-after-watch    Disable auto-merge after --watch

	Eval knobs (defaults match eval script defaults / training-faithful):
	  --batch-size N                 (default: 1)
	  --max-prompt-tokens N          (default: 1024)
	  --max-response-tokens N        (default: 40960)
  --token-check-interval N       (default: 4096)
  --min-step-tokens N            (default: 4096)
  --max-interventions N          (default: 5)
  --confidence-threshold FLOAT   (default: 0.0; 0=disable)
  --wait-conf-tail-tokens N      (default: 64)
  --verifier-max-prompt-length N (default: 16384)
  --verifier-max-new-tokens N    (default: 4096)
  --verifier-max-hint-tokens N   (default: 512)
  --stop-token-id ID             (default: 151645)
	  --vllm-gpu-mem-util FLOAT      (default: 0.90)
	  --vllm-max-model-len N         (default: max_prompt_tokens+max_response_tokens if 0; here default 40960)
	  --lmdeploy-backend BACKEND     (default: pytorch; only for --infer-engine lmdeploy)
	  --lmdeploy-session-len N       (default: 65536 when --64k/--align-opencompass-64k else vllm-max-model-len; only for lmdeploy)
	  --lmdeploy-max-batch-size N    (default: 128; only for lmdeploy)
	  --lmdeploy-log-level LEVEL     (default: WARNING; only for lmdeploy)
	  --lmdeploy-use-api-server      (lmdeploy only) start `lmdeploy serve api_server` and query it (OpenAI-style); closer to OpenCompass service eval
	  --lmdeploy-api-extra-cli STR   (lmdeploy only) extra CLI for api_server (e.g. \"--backend pytorch --session-len 65536 --max-batch-size 128\")
	  --stop-at-think-end            (lmdeploy only) stop actor when '</think>' appears
	  --degeneration-guard           (lmdeploy only) stop early on obvious repetition loops
	  --seed N                       (default: 0)

	After jobs finish, merge:
	  cat RUN_DIR/results/*.jsonl > RUN_DIR/merged.jsonl
EOF
}

jsonl=""
prompt_key="question"
base_model=""
verifier_lora=""
verifier_model=""
shards=""
out_dir=""

mode="both"
resume=0
run_dir_arg=""
resume_tag=""
eval=0
	eval_backend="local"
	eval_urls=""
	eval_workers=0
	eval_batch_size=64
	base_temperature=0.8
	repeat=1
	preset_64k=0
	align_opencompass_64k=0
	use_verifier_system_prompt=0
	infer_engine="vllm"
	job_prefix="rjob-eval-cogrpo-verifier"
	cluster="${cluster:-llmit_gpu}"
	image="${image:-registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605}"
	private_machine="${private_machine:-group}"
cpu="${cpu:-16}"
memory="${memory:-180000}"
conda_env="${conda_env:-oc_vllm}"
conda_sh="${conda_sh:-}"
dry_run=0
replace=0
debug=0
rdma=0
watch=0
watch_interval=300
merge_after_watch=1

	batch_size=1
	batch_size_set=0
	max_prompt_tokens=1024
	max_prompt_tokens_set=0
	max_response_tokens=40960
	max_response_tokens_set=0
	token_check_interval=4096
	min_step_tokens=4096
	max_interventions=5
	confidence_threshold=0.0
	wait_conf_tail_tokens=64
	verifier_max_prompt_length=16384
	verifier_max_new_tokens=4096
	verifier_max_hint_tokens=512
	stop_token_id=151645
	vllm_gpu_mem_util=0.90
	vllm_max_model_len=40960
	vllm_max_model_len_set=0
	lmdeploy_backend="pytorch"
	lmdeploy_session_len=0
	lmdeploy_session_len_set=0
	lmdeploy_max_batch_size=128
	lmdeploy_log_level="WARNING"
	lmdeploy_use_api_server=0
	lmdeploy_api_extra_cli=""
	stop_at_think_end=0
	degeneration_guard=0
	seed=0

	while [[ $# -gt 0 ]]; do
	  case "$1" in
	    -h|--help) usage; exit 0 ;;
    --jsonl) jsonl="$2"; shift 2 ;;
    --prompt-key) prompt_key="$2"; shift 2 ;;
    --base-model) base_model="$2"; shift 2 ;;
    --verifier-model) verifier_model="$2"; shift 2 ;;
    --verifier-lora) verifier_lora="$2"; shift 2 ;;
    --shards) shards="$2"; shift 2 ;;
	    --out-dir) out_dir="$2"; shift 2 ;;
	    --resume) resume=1; shift ;;
	    --run-dir) run_dir_arg="$2"; shift 2 ;;
	    --infer-engine) infer_engine="$2"; shift 2 ;;
	    --eval) eval=1; shift ;;
	    --eval-backend) eval_backend="$2"; shift 2 ;;
	    --eval-in-job) eval=1; eval_backend="rjob"; shift ;; # backward-compatible alias
	    --eval-urls) eval_urls="$2"; shift 2 ;;
	    --eval-workers) eval_workers="$2"; shift 2 ;;
	    --eval-batch-size) eval_batch_size="$2"; shift 2 ;;
	    --workers) eval_workers="$2"; shift 2 ;; # alias for eval
	    --64k) preset_64k=1; shift ;;
	    --align-opencompass-64k) align_opencompass_64k=1; shift ;;
	    --use-verifier-system-prompt) use_verifier_system_prompt=1; shift ;;
	    --base-temperature) base_temperature="$2"; shift 2 ;;
	    --repeat) repeat="$2"; shift 2 ;;
	    --mode) mode="$2"; shift 2 ;;
	    --job-prefix) job_prefix="$2"; shift 2 ;;
    --cluster) cluster="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --private-machine) private_machine="$2"; shift 2 ;;
    --cpu) cpu="$2"; shift 2 ;;
    --memory) memory="$2"; shift 2 ;;
    --conda-env) conda_env="$2"; shift 2 ;;
    --conda-sh) conda_sh="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --replace) replace=1; shift ;;
    --debug) debug=1; shift ;;
    --rdma) rdma=1; shift ;;
    --watch) watch=1; shift ;;
    --watch-interval) watch_interval="$2"; shift 2 ;;
	    --merge-after-watch) merge_after_watch=1; shift ;;
	    --no-merge-after-watch) merge_after_watch=0; shift ;;
	    --batch-size) batch_size="$2"; batch_size_set=1; shift 2 ;;
	    --max-prompt-tokens) max_prompt_tokens="$2"; max_prompt_tokens_set=1; shift 2 ;;
	    --max-response-tokens) max_response_tokens="$2"; max_response_tokens_set=1; shift 2 ;;
	    --token-check-interval) token_check_interval="$2"; shift 2 ;;
	    --min-step-tokens) min_step_tokens="$2"; shift 2 ;;
	    --max-interventions) max_interventions="$2"; shift 2 ;;
	    --confidence-threshold) confidence_threshold="$2"; shift 2 ;;
	    --wait-conf-tail-tokens) wait_conf_tail_tokens="$2"; shift 2 ;;
	    --verifier-max-prompt-length) verifier_max_prompt_length="$2"; shift 2 ;;
    --verifier-max-new-tokens) verifier_max_new_tokens="$2"; shift 2 ;;
	    --verifier-max-hint-tokens) verifier_max_hint_tokens="$2"; shift 2 ;;
	    --stop-token-id) stop_token_id="$2"; shift 2 ;;
	    --vllm-gpu-mem-util) vllm_gpu_mem_util="$2"; shift 2 ;;
	    --vllm-max-model-len) vllm_max_model_len="$2"; vllm_max_model_len_set=1; shift 2 ;;
	    --lmdeploy-backend) lmdeploy_backend="$2"; shift 2 ;;
	    --lmdeploy-session-len) lmdeploy_session_len="$2"; lmdeploy_session_len_set=1; shift 2 ;;
	    --lmdeploy-max-batch-size) lmdeploy_max_batch_size="$2"; shift 2 ;;
	    --lmdeploy-log-level) lmdeploy_log_level="$2"; shift 2 ;;
	    --lmdeploy-use-api-server) lmdeploy_use_api_server=1; shift ;;
	    --lmdeploy-api-extra-cli) lmdeploy_api_extra_cli="$2"; shift 2 ;;
	    --stop-at-think-end) stop_at_think_end=1; shift ;;
	    --degeneration-guard) degeneration_guard=1; shift ;;
	    --seed) seed="$2"; shift 2 ;;
	    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
	  esac
	done

[[ -n "${jsonl}" && -n "${base_model}" && -n "${shards}" && -n "${out_dir}" ]] || { usage; exit 2; }
[[ -f "${jsonl}" ]] || { echo "Missing jsonl: ${jsonl}" >&2; exit 2; }
[[ -f "${base_model}/config.json" ]] || { echo "base-model must be a HF dir with config.json: ${base_model}" >&2; exit 2; }
# Allow empty verifier config: vLLM backend can run verifier using the base model.
# (This matches online behavior when verifier_lora_path is empty.)
if [[ -n "${verifier_lora}" ]]; then
  [[ -f "${verifier_lora}/adapter_config.json" ]] || { echo "verifier-lora must contain adapter_config.json: ${verifier_lora}" >&2; exit 2; }
fi
if [[ -n "${verifier_model}" ]]; then
  [[ -f "${verifier_model}/config.json" ]] || { echo "verifier-model must be a HF dir with config.json: ${verifier_model}" >&2; exit 2; }
fi
if [[ -n "${verifier_model}" && "${infer_engine}" != "lmdeploy" ]]; then
  echo "ERROR: --verifier-model is only supported with --infer-engine lmdeploy (current eval script limitation)." >&2
  exit 2
fi
if [[ "${eval_backend}" != "local" && "${eval_backend}" != "rjob" ]]; then
  echo "ERROR: --eval-backend must be one of: local|rjob, got: ${eval_backend}" >&2
  exit 2
fi
if [[ "${eval}" == "1" ]] && [[ -z "${eval_urls}" ]]; then
  echo "ERROR: --eval requires --eval-urls (comma-separated CompassVerifier model URLs)." >&2
  exit 2
fi
if ! [[ "${eval_workers}" =~ ^[0-9]+$ ]] || [[ "${eval_workers}" -lt 0 ]]; then
  echo "--eval-workers must be a non-negative int, got: ${eval_workers}" >&2
  exit 2
fi
	if ! [[ "${eval_batch_size}" =~ ^[0-9]+$ ]] || [[ "${eval_batch_size}" -le 0 ]]; then
	  echo "--eval-batch-size must be a positive int, got: ${eval_batch_size}" >&2
	  exit 2
	fi
	if ! [[ "${repeat}" =~ ^[0-9]+$ ]] || [[ "${repeat}" -le 0 ]]; then
	  echo "--repeat must be a positive int, got: ${repeat}" >&2
	  exit 2
	fi
if [[ "${repeat}" -gt 1 ]] && [[ "${seed}" != "0" ]]; then
  echo "[WARN] --repeat=${repeat} with --seed=${seed} may produce identical repeats (deterministic sampling). Use --seed 0 for stochastic repeats." >&2
fi

	if ! [[ "${shards}" =~ ^[0-9]+$ ]] || [[ "${shards}" -le 0 ]]; then
	  echo "--shards must be a positive int, got: ${shards}" >&2
	  exit 2
	fi

	if [[ "${infer_engine}" != "vllm" && "${infer_engine}" != "lmdeploy" ]]; then
	  echo "ERROR: --infer-engine must be one of: vllm|lmdeploy, got: ${infer_engine}" >&2
	  exit 2
	fi
	if [[ "${infer_engine}" == "lmdeploy" ]] && [[ "${mode}" != "both" ]]; then
	  # lmdeploy backend supports control/exp/both at python level, but this launcher is mainly validated for both.
	  true
	fi

	if [[ "${align_opencompass_64k}" == "1" ]]; then
	  preset_64k=1
	  use_verifier_system_prompt=1
	  if [[ "${max_prompt_tokens_set}" != "1" ]]; then
	    max_prompt_tokens=65536
	  fi
	  if [[ "${max_response_tokens_set}" != "1" ]]; then
	    max_response_tokens=65536
	  fi
	fi

	if [[ "${preset_64k}" == "1" ]] && [[ "${vllm_max_model_len_set}" != "1" ]]; then
	  vllm_max_model_len=65536
	fi
	if [[ "${infer_engine}" == "lmdeploy" ]] && [[ "${lmdeploy_session_len_set}" != "1" ]]; then
	  if [[ "${preset_64k}" == "1" ]]; then
	    lmdeploy_session_len=65536
	  else
	    lmdeploy_session_len="${vllm_max_model_len}"
	  fi
	fi

	mkdir -p "${out_dir}/shards" "${out_dir}/results" "${out_dir}/logs"

base_name="$(basename "${jsonl}")"
base_stem="${base_name%.jsonl}"

if [[ "${resume}" == "1" ]]; then
  if [[ -n "${run_dir_arg}" ]]; then
    run_dir="${run_dir_arg}"
  else
    # Best-effort: pick latest run under out_dir with matching dataset stem.
    run_dir="$(ls -dt "${out_dir}/${base_stem}".* 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "ERROR: --resume requires an existing run dir. Provide --run-dir or ensure a prior run exists under: ${out_dir}/${base_stem}.*" >&2
    exit 2
  fi
  run_dir="$(cd "${run_dir}" && pwd)"
  run_dir_base="$(basename "${run_dir}")"
  if [[ "${run_dir_base}" == "${base_stem}."* ]]; then
    run_id="${run_dir_base#${base_stem}.}"
  else
    run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
  fi
  # Avoid rjob name collisions with the previous run's shard jobs.
  resume_tag="r$(date +%m%d%H%M%S)"
else
  run_id="$(date +%Y%m%d_%H%M%S)_${RANDOM}"
  run_dir="${out_dir}/${base_stem}.${run_id}"
  mkdir -p "${run_dir}"
fi

shard_dir="${run_dir}/shards"
mkdir -p "${shard_dir}"
mkdir -p "${run_dir}/results" "${run_dir}/logs"

job_list_file="${run_dir}/jobs.showname.txt"
job_map_file="${run_dir}/jobs.map.tsv"
if [[ "${resume}" == "1" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  job_list_file="${run_dir}/jobs.resume.${ts}.showname.txt"
  job_map_file="${run_dir}/jobs.resume.${ts}.map.tsv"
fi
: > "${job_list_file}"
: > "${job_map_file}"

meta_file="${run_dir}/meta.json"
META_FILE="${meta_file}" JSONL="${jsonl}" BASE_MODEL="${base_model}" VERIFIER_MODEL="${verifier_model}" VERIFIER_LORA="${verifier_lora}" PROMPT_KEY="${prompt_key}" SHARDS="${shards}" REPEAT="${repeat}" MODE="${mode}" EVAL="${eval}" RUN_ID="${run_id}" INFER_ENGINE="${infer_engine}" python3 - <<'PY'
import json
import os
from pathlib import Path

meta = Path(os.environ["META_FILE"])
data = {
  "jsonl": os.environ["JSONL"],
  "base_model": os.environ["BASE_MODEL"],
  "verifier_model": os.environ.get("VERIFIER_MODEL") or "",
  "verifier_lora": os.environ["VERIFIER_LORA"],
  "prompt_key": os.environ["PROMPT_KEY"],
  "shards": int(os.environ["SHARDS"]),
  "repeat": int(os.environ["REPEAT"]),
  "infer_engine": os.environ.get("INFER_ENGINE", "vllm"),
  "mode": os.environ["MODE"],
  "eval": bool(int(os.environ["EVAL"])),
  "run_id": os.environ["RUN_ID"],
}

if meta.exists():
  try:
    old = json.loads(meta.read_text(encoding="utf-8"))
  except Exception:
    old = {}
  # Hard guards for resume safety.
  for k in ("jsonl", "base_model", "verifier_model", "verifier_lora", "prompt_key", "shards", "repeat", "infer_engine"):
    if k in old and str(old.get(k)) != str(data.get(k)):
      raise SystemExit(f"meta mismatch for {k}: meta={old.get(k)!r} != current={data.get(k)!r}. Use a new out-dir or correct args.")
else:
  meta.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "Run dir: ${run_dir}"
if [[ "${resume}" != "1" ]] || [[ ! -f "${shard_dir}/shard_00000.jsonl" ]]; then
  echo "Splitting ${jsonl} -> ${shard_dir} (N=${shards}, repeat=${repeat})"
  JSONL="${jsonl}" SHARD_DIR="${shard_dir}" SHARDS="${shards}" REPEAT="${repeat}" python3 - <<'PY'
import os
import json

src = os.environ["JSONL"]
out_dir = os.environ["SHARD_DIR"]
n = int(os.environ["SHARDS"])
repeat = int(os.environ.get("REPEAT", "1"))

paths = [os.path.join(out_dir, f"shard_{i:05d}.jsonl") for i in range(n)]
fps = [open(p, "w", encoding="utf-8") for p in paths]
try:
    with open(src, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # Keep raw line if it's not JSON; still shard deterministically.
                obj = {"_raw": line}
            for r in range(max(1, repeat)):
                if isinstance(obj, dict):
                    o = dict(obj)
                    origin = o.get("origin_info")
                    # Keep origin_info as-is if user provided; but add repeat metadata at top-level for ease of access.
                    o["_orig_line_idx"] = idx
                    o["_repeat_id"] = r
                    # Also mirror into origin_info if origin_info is dict, so downstream scripts can use it.
                    if isinstance(origin, dict):
                        origin = dict(origin)
                        origin["_orig_line_idx"] = idx
                        origin["_repeat_id"] = r
                        o["origin_info"] = origin
                else:
                    o = {"value": obj, "_orig_line_idx": idx, "_repeat_id": r}
                fps[idx % n].write(json.dumps(o, ensure_ascii=False) + "\n")
finally:
    for fp in fps:
        fp.close()

print("OK shards:")
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        sz = sum(1 for _ in f)
    print(f"  {p} ({sz} lines)")
PY
else
  echo "[resume] Using existing shards under: ${shard_dir}"
fi

	hf_cache_dir="/mnt/shared-storage-user/opencompass-shared/model_weights/hf_hub"
	hf_datasets_cache="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/hf_datasets_cache"
	eval_py_vllm="/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/eval_co_grpo_with_verifier_v2.py"
	eval_py_lmdeploy="/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/eval_co_grpo_with_verifier_lmdeploy64k.py"
	cv_eval_py="/mnt/shared-storage-user/liuhongwei/main_works/repos/repro/scripts/compassverifier_eval_tail_jsonl.py"

_infer_done() {
  local out_jsonl=$1
  local shard_file=$2
  local mode=$3
  python3 - "$out_jsonl" "$shard_file" "$mode" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
shard_path = Path(sys.argv[2])
mode = sys.argv[3]

if not shard_path.exists():
  sys.exit(2)
expected = 0
with shard_path.open("r", encoding="utf-8") as f:
  for line in f:
    if line.strip():
      expected += 1
if expected == 0:
  # Empty shard: treat as already complete (avoid submitting a job that will fail on "no prompts").
  sys.exit(0)

if not out_path.exists():
  sys.exit(1)

required = []
if mode in ("control", "both"):
  required.append("control")
if mode in ("exp", "both"):
  required.append("exp")

n = 0
try:
  with out_path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      n += 1
      obj = json.loads(line)
      for k in required:
        if k not in obj or not isinstance(obj.get(k), dict):
          sys.exit(1)
except Exception:
  sys.exit(1)

sys.exit(0 if (expected > 0 and n == expected) else 1)
PY
}

_eval_done() {
  local out_jsonl=$1
  local mode=$2
  python3 - "$out_jsonl" "$mode" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
mode = sys.argv[2]

if not out_path.exists():
  sys.exit(1)

required = []
if mode in ("control", "both"):
  required.append("control")
if mode in ("exp", "both"):
  required.append("exp")

try:
  with out_path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      for k in required:
        side = obj.get(k)
        if not isinstance(side, dict):
          sys.exit(1)
        # Accept either score or explicit error as "done".
        if "cv_score" not in side and "cv_error" not in side:
          sys.exit(1)
except Exception:
  sys.exit(1)

sys.exit(0)
PY
}

_merged_eval_done() {
  local merged_jsonl=$1
  local mode=$2
  python3 - "$merged_jsonl" "$mode" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
mode = sys.argv[2]

if not out_path.exists():
  sys.exit(1)

required = []
if mode in ("control", "both"):
  required.append("control")
if mode in ("exp", "both"):
  required.append("exp")

try:
  with out_path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      obj = json.loads(line)
      for k in required:
        side = obj.get(k)
        if not isinstance(side, dict):
          sys.exit(1)
        if "cv_score" not in side and "cv_error" not in side:
          sys.exit(1)
      # Only need to validate one non-empty record.
      sys.exit(0)
except Exception:
  sys.exit(1)

sys.exit(1)
PY
}

_merge_results() {
  local run_dir=$1
  local shards=$2
  local merged_jsonl=$3
  echo "Merging shard outputs -> ${merged_jsonl}"
  RUN_DIR="${run_dir}" SHARDS="${shards}" MERGED="${merged_jsonl}" python3 - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
shards = int(os.environ["SHARDS"])

records = []
for shard_id in range(shards):
    p = run_dir / "results" / f"shard_{shard_id:05d}.jsonl"
    if not p.exists():
        raise SystemExit(f"missing shard output: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            shard_local_idx = obj.get("idx", line_no)
            obj["_shard_id"] = shard_id
            obj["_shard_local_idx"] = shard_local_idx
            # Prefer true original ordering if available (repeat-aware).
            origin = obj.get("origin_info") or {}
            try:
                orig_line_idx = int(origin.get("_orig_line_idx", obj.get("_orig_line_idx", -1)))
            except Exception:
                orig_line_idx = -1
            try:
                repeat_id = int(origin.get("_repeat_id", obj.get("_repeat_id", 0)))
            except Exception:
                repeat_id = 0
            if orig_line_idx >= 0:
                obj["_orig_idx"] = orig_line_idx
                obj["_repeat_id"] = repeat_id
            else:
                obj["_orig_idx"] = shard_local_idx * shards + shard_id
            records.append(obj)

records.sort(key=lambda x: (x.get("_orig_idx", 0), x.get("_repeat_id", 0), x.get("_shard_id", 0), x.get("_shard_local_idx", 0)))
merged = Path(os.environ.get("MERGED", str(run_dir / "merged.jsonl")))
with merged.open("w", encoding="utf-8") as w:
    for obj in records:
        w.write(json.dumps(obj, ensure_ascii=False) + "\n")

errs = [r for r in records if r.get("exp", {}).get("error")]
total_interventions = sum(r.get("exp", {}).get("num_interventions", 0) for r in records)
rt = [r.get("exp", {}).get("response_tokens_total", 0) for r in records]
if rt:
    rmin, ravg, rmax = min(rt), sum(rt) / len(rt), max(rt)
else:
    rmin = ravg = rmax = 0

print(f"merged_records={len(records)} errors={len(errs)} total_interventions={total_interventions} response_tokens_total(min/avg/max)={rmin}/{ravg:.2f}/{rmax}")
print(f"merged_path={merged}")
PY
}

_run_local_eval_on_merged() {
  local merged_jsonl=$1
  # Ensure internal model URLs bypass proxies.
  local hosts="localhost,127.0.0.1"
  if [[ -n "${eval_urls}" ]]; then
    local u
    IFS=',' read -ra _parts <<< "${eval_urls}"
    for u in "${_parts[@]}"; do
      u="${u#http://}"
      u="${u#https://}"
      u="${u%%/*}"
      u="${u%%:*}"
      if [[ -n "${u}" ]]; then
        hosts="${hosts},${u}"
      fi
    done
  fi
  export NO_PROXY="${hosts}${NO_PROXY:+,${NO_PROXY}}"
  export no_proxy="${NO_PROXY}"
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
  echo "[local_eval] NO_PROXY=${NO_PROXY}"

  echo "[local_eval] running CompassVerifier tail-only eval on: ${merged_jsonl}"
  tmp_out="$(mktemp "${merged_jsonl}.cv.XXXXXX")"
  PYTHONPATH="/mnt/shared-storage-user/liuhongwei/main_works/repos/repro:${PYTHONPATH:-}" python3 "${cv_eval_py}" \
    --in-jsonl "${merged_jsonl}" \
    --out-jsonl "${tmp_out}" \
    --prompt-key "${prompt_key}" \
    --mode "${mode}" \
    --eval-urls "${eval_urls}" \
    --workers "${eval_workers}" \
    --batch-size "${eval_batch_size}" \
    --progress
  mv -f "${tmp_out}" "${merged_jsonl}"
  echo "[local_eval] updated -> ${merged_jsonl}"
}

_sync_shards_from_merged() {
  local run_dir=$1
  local shards=$2
  local merged_jsonl=$3
  RUN_DIR="${run_dir}" SHARDS="${shards}" MERGED="${merged_jsonl}" python3 - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
shards = int(os.environ["SHARDS"])
merged = Path(os.environ["MERGED"])
out_dir = run_dir / "results"
out_dir.mkdir(parents=True, exist_ok=True)

groups = {i: [] for i in range(shards)}
with merged.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        try:
            sid = int(obj.get("_shard_id", -1))
        except Exception:
            sid = -1
        if sid < 0 or sid >= shards:
            # Best-effort: ignore records without shard annotations.
            continue
        try:
            sli = int(obj.get("_shard_local_idx", line_no))
        except Exception:
            sli = line_no
        obj["_shard_local_idx"] = sli
        groups[sid].append(obj)

for sid in range(shards):
    recs = groups.get(sid) or []
    recs.sort(key=lambda x: x.get("_shard_local_idx", 0))
    tmp = out_dir / f"shard_{sid:05d}.jsonl.tmp"
    final = out_dir / f"shard_{sid:05d}.jsonl"
    with tmp.open("w", encoding="utf-8") as w:
        for obj in recs:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(final)
print(f"[local_eval] synced {shards} shard result files from merged -> {out_dir}")
PY
}

_print_cv_summary() {
  local merged_jsonl=$1
  echo "CompassVerifier eval enabled; summary from merged.jsonl (cv_score fields):"
  MERGED="${merged_jsonl}" python3 - <<'PY'
import json
import os
from collections import defaultdict

p = os.environ["MERGED"]
c = []
e = []
per_c = defaultdict(list)
per_e = defaultdict(list)
with open(p, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        origin = r.get("origin_info") or {}
        try:
            k = int(origin.get("_orig_line_idx", r.get("_orig_line_idx", -1)))
        except Exception:
            k = -1
        cc = (r.get("control") or {}).get("cv_score")
        ee = (r.get("exp") or {}).get("cv_score")
        if isinstance(cc, (int, float)):
            c.append(float(cc))
            if k >= 0:
                per_c[k].append(float(cc))
        if isinstance(ee, (int, float)):
            e.append(float(ee))
            if k >= 0:
                per_e[k].append(float(ee))
def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0
macro_c = avg([avg(v) for v in per_c.values()]) if per_c else 0.0
macro_e = avg([avg(v) for v in per_e.values()]) if per_e else 0.0
print(
    json.dumps(
        {
            "control_scored": len(c),
            "control_micro_acc": avg(c),
            "control_macro_acc": macro_c,
            "exp_scored": len(e),
            "exp_micro_acc": avg(e),
            "exp_macro_acc": macro_e,
            "delta_micro": avg(e) - avg(c),
            "delta_macro": macro_e - macro_c,
        },
        ensure_ascii=False,
    )
)
PY
}

_job_status_summary() {
  local list_file=$1
  local total=0
  local running=0
  local waiting=0
  local succeed=0
  local failed=0
  local stopped=0
  local unknown=0
  local fatal=0
  local examples=()
  local fatal_examples=()

  _job_has_fatal_event() {
    local showname=$1
    local job_name=""
    local out=""

    out="$(rjob list --name "${showname}" 2>&1 || true)"
    job_name="$(echo "${out}" | awk 'index($0, "(showname=")>0 {print $5; exit}')"
    if [[ -z "${job_name}" ]]; then
      return 1
    fi
    ev="$(rjob events "${job_name}" 2>&1 || true)"
    if echo "${ev}" | grep -qiE "insufficient project quota|does not pass quotacheck|has no right to operate|forbidden|denied the request"; then
      echo "${ev}" | tail -n 6 >&2
      return 0
    fi
    return 1
  }

  while IFS= read -r showname; do
    [[ -z "${showname}" ]] && continue
    total=$((total + 1))
    out="$(rjob get --name "${showname}" 2>&1 || true)"
    if echo "${out}" | grep -qiE "no such|yielded nothing|not found"; then
      unknown=$((unknown + 1))
      continue
    fi
    main_line="$(echo "${out}" | head -n 1)"
    state="$(echo "${main_line}" | sed -E 's/.*: ([A-Za-z]+).*/\1/')"
    replica_line="$(echo "${out}" | grep -m 1 -E "replica .*: " || true)"
    replica_state=""
    if [[ -n "${replica_line}" ]]; then
      replica_state="$(echo "${replica_line}" | sed -E 's/.*: ([A-Za-z_]+).*/\1/')"
    fi

    case "${state}" in
      Running|Active)
        running=$((running + 1))
        ;;
      Inqueue|Starting|Pending)
        waiting=$((waiting + 1))
        if _job_has_fatal_event "${showname}"; then
          fatal=1
          failed=$((failed + 1))
          waiting=$((waiting - 1))
          if [[ ${#fatal_examples[@]} -lt 3 ]]; then
            fatal_examples+=("${showname}:${state}")
          fi
        fi
        ;;
      Succeed|Succeeded)
        succeed=$((succeed + 1))
        ;;
      Failed)
        failed=$((failed + 1))
        ;;
      Stopped)
        stopped=$((stopped + 1))
        ;;
      *)
        unknown=$((unknown + 1))
        ;;
    esac

    if [[ ${#examples[@]} -lt 4 ]] && ([[ "${state}" == "Inqueue" ]] || [[ "${state}" == "Starting" ]] || [[ "${state}" == "Pending" ]] || [[ "${state}" == "Running" ]]); then
      if [[ -n "${replica_state}" ]]; then
        examples+=("${showname}:${state}/${replica_state}")
      else
        examples+=("${showname}:${state}")
      fi
    fi
  done < "${list_file}"

  ts="$(date +'%Y-%m-%d %H:%M:%S')"
  echo "[${ts}] running task ${running}, waiting task ${waiting}, succeed ${succeed}, failed ${failed}, stopped ${stopped}, unknown ${unknown} (total ${total}) ${examples[*]-}"
  if [[ "${fatal}" == "1" ]]; then
    echo "[FATAL] Some jobs cannot be scheduled due to quota/permission errors: ${fatal_examples[*]-}" >&2
    return 2
  fi
  if [[ "${total}" -gt 0 ]] && [[ $((succeed + failed + stopped)) -ge "${total}" ]]; then
    return 0
  fi
  return 1
}

submitted=0
for i in $(seq 0 $((shards - 1))); do
  shard_file="${shard_dir}/shard_$(printf '%05d' "${i}").jsonl"
  out_jsonl="${run_dir}/results/shard_$(printf '%05d' "${i}").jsonl"
  log_file="${run_dir}/logs/shard_$(printf '%05d' "${i}").log"
  if [[ ! -s "${shard_file}" ]]; then
    echo "[skip] shard $(printf '%05d' "${i}") is empty; skip submit."
    : > "${out_jsonl}"
    continue
  fi
  if [[ -n "${resume_tag}" ]]; then
    job_name="${job_prefix}-${run_id}-${resume_tag}-$(printf '%05d' "${i}")"
  else
    job_name="${job_prefix}-${run_id}-$(printf '%05d' "${i}")"
  fi

  if [[ "${resume}" == "1" ]]; then
    if _infer_done "${out_jsonl}" "${shard_file}" "${mode}"; then
      if [[ "${eval}" != "1" ]] || [[ "${eval_backend}" == "local" ]] || _eval_done "${out_jsonl}" "${mode}"; then
        echo "[resume] shard $(printf '%05d' "${i}") already done; skip submit."
        continue
      fi
    fi
  fi

  if [[ "${replace}" == "1" ]]; then
    rjob delete --name "${job_name}" --force-all 2>/dev/null || true
  fi

  job_cmd=$(cat <<EOF
		set -eEo pipefail
		mkdir -p '${run_dir}' '${run_dir}/results' '${run_dir}/logs'
		LOG_FILE='${log_file}'
	# Capture ALL logs (including conda activation) to shared storage.
	exec > >(tee -a "\${LOG_FILE}") 2>&1
	if [[ "${debug}" == "1" ]]; then set -x; fi

	echo "[env] hostname=\$(hostname) pwd=\$(pwd) date=\$(date)"
	if [[ "${conda_env}" == "none" ]]; then
	  echo "[env] skipping conda activation (--conda-env none)"
	  python3 -V || true
	else
	  if [[ -n "${conda_sh}" ]]; then
	    if [[ -f "${conda_sh}" ]]; then
	      # shellcheck disable=SC1090
	      source "${conda_sh}"
	    else
	      echo "ERROR: --conda-sh provided but not found: ${conda_sh}" >&2
	      exit 2
	    fi
	  elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
	    source /opt/conda/etc/profile.d/conda.sh
	  elif [[ -f /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh ]]; then
	    source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh
	  elif command -v conda >/dev/null 2>&1; then
	    eval "\$(conda shell.bash hook)"
	  else
	    echo "ERROR: conda not found; set --conda-sh to the correct conda.sh in the image, or use --conda-env none." >&2
	    exit 2
	  fi
	  conda activate "${conda_env}"
	  echo "[env] conda_env=${conda_env}"
	  conda --version || true
	  python3 -V || true
	fi

# ---- Proxy hygiene (bypass proxies for internal model URLs) ----
_setup_no_proxy() {
  local urls="${1:-}"
  local hosts="localhost,127.0.0.1"
  if [[ -n "${urls}" ]]; then
    local u
    IFS=',' read -ra _parts <<< "${urls}"
    for u in "${_parts[@]}"; do
      u="${u#http://}"
      u="${u#https://}"
      u="${u%%/*}"
      u="${u%%:*}"
      if [[ -n "${u}" ]]; then
        hosts="${hosts},${u}"
      fi
    done
  fi
  export NO_PROXY="${hosts}${NO_PROXY:+,${NO_PROXY}}"
  export no_proxy="${NO_PROXY}"
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
  echo "[env] NO_PROXY=${NO_PROXY}"
}
_setup_no_proxy '${eval_urls}'

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=${hf_cache_dir}
export HF_HUB_CACHE=${hf_cache_dir}
export HUGGINGFACE_HUB_CACHE=${hf_cache_dir}
export HF_DATASETS_CACHE=${hf_datasets_cache}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1
	if [[ "${debug}" != "1" ]]; then
	  export VLLM_LOGGING_LEVEL=\${VLLM_LOGGING_LEVEL:-WARN}
	  export VERL_LOGGING_LEVEL=\${VERL_LOGGING_LEVEL:-WARN}
	  export RAY_LOGGING_LEVEL=\${RAY_LOGGING_LEVEL:-INFO}
	fi

	OUT_JSONL='${out_jsonl}'
	SHARD_FILE='${shard_file}'
	MODE='${mode}'
		EVAL='${eval}'
		EVAL_BACKEND='${eval_backend}'
		RESUME='${resume}'

		_infer_done_py() {
		  python3 - "\${OUT_JSONL}" "\${SHARD_FILE}" "\${MODE}" <<'PY'
import json
import sys
from pathlib import Path
out_path = Path(sys.argv[1])
shard_path = Path(sys.argv[2])
mode = sys.argv[3]
expected = 0
with shard_path.open("r", encoding="utf-8") as f:
  for line in f:
    if line.strip():
      expected += 1
if expected == 0:
  sys.exit(0)
if not out_path.exists():
  sys.exit(1)
required = []
if mode in ("control", "both"):
  required.append("control")
if mode in ("exp", "both"):
  required.append("exp")
n = 0
with out_path.open("r", encoding="utf-8") as f:
  for line in f:
    line = line.strip()
    if not line:
      continue
    n += 1
    obj = json.loads(line)
    for k in required:
      if k not in obj or not isinstance(obj.get(k), dict):
        sys.exit(1)
sys.exit(0 if (expected > 0 and n == expected) else 1)
PY
		}

	_eval_done_py() {
	  python3 - "\${OUT_JSONL}" "\${MODE}" <<'PY'
import json
import sys
from pathlib import Path
out_path = Path(sys.argv[1])
mode = sys.argv[2]
if not out_path.exists():
  sys.exit(1)
required = []
if mode in ("control", "both"):
  required.append("control")
if mode in ("exp", "both"):
  required.append("exp")
with out_path.open("r", encoding="utf-8") as f:
  for line in f:
    line = line.strip()
    if not line:
      continue
    obj = json.loads(line)
    for k in required:
      side = obj.get(k)
      if not isinstance(side, dict):
        sys.exit(1)
      if "cv_score" not in side and "cv_error" not in side:
        sys.exit(1)
sys.exit(0)
PY
	}

			_run_infer() {
				  tmp_out="\$(mktemp '${out_jsonl}.infer.XXXXXX')"
				  if [[ '${infer_engine}' == 'lmdeploy' ]]; then
				    export LMDEPLOY_SKIP_WARMUP=\${LMDEPLOY_SKIP_WARMUP:-1}
				    python3 '${eval_py_lmdeploy}' \
				      --base-model '${base_model}' \
				      $(if [[ -n "${verifier_model}" ]]; then echo "--verifier-model '${verifier_model}'"; fi) \
				      $(if [[ -n "${verifier_lora}" ]]; then echo "--verifier-lora '${verifier_lora}'"; fi) \
				      --prompts-file '${shard_file}' \
				      --prompt-key '${prompt_key}' \
				      --out-jsonl "\${tmp_out}" \
				      --mode '${mode}' \
				      $(if [[ "${lmdeploy_use_api_server}" == "1" ]]; then echo "--actor-transport openai --start-actor-api-server"; fi) \
				      $(if [[ -n "${lmdeploy_api_extra_cli}" ]]; then echo "--actor-api-extra-cli '${lmdeploy_api_extra_cli}'"; fi) \
				      $(if [[ "${use_verifier_system_prompt}" == "1" ]]; then echo "--use-verifier-system-prompt"; fi) \
				      $(if [[ "${stop_at_think_end}" == "1" ]]; then echo "--stop-at-think-end"; fi) \
				      $(if [[ "${degeneration_guard}" == "1" ]]; then echo "--degeneration-guard"; fi) \
				      --temperature '${base_temperature}' \
				      --max-prompt-tokens '${max_prompt_tokens}' \
				      --max-response-tokens '${max_response_tokens}' \
				      --token-check-interval '${token_check_interval}' \
				      --min-step-tokens '${min_step_tokens}' \
			      --max-interventions '${max_interventions}' \
			      --confidence-threshold '${confidence_threshold}' \
			      --wait-conf-tail-tokens '${wait_conf_tail_tokens}' \
			      --verifier-max-prompt-length '${verifier_max_prompt_length}' \
			      --verifier-max-new-tokens '${verifier_max_new_tokens}' \
			      --verifier-max-hint-tokens '${verifier_max_hint_tokens}' \
			      --stop-token-id '${stop_token_id}' \
			      --lmdeploy-backend '${lmdeploy_backend}' \
			      --lmdeploy-session-len '${lmdeploy_session_len}' \
			      --lmdeploy-max-batch-size '${lmdeploy_max_batch_size}' \
			      --lmdeploy-log-level '${lmdeploy_log_level}' \
			      --progress
			  else
			    python3 '${eval_py_vllm}' \
			      --backend vllm \
			      --mode '${mode}' \
			      --base-model '${base_model}' \
			      $(if [[ -n "${verifier_lora}" ]]; then echo "--verifier-lora '${verifier_lora}'"; fi) \
			      --prompts-file '${shard_file}' \
			      --prompt-key '${prompt_key}' \
			      --out-jsonl "\${tmp_out}" \
			      $(if [[ "${use_verifier_system_prompt}" == "1" ]]; then echo "--use-verifier-system-prompt"; fi) \
			      --temperature '${base_temperature}' \
			      --batch-size '${batch_size}' \
			      --max-prompt-tokens '${max_prompt_tokens}' \
			      --max-response-tokens '${max_response_tokens}' \
			      --token-check-interval '${token_check_interval}' \
			      --min-step-tokens '${min_step_tokens}' \
			      --max-interventions '${max_interventions}' \
			      --confidence-threshold '${confidence_threshold}' \
			      --wait-conf-tail-tokens '${wait_conf_tail_tokens}' \
			      --verifier-max-prompt-length '${verifier_max_prompt_length}' \
			      --verifier-max-new-tokens '${verifier_max_new_tokens}' \
			      --verifier-max-hint-tokens '${verifier_max_hint_tokens}' \
			      --stop-token-id '${stop_token_id}' \
			      --vllm-tp 1 \
			      --vllm-gpu-mem-util '${vllm_gpu_mem_util}' \
			      --vllm-max-model-len '${vllm_max_model_len}' \
			      --seed '${seed}' \
			      --progress
			  fi
			  mv -f "\${tmp_out}" "\${OUT_JSONL}"
			}

		_run_eval() {
		  echo "[cv_eval] running CompassVerifier tail-only eval..."
		  tmp_out="\$(mktemp '${out_jsonl}.cv.XXXXXX')"
		  python3 '${cv_eval_py}' \
		    --in-jsonl "\${OUT_JSONL}" \
		    --out-jsonl "\${tmp_out}" \
		    --prompt-key '${prompt_key}' \
		    --mode '${mode}' \
		    --eval-urls '${eval_urls}' \
		    --workers '${eval_workers}' \
		    --batch-size '${eval_batch_size}' \
		    --progress
		  mv -f "\${tmp_out}" "\${OUT_JSONL}"
		  echo "[cv_eval] updated -> \${OUT_JSONL}"
		}

		if [[ "\${RESUME}" == "1" ]] && _infer_done_py; then
		  echo "[resume] infer already complete: \${OUT_JSONL}"
		  if [[ "\${EVAL}" != "1" ]] || [[ "\${EVAL_BACKEND}" != "rjob" ]]; then
		    exit 0
		  fi
		  if _eval_done_py; then
		    echo "[resume] eval already complete: \${OUT_JSONL}"
		    exit 0
		  fi
		  echo "[resume] infer complete but eval missing; run eval only."
		  _run_eval
		  exit 0
		fi

		_run_infer
		if [[ "\${EVAL}" == "1" ]] && [[ "\${EVAL_BACKEND}" == "rjob" ]]; then
		  _run_eval
		fi
EOF
)

  echo "Submitting ${job_name} (shard ${i}/${shards})"
  echo "${job_name}" >> "${job_list_file}"
  submitted=$((submitted + 1))
  if [[ "${dry_run}" == "1" ]]; then
    echo "rjob submit --name='${job_name}' --gpu=1 --cpu='${cpu}' --memory='${memory}' --charged-group='${cluster}' --image='${image}' ..."
    printf "%s\t%s\t%s\t%s\t%s\n" "$(printf '%05d' "${i}")" "${job_name}" "" "${out_jsonl}" "${log_file}" >> "${job_map_file}"
  else
    extra_args=()
    if [[ "${rdma}" == "1" ]]; then
      extra_args+=(--custom-resources "rdma/mlnx_shared=1")
    fi
    if [[ "${private_machine}" != "none" && -n "${private_machine}" ]]; then
      extra_args+=(--private-machine "${private_machine}")
    fi
    submit_out="$(
      rjob submit \
      --name="${job_name}" \
      --gpu=1 \
      --cpu="${cpu}" \
      --memory="${memory}" \
      --charged-group="${cluster}" \
      --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
      --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
      --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
      --image="${image}" \
      --host-network=true \
      -P 1 \
      -e DISTRIBUTED_JOB=true \
      "${extra_args[@]}" \
      -- bash -lc "${job_cmd}" \
    )"
    echo "${submit_out}"
    created_name="$(echo "${submit_out}" | sed -nE 's/.*created rjob_name: ([^[:space:]]+).*/\1/p' | tail -n 1)"
    if [[ -n "${created_name}" ]]; then
      printf "%s\t%s\t%s\t%s\t%s\n" "$(printf '%05d' "${i}")" "${job_name}" "${created_name}" "${out_jsonl}" "${log_file}" >> "${job_map_file}"
    else
      # Keep a row anyway so users can still locate the job by showname.
      printf "%s\t%s\t%s\t%s\t%s\n" "$(printf '%05d' "${i}")" "${job_name}" "" "${out_jsonl}" "${log_file}" >> "${job_map_file}"
    fi
  fi
done

echo "Submitted ${submitted} job(s); they may queue independently."
if [[ "${resume}" == "1" ]]; then
  echo "[resume] submitted ${submitted} job(s) (skipped the rest as already complete)."
fi
echo "Outputs:"
echo "  Run dir: ${run_dir}"
echo "  Results: ${run_dir}/results/"
echo "  Logs:    ${run_dir}/logs/"
echo "  Jobs:    ${job_list_file}"
echo "  Job map: ${job_map_file}"
echo "Merge after all jobs finish:"
echo "  cat '${run_dir}/results/'*.jsonl > '${run_dir}/merged.jsonl'"

if [[ "${watch}" == "1" ]]; then
  if ! [[ "${watch_interval}" =~ ^[0-9]+$ ]] || [[ "${watch_interval}" -le 0 ]]; then
    echo "--watch-interval must be a positive int seconds, got: ${watch_interval}" >&2
    exit 2
  fi
  if [[ "${dry_run}" == "1" ]]; then
    echo "--dry-run set; skipping --watch polling."
    exit 0
  fi
  if [[ "${submitted}" -le 0 ]]; then
    echo "[resume] No jobs submitted; skipping --watch polling."
  else
  echo "Watching jobs every ${watch_interval}s until completion..."
  while true; do
    # NOTE: this script runs with `set -e`, so we must call the status function
    # in an `if` condition to avoid exiting on the expected non-zero return
    # while jobs are still running.
    if _job_status_summary "${job_list_file}"; then
      st=0
    else
      st=$?
    fi
    if [[ "${st}" -eq 0 ]]; then
      break
    fi
    if [[ "${st}" -eq 2 ]]; then
      echo "[watch] Exiting early due to fatal quota/permission error(s). Fix quota/charged-group and re-run with --resume." >&2
      exit 3
    fi
    sleep "${watch_interval}"
  done
  fi

  if [[ "${merge_after_watch}" == "1" ]]; then
    merged_jsonl="${run_dir}/merged.jsonl"
    _merge_results "${run_dir}" "${shards}" "${merged_jsonl}"
    if [[ "${eval}" == "1" ]] && [[ "${eval_backend}" == "local" ]]; then
      if _merged_eval_done "${merged_jsonl}" "${mode}"; then
        echo "[local_eval] merged.jsonl already has cv_score/cv_error; skip."
      else
        _run_local_eval_on_merged "${merged_jsonl}"
      fi
      _sync_shards_from_merged "${run_dir}" "${shards}" "${merged_jsonl}"
    fi
    if [[ "${eval}" == "1" ]]; then
      _print_cv_summary "${merged_jsonl}"
    fi
  fi
fi

# Support local eval even without --watch: if results already exist (e.g., --resume with submitted=0),
# merge + eval locally when requested.
if [[ "${eval}" == "1" ]] && [[ "${eval_backend}" == "local" ]] && [[ "${watch}" != "1" ]] && [[ "${dry_run}" != "1" ]]; then
  merged_jsonl="${run_dir}/merged.jsonl"
  all_done=1
  for i in $(seq 0 $((shards - 1))); do
    shard_file="${shard_dir}/shard_$(printf '%05d' "${i}").jsonl"
    out_jsonl="${run_dir}/results/shard_$(printf '%05d' "${i}").jsonl"
    if ! _infer_done "${out_jsonl}" "${shard_file}" "${mode}"; then
      all_done=0
      break
    fi
  done
  if [[ "${all_done}" != "1" ]]; then
    echo "[local_eval] infer not complete for all shards; skip local eval (run with --watch or resume after completion)." >&2
    exit 0
  fi
  if [[ ! -f "${merged_jsonl}" ]]; then
    _merge_results "${run_dir}" "${shards}" "${merged_jsonl}"
  fi
  if _merged_eval_done "${merged_jsonl}" "${mode}"; then
    echo "[local_eval] merged.jsonl already has cv_score/cv_error; skip."
    _sync_shards_from_merged "${run_dir}" "${shards}" "${merged_jsonl}"
    _print_cv_summary "${merged_jsonl}"
    exit 0
  fi
  _run_local_eval_on_merged "${merged_jsonl}"
  _sync_shards_from_merged "${run_dir}" "${shards}" "${merged_jsonl}"
  _print_cv_summary "${merged_jsonl}"
fi
