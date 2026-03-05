#!/usr/bin/env bash
set -eo pipefail

if [ -f /etc/profile.d/ssh-init.sh ]; then
  source /etc/profile.d/ssh-init.sh || true
fi

unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

cluster="${CLUSTER:-llmit_gpu}"
gpus="${GPUS:-8}"
nnodes="${NNODES:-1}"
exp_name="${EXP_NAME:-xt-verifier-pool8-$(date +%m%d%H%M)}"

model_path="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170}"
conda_env_path="${CONDA_ENV_PATH:-/mnt/shared-storage-user/liuhongwei/miniconda3/envs/repro}"
lmdeploy_path="${LMDEPLOY_PATH_OVERRIDE:-/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/lmdeploy_manually_fp8}"
model_name="${MODEL_NAME:-qwen2}"
adapters="${ADAPTERS:-}"
base_port="${BASE_PORT:-26000}"
session_len="${SESSION_LEN:-65536}"
max_batch_size="${MAX_BATCH_SIZE:-128}"
cache_max_entry_count="${CACHE_MAX_ENTRY_COUNT:-0.88}"

rjob_name="rjob-${exp_name}"

echo "[xt-verifier-pool8] name=${rjob_name}"
echo "[xt-verifier-pool8] model=${model_path}"
echo "[xt-verifier-pool8] model_name=${model_name}"
echo "[xt-verifier-pool8] adapters=${adapters:-<none>}"
echo "[xt-verifier-pool8] base_port=${base_port} gpus=${gpus}"

rjob submit \
  --name="${rjob_name}" \
  --gpu="${gpus}" \
  --memory="${RJOB_MEMORY:-512000}" \
  --cpu="${RJOB_CPU:-64}" \
  --charged-group="${cluster}" \
  --private-machine=group \
  --share-host-shm=True \
  --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
  --mount=gpfs://gpfs1/liuhongwei:/mnt/shared-storage-user/liuhongwei \
  --mount=gpfs://gpfs1/opencompass-shared:/mnt/shared-storage-user/opencompass-shared \
  --image=registry.h.pjlab.org.cn/ailab-llmit-llmit_gpu/liuhongwei_image:lhw-main-llmit-20260127162605 \
  -P "${nnodes}" \
  --host-network=true \
  -e DISTRIBUTED_JOB=true \
  -- bash -lc -- "
    set -eo pipefail &&
    unset https_proxy http_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY &&
    if [ ! -f /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh ]; then
      echo '[xt-verifier-pool8] ERROR: conda.sh missing' >&2 ;
      exit 127 ;
    fi &&
    source /mnt/shared-storage-user/liuhongwei/miniconda3/etc/profile.d/conda.sh &&
    conda activate '${conda_env_path}' &&
    export LMDEPLOY_PATH='${lmdeploy_path}' &&
    if [ ! -d \"${lmdeploy_path}\" ]; then
      echo \"[xt-verifier-pool8] ERROR: LMDEPLOY_PATH not found: ${lmdeploy_path}\" >&2 ;
      exit 2 ;
    fi &&
    export PYTHONPATH=\"${lmdeploy_path}:\${PYTHONPATH:-}\" &&
    export HF_DATASETS_OFFLINE=\${HF_DATASETS_OFFLINE:-1} &&
    export TRANSFORMERS_OFFLINE=\${TRANSFORMERS_OFFLINE:-1} &&
    export HF_EVALUATE_OFFLINE=\${HF_EVALUATE_OFFLINE:-1} &&
    export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-1} &&
    host_ip=\$(hostname -f) &&
    echo \"[xt-verifier-pool8] host_ip=\${host_ip}\" &&
    echo \"[xt-verifier-pool8] launching 8x lmdeploy api_server\" &&
    declare -a pids=() &&
    for i in \$(seq 0 7); do
      port=\$(( ${base_port} + i )) ;
      gpu=\${i} ;
      log_file=\"/tmp/verifier_api_\${i}.log\" ;
      echo \"[xt-verifier-pool8] shard=\${i} gpu=\${gpu} port=\${port} log=\${log_file}\" ;
      if [ -n \"${adapters}\" ]; then
        (
          export CUDA_VISIBLE_DEVICES=\${gpu} ;
          export LMDEPLOY_SKIP_WARMUP=1 ;
          exec lmdeploy serve api_server '${model_path}' \
            --backend pytorch \
            --tp 1 \
            --server-name 0.0.0.0 \
            --server-port \${port} \
            --session-len ${session_len} \
            --max-batch-size ${max_batch_size} \
            --cache-max-entry-count ${cache_max_entry_count} \
            --model-name '${model_name}' \
            --adapters '${adapters}'
        ) >\${log_file} 2>&1 &
      else
        (
          export CUDA_VISIBLE_DEVICES=\${gpu} ;
          export LMDEPLOY_SKIP_WARMUP=1 ;
          exec lmdeploy serve api_server '${model_path}' \
            --backend pytorch \
            --tp 1 \
            --server-name 0.0.0.0 \
            --server-port \${port} \
            --session-len ${session_len} \
            --max-batch-size ${max_batch_size} \
            --cache-max-entry-count ${cache_max_entry_count} \
            --model-name '${model_name}'
        ) >\${log_file} 2>&1 &
      fi ;
      pids+=(\$!) ;
    done &&
    echo \"[xt-verifier-pool8] waiting for /v1/models\" &&
    for i in \$(seq 0 7); do
      port=\$(( ${base_port} + i )) ;
      ok=0 ;
      for t in \$(seq 1 120); do
        if curl -sS --max-time 3 \"http://127.0.0.1:\${port}/v1/models\" >/dev/null 2>&1; then
          ok=1 ;
          break ;
        fi ;
        sleep 2 ;
      done ;
      if [ \${ok} -ne 1 ]; then
        echo \"[xt-verifier-pool8] ERROR: port \${port} not ready\" >&2 ;
        tail -n 80 \"/tmp/verifier_api_\${i}.log\" || true ;
        exit 3 ;
      fi ;
    done &&
    export HOST_IP_FOR_PRINT=\"\${host_ip}\" &&
    export BASE_PORT_FOR_PRINT=\"${base_port}\" &&
    python - <<'PY'
import json
import os
host = os.environ['HOST_IP_FOR_PRINT']
base = int(os.environ['BASE_PORT_FOR_PRINT'])
base_urls = [f'http://{host}:{base+i}' for i in range(8)]
openai_urls = [f'{u}/v1' for u in base_urls]
mapping = {str(i): base_urls[i % 8] for i in range(32)}
print('[xt-verifier-pool8] base_urls_csv=' + ','.join(base_urls), flush=True)
print('[xt-verifier-pool8] openai_urls_csv=' + ','.join(openai_urls), flush=True)
print('[xt-verifier-pool8] rank_url_dict_32=' + json.dumps(mapping, ensure_ascii=False, separators=(',', ':')), flush=True)
PY
    while true; do
      for pid in \"\${pids[@]}\"; do
        if ! kill -0 \"\${pid}\" 2>/dev/null; then
          echo \"[xt-verifier-pool8] ERROR: child process exited (pid=\${pid})\" >&2 ;
          exit 4 ;
        fi ;
      done ;
      sleep 15 ;
    done
  "
