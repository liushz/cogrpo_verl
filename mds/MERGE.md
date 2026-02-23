基线流程
7b Dense模型
- 起点模型：/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen2.5-7B/snapshots/e25af2efae60472008fbeaf5fb7c4274a87f78d4

- 冷启动脚本
```shell
set -x
XPUYU_PATH="/mnt/shared-storage-user/llmit/user/lvchengqi/projects/sft/xpuyu"
SAVE_DIR=/mnt/shared-storage-user/llmit/user/lvchengqi/ckpt/xpuyu/qwen2d5-7b_cold-start
if [ ! -d "$SAVE_DIR" ]; then
  mkdir -p "$SAVE_DIR"
fi
SCRIPT_NAME=$(basename "$0")
cp "$0" "${SAVE_DIR}/${SCRIPT_NAME}"
export PYTHONPATH=$XPUYU_PATH:"$XPUYU_PATH/_xtuner":$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
SP_SIZE=1
LR=0.00008
LR_MIN=0.000008
SEQ_LEN=131072
GLOBAL_BS=64
HF_MODEL_PATH=/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen2.5-7B/snapshots/e25af2efae60472008fbeaf5fb7c4274a87f78d4
CACHE_DIR=/mnt/shared-storage-user/llmit/user/lvchengqi/data/xpuyu_cache_dir_qwen2d5
DATA_PATH=/mnt/shared-storage-user/llmit/user/lvchengqi/data/math_20250911rc0/Enhance_Evolve_GPT_OSS_sft_data_with_system/processed/general_sys/Enhance-Evolve-GPT-OSS_sft_data_with_system.jsonl
echo "Save dir: ${SAVE_DIR}"
cp $0 $SAVE_DIR/
echo "Current home is ${HOME}"
/usr/local/bin/torchrun \
 --nproc-per-node=8  \
  --master_addr=${MASTER_ADDR} \
  --master_port=6000 \
  --nnodes=${NODE_COUNT} \
  --node_rank=${NODE_RANK} \
  -m \
  xpuyu.tools.sft \
  --llm ${HF_MODEL_PATH}  \
  --chat-template qwen_64k \
  --sp-size ${SP_SIZE} \
  --dset-file-types .jsonl \
  --dataset ${DATA_PATH} \
  --dset-formats processed \
  --dset-cache-dir ${CACHE_DIR} \
  --num-workers 16 \
  --dset-pack-level expand_soft \
  --global-pack \
  --max-length ${SEQ_LEN} \
  --mirco-batch-size 1 \
  --group-by-length \
  --global-batch-size ${GLOBAL_BS} \
  --lr $LR \
  --lr-min $LR_MIN \
  --wd 0.1 \
  --work-dir ${SAVE_DIR} \
  --checkpoint-interval -1 \
  --checkpoint-drop-optimizer \
  --hf-interval 0.2 \
  --log-interval 1 \
  --seed 1 \
  --selective-recompute 1.0 \
  --compile \
  --use-fa3 \
  --epoch 1
```
- 起点模型冷启动数据：/mnt/shared-storage-user/llmit/user/lvchengqi/data/math_20250911rc0/Enhance_Evolve_GPT_OSS_sft_data_with_system/processed/general_sys/Enhance-Evolve-GPT-OSS_sft_data_with_system.jsonl
- RL 起点模型：/mnt/shared-storage-user/llmit/user/chengguangran/model/cispo-cold-start-model/hf-170
- RL起点模型system prompt：已写入模型tokenizer_config，使用上述路径即可
You are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags.
- RL query 数据：/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-dapo-math-17k.parquet
- RL 训练脚本及参考配置：/mnt/shared-storage-user/llmit/user/chengguangran/projects/xtuner-cgr/crg_rl_projects/src/integration_project/scripts/rl_qwen25_7B_cold_start_grpo_dapo_math_260204_async.sh
30b MoE模型
- RL 起点模型：/mnt/shared-storage-user/llmit/user/lvchengqi/ckpt/xpuyu/qwen3-30ba3b_cold-start/20250924081143/hf-170
- RL query 数据：和集成验证流保持一致/mnt/shared-storage-user/llmit/user/lvchengqi/projects/moe_rl/xtuner_v1_projects/src/intern_s1_delivery/configs/data_configs/math_text_train_06-1.json
- RL训练脚本及参考配置：/mnt/shared-storage-
- user/llmit/user/chengguangran/projects/xtuner-cgr/crg_rl_projects/src/integration_project/scripts/rl_qwen3_30B_cold_start_grpo_rl_dataset_260204_async.sh
基线 Infra
- Xtuner codebase：https://gitlab.pjlab.org.cn/internlm/xtuner/-/tree/cgr/post-training-integration
- Lmdeploy codebase：/mnt/shared-storage-user/llmit/user/lvchengqi/projects/agent_rl/lmdeploy_manually_fp8
- 关键配置：
  - 异步采样：给定训练脚本默认开启了异步采样，如果需要关闭则设置
    - ENABLE_PARTIAL_ROLLOUT=0
    - TAIL_BATCH_CANDIDATE_STEPS=0
  - 开启重要性采样：rollout_is=RolloutImportanceSampling(rollout_is_level="token", rollout_is_mode="both", rollout_is_threshold=(5, 0), rollout_is_mask_threshold=(5, 0.5), rollout_is_veto_threshold=(20, 0)
  - lm_head开启fp32：fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1, lm_head_fp32=True)
  - MoE开启训推专家一致：enable_return_routed_experts = True
  - MoE关闭路由loss：model_cfg = Qwen3MoE30BA3Config(freeze_routers=True, balancing_loss_cfg=None)
  - 开启熵控制：entropy_control_cfg=dict(control_level="policy", upper_bound=0.75, upper_scale=2.0)
    - control_level：熵控制的细粒度，可选配置包括"policy" ｜ "group"，目前建议使用"policy" 
    - upper_bound&upper_scale：熵控制的上界，当熵大于upper_bound时，使用upper_scale系数对正优势样本进行放大
    - lower_bound&lower_scale：熵控制的下界，当熵小于lower_bound时，使用lower_scale系数对负优势样本进行放大
评测
7b Dense模型
- 训练日志：/mnt/shared-storage-user/llmit1/user/chengguangran/projects/xtuner-cgr/crg_rl_projects/work_dirs/Qwen2.5-7B-cold-start_dapo-math_is_pri_filter_async/20251226172004/20251226172055/
- 训练曲线：（左为aime2024）
[图片]
[图片]
- oc评测：
  - 评测配置：http://10.140.52.82:8080/job/api_eval_v3/3637/
  - 评测结果（80 step）：chengguangran-3637.csv
aime2024_acc
63.54
aime2025_acc
58.33
OlympiadBenchMath_acc
75.24
GPQA_diamond_acc
36.43
30b MoE模型
- 训练日志：/mnt/shared-storage-user/llmit1/user/chengguangran/projects/xtuner-cgr/crg_rl_projects/work_dirs/Qwen3-30Ba3b-cold-start_math-train_lmdeploy_is_pri_filter_adv_over/20251226110857/20251226110937
- 训练曲线：
[图片]
[图片]
[图片]
- oc评测：
  - 评测配置：http://10.140.52.82:8080/job/api_eval_v3/3613/pipeline-overview/
  - 评测结果（150 step）：chengguangran-3613.csv
aime2024_acc
86.56
aime2025_acc
85.73
OlympiadBenchMath_acc
85.79
GPQA_diamond_acc
55.18
