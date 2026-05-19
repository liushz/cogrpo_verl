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
system prompt:
```
You are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags.
```
- RL query 数据：/mnt/shared-storage-user/llmit/user/chengguangran/projects/verl-cgr/recipe/cispo/data/modified-dapo-math-17k.parquet
- RL 训练脚本及参考配置：/mnt/shared-storage-user/llmit/user/chengguangran/projects/xtuner-cgr/crg_rl_projects/src/integration_project/scripts/rl_qwen25_7B_cold_start_grpo_dapo_math_260204_async.sh
30b MoE模型
- RL 起点模型：/mnt/shared-storage-user/llmit/user/lvchengqi/ckpt/xpuyu/qwen3-30ba3b_cold-start/20250924081143/hf-170
- RL query 数据：和集成验证流保持一致/mnt/shared-storage-user/llmit/user/lvchengqi/projects/moe_rl/xtuner_v1_projects/src/intern_s1_delivery/configs/data_configs/math_text_train_06-1.json
- RL训练脚本及参考配置：/mnt/shared-storage-
- user/llmit/user/chengguangran/projects/xtuner-cgr/crg_rl_projects/src/integration_project/scripts/rl_qwen3_30B_cold_start_grpo_rl_dataset_260204_async.sh
基线 Infra
- Xtuner codebase：https://gitlab.pjlab.org.cn/internlm/xtuner/-/tree/cgr/post-training-integration
- Lmdeploy codebase（建议用自己可写的拷贝，避免误用他人目录）：/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/lmdeploy_manually_fp8
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
- 训练日志：/mnt/shared-storage-gpfs2/llmit1/user/chengguangran/projects/xtuner-cgr/crg_rl_projects/work_dirs/Qwen2.5-7B-cold-start_dapo-math_is_pri_filter_async/20251226172004/20251226172055/
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
- 训练日志：/mnt/shared-storage-gpfs2/llmit1/user/chengguangran/projects/xtuner-cgr/crg_rl_projects/work_dirs/Qwen3-30Ba3b-cold-start_math-train_lmdeploy_is_pri_filter_adv_over/20251226110857/20251226110937
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




## S2 Training

Rjob:

```
      
set -ex

gpu_group=llmit_gpu
namespace=ailab-llmit
num_gpus=64
num_nodes=$((num_gpus / 8))
job_name=interns2-preview-sft-0228b

config_file=/mnt/shared-storage-user/llmit/user/liujiangning/projects/crg_rl_projects/src/lagent_rl/scripts/s2_preview_sft/thinker_sft_0228b/thinker_sft_0228b.py

rjob submit \
    --name=${job_name} \
    --gpu=8 --memory=1800000 --cpu=120 \
    --charged-group=${gpu_group} \
    --namespace ${namespace} \
    --private-machine=group \
    -P ${num_nodes} \
    --image registry.h.pjlab.org.cn/ailab-llmrazor-llmrazor_gpu/xtuner_tmp:pt26_20251111_dea98b8_grouped_router_topk1_addoss \
    --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared \
    --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
    --mount=gpfs://gpfs2/sfteval:/mnt/shared-storage-user/sfteval \
    --mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
    --mount=gpfs://gpfs2/llmit1:/mnt/shared-storage-user/llmit1 \
    --host-network=true \
    --gang-start=true \
    --custom-resources rdma/mlnx_shared=8  \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    -e DISTRIBUTED_JOB=true \
    -- bash -c "
    cd /mnt/shared-storage-user/llmit/user/liujiangning/projects/interns2_sft/xtuner
    bash scripts/sft_intern_s1_vl_entrypoint.sh ${config_file}"
```


Train.py

```python
      
from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
)
from xtuner.v1.module.rope.rope import RopeScalingConfig
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3VLMoE30BA3Config
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
import json
import os
import shutil
from xtuner.v1.model.compose.qwen3_vl.modeling_qwen3_vl import QWEN3VL_COMPILE_CFG
QWEN3VL_COMPILE_CFG.pop("xtuner.v1.model.compose.qwen3_vl.modeling_vision.Qwen3VLVisionLayer.forward")

# 路径配置
ceph_config = "/mnt/shared-storage-user/llmit1/user/liujiangning/petreloss.conf"
meta_data_path = '/mnt/shared-storage-user/llmit/user/liujiangning/projects/crg_rl_projects/src/lagent_rl/scripts/s2_preview_sft/thinker_sft_0228b/thinker_sft_0228b.json'
model_path = '/mnt/shared-storage-user/puyudelivery/user/puyudilivery/ckpts/xtuner_saved_model/interns1_1_mini_official/interns1_1_mini_cpt_based_mlp_warmup_bs512_epoch1_maxlr2e-5_minlr2e-6_max16k-hf/20260203162506/hf-23818'
work_dir = "/mnt/shared-storage-user/llmit1/user/liujiangning/exp/s2_preview/sft/work_dirs/cpt0203_thinker_sft_0228b"
tokenizer_cache_dir = "/mnt/shared-storage-user/llmit1/user/liujiangning/xtuner_tokenizer_cache_dir/thinker_sft_0228b_1"

# 将当前配置文件拷贝到work_dir
if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)
current_file = __file__
shutil.copy(current_file, work_dir)

# 训练超参数
sample_max_length = 65536
pack_max_length = 65536
processor_path = model_path
rand_video_max_frames = 24
add_vision_id = True
num_workers = 8
global_batch_size = 64
total_epoch = 1
hf_interval = 500
hf_max_keep = 20
checkpoint_interval = 500
checkpoint_maxkeep = 20
lr = 3e-5
lr_min = 1e-6
weight_decay = 0.05
warmup_ratio = 0.1
recompute_ratio = 1.0
loss_reduction = "square"
enable_3d_rope = False
max_pixels = 16777216   # 16384 * 32 * 32
sp_size = 2

# model config
model_cfg = Qwen3VLMoE30BA3Config(text_config=Qwen3MoE30BA3Config())
model_cfg.vision_config.depth = 24
model_cfg.vision_config.hidden_size = 1024
model_cfg.vision_config.intermediate_size = 4096
model_cfg.vision_config.deepstack_visual_indexes = []

model_cfg.projector_config.vision_hidden_size = 1024
model_cfg.projector_config.deepstack_visual_indexes = []

model_cfg.text_config.max_position_embeddings = 262144
model_cfg.text_config.rope_theta = 5000000
model_cfg.text_config.rope_scaling_cfg = RopeScalingConfig(
            fope_init_factor=0.5,
            fope_sep_head=True,
            num_inv_freq=None,
            )
model_cfg.text_config.vocab_size=155008

# dataset config
if ceph_config is not None:
    oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
else:
    oss_loader_cfg = None

ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    tokenize_fn = Qwen3VLTokenizeFnConfig(
        max_length=sample_max_length,
        processor_path=processor_path,
        min_pixels=_data.get('min_pixels', None),
        max_pixels=max_pixels,
        video_min_total_pixels=_data.get('video_min_total_pixels', None),
        video_max_total_pixels=_data.get('video_max_total_pixels', None),
        video_min_frames=_data.get('video_min_frames', None),
        video_max_frames=_data.get('video_max_frames', None),
        fps=_data.get('fps', None),
        rand_video_max_frames=rand_video_max_frames,
        add_vision_id=add_vision_id,
        system_message=_data.get('system_message', None),
        hash=_data.get('hash', None),
        enable_3d_rope=enable_3d_rope,
        oss_loader_cfg=oss_loader_cfg,
        debug=False,
        oss_time_log_thr=10
    )

    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          class_name='VLMJsonlDataset',
                                          enable_sequential_sampler=True,  # 为了保证可复现性，使用顺序采样，入表的数据已经全局shuffle
                                          cache_tag='cache_tags_v1',
                                          cache_dir=tokenizer_cache_dir),
                 "tokenize_fn": tokenize_fn
                 }
    dataset_config.append(_data_cfg)

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_level='soft',
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=20,
)
# optimizer and lr config
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(recompute_ratio=recompute_ratio,
                      torch_compile=True,
                      checkpoint_preserve_rng_state=False)

resume_cfg = ResumeConfig(auto_resume=True)

# trainer config
trainer = TrainerConfig(
    sp_size=sp_size,
    load_from=model_path,
    resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker='tensorboard',
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    hf_max_keep=hf_max_keep,
    work_dir=work_dir,
)
```
