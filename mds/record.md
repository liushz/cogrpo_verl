# Experiment Progress Record

> Timezone: `Asia/Hong_Kong (UTC+8)`

## 2026-03-03

### 11:25（`0a461` 挂掉：Control stream batch size mismatch）

- 现象：`rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-0a461` 出现单 replica 失败，最终 rjob 状态 `Failed`。
- 失败栈（replica: `...-c4ca4`）：
  - `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 32 but got size 67 ...`
  - 位置：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`（by_step dual_stream rollout 的 control stream 组装阶段）
- 根因：control stream 用 `_extract_vllm_outputs()` **flatten** 了 `output.outputs[*]`，当 vLLM 偶发返回 “一个 prompt 多个 outputs”（疑似 stale `n/best_of`）时，`control_responses` 的 batch 维度会变成 `> batch_size`，最终在 `torch.cat([idx, control_responses], dim=-1)` 处炸掉。

**修复（已落代码）**

- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`：control stream 改为 **每个 prompt 只取第一个 output**（与 exp stream 处理一致），并在极端情况下对输出做 trunc/pad 到 `batch_size`，避免长跑随机 crash。

**动作**

- 已停止：`rjob stop rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-0a461`（释放 32 卡资源，避免残留 3/4 replicas 空转）。

### 11:39（重提：`0a461` 续训 v5r1）

- 目标：从 `v5/global_step_40` 续训（不丢 60h+ 进度），并带上 control stream 多输出修复。
- 提交：
  - rjob: `rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-3ae56`
  - showname: `rjob_cogrpo32_hf170_lora2445_dapo17k_pl2048_cfK2E2_int2_32k_v5r1`
  - resume_dir: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/checkpoints/co_grpo_v2/cogrpo32-hf170-lora2445-dapo17k-pl2048-cfK2E2-int2-32k-v5`（`resume_mode=auto` 会自动读取 `latest_checkpointed_iteration.txt`）
  - 状态（提交后）：`Inqueue/STARTING`

## 2026-03-02

### 11:15（巡检）

- 32 卡 Running：`0a461` / `2a754(baseline)`
- `0a461`（LoRA2445-int2，rollout.n=8，actor_update_streams=both）
  - Job: `rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-0a461`（Running）
  - SwanLab last step: `37`（~`2026-03-02 11:04`）
    - `co_grpo/cf_delta_mean=+0.1818`
    - `co_grpo/cf_delta_pos_ratio=0.1818`
    - `co_grpo/cf_diff_zero_ratio=0.7636`
    - `co_grpo/verifier_help_rate=0.1064`
    - `co_grpo/exp_hint_len_mean=76.21`
    - `co_grpo/control_reward/mean=-0.1816`
    - `co_grpo/exp_reward/mean=-0.1797`
    - `co_grpo/effective_control_weight=0.3074` / `effective_exp_weight=0.6926`
- `2a754` baseline（max_interventions=0，无 verifier 介入）
  - Job: `rjob-grpo32-hf170-baseline-by-step-exp-dapo1-2a754`（Running）
  - SwanLab last step: `92`（~`2026-03-02 11:03`）
    - `co_grpo/exp_reward/mean=-0.3389`
    - `co_grpo/exp_response_len_mean=22767.43`
    - `co_grpo/exp_hint_len_mean=0.0`

### 23:07（动作：停止 hf181 tp=2，改 tp=1 重提）

- 背景：`tp=2`（`8cc94`）实测 `timing_s/gen`/`perf/time_per_step` 显著慢于历史 `tp=1`，且 GPU util 仅 ~50–60%，不符合“提速”预期（更像增加通信开销）。
- 停止：
  - `rjob stop rjob-cogrpo32-hf181-nolora-dapo17k-pl2048-cf-8cc94`（showname: `..._tp2_v0`）
- 重提（tp=1，其余配置保持一致）：
  - 新 rjob: `rjob-cogrpo32-hf181-nolora-dapo17k-pl2048-cf-ca10a`
  - showname: `rjob_cogrpo32_hf181_nolora_dapo17k_pl2048_cfK2E2_int2_32k_tp1_v0`
  - 状态（提交后）：`Inqueue/STARTING`（4 replicas）

### 23:32（动作：停止 baseline）

- 停止（释放队列/资源）：
  - `rjob stop rjob-grpo32-hf170-baseline-by-step-exp-dapo1-2a754`

### 23:53（动作：提交 “无反事实 / 旧 reward(global_gap)” 对照实验）

- 目的：对比 `cf_branch` 的收益（主对照参考：`0a461`）。
- 配置要点：
  - `exp8 + control8`：`rollout.n=8`（双流各 8）
  - `max_interventions=2`
  - `verifier_credit_assignment=global_gap`（旧方式，不跑反事实）
  - `cf_branch_prob=0.0`（显式关闭 cf 采样）
  - `tp=1`
- 新 rjob：
  - rjob: `rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-9726d`
  - showname: `rjob_cogrpo32_hf170_lora2445_dapo17k_pl2048_gap_int2_n8_32k_tp1_v0`
  - 状态（提交后）：`Inqueue/STARTING`（4 replicas）

### 12:15（XTuner step=5 提交）

- 8 卡 step=5（bsz=8，32k 输出不变，开启 trajectory 内 CoGRPO debug dump）
  - Job: `rjob-xtcf8-bsz8-s5-03021157-34364629`（当前：`Inqueue/Unschedulable`）
- 4 卡 step=5（同配置，降低资源以便更快排队）
  - Job: `rjob-xtcf4-bsz8-s5-03021212-54585196`（当前：`Inqueue/Unschedulable`）

## 2026-03-01

### 14:44（巡检）

- 32 卡 Running（当时 4 个）：`eee17` / `0a461` / `5be6e` / `2a754(baseline)`
- 关键结论（当时）：`0a461`（LoRA2445-int2, n=8, both-stream update）当前最“work”；`eee17` 信号极稀疏、性价比低；`5be6e`（exp16）最慢且出现指标异常风险；baseline 作为对照保留。
- 主要风险：vLLM rollout 侧出现大量 `KV cache space` 不足 → `preempted by RECOMPUTE`，会显著拖慢 `timing_s/gen` / `perf/time_per_step`。

### 15:00（动作：停止 `eee17`）

- 15:00:33 请求停止：`rjob stop rjob-cogrpo32-hf170-lora1705-dapo17k-pl2048-eee17`
- 15:01:13 确认已完全停止：`rjob get ...` 显示 `Stopped`（4/4 replicas stopped）

**停止前最后一次 metrics 快照（SwanLab last scalar / rank0 log 对齐）**

- Job: `rjob-cogrpo32-hf170-lora1705-dapo17k-pl2048-eee17`（showname: `...-d316a`）
- SwanLab run: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/swanlab/run-20260226_084330-26gb7gp9ystxiykre6fwx`
- rank0 log: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/logs/verl_log_cogrpo32-hf170-lora1705-dapo17k-pl2048-cfK2E2-int2-32k-v4_rank0.txt`
- last step: `50`
  - `co_grpo/exp_hint_len_mean=74.789`
  - `co_grpo/verifier_help_rate=0.1084`
  - `co_grpo/cf_delta_mean=+0.0656`
  - `co_grpo/cf_delta_pos_ratio=0.0492`
  - `co_grpo/cf_diff_zero_ratio=0.9508`（uplift 极稀疏）
  - `co_grpo/control_reward/mean=-0.1855`
  - `co_grpo/exp_reward/mean=-0.1328`
  - `perf/time_per_step=5325s`（~1.48h/step）
  - rank0 累计告警：`no-decision warnings=32`；`KV recompute warnings=203`
- checkpoint 目录（已落盘）：`/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/checkpoints/co_grpo_v2/cogrpo32-hf170-lora1705-dapo17k-pl2048-cfK2E2-int2-32k-v4`（含 `global_step_20/40`）

### 15:01（当前 Running 32 卡实验：3 个）

#### A) `0a461`（LoRA2445-int2，rollout.n=8，actor_update_streams=both）

- Job: `rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-0a461`
- SwanLab run: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/swanlab/run-20260227_233638-a6wre60v6f85wlhnfei8a`
- rank0 log: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/logs/verl_log_cogrpo32-hf170-lora2445-dapo17k-pl2048-cfK2E2-int2-32k-v5_rank0.txt`
- last step: `24`
  - `co_grpo/exp_hint_len_mean=71.990`
  - `co_grpo/verifier_help_rate=0.1074`
  - `co_grpo/cf_delta_mean=+0.0714`
  - `co_grpo/cf_delta_pos_ratio=0.1071`
  - `co_grpo/cf_diff_zero_ratio=0.8214`
  - `co_grpo/control_reward/mean=-0.0879`
  - `co_grpo/exp_reward/mean=-0.0918`
  - `perf/time_per_step=5155s`（~1.43h/step）
  - rank0 累计告警：`no-decision warnings=31`；`KV recompute warnings=80`
- 观察：这条线目前 uplift 更“健康”（`cf_delta_mean` 为正、且 `cf_delta_pos_ratio` 较高），作为主线继续跑。

#### B) `5be6e`（LoRA2445-int2-exp16，rollout.n=16，actor_update_streams=exp）

- Job: `rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-5be6e`
- SwanLab run: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/swanlab/run-20260227_194535-6v7878zxqlvbu4rj0g0kj`
- rank0 log: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/logs/verl_log_cogrpo32-hf170-lora2445-dapo17k-pl2048-cfK2E2-int2-exp16-32k-v5_rank0.txt`
- last step: `19`
  - `co_grpo/exp_hint_len_mean≈76.1`（最后一次记录在 step18）
  - `co_grpo/verifier_help_rate=0.00488`（显著偏低，且和 uplift 指标不一致，疑似存在口径/统计子集异常风险）
  - `co_grpo/cf_delta_mean=-0.0169`
  - `co_grpo/cf_delta_pos_ratio=0.1102`
  - `co_grpo/cf_diff_zero_ratio=0.7627`
  - `co_grpo/control_reward/mean=-0.1504`
  - `co_grpo/exp_reward/mean=-0.1494`
  - `perf/time_per_step=7199s`（~2.0h/step）
  - rank0 累计告警：`no-decision warnings=30`；`KV recompute warnings=888`（极高，是最慢的主要原因）
- 观察：更慢、且出现 `verifier_help_rate` 近乎为 0 的风险信号；建议后续优先定位该指标与 “插 hint / uplift” 的一致性问题，再决定是否继续跑到 `save_freq=20`。

### 22:50（动作：停止 `5be6e`）

- 22:50:28 请求停止：`rjob stop rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-5be6e`
- 22:51:01 确认已完全停止：`rjob get ...` 显示 `Stopped`（4/4 replicas stopped）

**停止前最后一次 metrics 快照（SwanLab last scalar / rank0 log 对齐）**

- Job: `rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-5be6e`
- SwanLab run: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/swanlab/run-20260227_194535-6v7878zxqlvbu4rj0g0kj`
- rank0 log: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/logs/verl_log_cogrpo32-hf170-lora2445-dapo17k-pl2048-cfK2E2-int2-exp16-32k-v5_rank0.txt`
- last step: `23`
  - `co_grpo/exp_hint_len_mean=76.455`
  - `co_grpo/verifier_help_rate=0.00293`（持续偏低）
  - `co_grpo/cf_delta_mean=-0.1034`
  - `co_grpo/cf_delta_pos_ratio=0.1034`
  - `co_grpo/cf_diff_zero_ratio=0.7155`
  - `co_grpo/control_reward/mean=-0.0884`
  - `co_grpo/exp_reward/mean=-0.0918`
  - `perf/time_per_step=6906s`（~1.92h/step）
  - `timing_s/update_verifier=8.90s`（本次已不再出现早期的 200s+ verifier update）
  - rank0 累计告警：`no-decision warnings=30`；`KV recompute warnings=1066`（极高，拖慢主因）

#### C) `2a754` baseline（max_interventions=0，无 verifier 介入，用于对照）

- Job: `rjob-grpo32-hf170-baseline-by-step-exp-dapo1-2a754`
- SwanLab run: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/swanlab/run-20260228_173216-h9o2ovtflyhqss3zdw1tw`
- rank0 log: `/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/logs/verl_log_grpo32-hf170-baseline-by_step-exp-dapo17k-pl2048-b128-r16-32k-0228v7_rank0.txt`
- last step: `47`
  - `co_grpo/exp_hint_len_mean=0`（无介入）
  - `co_grpo/exp_reward/mean=-0.2412`
  - `co_grpo/exp_response_len_mean=21188.95`
  - `perf/time_per_step=1621s`（~27min/step）
  - rank0 累计告警：`no-decision warnings=0`；`KV recompute warnings=0`

### 15:47（提交新实验：hf181 shared-base 对照 0a461）

- 目的：作为 `0a461`（hf170 + LoRA2445 解耦）的 **non-decoupled / shared-base** 对照；base 用混训全量模型 `hf-181`，verifier 更新直接作用在 shared base 上（`verifier_lora_path="" -> verifier_update_base=True`）。
- 性能设置：`tp=2`（`actor_rollout_ref.rollout.tensor_model_parallel_size=2`），用于缓解长上下文下 vLLM 的 KV-cache `preempted by RECOMPUTE`（期望降低重算、改善 step 时长/抖动）。

**提交命令**

- `bash run_rjob_cogrpo_32gpu.sh --exp-name cogrpo32-hf181-nolora-dapo17k-pl2048-cfK2E2-int2-32k-tp2-v0 --model-path <hf-181> --no-verifier-lora --tp 2`

**rjob 信息**

- rjob id: `rjob-cogrpo32-hf181-nolora-dapo17k-pl2048-cf-8cc94`
- showname: `rjob_cogrpo32_hf181_nolora_dapo17k_pl2048_cfK2E2_int2_32k_tp2_v0`
- 状态（提交后立刻查看）：`Inqueue`（replicas: `STARTING`）
- model_path:
  - `/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/ckpt/cold_start_full_qwen2d5-7b/20260223230703_mixed_full/20260224105827/hf-181`

## 2026-03-01

### XTuner 8GPU smoke (bsz=8, 32k output)

- Script: `run_rjob_xtuner_cf_branch_8gpu_bsz8.sh`
- rjob: `rjob-xtcf8-bsz8-03011544-91226602`
- xtuner: `cogrpo-v2-xtuner-control-exp@6706a817`
- Backend: lmdeploy (fp8 copy)
  - `LMDEPLOY_PATH=/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/lmdeploy_manually_fp8`
- Key env:
  - `GLOBAL_BATCH_SIZE=8`, `PROMPT_REPEAT_K=16`, `ROLLOUT_STEPS=1`, `TRAIN_OPTIMIZER_STEPS=1`
  - `MAX_PROMPT_LENGTH=2048`, `MAX_RESPONSE_LENGTH=32768`, `PACK_MAX_LENGTH=35840`
  - `COGRPO_TOKEN_CHECK_INTERVAL=2048`, `COGRPO_MIN_STEP_TOKENS=2048`, `COGRPO_MAX_INTERVENTIONS=2`
  - `COGRPO_CF_BRANCH_K=2`, `COGRPO_CF_BRANCH_MAX_EVENTS_PER_SAMPLE=2`
  - `COGRPO_KEEP_CF_ROLLOUTS=0`, `COGRPO_VERIFIER_SKIP_TRUNCATED=1`, `COGRPO_VERIFIER_LORA_ENABLE=0`

**Result**

- Status: `Succeeded`（1 rollout + 1 optimizer step，32k 输出保持不变）
- Work dir: `/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner/work_dirs/hf-170_data_lmdeploy/20260301225141`
  - rank0 log: `/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner/work_dirs/hf-170_data_lmdeploy/20260301225141/rank_0.log`
  - training log: `/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner/work_dirs/hf-170_data_lmdeploy/training_log_030122.txt`
  - trajectory: `/mnt/shared-storage-user/llmit/user/liuhongwei/verifier_llmit/xtuner/work_dirs/hf-170_data_lmdeploy/20260301225141/rollout_idx_1_trajectory.jsonl`
- Key scalars（log tail）：
  - `response_len/mean=20629.97`, `response_len/max=32768.00`
  - `advantages/std=0.87634`, `advantages/pos_ratio=0.25852`
  - `co_grpo/exp_reward_mean=-1.33818`, `co_grpo/effective_control_weight=0.50`, `co_grpo/effective_exp_weight=0.50`
- Note: 结束时出现 `Raylet is terminated` / `store.cc Disconnecting client`，目前看更像 Ray teardown 噪声（raylet.out 未见 OOM/FATAL）。 

## 2026-03-02

### 11:05（32 卡训练巡检）

**rjob 状态**

- Running:
  - `rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-0a461`
  - `rjob-grpo32-hf170-baseline-by-step-exp-dapo1-2a754`
- Inqueue / STARTING:
  - `rjob-cogrpo32-hf181-nolora-dapo17k-pl2048-cf-8cc94`（tp=2 shared-base 对照；尚未产生日志文件）
- Stopped:
  - `rjob-cogrpo32-hf170-lora1705-dapo17k-pl2048-eee17`
  - `rjob-cogrpo32-hf170-lora2445-dapo17k-pl2048-5be6e`

**A) `0a461`（LoRA2445-int2, n=8, both-stream update）**

- last step: `37`（SwanLab）
  - `co_grpo/verifier_help_rate=0.1064`
  - `co_grpo/exp_hint_len_mean=76.210`
  - `co_grpo/cf_delta_mean=+0.1818`
  - `co_grpo/cf_delta_pos_ratio=0.1818`
  - `co_grpo/cf_diff_zero_ratio=0.7636`
  - `co_grpo/control_reward/mean=-0.1816`
  - `co_grpo/exp_reward/mean=-0.1797`
  - `timing_s/update_verifier=6.01s`
  - `perf/time_per_step=5851s`（~1.63h/step）
- rank0 累计告警（本地 log 统计）：`no-decision warnings=32`；`KV recompute warnings=142`
- ckpt 落盘：当前仅见 `global_step_20`（save_freq=20，等 step>=40 才会有 `global_step_40`）

**B) `2a754` baseline（max_interventions=0）**

- last step: `92`（SwanLab）
  - `co_grpo/exp_reward/mean=-0.3389`
  - `co_grpo/exp_hint_len_mean=0`
  - `co_grpo/exp_response_len_mean=22767.43`
  - `perf/time_per_step=1677s`（~28min/step）
- rank0 累计告警：`no-decision warnings=0`；`KV recompute warnings=0`
- ckpt 落盘：已有 `global_step_20/40/60/80`（`latest_checkpointed_iteration.txt` 最新写入在 `global_step_80`）

### 13:02（Day1 Smoke Test 排雷：风险总结 / TODO）

#### 本轮新增观测与 debug（默认不影响 32 卡；仅 env 开关启用）

- Actor Mask 梯度泄漏自检（确认 hint token 的梯度≈0）
  - 代码：`verl/workers/actor/dp_actor.py`（`VERL_DEBUG_MASK_GRAD=1`）
  - 指标：`debug/masked_logprob_grad_absmax`、`debug/masked_token_ratio`
- shared-base 冲突监控（Δθ sketch）
  - 代码：`verl/workers/fsdp_workers.py`（`VERL_CONFLICT_DEBUG=1`）
  - 指标：`conflict/cos_actor_verifier`、`conflict/norm_ratio_verifier_over_actor`
- Actor vs Verifier metrics 分离（避免 shared-base 下 `metrics.update()` 覆盖）
  - 代码：`verl/workers/fsdp_workers.py:update_verifier()` 统一前缀 `verifier/*`
- cf_branch “信号稀疏”补充口径
  - 代码：`verl/trainer/ppo/ray_trainer.py`
  - 新增：`co_grpo/cf_delta_zero_ratio`、`co_grpo/cf_delta_neg_ratio`（含 `_untrunc` 版本）
- resume 一致性修复（LoRA）
  - 问题：resume 后训练侧已 load checkpoint LoRA，但 vLLM 仍用旧 adapter，导致 verifier PPO 行为策略不对齐
  - 处理：`ray_trainer.py:_load_checkpoint()` 增加一次性 `save_verifier_lora + reload_verifier_lora` 到 `verifier_lora_latest`
- resume 完整性修复（LoRA optimizer/scheduler）
  - 问题：verifier LoRA 用独立 AdamW + scheduler，但之前不会随 checkpoint 保存/恢复（动量会重置，影响稳定性/可复现性）
  - 处理：`verl/workers/fsdp_workers.py:save_checkpoint/load_checkpoint()` 增加 `verifier_optim_world_size_*_rank_*.pt` 与 `verifier_extra_state_world_size_*_rank_*.pt`
- 离线分析脚本（配合 dual dump）
  - `scripts/analyze_hint_truncation_from_dual_dump.py`：量化 “插 hint 导致截断比例” + “未截断子集 uplift”
  - `scripts/verifier_decision_margin.py`：LoRA vs base 的 `<GO>/<WAIT>` margin/entropy + no-decision finish_reason
  - `scripts/extract_update_metrics_table.py`：导出 per-step `actor/*` vs `verifier/*` 指标表（CSV）
  - `scripts/report_lora_vs_base.py`：dual dump 分解（新增 `delta_neg_ratio`）

#### 当前主要风险（会导致 “no-decision 高 / cf_diff_zero 高 / 训练不稳定 / crash”）

1) **hf181 shared-base：Verifier 输出 4k 限制导致格式/决策截断**
   - 现象：`[Verifier] High no-decision rate` 频繁；且 `cf_diff_zero_ratio` 偏高（如 step14: `0.840`）。
   - 高概率根因：hf181 冷启动+混训的 student_response 平均长度 20k+；插 hint 后更长；Verifier 侧 `max_new_tokens=4096` 更易触发 `finish_reason=length`，导致无法解析 `<GO>/<WAIT>`。
   - 必做：对 shared-base（hf181）单独提高 verifier 输出 token（或至少在 smoke 里拉到 8k/16k 观察 no-decision 是否断崖下降）。

2) **“插 hint → 变长 → 截断” 可能把有益 hint 的增益掩盖掉**
   - 建议：优先统计 **control 不截断但 exp 截断** 的比例，并只在 **前后都不截断** 的子集上统计 uplift（`Δreward`）。
   - 工具：`scripts/analyze_hint_truncation_from_dual_dump.py`（注意旧 dump 可能不可靠，见第 3 点）。

3) **旧 dual dump 的 control 可能被 cf_control 覆盖/污染，导致配对统计失真**
- 例：`cogrpo32-hf181-nolora-...-v5-fixalloc1` 的 `hint_truncation_gain.csv` 显示 `control_samples=3412` vs `exp_samples=1024` 且 `suspect_overwritten_control_dump=True`；

4) **离线 AIME eval 的宏平均（macro）与 `n_questions` 统计口径有坑（已修复）**
   - 现象：历史 `cv_metrics.json` 出现 `n_questions=0`、`*_macro_acc=0`（但 `*_micro_acc` 正常）。
   - 根因：本地 eval runner 会把 repeat 元信息写在 `origin_info.origin_info._orig_line_idx`，而 `scripts/compassverifier_eval_tail_jsonl.py` 只读 `origin_info._orig_line_idx`，导致无法按题聚合 macro。
   - 修复：`scripts/compassverifier_eval_tail_jsonl.py` 兼容读取嵌套的 `_orig_line_idx`。
   - 注意：旧的 `cv_metrics.json` 需要对同一份 `merged.jsonl` 重新跑一次 cv_eval 才能得到正确 macro。
   - 另外：`delta_micro/delta_macro` 只有在 `--mode both`（同一份 jsonl 同时包含 control+exp）时才有意义；exp-only 的 delta 会退化成 `exp_micro-0`。

5) **离线 eval 可能出现“输出跑满 65k token”导致结果不可用（已加 guard 开关）**
   - 现象：历史本地 AIME eval（base/control）出现 `response_tokens_total` 均值接近 `max_response_tokens=65536`，且尾部呈现明显重复（模型不出 EOS）。
   - 影响：耗时巨大、acc 失真（绝大多数样本被当成无效/未作答）。
   - 处理：`run_local_eval_cogrpo_verifier_8gpu.sh` 默认开启 `DEGENERATION_GUARD=1`（传 `--degeneration-guard`），并增强了单行重复检测；必要时建议把 `MAX_RESPONSE_TOKENS` 降到 32k/8k 做快速 sanity。

#### 离线评测脚本（Checkpoint/LoRA 对齐线上 vLLM）

- 新增：`run_local_eval_cogrpo_gs20_vllm_8gpu.sh`
  - 默认评测：actor `global_step_20/actor/huggingface` + verifier `verifier_lora/global_step_20`
  - 引擎：`eval_co_grpo_with_verifier_v2.py --backend vllm`（by_step 插 hint，与线上 rollout 逻辑一致）
  - 默认关键参数对齐线上该 run：`max_model_len=35840` / `max_response_tokens=32768` / `token_check_interval=2048` / `max_interventions=2` / `verifier_max_new_tokens=4096`
     同时 control 样本缺少 `context_exhausted/last_finish_reason` 等字段。
   - 结论：该 run 的 “hint-induced truncation / exp-control 配对” 统计 **不可信**；需要用新代码重跑后再做表 B/C。

4) **vLLM KV cache 不足 → Preempted by RECOMPUTE（严重拖慢 step，且可能诱发底层 CUDA 异常）**
   - 现象：rank0 log 出现大量 `preempted by PreemptionMode.RECOMPUTE`；step 时间 1h+。
   - 缓解策略：提高 `tensor_parallel_size`（如 tp=2）、减少并发/bsz/rollout.n、调整 vLLM KV 配置（优先不影响效果的前提下做 tp=2）。

5) **CUDA illegal memory access（致命 crash）**
   - 例：`cogrpo32-hf181-nolora-...-v5-fixalloc1` 在 step14 后 crash（vLLM rollout 侧报 `illegal memory access`，随后 `mem_get_info` 也触发同错误）。
   - 建议：先在 4/8 卡 smoke 上加 `CUDA_LAUNCH_BLOCKING=1` + 关闭可疑 cache（如 prefix cache）复现定位，再放大到 32 卡。

#### TODO（按优先级：先验证再扩跑）

- [P0] 4/8 卡 smoke：打开 `VERL_DEBUG_MASK_GRAD=1`，确认 masked token 的梯度指标≈0（避免 Advantage/Mask 误判）。
- [P0] shared-base（hf181）单独拉高 verifier `max_new_tokens`，同时开 `VERL_VERIFIER_DEBUG_LOG=1` 抓 `finish_reason` 分布，验证 no-decision 是否主要由 `length` 截断导致。
- [P0] 用 “新干净的 dual dump” 跑：
  - `scripts/analyze_hint_truncation_from_dual_dump.py`（先回答：插 hint 是否显著提高截断比例？）
  - `scripts/verifier_decision_margin.py`（回答：LoRA vs base 决策位 margin/entropy 是否显著不同？）
- [P1] shared-base 冲突证据：在 smoke 打开 `VERL_CONFLICT_DEBUG=1`，沉淀 `conflict/*` 指标（论文可用）。
- [P1] 导出表 A：用 `scripts/extract_update_metrics_table.py` 生成 per-step `actor/*` vs `verifier/*` 的 CSV，避免指标覆盖造成误判。

---

## Engineering Notes（便于复盘/写论文/排障）

### 1) Batch Size 对齐（batch alignment）做了什么？为什么重要？

**对齐目标**：让每个 DP rank 的 **样本数 / micro-step 数** 完全一致，避免 FSDP/NCCL collective 因为某些 rank “少跑/多跑/先结束/后结束”而卡死；同时尽量让每个 rank 的 **token 总量** 相近，降低 straggler（跨节点 all-reduce 等待）。

1. **样本数对齐（硬约束，启动前就 fail fast）**
   - 在 launcher 里强校验：`train_batch_size * rollout.n` 必须能被 `dp_size` 整除；并且归一化后的 `normalized_mini` 必须能被 `micro_bsz_per_gpu` 整除。  
     位置：`scripts/run_multinodes_cgrpo_v2.sh`（FSDP normalization sanity checks）。
   - FSDP worker 内部也会做同样的归一化/整除校验（防止外部脚本漏传/改错）：  
     位置：`verl/workers/fsdp_workers.py`（`ppo_mini_batch_size` 与 `ppo_micro_batch_size_per_gpu` 的 normalize + assert）。

2. **token 对齐（软对齐，降低跨节点 straggler）**
   - 训练 step 前在 driver 侧对 batch 进行重排：把每条样本的 `attention_mask` seqlen 做分桶/分区，使各 rank “总 token” 尽量均衡。  
     位置：`verl/trainer/ppo/ray_trainer.py` → `RayPPOTrainer._balance_batch()`（日志里 `global_seqlen/*`）。

3. **动态 micro-batching（避免长上下文下某些阶段 OOM）**
   - rollout/recompute logprob/actor update 都支持 `use_dynamic_bsz` 与 `*_micro_batch_size_per_gpu`，核心思路是把一个大 batch 拆成多个 micro-step，保证显存峰值可控。  
     入口：`scripts/run_multinodes_cgrpo_v2.sh` 参数 `actor_rollout_ref.{actor|ref|rollout}.*_use_dynamic_bsz`；落地在 `verl/workers/fsdp_workers.py` 的 `compute_log_prob()` 等路径。

### 2) 动态奖励信号聚合（reward aggregation）在分布式下怎么做？

这里的“动态”主要有两层：
1) **来源动态**：control/exp 双流（global_gap） vs EXP-only + event-level 反事实（cf_branch）。  
2) **传输动态**：为了避免跨节点传输 32k 超大对象，reward 侧会按“只需要的字段/只需要的 token window”做裁剪。

**核心数据流（简化）**
- rollout worker（vLLM）产生：`exp`（主线）以及（在 cf_branch 下）每个 event 的 `cf_control_batch`（without-hint continuation）。  
- driver 侧 reward_manager 计算 outcome reward（通常是 `math_dapo/aime` 的 ±1 或 CompassVerifier 的 bool→±1），写入 `token_level_rewards`。  
- driver 侧 `compute_advantage()` 做 GRPO/CoGRPO 的 group normalize / curriculum mixing（不需要 NCCL，batch 在 driver 上完整可见）。  
  位置：`verl/trainer/ppo/ray_trainer.py` → `compute_advantage()` / `core_algos.compute_*`.

**分布式效率关键点（避免 Ray/网络成为瓶颈）**
- **cf_branch 的反事实 batch 只保留 tail window**：因为 CompassVerifier/答案抽取通常只看 response 尾部，传 32k 全量会让 Ray object store/网络爆炸。  
  位置：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`：
  - `_CF_CONTROL_BATCH_DROP_NONTENSOR_KEYS`：丢弃长 decoded 字符串/诊断字段；
  - `_build_cf_reward_tail_batch(...)`：保留 `prompts` + `responses` 的 tail tokens + `attention_mask`（用于 reward 评估）。
- **默认不启用 reward_async（Ray remote reward）**：长上下文下把 `DataProto` 丢给 Ray task 会序列化多 GB 对象并触发 object store spill（TB 级 I/O）→ 实际表现就是“训练卡住/极慢”。  
  位置：`scripts/run_multinodes_cgrpo_v2.sh`（`VERL_REWARD_ASYNC` 的说明与默认 False）。

### 3) 跨节点聚合是怎么做的？遇到过哪些“严重阻塞”？

分三类看（对应三套“通信系统”）：

1) **NCCL / torch.distributed collectives（最容易“看起来像死锁”）**
   - 触发条件：某些 rank 成为 straggler（比如 vLLM 侧 KV preemption → gen 极慢），其余 rank 在 FSDP all-reduce/all-gather 等 collective 等待，最终 hit timeout。  
   - 缓解：
     - `RayPPOTrainer._balance_batch()` 做 seqlen 均衡；
     - 把 `VERL_NCCL_TIMEOUT_SEC` 拉大（multi-node + 32k 的单 step 可能 >30min）；入口：`verl/workers/fsdp_workers.py:_get_dist_timeout()`；
     - 需要定位时开 `TORCH_NCCL_TRACE_BUFFER_SIZE`（flight recorder）。

2) **Ray object store / spill（典型现象：GPU 利用率低但系统“很慢”）**
   - 触发条件：跨节点传输/存储多 GB 的 `DataProto`（特别是 32k response + decoded strings + async reward）。  
   - 缓解：
     - rjob 侧挂载 host `/dev/shm`（脚本里 `--share-host-shm=True`），并显式加大 `VERL_RAY_OBJECT_STORE_MEMORY_GB`；
     - cf_branch 侧裁剪 cf batch（见上）；
     - 禁用 reward_async。

3) **vLLM TP 侧通信（容易出现“诡异 hang/不稳定”）**
   - 当前 rollout 里强制 `disable_custom_all_reduce=True`，避免 vLLM 自研 all-reduce 在多节点/特定环境下的不稳定行为。  
     位置：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`（vLLM `LLM(...)` 初始化参数）。

### 4) verl + vLLM 在线 RL：显存是怎么“分账”的？常见 OOM 怎么定位？

**同一张卡上主要有两类“吃显存”的实体：**
1) **FSDP2 actor（训练）**：sharded 权重 + 梯度 + optimizer state + activation（有 gradient checkpointing 会显著降 activation 峰值）。  
2) **vLLM engine（推理/rollout/verifier）**：推理权重 + KV cache（大头）+ 临时 workspace。KV cache 上限由 `gpu_memory_utilization`、`max_num_seqs`、`max_num_batched_tokens` 强约束。

**这个 repo 针对“最常见/最难复现”的 OOM 做过的几类修复/兜底（有注释可追溯）：**
- vLLM memory pool 与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true` 冲突会直接 assert：launcher 会主动剔除该配置。  
  位置：`scripts/run_multinodes_cgrpo_v2.sh`（对 `PYTORCH_CUDA_ALLOC_CONF` 的清洗）。
- `max_num_batched_tokens` 过大在长上下文下可能触发 vLLM `illegal memory access`：默认对 batched_tokens 做稳定性 cap（`max_model_len + 8192`），除非用户显式 override。  
  位置：`scripts/run_multinodes_cgrpo_v2.sh`（cap 逻辑的注释）。
- prefix cache 在 by_step 下可能导致 cache 增长/回收不彻底 → 长跑 OOM：by_step 默认关 prefix cache，并在 sleep/free_cache 时 best-effort reset。  
  位置：`mds/DETAILED_REWARD.md`（原因与风险）；`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`（相关逻辑）。
- 观测层面：`verl/utils/debug/performance.py` 的 `mem_get_info()` 能统计到 vLLM 低层 allocator 的真实占用（PyTorch 的 `memory_reserved/allocated` 经常看不见 vLLM）。  

**定位建议（实操顺序）**
1) 看报错栈：是 PyTorch OOM（训练/重算 logprob）还是 vLLM OOM/illegal access（rollout）；
2) 对 vLLM：先降并发（`max_num_seqs/max_num_batched_tokens`），再调 `gpu_memory_utilization`、再看是否 prefix-cache 相关；
3) 对训练侧：先降 `micro_bsz_per_gpu` 或开 `use_dynamic_bsz`，再考虑 offload（如果允许）。

### 5) 32 卡 4 节点：每张卡在做什么？（以当前 CoGRPO/GRPO 为例）

**Ray 层面**
- 4 个 placement group（每节点 1 个），每个 PG 有 8 个 bundle（1 GPU + 1 CPU），总计 32 个 worker slot。  
  位置：`verl/single_controller/ray/base.py:RayResourcePool.get_placement_groups()`。

**Worker 角色层面（hybrid engine）**
- 默认（`trainer.parallel_control_exp=False`，例如 cf_branch）：所有 GPU 在同一个 `global_pool`。  
  - 只会起 `actor_rollout` worker group（GRPO/CoGRPO 下 `use_critic=False`，所以不会起 critic）。  
  - 每个 GPU rank 运行一个 `ActorRolloutRefWorker(role=\"actor_rollout\")`：既做 FSDP2 actor 更新，也跑 vLLM rollout + verifier 生成（同进程/同卡）。  
    入口：`verl/trainer/main_ppo.py` + `verl/trainer/ppo/ray_trainer.py:init_workers()`。
- 双流并行（`trainer.parallel_control_exp=True`，global_gap 时常用）：每节点 GPU 一分为二：`exp_pool` vs `control_pool`。  
  - `actor_rollout`（exp）在 exp_pool；`actor_rollout_control`（control）在 control_pool；control side 强制 `rollout.n=1` 且禁 prefix cache，保证更轻、更稳。  
    入口同上（`main_ppo.py` 的 resource_pool_spec + `ray_trainer.py` 的 control_config patch）。

### 6) 为什么这里 RL 选 FSDP（fsdp2），而不是 DeepSpeed？

就当前 repo 的“真实实现”而言：
- `verl/trainer/main_ppo.py` 明确只支持两条训练后端：`fsdp/fsdp2` 和 `megatron`（没有 deepspeed 分支）。
- 选择 FSDP2 的工程动机更偏“可控/易集成”：
  - hybrid engine 里要频繁做：rollout（vLLM）↔ 训练（FSDP）之间的权重同步、以及 checkpoint/save/load（尤其是 verifier LoRA 的单独保存/恢复），FSDP 的 state_dict / device_mesh 路径在本 repo 已打通；
  - DeepSpeed/ZeRO 引入后，权重聚合/同步到 vLLM、以及 LoRA runtime updating 的一致性，需要额外一整套 glue code 与更多 failure mode。
