# LoRA 解耦 vs Shared-base（同 base）细粒度分析：产出与复现

本目录的目标是把「为什么 LoRA verifier 更 work，而 shared-base 不 work」落成可量化的表格/图，并且做到**线上逻辑可复用**（prompt 模板、dump 字段一致）。

## 需要的训练侧开关（只影响观测，不改默认行为）

### 1) 观测校准：Actor vs Verifier 指标不互相覆盖
- 已实现：`verl/workers/fsdp_workers.py:update_verifier()` 会把 verifier update 的 metrics 全部加前缀 `verifier/`。
- 预期：SwanLab 同时出现 `actor/*` 与 `verifier/actor/*`。

### 2) dual_rollout_data dump 字段（离线 margin/no-decision 必需）
- 已实现：`verl/workers/reward_manager/naive.py` dump 中新增字段：
  - `question`
  - `student_response_policy`（用于 verifier prompt；exp stream 会剥离 hint token）
  - `student_response_full`（exp: 含 hint；control: 原样）
- 可选（体积很大）：`VERL_DUAL_DUMP_INCLUDE_VERIFIER_PROMPT=1` 会额外落盘 `verifier_prompt_user`。
- 可选（兼容旧字段）：`VERL_DUAL_DUMP_DUPLICATE_FULL_RESPONSE=1` 会额外落盘 legacy `full_response`（会重复占用空间）。

### 3) shared-base 的「更新冲突」量化（Table D）
仅在 `verifier.update_base=True` 且开启下面 env 时生效：
- `VERL_CONFLICT_DEBUG=1`
- `VERL_CONFLICT_DEBUG_FREQ=1`（建议 debug=1；32 卡建议 >=5）
- 可调：`VERL_CONFLICT_DEBUG_K=4096`、`VERL_CONFLICT_PARAM_NAMES=...`

输出指标（SwanLab）：
- `conflict/cos_actor_verifier`
- `conflict/norm_ratio_verifier_over_actor`

### 4) Phase 5（可选）：shared-base 稳定性对照（不默认开启）
- `VERL_VERIFIER_LR_SCALE=0.1`：仅在 `update_verifier()` 内临时缩放 verifier update 的 lr（shared optimizer 会在本次 update 结束后恢复）。
- `VERL_VERIFIER_GRAD_CLIP=0.1`：仅覆盖 verifier update 的 `grad_clip`。

## 表/图产出（用于汇报）

### 表 A：Actor vs Verifier 更新指标分离表（每 step）
从 SwanLab 导出 per-step 指标（CSV）。

```bash
cd repos/repro
python scripts/extract_update_metrics_table.py \
  --run-dir /path/to/swanlab/run-xxxx
```

输出：
- `<run_dir>/update_metrics.csv`

### 表 B：收益分解（dual_rollout_data）
从 `dual_rollout_data` 的 exp/control batch dump 统计：
- `hint_rate`
- `Δreward(exp-control)` 分布/均值/分位数/Δ>0 覆盖率
- 相关性：Δ vs `hint_len` / `exp_resp_len`（Pearson + Spearman）
- finish_reason（eos/length/stop/other）占比
- 分桶表：按 `hint_len` / `exp_resp_len` 分位做 bucket 统计

```bash
cd repos/repro
python scripts/report_lora_vs_base.py \
  --base-exp <shared_base_exp_name> \
  --lora-exp <lora_exp_name> \
  --base-batches 5 10 15 20 \
  --lora-batches 5 10 15
```

输出（默认在 `repos/repro/outputs/...`）：
- `reward_decomposition.csv`
- `sample_deltas.csv`
- `bucket_delta_stats.csv`
- `delta_hist.png` / `delta_vs_hintlen.png` / `delta_vs_resplen.png` 等

### 表 C：决策位可优化性（margin/entropy + no-decision）
离线对同一批 `question + student_response_policy` 比较 base vs LoRA：
- `<GO>` vs `<WAIT>` 的 margin/entropy（teacher forcing）
- generation 的 no-decision、finish_reason、malformed final-like 诊断

```bash
cd repos/repro
python scripts/verifier_decision_margin.py \
  --inputs /path/to/dual_rollout_data/<exp>/exp/batch_5.json \
  --base-model /path/to/hf-170-or-hf-181 \
  --lora-adapter /path/to/lora/checkpoint-2445 \
  --max-samples 128 \
  --out-dir outputs/verifier_margin_demo
```

输出：
- `decision_margin_per_sample.csv`
- `decision_margin_summary.csv`
- `margin_hist.png` / `entropy_hist.png`

### 表 D：优化冲突（shared-base）
开启 `VERL_CONFLICT_DEBUG=1` 后，直接从 SwanLab 或表 A 导出：
- `conflict/cos_actor_verifier`
- `conflict/norm_ratio_verifier_over_actor`

