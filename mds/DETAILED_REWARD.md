# Co-GRPO v2：Verifier 细粒度奖励与 Credit Assignment（设计 → 实现 → 复盘）

更新时间：2026-02-12  
适用 repo：`/mnt/shared-storage-user/liuhongwei/main_works/repos/repro`

本文目标：
- 先按时间顺序梳理两代方案：`global_gap`（粗粒度）→ `cf_branch`（事件级 uplift）
- 再记录当前 repo 的**真实实现**：字段、公式、脚本参数、常见坑与 debug 信号

---

## 1. 目标与记号

我们要训练的是 Verifier 的“介入决策 + hint 生成”（输出 `<GO>` 或 `<WAIT> hint`），希望对每次介入事件给到尽可能接近：

`Q(s, WAIT) - Q(s, GO)`

为方便描述，定义：
- 对每个样本 `i`（一个 prompt）：
  - 主线 EXP（with hint）最终响应：`y_main_i`
  - outcome reward：`R_main_i = R(y_main_i)`
- 对主线中第 `j` 次**真实插入 hint** 的事件（event）：
  - 插入前 prefix 状态：`s_{i,j}`（prompt + response_prefix_tokens）
  - 反事实分叉（without 当前 hint）最终响应：`y0_{i,j}`
  - 反事实 reward：`R0_{i,j} = R(y0_{i,j})`

注意：`cf_branch` 的 “without-hint” 分叉是**事后生成**（主线跑完后再跑），但分叉起点是插入前 prefix 的快照，所以不使用未来 token，仍是有效的 counterfactual baseline。

---

## 2. 历史方案 V1：双流（control+exp）+ 样本级 gap 均分（`verifier_credit_assignment=global_gap`）

### 2.1 生成方式

- 对同一批 prompts 同时生成两条完整响应：
  - `control`：不插 hint
  - `exp`：按 `token_check_interval/min_step_tokens/max_interventions/confidence_threshold` 周期触发 verifier，插 hint 后继续生成
- 两条流各自生成到结束（EOS/stop 或长度截断），用同一 reward_fn 得到：
  - `R_control_i` 与 `R_exp_i`

### 2.2 outcome gap + shaping（样本级）

- 基础差值：`raw_gap_i = R_exp_i - R_control_i`
- shaping（两种模式）：
  - `verifier_reward_mode=gap`：`clamped_gap = clamp(raw_gap_i, [-1,1])`
  - `verifier_reward_mode=headroom`：  
    `headroom = clamp(1 - clamp(R_control_i,[0,1]), min=headroom_min)`  
    `clamped_gap = clamp(raw_gap_i / headroom, [-1,1])`
- 正向 boost（`improve_coef`）：
  - `positive_gap = clamp(clamped_gap, [0,0.5])`
  - `verifier_outcome_reward_i = clamped_gap + improve_coef * positive_gap`
- cost（只在没有正增益时惩罚，避免抑制有效干预）：
  - `cost_i = lambda_freq * num_interventions_i + lambda_len * hint_token_counts_i/100`
  - `verifier_outcome_reward_i -= cost_i   (仅当 clamped_gap <= 0)`

### 2.3 credit assignment（粗粒度的根源）

- rollout 侧收集每次**真实插入 hint** 的 verifier 轨迹（`verifier_batch`）
- trainer 侧把样本级 `verifier_outcome_reward_i` **均分**给该样本所有事件：
  - `step_reward_{i,j} = verifier_outcome_reward_i / (#events_in_sample_i)`
- 再平铺到 token-level（按 response_mask 均摊），最后做 GRPO group normalize（旧版通常是 `group_index = parent_uid:step_idx`）

### 2.4 主要问题

- 一个样本可能插 0~N 次 hint，但 reward 只有一个样本级差值，最后还要均分 → 每个事件信号噪声很大
- N 次干预里有正有负，会互相抵消 → credit 非常不稳定
- `control` 与 `exp` 从开头就可能因为采样随机性/长度差分叉 → `R_control` 并不是“该插入点的局部反事实”，归因更不精确

---

## 3. 新方案 V2：事件级 uplift（`verifier_credit_assignment=cf_branch|cf`）

核心思路：只跑主线 EXP（with hint），然后对每个 event 额外生成一条 “without 当前 hint” 的反事实分叉，用 uplift 做 credit。

### 3.1 生成方式（主线 + 反事实分叉）

- 主线：正常跑 EXP（插 hint 后继续生成到结束）→ 得到 `R_main_i`
- 对每次真实插入 hint 的锚点 `s_{i,j}`：
  - 额外跑一条 without-hint continuation（从插入前 prefix 继续生成，但不插这次 hint）→ 得到 `R0_{i,j}`

当前实现是：**每个 event 只生成 1 条** `R0_{i,j}`（`n=1`），不做多次采样平均；想降低方差可以后续扩展成 `K>1`。

### 3.2 事件级 step reward（credit）

对每个 event：

`Δ_{i,j} = R_main_i - R0_{i,j} - cost_{i,j}`

其中 cost 常用：
- `cost_{i,j} = lambda_freq + lambda_len * hint_len_tokens/100`

备注：`R_main_i` 在同一个样本的所有事件上复用（省算力），而 baseline `R0_{i,j}` 是每个事件自己的。

### 3.3 为什么它比 global_gap 更细

- baseline 是“同一插入点 prefix 上的局部反事实”，比“全局 control vs exp”更接近因果归因
- 不需要把一个样本的 outcome gap 均分给多次干预 → 事件级 reward 更尖、更可学

---

## 4. 当前 repo 实现（co_grpo_v2，按时间顺序记录）

### 4.1 2026-02-09/10：global_gap（双流）阶段

- 训练信号：样本级 gap → 均分到每个 event（粗粒度）
- 典型现象：插入靠后/甚至 `</think>` 后插入时，可能“马后炮”，对准确率常是负贡献

### 4.2 2026-02-10/11：插入时机自洽（禁止 post-`</think>` 插入）

规则（rollout 侧执行）：
- **禁止**在 `</think>` 后存在正文时插 hint（不做马后炮）
- 如果触发 verifier 但当前位置已经 post-think：
  - 允许**一次**回滚到 `</think>` 前（prethink rollback）再插入
  - 若仍无法安全插入则跳过该次干预

对应实现文件：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

### 4.3 2026-02-11：实现 cf_branch（事件级 uplift）

#### Rollout（记录 event + 事后生成 cf_control_batch）

文件：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

- 每次 `hint_applied=True` 时记录 event：
  - `event_uid = f"{parent_sample_uid}:{hint_idx}"`
  - `initial_state`：插入前 prefix 快照（`response_tokens/loss_masks/hints/_prethink_rollback_used`）
  - 诊断字段：`prefix_len/wait_confidence/wait_avg_logprob/hint_token_count/prethink_anchor/state_hash_bucket`
- 主线结束后，批量生成 without-hint continuation：
  - 调用一次 `dual_stream_rollout(stream_mode="exp", cf_branching=False, collect_verifier_trajectories=False, initial_state_overrides=...)`
  - 得到 `meta_info["cf_control_batch"]`（每个 event 一条，`n=1`）

cf batch 的 join 字段（non_tensor）：
- `cf_event_uid`：对应 event_uid
- `cf_parent_sample_uid`：对应主线样本的 sample_uid（优先用它 join，抗 batch reorder）
- `cf_parent_idx`：fallback（仅在 batch 未 reorder 时可靠）
- 以及 `cf_parent_uid/cf_step_idx`

#### Trainer（按 event_uid 做 uplift）

文件：`verl/trainer/ppo/ray_trainer.py`

- `use_cf_branch = algorithm.verifier_credit_assignment in {"cf_branch","cf"}` 时：
  - EXP-only：不再算全局 control stream reward（节省一条全量 rollout）
  - 先算 `R_main_i`（exp_batch reward）
  - 再算 `R0_{i,j}`（cf_eval_batch reward）
  - 用 `event_uid` 做映射：`reward_by_event_uid[event_uid] = R0`
  - 对每条 verifier 轨迹（只包含 hint_applied=True 的事件）算 `Δ` 并写入 `verifier_train_batch`
- 工程修复（cf 分叉 batch 缺少 reward metadata）：
  - rollout worker 内生成的 `cf_control_batch` 不含 `data_source/reward_model/extra_info`
  - trainer 侧在 reward 前用 `cf_parent_sample_uid -> batch.sample_uid`（或 fallback `cf_parent_idx`）把 metadata 回填

### 4.4 2026-02-12：分布式稳定性修复（必写进复盘）

#### (1) meta_info side-batch 跨 rank 丢失（导致 verifier_train_batch_size 只剩 rank0）

- 症状：
  - dump 里 exp 真正插了很多 hints，但 `verifier/train_batch_size` 很小（接近 rank0 事件数）
  - cf_control_batch 也只有少量样本
- 根因：`DataProto.concat(...)` 没有把 `meta_info[key]` 里装的 `DataProto` 跨 rank 合并
- 修复文件：`verl/protocol.py`（concat 时对 meta_info 中的 DataProto 再 concat 一次）

#### (2) repeat_interleave 导致每 rank 长度极不均衡 → NCCL watchdog timeout

- 症状：日志里 `global_seqlen/min` 很小但 `max` 极大，随后 allreduce watchdog timeout
- 修复文件：`verl/trainer/ppo/ray_trainer.py`
  - Co-GRPO 的 prompt repeat 改为 tile（`interleave=False`），避免“同 prompt 重复成片落到一个 rank”

#### (3) malformed_like_outputs 指标口径修复 + 可观测性增强

- 旧口径会把正常 `<GO>` 误判为 malformed
- 现口径：只有**解析不到 `<GO>/<WAIT>` 决策行**才算 malformed_like
- 增加 metrics：`verifier_no_valid_decision_rate` / `verifier_no_valid_decision_final_like_rate`

---

## 5. Prefix-cache（vLLM enable_prefix_caching）开关：为什么 by_step 默认关？开了可能有什么问题？

开关位置：
- rollout 配置：`rollout.enable_kv_cache_optimization`
- 映射到 vLLM：`enable_prefix_caching=<enable_kv_cache_optimization>`
- 启动脚本默认逻辑（文件：`scripts/run_multinodes_cgrpo_v2.sh`）：
  - `verifier_intervention_mode=by_step` 时默认 `rollout_enable_kv_cache_optimization=False`
  - 可用环境变量 `ROLLOUT_ENABLE_KV_CACHE_OPTIMIZATION=True|False` 强制覆盖（debug 脚本 `run_dev_debug_cogrpo_8gpu.sh` 默认强制 True）

为什么 by_step 默认关（历史经验/风险点）：
- **显存与 cache 增长**：by_step 会多次调用 vLLM `generate()`，prefix-cache 会缓存大量中间 prefix block；部分 vLLM 版本存在 reset/回收不干净 → 长跑可能 OOM
- **sleep/offload 交互问题**：训练侧常用 `sleep(level=1)`/offload 释放 KV/权重；某些版本需要先 `reset_prefix_cache()`（见 async server 的注释），否则可能出现不稳定/异常  
  - rollout worker 已做 best-effort 兜底：当 `enable_prefix_caching=True` 时，在 `sleep/free_cache_engine` 前尝试调用 `reset_prefix_cache()`（文件：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`）
- **权重/LoRA 更新的陈旧 cache 风险**：RL 训练中 actor 权重/LoRA 会不断同步，如果 prefix-cache 未被正确清空，理论上可能复用“旧权重下的 KV”，导致输出不一致（需要用 `init_cache_engine/free_cache_engine` 或显式 reset 来兜底）

什么时候建议开：
- 本地小规模 debug（想加速 by_step，且能接受偶发不稳定）
- 或已确认当前 vLLM 版本在你们场景下不会出现 cache leak / sleep 问题，并且每次权重同步后 cache 能正确清空

---

## 6. 关键参数表（按脚本/实现真实含义）

主要入口脚本：`scripts/run_multinodes_cgrpo_v2.sh`、`run_dev_debug_cogrpo_8gpu.sh`

- `response_n`：response 长度倍率，`max_response_length = 1024 * response_n`
- `token_check_interval`：by_step 每次增量生成的目标跨度（单位：**模型生成 token**）
- `min_step_tokens`：每步最小生成 token（通常与 token_check_interval 设成一致）
- `max_interventions`：每条样本最多插 hint 次数（上限）
- `verifier_max_hint_tokens`：单条 hint 允许的最大 token（用于 verifier 端生成约束）
- `estimated_hint_tokens`：给 max_model_len 预留 hint headroom 的估计（过小会导致 context_exhausted）
- `confidence_threshold`：默认 `0.0` 关闭；>0 时启用 “WAIT 置信度过滤”（低于阈值的 WAIT 会被挡掉）
- `verifier_credit_assignment`：
  - `global_gap`：双流全局差值均分（旧）
  - `cf_branch|cf`：事件级 uplift（新，EXP-only）
- `cf_branch_prob`：记录 event 并生成反事实 baseline 的概率（默认 1.0）
- `cf_branch_max_events_per_sample`：每条样本最多记录多少个 event 做 cf（<=0 时默认用 `max_interventions`）
- `cf_branch_state_hash_mod`：event 记录的 state-hash 桶数（用于 group normalize 分桶/诊断）

并行相关（cf_branch 下会被忽略）：
- `parallel_control_exp=True`：control+exp 并行双流（global_gap 用）
- `CONTROL_ROLLOUT_GPUS_PER_NODE` / `EXP_ROLLOUT_GPUS_PER_NODE`

---

## 7. Debug 观察清单（建议每次跑都看）

1) “有没有真的在插？”
- dump：`.../dual_rollout_data/.../exp/batch_*.json` 里 `hints` 非空且与插入位置一致
- 训练指标：`co_grpo/exp_hint_len_mean`（exp_hint_len 不应为 0）

2) “cf baseline 数量是否对齐？”
- `timing_info.exp_metrics.cf_event_count` ≈ `len(cf_control_batch)`
- `verifier/train_batch_size` 应接近所有 rank 的 `hint_applied` 总数（不应只剩 rank0）

3) “verifier 输出格式是否跑偏？”
- `verifier_no_valid_decision_rate` 偏高时，优先怀疑 verifier prompt/template 或 stop/截断配置

4) “是否存在 NCCL straggler？”
- seqlen 极不均衡（min 很小 max 很大）且随后 watchdog → 先检查 prompt repeat 分配方式与 response 超长样本比例

---

## 8. 后续可选增强（不影响当前实现的复盘）

- 反事实多采样：每个 event 生成 `K>1` 条 without-hint，做均值 baseline 降方差（代价：算力 ×K）
- common random numbers：用 `hash(event_uid)` 派生 seed，让主线/分叉随机性更一致
- 事件选择：只对前 K 次干预、或 wait_confidence 最低的干预做 cf（控制算力）



对，目前 outcome reward 只有 {0,1} 的话：

你现在的 cf_branch（每个 event 只补 1 条 R0）会让每个事件的 credit 基本变成 Δ ∈ {-1,0,1} - cost，方差很大、粒度很粗。
它比 global_gap 仍然强在“归因位置对了”（同 prefix 的局部反事实），但学习信号的分辨率确实受限于二值 reward。
要把粒度做细，最直接的办法不是“再写更复杂的公式”，而是把二值 reward 变成“概率估计”：

方案 A（最推荐）：每个 event 采样 K 条反事实，学 P(correct|prefix)
把 R0(event) 从 1 条改成 K 条（同一个 prefix，随机采样 K 次 continuation）：

R0_hat = mean_k R0_k，它就是 P(correct | prefix, no-hint) 的估计
Δ = R_main - R0_hat - cost
这样 Δ 的粒度立刻从三值变成 1/K 的步进（例如 K=8 就是 0.125 粒度），并且方差按 ~1/K 下降
代价：算力 ×K。通常可以只对“有插入的 event”做、且只对前 1~2 次干预/低置信度干预做 K-sample 控算力。

方案 B（更准但更贵）：局部 fork 做 R1_hat - R0_hat
不要用整条主线最终 R_main 去给每个 event 复用，而是在 event 处真正分叉两边都采样：

with-hint 分支采样 K1 条：R1_hat
without-hint 分支采样 K0 条：R0_hat
Δ = R1_hat - R0_hat - cost
这能更“因果隔离”某一次 hint 的贡献（不被后续 hint 的连锁效应影响），但算力接近每个 event 额外跑 (K0+K1) 条。

方案 C（换 reward）：把最终 reward 从 0/1 换成连续分数
比如引入 RM/LLM judge 评分、或更细的 rule-based（步骤正确性/答案接近度/格式可解析度等）。这能从根上解决粒度问题，但工程/稳定性成本通常比 A 大。

如果你只想“最小改动、立刻让信号变细”，就做 方案 A：每个 event 的反事实 K-sample（K=4 或 8）。
