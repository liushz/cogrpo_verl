# Co-GRPO / CoGRPO FAQ（可能被问到的细节问题 + 标准答案）

Last updated: 2026-03-02  
Scope: 以本 repo（`/mnt/shared-storage-user/liuhongwei/main_works/repos/repro`）当前实现为准，重点覆盖 `by_step + cf_branch + LoRA/shared-base` 的线上同款训练与离线分析。

> 使用场景：
> - 论文/汇报答辩（reviewer 风格问题）
> - code review / 线上排障（实现细节问题）
> - 实验对照设计（Ablation / compute-matched baseline）

---

## 0) 一句话“导航图”（看代码先看哪几个文件）

- **训练主循环 / reward / cf_branch credit assignment / metrics**：`verl/trainer/ppo/ray_trainer.py`
- **advantage estimator（GRPO / Co-GRPO）**：`verl/trainer/ppo/core_algos.py`
- **by_step rollout + 插 hint + cf event 记录**：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- **线上同款“hint 插入/回滚/安全锚点/模板”**：`verl/workers/rollout/vllm_rollout/verifier_hint_injection.py`
- **分布式/FSDP worker（logprob 重算、verifier update、resume 同步）**：`verl/workers/fsdp_workers.py`
- **启动器（32k/长上下文稳定性、Ray object store、max_model_len headroom）**：`scripts/run_multinodes_cgrpo_v2.sh`
- **rjob 封装（32 GPU/4 节点参数口径）**：`run_rjob_cogrpo_32gpu.sh`
- **reward adapter（math_dapo / AIME / CompassVerifier / ±1）**：`verl/utils/reward_score/__init__.py` + `verl/utils/reward_score/math_dapo.py`
- **dump/离线分析相关开关**：`mds/DETAILED_REWARD.md`、`mds/LORA_VS_SHARED_BASE_REPORT.md`

---

## 1) 算法/方法层：Co-GRPO 到底是什么？和 GRPO 区别在哪？

### Q1：Co-GRPO 和 GRPO 的最小差异是什么？
**A：**GRPO 只有一个 policy（或 actor）自己生成、自己拿 outcome reward，然后做“同 prompt 组内”相对优势（group normalize）。  
Co-GRPO 在此基础上引入 Verifier 干预：EXP 流的 generation 过程中 Verifier 输出 `<GO>/<WAIT>` + hint，影响后续 token；并且 Verifier 自身也能被训练（LoRA 或 shared-base）。

实现判断入口：
- `algorithm.adv_estimator=grpo`：走 `AdvantageEstimator.GRPO`（`verl/trainer/ppo/core_algos.py:compute_grpo_outcome_advantage`）。
- `algorithm.adv_estimator=co_grpo`：走 `AdvantageEstimator.CO_GRPO`（`verl/trainer/ppo/core_algos.py:compute_co_grpo_advantage`），rollout 侧会触发 CoGRPO 的 by_step 逻辑（`ray_trainer.fit()` 里调用 `actor_rollout_wg.dual_stream_rollout(...)`）。

### Q2：为什么说“Co-GRPO 训练信号更稳/更可解释”？
**A：**可解释性来自“Verifier 显式输出干预文本（hint）”，稳定性来自“相对优势/对照”的归因：  
- `global_gap`：control vs exp 的 outcome 差值（粗但直观）。  
  实现：`verl/trainer/ppo/ray_trainer.py` 的 control/exp reward 计算 + `raw_relative_rewards = R_exp - R_control`。
- `cf_branch`（v2 推荐）：把 credit 从“整条样本的差值”变成“每次 intervention 的局部 uplift”，即在插入点 prefix 上做 no-hint continuation baseline。  
  实现：`ray_trainer.py` cf_branch reward mapping（`Δ = R_main(sample_uid) - R0(event_uid) - cost`）。

### Q3：control/exp 双流一定都会跑吗？
**A：**不一定。
- `global_gap` 时常见是 `stream_mode="both"`（control+exp），并可配 `trainer.parallel_control_exp=True` 把两条流分配到不同 GPU 池。
- `cf_branch` 时默认 **EXP-only**，只在需要 actor update 也吃 control（`algorithm.actor_update_streams=both`）时才额外跑全局 control。  
  实现：`ray_trainer.py` 里根据 `actor_update_streams` 决定 `want_control_stream`，并把 `stream_mode` 设为 `exp` 或 `both`。

### Q4：cf_branch 具体在优化什么？公式是什么？
**A：**每个 intervention event `e` 的 credit：
- `R_main(s)`：该样本 EXP 主轨迹的 outcome reward（同一个 sample 的所有 event 共享）
- `R0(e)`：从插入前 prefix 出发、**不插当前 hint** 的 counterfactual continuation reward
- `cost(e)`：频率/长度惩罚（可选）

当前实现（cf_branch）：
- `diff = R_main - R0`
- `delta = diff - (lambda_freq + lambda_len * hint_len/100)`
并把 `delta` 均摊到该 event 的 verifier 输出 token 上，再做一次 GRPO group normalize（用 group_index 分组）。

对应代码：`verl/trainer/ppo/ray_trainer.py`（`keep_idxs/step_rewards/costs/diffs` 这段）。

### Q5：reward 是二值（±1）的话，cf_branch 的信号会不会太粗？
**A：**会偏粗，这是客观事实：`R_main, R0 ∈ {+1,-1}` 时，`diff` 只有少数几个离散值（再减 cost）。  
解决路径主要有两类（本 repo 都有落点/接口）：
1) **提高 baseline 估计精度**：把 `cf_branch_k` 提高到 `K>1`，对同一 `event_uid` 采样 K 次 counterfactual，训练时用均值当 `R0(e)`（代码里已经做了按 `cf_event_uid` 平均）。  
   - 参数：`algorithm.cf_branch_k`
   - 聚合：`ray_trainer.py` 的 `reward_sum/reward_count -> reward_by_event_uid`
2) **换更细粒度的 reward**：例如概率型/打分型 RM（而不是 strict correctness），或者在 reward_extra_info 里引入额外 signal（需要 reward 端支持）。

### Q6：global_gap 的 shaping（headroom / improve_coef / penalty）和 cf_branch 的 cost 一样吗？
**A：**不一样（这是常见问点）。
- `global_gap` 里：先 `clamp(R_exp - R_control)`（或用 `headroom` 归一化），再对正 gap 做 boost（`improve_coef`），最后 **只在 “无提升” 时** 才扣 intervention penalty。  
  对应：`verl/trainer/ppo/ray_trainer.py` 的 `clamped_gap/positive_gap/improve_coef` 和 `torch.where(clamped_gap > 0, ..., -cost)`。
- `cf_branch` 里：当前实现是 event-level `delta = diff - cost`（即 cost 直接作用在每个 event 上），并且可选跳过 truncated event（仅对 verifier 更新跳过，actor 仍更新）。  
  对应：`ray_trainer.py` cf-branch 段落（`cost = lambda_freq + ...`）。

### Q7：`tie_no_intervention_weight` 这个参数是做什么的？
**A：**目前是 **legacy 配置项**：launcher 里仍会传（`+algorithm.verifier_reward_weighting.tie_no_intervention_weight=...`），但训练侧当前只消费了 `improve_coef`，没有使用 tie 权重。  
如果要启用，需要在 `ray_trainer.py` 的 verifier reward shaping 或 cf_branch delta 处显式加入该逻辑。

---

## 2) Rollout/插入逻辑：Verifier 怎么介入？怎么保证“像线上”？

### Q8：Verifier 的 prompt / system prompt 是哪里定义的？线上/线下如何对齐？
**A：**线上 canonical 模板放在 `verl/workers/rollout/vllm_rollout/verifier_hint_injection.py`：
- `VERIFIER_SYSTEM_PROMPT`
- `VERIFIER_INTERVENE_PROMPT`
- `build_verifier_user_prompt(question, student_response)`

by_step rollout 使用同一个模板构造 verifier 输入（在 `vllm_rollout_spmd.py` 调用 `build_verifier_user_prompt`）。

### Q9：Verifier 输出的严格格式是什么？如何解析？
**A：**严格要求（模板里写死）：
- 先 `<think>...</think>`
- 然后 **只输出一行决策**：`<GO>` 或 `<WAIT> <hint>`

解析逻辑在 `vllm_rollout_spmd.py:_extract_verifier_hint()`：
- 优先在最后一个 `</think>`（或等价 “end of thought” marker）之后寻找 `<GO>/<WAIT>`
- 对 `<WAIT>` 的 hint 做 sanitize：不能包含 `<think>/<GO>/<WAIT>/output format/final answer/```...` 等 marker，否则当作无效决策
- 如果存在 think block 但 tail 没找到合法 decision，会直接判 malformed（避免误解析 `<think>` 里的 quoted tag）。

### Q10：为什么要这么“严格解析”，会不会把有效 hint 丢掉？
**A：**这是 trade-off：严格解析的目标是防止 verifier 把“元指令/格式说明/最终答案”泄露进 hint（会污染 actor 上下文、也会污染 cold-start 数据）。  
实际调优时通常更接受“少插但干净”，否则会出现：
- hint 变成 prompt 本身的复读（没增益）
- hint 直接泄露答案（reward hacking）
- no-decision 暴涨（训练信号断崖下降）

### Q11：什么情况下会触发 “no-decision”？为什么它很致命？
**A：**no-decision = verifier 输出里解析不到合法 `<GO>/<WAIT>` 决策行。  
致命点：by_step 需要 verifier 决策决定是否插入；no-decision 多时，相当于系统退化为“无稳定干预/无稳定 credit”。

排查/证据：
- rollout 侧会统计 `verifier_no_valid_decision` / `verifier_no_valid_decision_final_like`（`vllm_rollout_spmd.py` 里汇总到 `exp_metrics`）。
- 常见根因是 verifier 被 `max_new_tokens` 截断（finish_reason=length），决策行没来得及输出。

### Q12：插 hint 的位置怎么选？为什么要“禁止马后炮”？
**A：**位置策略在 `verifier_hint_injection.py:insert_hint_tokens()`：
1) **优先插在最后一个 `</think>` 前**（且要求 `</think>` 后基本为空白），保证 hint 进入思考段，不干扰最终 answer 段。
2) 如果已经进入 post-think（`is_post_think_finalized=True`），直接跳过本次插入（记 `_hint_skipped_late_stage`）。
3) 否则尝试找 EOS marker 或 append（作为 fallback）。

“禁止马后炮”目的：避免在模型已经写完最终答案后再插提示，这会制造虚假的 uplift/伪相关（甚至促使模型学会“写完再补”）。

### Q13：什么是 “prethink rollback”？为什么要回滚？
**A：**当已经出现 `</think>` 且后面有 substantive 内容时，认为进入 final 段，此时插 hint 很可能无效/有害。  
所以系统允许 **每条样本最多一次** rollback 到 `</think>` 前（`rollback_prethink_once`），给 verifier 一个“仍可干预”的锚点，但不允许无限回滚。

对应：`verifier_hint_injection.py:rollback_prethink_once()`；rollout 侧在必要时调用 `_rollback_prethink_once(state, tokenizer)`。

### Q14：为什么 hint 必须是一人称、同语言、短句？这些是怎么被 enforce 的？
**A：**
- 强 enforce：在 prompt 模板里写死（`verifier_hint_injection.py:VERIFIER_INTERVENE_PROMPT`）。
- 弱 enforce：插入前做 sanitize / dedup / banned phrase 过滤（`vllm_rollout_spmd.py:InterventionPolicy.should_intervene()` + `_extract_verifier_hint()`）。
  - `InterventionPolicy` 会去重（与历史 hints 文本相似度 ≥0.9 就拒绝）
  - 禁止 “output format / final answer / plus one short guidance hint ...” 等元提示进入上下文

### Q15：`confidence_threshold` 是什么？用不用它会影响什么？
**A：**这是一个可选门控（默认 0 关闭）。  
当 Verifier 输出 `<WAIT>` 时，会用 `<WAIT>` 决策行附近 token 的平均 logprob 估一个 `wait_confidence`（`exp(avg_logprob)`），如果 `wait_confidence < confidence_threshold` 则拒绝干预。  
实现：
- 估计：`vllm_rollout_spmd.py:_compute_wait_confidence(...)`
- 门控：`InterventionPolicy.should_intervene()`（threshold>0 才启用）

直觉：这是“减少低把握的 WAIT”以降低错误干预。但它也可能把冷启动 verifier 的有效信号挡掉，所以线上默认一般会先关。

---

## 3) Loss/Mask：怎么保证 actor 不学会“吐 hint”？怎么保证 credit 不串？

### Q16：actor 会不会学会生成 `<WAIT>`/hint 文本本身？
**A：**正常不会（设计上明确隔离）。
- by_step rollout 会为 hint token 打 `loss_mask=0`（代表这段不是 actor action，不进 actor loss）。
- 训练侧把 `response_mask` 设为 `exp_loss_mask[:, -resp_len:]`（而不是 `exp_response_mask`），确保优势/损失只作用于 policy 生成 token。  
  对应：`verl/trainer/ppo/ray_trainer.py` 里对 `response_mask` 的设置；以及 `compute_advantage()` 在 Co-GRPO 分支里优先用 `data.batch["response_mask"]`。

### Q17：为什么要把 `response_mask` 强行替换成 `exp_loss_mask`？
**A：**如果用 EOS-based `exp_response_mask`，hint token 会被当作“on-policy token”参与 advantage/PG loss，训练会鼓励 actor 复读 hint（灾难性）。  
所以这里是强约束：**对 actor 来说 hint 只是条件信息，不是动作。**

### Q18：shared-base（verifier.update_base=True）时，为什么要 “actor 先 update，再 verifier update”？
**A：**shared-base 模式下 verifier update 会改动同一套 base weights。若先更新 verifier，再更新 actor，会造成行为策略对齐问题（old_log_probs 对不上、KL 爆、首步不稳）。  
实现：`ray_trainer.py` 里注释 “Shared-base verifier update changes actor weights too. update actor first.”；`fsdp_workers.py:update_verifier()` 也有清梯度/避免污染的兜底。

### Q19：cf_branch 下 verifier 的 group normalize 是怎么做的？为什么 group_index 里要塞 bucket？
**A：**verifier training 也用 GRPO outcome advantage 做 group normalize（为了在同类 event 间做相对优势、降低方差）。  
当前 `group_index` 形如：
`{parent_uid}:{step_idx}:{prefix_bucket}:{conf_bucket}:{hash_bucket}`

这些 bucket 的目的：
- prefix_bucket：按 prefix_len 粗分（例如每 2048 tokens 一个桶），避免“不同阶段”的 verifier event 混成一组
- conf_bucket：按 wait_confidence 粗分（可选）
- hash_bucket：按 prefix token hash mod 分桶，进一步把极其不同的状态分开

对应：`ray_trainer.py` cf_branch 构造 `group_index.append(...)`。

---

## 4) 分布式/系统：为什么 32 卡 4 节点这样分工？哪里会卡？怎么解释性能瓶颈？

### Q20：32 卡 4 节点时，每张卡实际跑的是什么角色？
**A：**默认（cf_branch / 非 parallel_control_exp）会起一个 `global_pool`，每张卡一个 `ActorRolloutRefWorker(role="actor_rollout")`：同卡同时承担
- FSDP2 actor 更新（训练）
- vLLM rollout（推理，含 verifier 推理）
- 必要时的 logprob 重算（FSDP actor 前向）

入口：
- `verl/trainer/main_ppo.py` 里构造 `resource_pool_spec`（parallel_control_exp 为 False 时只建 global_pool）
- `verl/trainer/ppo/ray_trainer.py:init_workers()` 里 spawn `actor_rollout` worker group

### Q21：什么时候会把 GPU 分成 exp_pool/control_pool？
**A：**只有当 `trainer.parallel_control_exp=True`（通常配 `global_gap` 双流）时：每节点 GPU 会拆成两份：
- `actor_rollout` 在 exp_pool
- `actor_rollout_control` 在 control_pool（强制 `rollout.n=1` 且禁 prefix cache，尽量轻量且不引入不稳定因素）

对应：`main_ppo.py` 的 mapping + `ray_trainer.py:init_workers()` 里对 control_config 的 patch。

### Q22：分布式“高效聚合”的关键实现点是什么？
**A：**三类 bottleneck 各有对策（都在本 repo 落地了）：
1) **NCCL collective 等 straggler**：用 seqlen balance 重排 batch（`RayPPOTrainer._balance_batch()`），并允许加大 `VERL_NCCL_TIMEOUT_SEC`。
2) **Ray object store spill**：cf_branch 会裁剪 cf_control batch（tail window + 丢掉长字符串字段），并默认 `reward_async=False`；rjob 建议挂 host `/dev/shm` + 提高 object store。
3) **vLLM TP 自研通信不稳**：vLLM 初始化时 `disable_custom_all_reduce=True`（`vllm_rollout_spmd.py`）。

### Q23：为什么有时 GPU 利用率很低，看起来“并发没打满”？
**A：**最常见不是 compute 不够，而是 “I/O / spill / scheduler 限制”：
- Ray object store spill 会让数据在磁盘来回（GPU 在等）
- vLLM KV cache 不足会频繁 `preempted by RECOMPUTE`（同样表现为 GPU 利用率低、step 很慢）
- `max_num_batched_tokens`/`max_num_seqs` 过小会限制 vLLM 并发

定位路径：
- 看 rank0 log 的 `timing_s/gen`、`perf/throughput`、以及 vLLM 的 preempt/recompute 告警
- 看 `scripts/run_multinodes_cgrpo_v2.sh` 的 Ray object store 参数与 `reward_async` 状态

### Q24：为什么 by_step 默认关 prefix-cache？什么时候可以开？
**A：**by_step 会多次增量 generate，prefix-cache 容易造成 cache 增长/回收不干净，长跑风险高（OOM / 不稳定）。  
本 repo 默认：by_step 时 `rollout_enable_kv_cache_optimization=False`（除非你显式 `ROLLOUT_ENABLE_KV_CACHE_OPTIMIZATION=True`）。  
对应：`scripts/run_multinodes_cgrpo_v2.sh`。

### Q25：为什么要在 `max_interventions=0` 时仍然给 `max_model_len` 加 `base_headroom=1024`？
**A：**经验性稳定性修复：当 prompt+response 正好卡在 max_model_len 边界时，某些 vLLM 版本/场景下出现过 `CUDA illegal memory access`。  
所以即使没有 interventions，也让 `max_model_len` 比 `max_prompt_length + max_response_length` 大一点，避免 exact-boundary。  
对应：`scripts/run_multinodes_cgrpo_v2.sh`（base_headroom 逻辑与注释）。

---

## 5) Reward/Eval：reward 是怎么算的？为什么 reward mean 可能为负？

### Q26：训练中 reward 的来源是什么？是 RM 还是规则？
**A：**取决于 `data_source`：
- `math_dapo*` / `aime*`：默认用 `math_dapo.compute_score()` 或 CompassVerifier（若配置了 reward endpoints 且能拿到 question）。  
  入口：`verl/utils/reward_score/__init__.py`（dispatch），`verl/utils/reward_score/math_dapo.py`（答案抽取/±1）。
- reward manager 调用链：`verl/trainer/ppo/reward.py:load_reward_manager()` → `verl/workers/reward_manager/*`.

### Q27：为什么 GRPO baseline 里 `co_grpo/exp_reward/mean` 不一定 > 0？
**A：**在 `math_dapo/aime` 口径下 outcome reward 是 ±1（对=+1/错=-1）。  
因此 `reward_mean = 2*acc - 1`：只有当 accuracy > 50% 时 reward_mean 才 > 0；否则就是负的（不是 bug）。

### Q28：cf_branch 为什么只用 tail tokens 去算 reward？会不会算错？
**A：**这是成本/可扩展性的工程选择：大量 RM/答案抽取只需要 response 尾部（`Answer:` 或最终 boxed），传全量 32k 会极度昂贵。  
cf_branch 的 rollout worker 会把 counterfactual continuation 裁成 “prompts + response tail window + attention_mask” 用于 reward（并丢掉 decoded strings 等大字段）。

如果你的 reward 需要完整推理链（例如过程打分），就不应只截 tail：可以提高 `cf_branch_reward_tail_tokens` 或改成 full-response reward（代价是 Ray/网络/存储显著变重）。

---

## 6) 常见 failure modes：被问到就照这个答（并给出定位路径）

### Q29：为什么 `no-decision rate` 很高？是 verifier 不会输出 `<GO>/<WAIT>` 吗？
**A：**高概率是 verifier 输出被截断（`max_new_tokens` 太小），导致决策行没出现。  
快速验证：
- rollout dump 里看 verifier 输出是否大量 `finish_reason=length`（或 `verifier_traj["max_new_tokens"]` 对照）
- 指标：`verifier_no_valid_decision` 和 `verifier_no_valid_decision_final_like`

解决优先级：
1) 先把 verifier `max_new_tokens` 拉到能稳定吐出决策（8k/16k/32k 逐级）
2) 再谈 hint 质量/增益

### Q30：为什么 cf_branch 的 `cf_diff_zero_ratio` 很高？
**A：**最常见原因是 reward 粗（±1）+ baseline 只采样 1 次：很多 event 的 `R_main == R0`。  
工程上可以用 `cf_branch_k>1`（对同一 event 多采样）把 `R0` 变成概率估计；或者换更细 reward。

### Q31：为什么 “插 hint 后反而更容易截断”？怎么让统计可信？
**A：**插 hint 会增加上下文长度，确实会推高 `gen_budget_exhausted/context_exhausted` 概率，导致 “真实增益被截断掩盖”。  
本 repo 的 cf_branch verifier credit 支持：
- 统计 truncated event 比例（`co_grpo/cf_trunc_event_ratio`）
- 可选跳过 truncated event 的 verifier update（`VERL_VERIFIER_SKIP_TRUNCATED=1` 或 `verifier.skip_truncated_samples=True`）

答辩口径建议：把 uplift 分成 “untruncated 子集” 与 “truncated 子集”分别报告（代码里已有 `*_untrunc` 指标）。

### Q32：为什么 sample_idx/样本对不上？怎么保证配对正确？
**A：**会对不上的原因通常有三件事叠加：
1) prompt 会被 repeat（`rollout.n`），单个 dataset item 对应多个 `sample_uid`
2) 为了平衡 token，batch 在 driver 会 reorder（`balance_batch=True`）
3) cf_control_batch 是 rollout worker 侧生成的，需要在 driver 上重新 attach reward metadata

当前实现的“正确配对键”：
- prompt 级：`uid`（prompt group id，用于 GRPO 分组）
- repeat 级：`sample_uid`（每个 repeat 唯一；用于 cf_branch / dump join）
- event 级：`event_uid = {sample_uid}:{hint_idx}`
- cf join：优先用 `cf_parent_sample_uid -> batch_sample_uid -> idx` 映射（见 `ray_trainer.py` 里 cf_eval_batch attach meta 的逻辑）

结论：不要指望 “sample_idx” 能跨所有 dump/脚本稳定；以 `sample_uid/event_uid` 做 join 才稳。

### Q33：为什么会出现 “control dump 被覆盖/污染”？现在怎么避免？
**A：**曾经的坑是 cf_control 的 reward/dump 被误当作 control stream 写进同一个目录/同一口径字段里。  
当前做法：
- 给 cf_control 明确打 `stream_type="cf_control"`（`ray_trainer.py`）
- control_batch 会剔除 exp-only 字段，且 control rollout diagnostics 会映射成同名 key 方便对比（`ray_trainer.py`）

### Q34：为什么 vLLM 会出现 `preempted by RECOMPUTE`？怎么解释它对训练效率的影响？
**A：**KV cache 不足 → vLLM 为了容纳更多 active sequences 会驱逐 KV 并在后续重算（RECOMPUTE），本质是“用算力换显存”。  
表现：
- `timing_s/gen` 显著上涨
- GPU 利用率可能不高（在等调度/重算/串行瓶颈）

常见对策（不改算法效果优先）：
- 提高 TP（`--tp 2`）让 KV/权重切分更合理
- 降 `max_num_seqs` / 降 `rollout.n` / 降 batch
- 调整 `max_num_batched_tokens`（但注意稳定性 cap）

---

## 7) “为什么不用 DeepSpeed？”（标准答法）

### Q35：为什么 verl RL 不用 DeepSpeed/ZeRO，而用 FSDP2？
**A（就本 repo 的事实口径）**：
1) 代码路径：`verl/trainer/main_ppo.py` 只支持 `fsdp/fsdp2` 或 `megatron`（没有 deepspeed 后端分支）。
2) 工程复杂度：CoGRPO 是 hybrid engine（训练+rollout+verifier 同时存在），需要频繁做
   - rollout(vLLM) ↔ 训练(FSDP) 的权重/LoRA 同步
   - checkpoint/save/load（尤其 verifier LoRA 单独保存/恢复）
   - long-context 下的稳定性兜底（mask、回滚、裁剪、object store 控制）
   这些 glue code 在本 repo 已围绕 FSDP2 打通；引入 ZeRO 需要重新实现一套 “聚合权重→喂给 vLLM / LoRA 热更新 / resume 对齐” 的链路，并带来更多 failure mode。

---

## 8) 实验/对照（reviewer 最爱问的公平性问题）

### Q36：cf_branch 更贵，怎么做 compute-matched baseline？
**A：**有两个常用口径：
- **固定 wall-clock/GPU-hours**：让 baseline 多跑 step（或更大 batch）对齐总算力
- **固定 token budget**：对齐 “生成 token 总数（含 cf_control）” 或对齐 “reward 调用次数”

建议在实验计划里明确写出对齐口径（见 `mds/plan/CO_GRPO_PAPER_PLAN_128H200.md` 的 E0m 设想）。

### Q37：LoRA 解耦 vs shared-base 的公平对照怎么做？
**A：**对齐点：
- actor base 相同（同起点 ckpt）
- 同一套 verifier prompt / 插入规则（使用 `verifier_hint_injection.py`）
- 相同 rollout 配置（response_n/max_prompt_length/token_check_interval/max_interventions/cf_branch_k）
- 只改 verifier 的更新方式：
  - LoRA：`verifier.lora_path` 非空 + `verifier.update_base=False`
  - shared-base：`verifier.lora_path=""` + `verifier.update_base=True`

并且建议额外开观测：
- `VERL_CONFLICT_DEBUG=1`（shared-base 的冲突量化，见 `mds/LORA_VS_SHARED_BASE_REPORT.md`）

---

## 9) 还能被追问的“尖锐问题”（给一句话答案）

- **Q：会不会 reward hacking（hint 泄露答案）？**  
  A：prompt 里禁止泄露最终答案 + hint 解析/过滤会丢弃包含 “final answer/格式指令/tag/代码块”等元文本；目前 `\\boxed{}` 更多用于 final-like 诊断计数（不是硬过滤），如果担心泄露可以再加一层 hint 过滤。actor loss mask 也不会学习 hint 文本。

- **Q：为什么不用 token-level reward/过程监督？**  
  A：当前实现以 outcome reward 为主是为了线上可扩展与稳定；cf_branch 提供 event-level 归因，已能显著提升 credit 精度；过程 reward 可以作为后续增强但会显著增加 RM 成本与工程复杂度。

- **Q：Verifier 真的“学会了何时插”还是只是长度/截断 bias？**  
  A：需要用 `cf_delta_*_untrunc`、finish_reason 分桶、以及 margin/entropy（表 C）来拆解；repo 已提供相应 dump 字段与分析脚本（见 `mds/LORA_VS_SHARED_BASE_REPORT.md`）。
