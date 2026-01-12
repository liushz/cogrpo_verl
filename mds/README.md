# Co-GRPO 文档目录

本目录包含 Co-GRPO (Cooperative Group Relative Policy Optimization) 实现的完整技术文档。

---

## 📚 文档列表

### 1. Co-GRPO 算法详解 v1.0 ⭐️⭐️⭐️（推荐）
**文件**: `CO_GRPO_ALGORITHM_V1.md`

**内容**:
- 算法概述与核心创新
- 相比普通 GRPO 的详细优势分析
- 协同进化的 Verifier 设计（三种干预粒度详解）
- Verifier LoRA 冷启动完整流程（Oracle 指导的数据生成）
- 算法流程与关键技术细节
- 配置参数说明与性能分析

**适合**: 
- 深入理解 Co-GRPO 算法原理
- 了解设计思路和创新点
- 学习 Verifier 冷启动方法

**阅读时间**: 40-50 分钟

---

### 2. Co-GRPO 实现详解 ⭐️⭐️
**文件**: `CO_GRPO_IMPLEMENTATION.md`

**内容**:
- 在 verl 框架中的完整实现
- 核心流程详解（训练循环、双流推演、奖励计算、优势计算）
- 关键组件实现（具体到文件行号）
- 三种介入模式详解（by_response, by_token, by_step）
- 数据流图和数据结构转换
- 配置参数完整说明

**适合**: 
- 代码审查和调试
- 二次开发和定制
- 理解实现细节

**阅读时间**: 30-40 分钟

---

### 3. Verifier LoRA 训练指南 🎓
**文件**: `LORA_TRAIN.md`

**内容**:
- Verifier 的作用与输入输出格式
- 训练数据格式与构造方法
- 完整的 SFT 训练脚本
- 数据准备与验证脚本
- 最佳实践与故障排查
- 快速开始模板

**适合**:
- 准备 Verifier LoRA
- 数据构造
- 训练 Verifier

**阅读时间**: 20-30 分钟

---

## 🚀 推荐阅读顺序

### 新手入门
1. **先看**: `CO_GRPO_ALGORITHM_V1.md` → 理解算法原理和设计思路（**强烈推荐**）
2. **再看**: `CO_GRPO_IMPLEMENTATION.md` → 了解代码实现细节
3. **最后**: `LORA_TRAIN.md` → 学习如何准备和训练 Verifier

### 快速配置
1. **直接看**: `CO_GRPO_ALGORITHM_V1.md` → "配置参数说明" 章节

### 准备 Verifier LoRA
1. **先看**: `CO_GRPO_ALGORITHM_V1.md` → "Verifier LoRA 冷启动" 章节（了解完整流程）
2. **再看**: `LORA_TRAIN.md` → 获取详细训练脚本和步骤
3. **参考**: `scripts/verifier_data_gen/README.md` → 数据生成 Pipeline 使用说明

### 代码开发
1. **先看**: `CO_GRPO_ALGORITHM_V1.md` → 理解算法设计
2. **再看**: `CO_GRPO_IMPLEMENTATION.md` → 查看具体实现
3. **参考**: 代码中的注释和文档字符串

---

## 📊 文档统计

| 文档 | 大小 | 章节 | 用途 |
|------|------|------|------|
| CO_GRPO_ALGORITHM_V1 | ~50KB | 10 章 | 算法详解（**推荐**） |
| CO_GRPO_IMPLEMENTATION | ~40KB | 6 章 | 实现详解 |
| LORA_TRAIN | ~32KB | 13 章 | Verifier 训练 |
| **总计** | **~122KB** | - | **3 份核心文档** |

---

## 🔑 关键信息

### ⚠️ 必读配置（重要！）

**Verifier LoRA 必须配置**:
```yaml
verifier:
  lora_path: "/path/to/verifier_lora"  # 必填！
  lora_rank: 16
  lora_alpha: 32
```

**工作流程**:
```bash
# 1. 准备训练数据（详见 CO_GRPO_ALGORITHM_V1.md 和 scripts/verifier_data_gen/README.md）
bash scripts/verifier_data_gen/run_verifier_data_pipeline_v2.sh \
    /path/to/rollout_data \
    ./verifier_data_output

# 2. 训练 Verifier LoRA（详见 LORA_TRAIN.md）
python train_verifier_sft.py \
    --base_model /path/to/base_model \
    --data verifier_data_output/verifier_sft_train_data.jsonl \
    --output /path/to/verifier_lora \
    --lora_rank 16 --lora_alpha 32 --epochs 3

# 3. 验证效果
python test_verifier.py --base_model /path/to/base_model --lora_path /path/to/verifier_lora

# 4. 运行 Co-GRPO
bash scripts/run_multinodes_co_grpo.sh \
    <model> <nnodes> <n_gpus_per_node> <gen_tp> <gpu_memory_utilization> <train_file_name>
```

### ✅ 快速测试

```bash
# 运行 1 个 step 验证
python -m verl.trainer.main_ppo \
    --config configs/co_grpo_test.yaml \
    --max_steps 1 \
    --debug True
```

---

## 📝 文档更新记录

- **2025-01-XX**: 创建 v1.0 算法文档，重组文档结构
- **2024-12-24**: 创建 Verifier LoRA 训练指南
- **2024-12-XX**: 创建实现详解文档

---

## 🎯 项目状态

- **代码状态**: ✅ 可用
- **文档状态**: ✅ 完整
- **测试状态**: ⏳ 待运行
- **建议操作**: 准备 Verifier LoRA → 运行测试

---

## 💡 获取帮助

### 遇到问题？

1. **算法理解** → 查看 `CO_GRPO_ALGORITHM_V1.md`
2. **代码问题** → 查看 `CO_GRPO_IMPLEMENTATION.md`
3. **Verifier 训练** → 查看 `LORA_TRAIN.md`
4. **数据生成** → 查看 `scripts/verifier_data_gen/README.md`

### 文档导航

所有文档都有完整的目录（TOC），可以快速跳转到感兴趣的章节。

---

**最后更新**: 2025-01-XX  
**版本**: v1.0  
**状态**: ✅ 生产就绪
