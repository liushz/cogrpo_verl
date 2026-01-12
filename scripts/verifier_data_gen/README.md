# Verifier LoRA 冷启动数据生成 Pipeline v2.0

基于GRPO Rollout数据，使用Oracle模型进行"最佳干预点定位"，生成Verifier LoRA的SFT训练数据。

## 核心特性

1. **最佳干预点定位**：不再简单找第一个错误，而是找到"最佳介入时刻"
2. **双重干预模式**：
   - **Warning (预警)**：在错误发生前介入（5-10%）
   - **Correction (纠错)**：在错误发生后立即纠正（10-15%）
3. **动态语义粒度**：基于语义原子（句号、逗号、逻辑词）而非固定切分
4. **高信噪比数据**：80% GO样本，确保Verifier"平时沉默，关键时刻介入"

## 快速开始

### 基本用法

```bash
cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro

# 运行完整pipeline
bash scripts/verifier_data_gen/run_verifier_data_pipeline_v2.sh \
    /path/to/rollout_data \
    ./verifier_data_output
```

### 参数说明

```bash
bash scripts/verifier_data_gen/run_verifier_data_pipeline_v2.sh \
    <ROLLOUT_DATA_DIR> \      # GRPO Rollout数据目录
    <OUTPUT_DIR> \            # 输出目录
    [rerun_step2]             # 可选：强制重新运行Step 2
```

### 环境变量

```bash
# 测试模式：只处理前N个样本
export TEST_MODE_MAX_CASES=100

# 运行pipeline
bash scripts/verifier_data_gen/run_verifier_data_pipeline_v2.sh ...
```

## Pipeline步骤

### Step 1: 数据提取与分类
- 读取所有rollout JSONL文件
- 根据`acc`字段分类：正确 → Pool A, 错误 → Pool B
- 输出：`pool_a_correct.jsonl`, `pool_b_incorrect.jsonl`

### Step 2: 最佳干预点定位
- 使用Oracle模型（gpt-oss-120b）分析错误路径
- Oracle输出JSON：`intervention_type`, `insert_after_snippet`, `verifier_content`
- 支持批量并行处理和断点续传
- 输出：`pool_b_with_interventions.jsonl`

### Step 3: 负样本组装
- 根据`insert_after_snippet`定位插入点
- 提取插入点之前的context
- 组装成messages格式
- 输出：`negative_samples_warning.jsonl`, `negative_samples_correction.jsonl`

### Step 4: 正样本生成
- 使用语义切分对正确response进行切分
- 为每个切分点生成`<GO>`标注
- 输出：`positive_samples_go.jsonl`

### Step 5: 数据混合与平衡
- 按比例混合所有样本（80% GO, 10% Warning, 10% Correction）
- 随机打乱
- 验证数据格式
- 输出：`verifier_sft_train_data.jsonl`

## 单独运行步骤

如果需要单独运行某个步骤：

```bash
# Step 1
python3 scripts/verifier_data_gen/step1_extract_rollout_data.py \
    --rollout_dir /path/to/rollout \
    --output_dir ./intermediate \
    --test_mode 100

# Step 2
python3 scripts/verifier_data_gen/step2_optimal_intervention_localization.py \
    --input_file ./intermediate/pool_b_incorrect.jsonl \
    --output_file ./intermediate/pool_b_with_interventions.jsonl \
    --prompt_template scripts/verifier_data_gen/prompts/optimal_intervention_prompt.txt \
    --oracle_urls "http://100.101.167.1:24000/v1,http://100.101.167.1:24002/v1,..." \
    --batch_size 50 \
    --max_workers 10 \
    --resume

# Step 3
python3 scripts/verifier_data_gen/step3_assemble_negative_samples.py \
    --input_file ./intermediate/pool_b_with_interventions.jsonl \
    --output_dir ./intermediate/step3_output

# Step 4
python3 scripts/verifier_data_gen/step4_generate_positive_samples.py \
    --input_file ./intermediate/pool_a_correct.jsonl \
    --output_file ./intermediate/positive_samples_go.jsonl \
    --samples_per_item 3

# Step 5
python3 scripts/verifier_data_gen/step5_balance_and_finalize.py \
    --positive_file ./intermediate/positive_samples_go.jsonl \
    --warning_file ./intermediate/step3_output/negative_samples_warning.jsonl \
    --correction_file ./intermediate/step3_output/negative_samples_correction.jsonl \
    --output_file ./verifier_sft_train_data.jsonl \
    --positive_ratio 0.80 \
    --warning_ratio 0.10 \
    --correction_ratio 0.10
```

## 配置参数

### Oracle API配置
在`run_verifier_data_pipeline_v2.sh`中修改：
```bash
ORACLE_MODEL_URLS="http://100.101.167.1:24000/v1,http://100.101.167.1:24002/v1,..."
BATCH_SIZE=50
MAX_WORKERS=10
```

### 数据比例配置
```bash
POSITIVE_RATIO=0.80      # GO样本占比
WARNING_RATIO=0.10       # Warning样本占比
CORRECTION_RATIO=0.10    # Correction样本占比
```

### 质量过滤参数
在Step 3和Step 4中：
```bash
--min_context_length 50   # 最小context长度
--max_context_length 2048 # 最大context长度
```

## 输出数据格式

最终训练数据格式（messages格式）：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Question: ...\n\n[context before insertion]"
    },
    {
      "role": "assistant",
      "content": "<GO>" 或 "<WAIT> [指导内容]"
    }
  ],
  "intervention_type": "GO" | "Warning" | "Correction"
}
```

## 故障排查

### Oracle API调用失败
- 检查URL是否可访问
- 检查网络连接
- 增加`max_retries`和`timeout`参数

### 锚点匹配失败
- 检查`insert_after_snippet`是否在原文中唯一
- 查看`pool_b_with_interventions.jsonl`中的`oracle_response`字段
- 可能需要调整Oracle prompt以提高锚点质量

### 数据比例不平衡
- 检查Pool A和Pool B的样本数量
- 调整`SAMPLES_PER_ITEM`参数增加正样本
- 手动调整Step 5中的比例参数

## 性能优化

1. **并行处理**：Step 2支持批量并行调用Oracle API
2. **断点续传**：使用`--resume`参数可以从中断处继续
3. **测试模式**：使用`TEST_MODE_MAX_CASES`先测试小规模数据

## 依赖

- Python 3.8+
- requests
- 标准库：json, os, sys, re, random, concurrent.futures

## 文件结构

```
scripts/verifier_data_gen/
├── README.md                                    # 本文档
├── run_verifier_data_pipeline_v2.sh            # 主pipeline脚本
├── step1_extract_rollout_data.py               # 步骤1: 数据提取
├── step2_optimal_intervention_localization.py  # 步骤2: 干预点定位
├── step3_assemble_negative_samples.py         # 步骤3: 负样本组装
├── step4_generate_positive_samples.py          # 步骤4: 正样本生成
├── step5_balance_and_finalize.py              # 步骤5: 数据平衡
├── prompts/
│   └── optimal_intervention_prompt.txt         # Oracle prompt模板
└── utils/
    ├── __init__.py
    ├── api_client.py                           # Oracle API客户端
    ├── data_parser.py                          # 数据解析工具
    ├── semantic_splitter.py                    # 语义切分工具
    └── json_validator.py                      # JSON验证工具
```

## 示例输出

Pipeline完成后会输出：
- 总样本数
- GO/Warning/Correction样本数和占比
- 平均context长度
- 数据质量报告
- 样本示例

## 注意事项

1. Oracle API调用可能较慢，建议使用批量并行处理
2. 确保有足够的磁盘空间存储中间文件
3. 建议先用测试模式验证pipeline是否正常工作
4. 最终数据需要验证格式是否符合训练要求

