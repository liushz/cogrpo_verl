#!/bin/bash
set -euo pipefail

# Verifier LoRA 冷启动数据生成 Pipeline v2.0
# 基于GRPO Rollout数据，使用Oracle模型进行"最佳干预点定位"

cd "$(dirname "$0")/../.."

# ========== 配置参数 ==========
ROLLOUT_DATA_DIR="${1:-/mnt/shared-storage-user/llmit/user/liuhongwei/rl_llmit/verl_train/checkpoints/grpo/interns1-8b-hf-1951_passrate_math_merged_32k_20251225_214441/rollout_w_answer}"
OUTPUT_DIR="${2:-./verifier_data_output}"

# Oracle API URLs (从server.log提取)
ORACLE_MODEL_URLS="http://100.101.167.1:24000/v1,http://100.101.167.1:24002/v1,http://100.101.167.1:24003/v1,http://100.101.167.1:24001/v1,http://100.100.41.1:24004/v1,http://100.100.41.1:24005/v1,http://100.100.41.1:24006/v1,http://100.100.41.1:24007/v1"

# 数据比例配置
POSITIVE_RATIO=0.80
WARNING_RATIO=0.10
CORRECTION_RATIO=0.10

# 处理参数
TEST_MODE_MAX_CASES="${TEST_MODE_MAX_CASES:-0}"  # 0表示处理全部
BATCH_SIZE=50
MAX_WORKERS=10
SAMPLES_PER_ITEM=3

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROMPT_TEMPLATE="${SCRIPT_DIR}/prompts/optimal_intervention_prompt.txt"

# ========== 检查输入 ==========
if [ ! -d "$ROLLOUT_DATA_DIR" ]; then
    echo "❌ Error: Rollout data directory not found: $ROLLOUT_DATA_DIR"
    exit 1
fi

if [ ! -f "$PROMPT_TEMPLATE" ]; then
    echo "❌ Error: Prompt template not found: $PROMPT_TEMPLATE"
    exit 1
fi

# ========== 创建输出目录 ==========
mkdir -p "$OUTPUT_DIR"
INTERMEDIATE_DIR="${OUTPUT_DIR}/intermediate"
mkdir -p "$INTERMEDIATE_DIR"

echo "=========================================="
echo "Verifier LoRA 冷启动数据生成 Pipeline v2.0"
echo "=========================================="
echo "Rollout Data Dir: $ROLLOUT_DATA_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Test Mode: ${TEST_MODE_MAX_CASES} (0 = all)"
echo "=========================================="
echo ""

# ========== Step 1: 数据提取与分类 ==========
echo "Step 1: Extracting and classifying rollout data..."
STEP1_OUTPUT="${INTERMEDIATE_DIR}/step1_output"
mkdir -p "$STEP1_OUTPUT"

if [ ! -f "${STEP1_OUTPUT}/pool_a_correct.jsonl" ] || [ ! -f "${STEP1_OUTPUT}/pool_b_incorrect.jsonl" ]; then
    python3 "${SCRIPT_DIR}/step1_extract_rollout_data.py" \
        --rollout_dir "$ROLLOUT_DATA_DIR" \
        --output_dir "$STEP1_OUTPUT" \
        --test_mode "$TEST_MODE_MAX_CASES"
else
    echo "  Step 1 output already exists, skipping..."
fi

if [ ! -f "${STEP1_OUTPUT}/pool_a_correct.jsonl" ] || [ ! -f "${STEP1_OUTPUT}/pool_b_incorrect.jsonl" ]; then
    echo "❌ Error: Step 1 failed to generate output files"
    exit 1
fi

echo "✅ Step 1 completed"
echo ""

# ========== Step 2: 最佳干预点定位 ==========
echo "Step 2: Locating optimal intervention points using Oracle..."
STEP2_OUTPUT="${INTERMEDIATE_DIR}/pool_b_with_interventions.jsonl"

if [ ! -f "$STEP2_OUTPUT" ] || [ "${3:-}" == "rerun_step2" ]; then
    python3 "${SCRIPT_DIR}/step2_optimal_intervention_localization.py" \
        --input_file "${STEP1_OUTPUT}/pool_b_incorrect.jsonl" \
        --output_file "$STEP2_OUTPUT" \
        --prompt_template "$PROMPT_TEMPLATE" \
        --oracle_urls "$ORACLE_MODEL_URLS" \
        --batch_size "$BATCH_SIZE" \
        --max_workers "$MAX_WORKERS" \
        --resume
else
    echo "  Step 2 output already exists, skipping..."
    echo "  (Use 'rerun_step2' as 3rd argument to force rerun)"
fi

if [ ! -f "$STEP2_OUTPUT" ]; then
    echo "❌ Error: Step 2 failed to generate output file"
    exit 1
fi

echo "✅ Step 2 completed"
echo ""

# ========== Step 3: 负样本组装 ==========
echo "Step 3: Assembling negative samples..."
STEP3_OUTPUT="${INTERMEDIATE_DIR}/step3_output"
mkdir -p "$STEP3_OUTPUT"

if [ ! -f "${STEP3_OUTPUT}/negative_samples_warning.jsonl" ] || [ ! -f "${STEP3_OUTPUT}/negative_samples_correction.jsonl" ]; then
    python3 "${SCRIPT_DIR}/step3_assemble_negative_samples.py" \
        --input_file "$STEP2_OUTPUT" \
        --output_dir "$STEP3_OUTPUT"
else
    echo "  Step 3 output already exists, skipping..."
fi

if [ ! -f "${STEP3_OUTPUT}/negative_samples_warning.jsonl" ] || [ ! -f "${STEP3_OUTPUT}/negative_samples_correction.jsonl" ]; then
    echo "❌ Error: Step 3 failed to generate output files"
    exit 1
fi

echo "✅ Step 3 completed"
echo ""

# ========== Step 4: 正样本生成 ==========
echo "Step 4: Generating positive samples (GO)..."
STEP4_OUTPUT="${INTERMEDIATE_DIR}/positive_samples_go.jsonl"

if [ ! -f "$STEP4_OUTPUT" ]; then
    python3 "${SCRIPT_DIR}/step4_generate_positive_samples.py" \
        --input_file "${STEP1_OUTPUT}/pool_a_correct.jsonl" \
        --output_file "$STEP4_OUTPUT" \
        --samples_per_item "$SAMPLES_PER_ITEM"
else
    echo "  Step 4 output already exists, skipping..."
fi

if [ ! -f "$STEP4_OUTPUT" ]; then
    echo "❌ Error: Step 4 failed to generate output file"
    exit 1
fi

echo "✅ Step 4 completed"
echo ""

# ========== Step 5: 数据混合与平衡 ==========
echo "Step 5: Balancing and finalizing training data..."
FINAL_OUTPUT="${OUTPUT_DIR}/verifier_sft_train_data.jsonl"

python3 "${SCRIPT_DIR}/step5_balance_and_finalize.py" \
    --positive_file "$STEP4_OUTPUT" \
    --warning_file "${STEP3_OUTPUT}/negative_samples_warning.jsonl" \
    --correction_file "${STEP3_OUTPUT}/negative_samples_correction.jsonl" \
    --output_file "$FINAL_OUTPUT" \
    --positive_ratio "$POSITIVE_RATIO" \
    --warning_ratio "$WARNING_RATIO" \
    --correction_ratio "$CORRECTION_RATIO"

if [ ! -f "$FINAL_OUTPUT" ]; then
    echo "❌ Error: Step 5 failed to generate output file"
    exit 1
fi

echo "✅ Step 5 completed"
echo ""

# ========== 最终统计 ==========
echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "=========================================="
echo "Final output: $FINAL_OUTPUT"
echo ""
echo "File size: $(du -h "$FINAL_OUTPUT" | cut -f1)"
echo "Line count: $(wc -l < "$FINAL_OUTPUT")"
echo ""
echo "Intermediate files saved in: $INTERMEDIATE_DIR"
echo "=========================================="

