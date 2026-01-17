#!/bin/bash
# 快速验证Verifier LoRA修复是否生效

echo "========================================"
echo "Verifier LoRA Fix - 快速验证脚本"
echo "========================================"
echo ""

LOG_DIR="/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/logs"

# 查找最新日志
LATEST_LOG=$(ls -t "$LOG_DIR"/verl_log_*.txt 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 没有找到日志文件，请先启动实验"
    echo ""
    echo "启动命令："
    echo "  cd /mnt/shared-storage-user/liuhongwei/main_works/repos/repro"
    echo "  bash run_cgrpo_mini_cluster.sh"
    exit 1
fi

echo "📄 最新日志: $(basename $LATEST_LOG)"
LOG_TIME=$(stat -c %y "$LATEST_LOG" | cut -d'.' -f1)
echo "⏰ 修改时间: $LOG_TIME"
LOG_SIZE=$(du -h "$LATEST_LOG" | cut -f1)
echo "📊 文件大小: $LOG_SIZE"
echo ""

# 1. 检查LoRA重载成功
echo "1️⃣  检查Verifier LoRA重载："
RELOAD_SUCCESS=$(grep -c "Successfully reloaded Verifier LoRA via AsyncLLMEngine API" "$LATEST_LOG" 2>/dev/null || echo "0")
RELOAD_FAIL=$(grep -c "does not support add_lora" "$LATEST_LOG" 2>/dev/null || echo "0")

if [ "$RELOAD_SUCCESS" -gt 0 ]; then
    echo "   ✅ 成功: $RELOAD_SUCCESS 次"
else
    echo "   ❌ 未找到成功记录"
fi

if [ "$RELOAD_FAIL" -gt 0 ]; then
    echo "   ⚠️  仍有失败: $RELOAD_FAIL 次"
else
    echo "   ✅ 没有失败警告"
fi
echo ""

# 2. 检查训练步骤
echo "2️⃣  检查训练进度："
STEPS=$(grep -o "training/global_step:[0-9.]*" "$LATEST_LOG" | tail -1 | cut -d: -f2)
if [ -n "$STEPS" ]; then
    STEP_INT=$(echo "$STEPS" | cut -d. -f1)
    echo "   当前步骤: $STEPS"
    
    if [ "$STEP_INT" -gt 2 ]; then
        echo "   ✅ 超过2步，修复成功！"
    elif [ "$STEP_INT" -eq 2 ]; then
        echo "   ⏳ 刚好2步，继续等待..."
    else
        echo "   ⚠️  步数少于2，检查是否在初始化"
    fi
else
    echo "   ⏳ 还未开始训练（或正在初始化）"
fi
echo ""

# 3. 检查Verifier状态
echo "3️⃣  检查Verifier状态："
VERIFIER_ENABLED=$(grep "verifier/enabled" "$LATEST_LOG" | tail -1 | grep -o "enabled:[0-9.]*" | cut -d: -f2)
VERIFIER_LR=$(grep "verifier/lr" "$LATEST_LOG" | tail -1 | grep -o "lr:[0-9.e-]*" | cut -d: -f2)
VERIFIER_BATCH=$(grep "verifier/train_batch_size" "$LATEST_LOG" | tail -1 | grep -o "batch_size:[0-9.]*" | cut -d: -f2)

if [ -n "$VERIFIER_ENABLED" ]; then
    echo "   Verifier启用: $VERIFIER_ENABLED"
fi
if [ -n "$VERIFIER_LR" ]; then
    echo "   Verifier学习率: $VERIFIER_LR"
fi
if [ -n "$VERIFIER_BATCH" ]; then
    echo "   训练batch大小: $VERIFIER_BATCH"
fi
echo ""

# 4. 检查同步时间
echo "4️⃣  检查LoRA同步时间："
RELOAD_TIME=$(grep "verifier/lora_reload_s" "$LATEST_LOG" | tail -1 | grep -o "reload_s:[0-9.]*" | cut -d: -f2)
if [ -n "$RELOAD_TIME" ]; then
    echo "   重载时间: ${RELOAD_TIME}秒"
    
    # 检查是否合理
    RELOAD_FLOAT=$(echo "$RELOAD_TIME" | awk '{print $1}')
    if (( $(echo "$RELOAD_FLOAT < 10.0" | bc -l) )); then
        echo "   ✅ 时间合理（< 10秒）"
    else
        echo "   ⚠️  时间较长（>= 10秒）"
    fi
else
    echo "   ⏳ 还未进行LoRA同步"
fi
echo ""

# 5. 总体评估
echo "========================================"
echo "📊 总体评估："

TOTAL_ISSUES=0

# 检查各项指标
if [ "$RELOAD_SUCCESS" -eq 0 ]; then
    echo "❌ LoRA重载未成功或未开始"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
elif [ "$RELOAD_FAIL" -gt 0 ]; then
    echo "⚠️  LoRA重载仍有部分失败"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
fi

if [ -n "$STEP_INT" ] && [ "$STEP_INT" -le 2 ]; then
    echo "⏳ 训练还未超过2步，继续等待"
fi

if [ "$TOTAL_ISSUES" -eq 0 ] && [ -n "$STEP_INT" ] && [ "$STEP_INT" -gt 2 ]; then
    echo ""
    echo "🎉 修复成功！"
    echo ""
    echo "✅ Verifier LoRA能够正常重载"
    echo "✅ 训练能够持续进行（超过2步）"
    echo "✅ 所有指标正常"
    echo ""
    echo "下一步："
    echo "  1. 让训练继续运行"
    echo "  2. 监控训练质量指标"
    echo "  3. 等待训练完成"
elif [ "$TOTAL_ISSUES" -eq 0 ]; then
    echo ""
    echo "⏳ 初始化中，指标正常"
    echo ""
    echo "请等待5-10分钟后重新运行此脚本"
else
    echo ""
    echo "❌ 发现 $TOTAL_ISSUES 个问题"
    echo ""
    echo "请检查："
    echo "  1. 代码是否已更新（git log）"
    echo "  2. 是否使用了正确的分支"
    echo "  3. 日志文件中的详细错误"
    echo ""
    echo "详细日志: tail -100 $LATEST_LOG"
fi

echo "========================================"
