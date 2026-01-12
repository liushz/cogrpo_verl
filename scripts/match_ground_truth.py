#!/usr/bin/env python3
"""
从 parquet 数据中匹配 ground_truth 答案并添加到 jsonl 文件中
"""
import json
import pandas as pd
import re
from pathlib import Path
import argparse
import sys


def build_question_to_answer_mapping(parquet_path):
    """从 parquet 文件建立问题到答案的映射"""
    print(f"正在读取 parquet 文件: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    question_to_answer = {}
    for idx, row in df.iterrows():
        # 从 prompt 中提取 user 消息的问题
        user_content = None
        for msg in row['prompt']:
            if msg['role'] == 'user':
                user_content = msg['content']
                break
        
        # 如果没有，从 extra_info 中获取
        if user_content is None and 'question' in row['extra_info']:
            user_content = row['extra_info']['question']
        
        if user_content and 'reward_model' in row and 'ground_truth' in row['reward_model']:
            # 清理问题文本用于匹配（去除多余空白）
            clean_question = re.sub(r'\s+', ' ', user_content.strip())
            question_to_answer[clean_question] = row['reward_model']['ground_truth']
    
    print(f"建立了 {len(question_to_answer)} 个问题到答案的映射")
    return question_to_answer


def extract_user_question(input_text):
    """从 input 字段中提取 user 消息的问题"""
    # 查找 "user\n" 之后的内容
    user_match = re.search(r'user\n(.*?)(?:\nassistant|$)', input_text, re.DOTALL)
    if user_match:
        user_content = user_match.group(1).strip()
        clean_question = re.sub(r'\s+', ' ', user_content.strip())
        return clean_question
    return None


def process_jsonl_file(input_path, output_path, question_to_answer):
    """处理单个 jsonl 文件，添加 ground_truth 字段"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    matched_count = 0
    total_count = 0
    not_matched = []
    
    print(f"\n处理文件: {input_path}")
    print(f"输出到: {output_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                total_count += 1
                
                # 从 input 中提取 user 消息
                input_text = data.get('input', '')
                clean_question = extract_user_question(input_text)
                
                ground_truth = None
                if clean_question and clean_question in question_to_answer:
                    ground_truth = question_to_answer[clean_question]
                    matched_count += 1
                elif clean_question:
                    not_matched.append((line_num, clean_question[:100]))
                
                # 添加 ground_truth 字段
                data['ground_truth'] = ground_truth
                
                # 写入新文件
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析错误: {e}")
                continue
    
    print(f"  总数据: {total_count} 条")
    print(f"  成功匹配: {matched_count} 条 ({matched_count/total_count*100:.1f}%)")
    print(f"  未匹配: {len(not_matched)} 条")
    
    if not_matched and len(not_matched) <= 10:
        print(f"\n未匹配的问题:")
        for idx, q in not_matched:
            print(f"  行 {idx}: {q}...")
    elif not_matched:
        print(f"\n前5条未匹配的问题:")
        for idx, q in not_matched[:5]:
            print(f"  行 {idx}: {q}...")
    
    return matched_count, total_count


def main():
    parser = argparse.ArgumentParser(description='从 parquet 数据中匹配 ground_truth 答案并添加到 jsonl 文件中')
    parser.add_argument('--parquet', type=str, required=True,
                        help='parquet 数据文件路径')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入 jsonl 文件目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出 jsonl 文件目录')
    parser.add_argument('--pattern', type=str, default='*.jsonl',
                        help='文件匹配模式 (默认: *.jsonl)')
    
    args = parser.parse_args()
    
    # 建立问题到答案的映射
    question_to_answer = build_question_to_answer_mapping(args.parquet)
    
    # 处理目录下的所有文件
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    jsonl_files = list(input_dir.glob(args.pattern))
    
    if not jsonl_files:
        print(f"错误: 在 {input_dir} 中未找到匹配 {args.pattern} 的文件")
        sys.exit(1)
    
    print(f"\n找到 {len(jsonl_files)} 个文件需要处理")
    
    total_matched = 0
    total_processed = 0
    
    for input_file in sorted(jsonl_files):
        output_file = output_dir / input_file.name
        matched, processed = process_jsonl_file(input_file, output_file, question_to_answer)
        total_matched += matched
        total_processed += processed
    
    print(f"\n{'='*80}")
    print(f"处理完成:")
    print(f"  总文件数: {len(jsonl_files)}")
    print(f"  总数据: {total_processed} 条")
    print(f"  总匹配: {total_matched} 条 ({total_matched/total_processed*100:.1f}%)")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

