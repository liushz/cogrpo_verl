# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
# 先导入标准库的 math，避免被同目录下的 math.py 覆盖
# 临时移除当前目录，确保导入标准库的 math
import sys
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
# 导入标准库的 math 和 random
import math  # 标准库的 math
import random

# ========== DEBUG: Track function calls ==========
_debug_call_count = 0

verify_prompt = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT 
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>: 
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""


def _postprocess_llm_response(llm_response):
    thinking_finish_words=["<conclude>", "**Final Answer**", "</think>"]
    for thinking_finish_word in thinking_finish_words:
        if thinking_finish_word in llm_response:
            llm_response = llm_response.split(thinking_finish_word)[-1]

    # only keep last 10 lines
    num_lines = len(llm_response.split("\n"))
    if num_lines > 10:
        llm_response = "\n".join(llm_response.split("\n")[-10:])

    if len(llm_response) > 1000:
        llm_response = llm_response[-1000:]

    return llm_response

def compute_score(question, gold_answer, llm_response, reward_model_clients):
    """The scoring function for CompassVerifier.

    Args:
        question: the question
        gold_answer: the gold answer
        llm_response: the response from the LLM
        reward_model_clients: list of OpenAI client instances

    Returns:
        int: 1 if correct (A), 0 otherwise (B or C)

    Raises:
        ValueError: If reward_model_clients is None or empty
        RuntimeError: If API call fails
    """
    global _debug_call_count
    _debug_call_count += 1
    call_id = _debug_call_count

    # Debug verbosity (opt-in via env)
    debug_enabled = bool(os.environ.get("VERL_DEBUG"))

    if debug_enabled:
        print(f"[CompassVerifier #{call_id}] ENTER compute_score", file=sys.stderr)
        print(f"[CompassVerifier #{call_id}] reward_model_clients type: {type(reward_model_clients)}", file=sys.stderr)
        print(f"[CompassVerifier #{call_id}] reward_model_clients: {reward_model_clients}", file=sys.stderr)

    # 检查 reward_model_clients；离线或未配置时直接返回0，避免训练中断
    if not reward_model_clients or not isinstance(reward_model_clients, list):
        if debug_enabled:
            print(f"[CompassVerifier #{call_id}] reward_model_clients missing or invalid, returning 0", file=sys.stderr)
        print("reward_model_clients missing or invalid, returning 0 reward to continue training")
        return 0
    
    # 过滤掉 None 或无效的客户端
    valid_clients = []
    for client in reward_model_clients:
        try:
            base_url = getattr(getattr(client, "_client", None), "base_url", None)
            if client is not None and base_url and str(base_url).startswith(("http://", "https://")):
                valid_clients.append(client)
        except Exception:
            continue
    if not valid_clients:
        if debug_enabled:
            print(f"[CompassVerifier #{call_id}] No valid reward model clients after filtering, returning 0", file=sys.stderr)
        print("No valid reward model clients after filtering, returning 0 reward")
        return 0

    if debug_enabled:
        print(f"[CompassVerifier #{call_id}] Found {len(valid_clients)} valid clients", file=sys.stderr)

    # 随机选择一个客户端
    api_client = random.choice(valid_clients)
    api_base_url = getattr(getattr(api_client, "_client", None), "base_url", None)
    if debug_enabled:
        print(f"[CompassVerifier #{call_id}] Using API endpoint: {api_base_url}", file=sys.stderr)
    
    # 构建 prompt
    try:
        llm_response = _postprocess_llm_response(llm_response)
        prompt_content = verify_prompt.replace("{question}", str(question)).replace("{gold_answer}", str(gold_answer)).replace("{llm_response}", str(llm_response))
    except Exception as e:
        raise ValueError(f"Failed to format prompt: {e}")
    
    # API 调用，添加错误处理
    try:
        if debug_enabled:
            print(f"[CompassVerifier #{call_id}] Calling API...", file=sys.stderr)
        response = api_client.chat.completions.create(
            model="cv-32b",
            messages=[
                {"role": "user", "content": prompt_content},
            ],
            temperature=0.0,
        )
        if debug_enabled:
            print(f"[CompassVerifier #{call_id}] API response received", file=sys.stderr)
    except Exception as e:
        if debug_enabled:
            print(f"[CompassVerifier #{call_id}] API call FAILED: {e}", file=sys.stderr)
        raise RuntimeError(f"API call failed: {e}")

    # 检查响应有效性
    if not response or not hasattr(response, 'choices') or not response.choices:
        if debug_enabled:
            print(f"[CompassVerifier #{call_id}] Invalid API response: no choices", file=sys.stderr)
        raise RuntimeError("Invalid API response: no choices")

    if not response.choices[0] or not hasattr(response.choices[0], 'message'):
        if debug_enabled:
            print(f"[CompassVerifier #{call_id}] Invalid API response: no message", file=sys.stderr)
        raise RuntimeError("Invalid API response: no message")

    content = response.choices[0].message.content
    if not content:
        if debug_enabled:
            print(f"[CompassVerifier #{call_id}] Invalid API response: empty content", file=sys.stderr)
        raise RuntimeError("Invalid API response: empty content")

    if debug_enabled:
        print(f"[CompassVerifier #{call_id}] API returned: '{content.strip()}'", file=sys.stderr)

    # 判断结果：A (CORRECT) 返回 1，其他返回 0
    content_upper = content.strip().upper()
    if content_upper == "A" or "CORRECT" in content_upper:
        if debug_enabled:
            print(f"[CompassVerifier #{call_id}] RETURNING 1 (CORRECT)", file=sys.stderr)
        return 1
    else:
        if debug_enabled:
            print(f"[CompassVerifier #{call_id}] RETURNING 0 (INCORRECT)", file=sys.stderr)
        return 0



if __name__ == "__main__":
    from openai import OpenAI
    question = "What is the capital of France?"
    gold_answer = "Paris"
    llm_response = "The capital of France is Paris."
    
    # 处理环境变量中的 URLs
    urls_str = os.environ.get("REWARD_MODEL_URLS", "")
    urls = [url.strip() for url in urls_str.split(",") if url.strip()]  # 过滤空字符串
    
    if not urls:
        print("警告: REWARD_MODEL_URLS 环境变量未设置或为空")
        print("请设置环境变量: export REWARD_MODEL_URLS='http://host1:port1/v1,http://host2:port2/v1'")
        reward_model_clients = None
    else:
        print(f"找到 {len(urls)} 个 URL: {urls}")
        reward_model_clients = [OpenAI(base_url=url, api_key="NONE") for url in urls]
    
    # 测试代码
    if reward_model_clients:
        for i in range(10):
            test_gold_answer = random.choice(["Paris", "Berlin", "Madrid", "Rome", "Athens"])
            print(f"\n测试 {i+1}: gold_answer = {test_gold_answer}")
            try:
                result = compute_score(question, test_gold_answer, llm_response, reward_model_clients)
                print(f"结果: {result} ({'正确' if result == 1 else '错误'})")
            except Exception as e:
                print(f"错误: {e}")
    else:
        print("\n跳过测试: 没有可用的客户端")
