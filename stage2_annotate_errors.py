#!/usr/bin/env python3
"""
Error Annotation Tool for LLM Math Reasoning Mistakes

This script uses a DeepSeek API (or OpenAI-compatible) to annotate error samples
with a six-category error taxonomy: PUE, CE, RE, CPE, COE, OE.

Usage:
    export DEEPSEEK_API_KEY="your_api_key"
    python annotate_errors.py --input_file errors.jsonl --output_file labeled.jsonl

    # With existing labels for reuse
    python annotate_errors.py --input_file errors.jsonl --output_file labeled.jsonl \\
        --existing_label_file partial_labels.jsonl
"""

import os
import sys
import json
import re
import time
import argparse
from openai import OpenAI

# ===================== Default Configuration =====================
DEFAULT_API_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRY = 3
DEFAULT_SLEEP_INTERVAL = 2

VALID_TYPES = {"PUE", "CE", "RE", "CPE", "COE", "OE"}

# ===================== System Prompt (Error Taxonomy) =====================
SYSTEM_PROMPT = """
你是数学大模型错误标注专家。你需要根据给定的题目、标准解法（steps）和模型解答（model_output），判断模型解答中「最早出现的核心错误」属于以下哪一个类别。只输出纯净JSON，不带其他文字。

══════════════════════════════════════
错误类型体系（六类，严格互斥）
══════════════════════════════════════

1. PUE — 问题理解错误
模型在理解题目语义、约束条件或问题目标阶段出错，导致后续解题建立在错误的问题表征之上。
判定标准：忽略关键条件、错误理解问题要求（如优化方向反）、将原问题转化为不同问题、变量/对象语义理解错误。
判别核心：“做错题 / 做成另一个题”。

2. CE — 概念错误
模型在数学概念理解或公式选择阶段发生错误。
判定标准：使用错误公式、混淆数学概念（如排列 vs 组合）、定理/公式适用条件错误、建模时公式选择错误。
互斥规则：若错误涉及公式或知识选择错误，即使后续推理受影响，也优先归入 CE，而不归入 RE。
证据约束：CE 要求明确指出被错误使用的数学概念、定理或公式。若无法明确指出哪个概念或公式被误用，则不得归入 CE。
冲突消解：若同时存在概念错误和推理错误，优先归入 CE。

3. RE — 推理错误
模型在已调用正确数学知识的前提下，推理结构出现错误。
判定标准：推理步骤缺失且影响结论、推理跳步导致逻辑断裂、前后推导不一致、中间结论无法由前一步推出。
判别核心：“知识正确，但推不出来或推错链条”。
互斥规则：如果错误根源是知识选错，则标 CE，不标 RE。若 CE 和 RE 同时存在，优先标 CE。

4. CPE — 计算执行错误
模型在符号运算或数值计算执行阶段出错，但前置知识与推理均正确。
判定标准：加减乘除错误、代数化简错误、符号运算错误（如正负号）、正确方法但计算结果错误。
门控条件：CPE 仅在前序推理步骤均可由标准数学规则支持时适用。若推理本身有误，优先检查 CE 或 RE。
不包括：概念错误(CE)、推理错误(RE)。

5. COE — 计数与枚举错误
模型在处理离散结构（计数、枚举、组合、概率事件列举）时发生遗漏、重复或分类错误。
判定标准：漏计、重复计数、分类不完整、枚举方式错误、概率事件遗漏。

6. OE — 输出错误/幻觉
模型输出无法归入以上五类的生成失败现象。
判定标准：编造题目未给出的信息、输出与题目无关的内容、推导完全不可解释或不可验证、输出内容无法构成有效推理。
强约束：OE 是严格兜底类别。OE 仅在无法定位任何具体错误步骤时使用。若错误可追溯至 CE/RE/CPE/COE，则不得标为 OE。
OE 不得用于部分可追溯的推理错误，也不得用于仅因推理复杂或冗长导致难以理解的情况。

══════════════════════════════════════
判定优先级（必须严格遵循）
══════════════════════════════════════
1. 首先检查是否错在理解题目本身 → 若是，标 PUE。
2. 否则检查是否公式/概念选错（需明确指出哪个概念/公式被误用）→ 若是，标 CE。
3. 否则检查是否推理步骤缺失或逻辑矛盾 → 若是，标 RE。
4. 否则检查是否纯计算错误（前序推理必须正确）→ 若是，标 CPE。
5. 否则检查是否计数/枚举/概率事件遗漏 → 若是，标 COE。
6. 若以上均不适用，标 OE（极少使用，且不得用于部分可追溯的推理错误）。

══════════════════════════════════════
输出格式（严格遵守）
══════════════════════════════════════
{
  "error_type": "PUE / CE / RE / CPE / COE / OE",
  "reason": "1-2句话，指出最早的关键错误及判定依据。"
}
"""

# ===================== Utility Functions =====================
def clean_text(text: str) -> str:
    """Clean DeepSeek tokenizer special symbols (Ġ -> space, Ċ -> newline)."""
    if not text:
        return ""
    return text.replace('Ġ', ' ').replace('Ċ', '\n').strip()

def label_single_sample(client, model, problem, steps, model_output, max_tokens, temperature, max_retry, sleep_interval):
    user_prompt = f"""
题目：
{problem}

标准解法：
{steps}

模型解答：
{model_output}

请按规则标注最早出现的核心错误，仅输出JSON。
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    for retry_idx in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw_output = resp.choices[0].message.content.strip()
            finish_reason = resp.choices[0].finish_reason

            if not raw_output or finish_reason == "length":
                raise Exception("Output empty or truncated")

            # Remove markdown code blocks
            if "```" in raw_output:
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()

            # Extract JSON
            left = raw_output.find("{")
            right = raw_output.rfind("}") + 1
            if left == -1 or right <= left:
                raise Exception("No valid JSON found")

            json_str = raw_output[left:right]
            result = json.loads(json_str)

            if "error_type" not in result or "reason" not in result:
                raise Exception("Missing fields: error_type / reason")

            error_type = result["error_type"].strip().upper()
            if error_type not in VALID_TYPES:
                raise Exception(f"Invalid error type: {error_type}")
            result["error_type"] = error_type

            return result

        except Exception as e:
            print(f"  Retry {retry_idx+1}/{max_retry} failed: {str(e)[:60]}")
            time.sleep(sleep_interval)
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Annotate LLM math reasoning errors with a six-category taxonomy."
    )
    parser.add_argument("--input_file", required=True,
                        help="Path to input JSONL file with error samples (must contain 'problem', 'steps', 'model_output')")
    parser.add_argument("--output_file", required=True,
                        help="Path to output JSONL file (will be appended)")
    parser.add_argument("--existing_label_file", default=None,
                        help="Optional existing labeled file to reuse (skip API calls for matching sample_id)")
    parser.add_argument("--api_key", default=None,
                        help="DeepSeek API key (if not set, reads from DEEPSEEK_API_KEY env var)")
    parser.add_argument("--api_url", default=DEFAULT_API_URL,
                        help=f"API base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens for response (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max_retry", type=int, default=DEFAULT_MAX_RETRY,
                        help=f"Max retries per sample (default: {DEFAULT_MAX_RETRY})")
    parser.add_argument("--sleep_interval", type=float, default=DEFAULT_SLEEP_INTERVAL,
                        help=f"Sleep between requests (default: {DEFAULT_SLEEP_INTERVAL})")
    parser.add_argument("--fail_log", default=None,
                        help="File to write sample IDs that failed after all retries (optional)")

    args = parser.parse_args()

    # Determine API key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: API key not provided. Set DEEPSEEK_API_KEY environment variable or use --api_key.")
        sys.exit(1)

    client = OpenAI(base_url=args.api_url, api_key=api_key)

    # 1. Read input samples
    print(f"Reading samples from: {args.input_file}")
    samples = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Total samples: {len(samples)}")

    # 2. Load existing labels if provided
    existing_labels = {}
    if args.existing_label_file and os.path.exists(args.existing_label_file):
        with open(args.existing_label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                sid = item.get("sample_id")
                if sid:
                    existing_labels[sid] = item
        print(f"Loaded existing labels: {len(existing_labels)} (will reuse, no API calls)")

    # 3. Resume from output file
    finished_ids = set(existing_labels.keys())
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    finished_ids.add(json.loads(line)["sample_id"])
                except:
                    pass
        print(f"Output file already has: {len(finished_ids) - len(existing_labels)} entries (resuming)")

    print(f"Total samples to skip: {len(finished_ids)}")

    # 4. Process samples
    success = 0
    fail_ids = []
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.output_file, "a", encoding="utf-8") as fout:
        for idx, item in enumerate(samples, 1):
            sid = item.get("sample_id", f"sample_{idx}")

            # Skip if already processed
            if sid in finished_ids:
                success += 1
                continue

            print(f"\n[{idx}/{len(samples)}] Annotating {sid}")

            # Clean text fields (handle DeepSeek special tokens)
            problem = clean_text(item.get("problem", ""))
            steps = clean_text(item.get("steps", item.get("solution", "")))
            model_out = clean_text(item.get("model_output", ""))

            result = label_single_sample(
                client, args.model, problem, steps, model_out,
                args.max_tokens, args.temperature, args.max_retry, args.sleep_interval
            )

            if result:
                item["error_type"] = result["error_type"]
                item["error_reason"] = result["reason"]
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()
                success += 1
                print(f"  Success: {result['error_type']}")
            else:
                fail_ids.append(sid)
                print("  Failed after retries, skipped")

            time.sleep(args.sleep_interval)

    # 5. Save fail log if specified
    if args.fail_log and fail_ids:
        with open(args.fail_log, "w", encoding="utf-8") as f:
            f.write("\n".join(fail_ids))
        print(f"Failed IDs logged to: {args.fail_log}")

    print("\n" + "="*65)
    print(f"Annotation finished. Success: {success}, Failed: {len(fail_ids)}")
    if args.existing_label_file:
        print(f"Reused existing labels: {len(existing_labels)}")
    print(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    main()