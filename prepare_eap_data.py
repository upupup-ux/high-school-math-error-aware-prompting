#!/usr/bin/env python3
"""
Prepare EAP data for Stage 3 experiments.

This script performs three preparation tasks:
1. Generate problem summaries (for EAP-Summary)
2. Generate structural fingerprints (for EAP-Formula)
3. Predict error types and causes (for EAP-ErrorCause)

Usage:
    # 1. Generate summaries for error database
    python prepare_eap_data.py --task summary --input errors.jsonl --output errors_with_summary.jsonl

    # 2. Generate fingerprints for error database
    python prepare_eap_data.py --task fingerprint --input errors.jsonl --output errors_with_fingerprint.jsonl

    # 3. Predict errors for test samples
    export DEEPSEEK_API_KEY="your_api_key"
    python prepare_eap_data.py --task predict_error --input test.jsonl --output test_with_pred.jsonl

    # 4. Run all tasks sequentially (summary + fingerprint on error_db, predict on test)
    python prepare_eap_data.py --task all --error_db errors.jsonl --test_file test.jsonl \
        --output_dir ./prepared_data/
"""

import os
import sys
import json
import re
import time
import argparse
from collections import Counter
from openai import OpenAI

# ===================== Default Configuration =====================
DEFAULT_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEFAULT_API_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRY = 3
DEFAULT_SLEEP_INTERVAL = 0.8

# ===================== Task 1: Summary Generation =====================
SUMMARY_SYSTEM_PROMPT = """你是一个数学问题分析专家。你需要将每道数学题提炼为一句话的数学结构摘要。

摘要规则：
1. 仅保留核心数学对象（如"椭圆"、"离心率"、"正弦函数"）
2. 仅保留关键约束条件（如"焦距为2"、"角A=60°"）
3. 仅保留求解目标（如"求离心率"、"求最大值"）
4. 忽略具体数值、人名、故事背景等无关信息
5. 使用简洁的数学语言，长度控制在20-50字

输出格式（严格遵守，只输出JSON，不要其他内容）：
{
  "summary": "一句话的数学结构摘要"
}"""

def generate_summary(client, problem_text, model_name, max_retry, sleep_interval):
    user_content = f"请为以下数学题生成数学结构摘要：\n\n题目：\n{problem_text}\n\n仅输出JSON。"
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    for attempt in range(1, max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}}
            )
            raw = resp.choices[0].message.content.strip()

            if "```" in raw:
                raw = raw.replace("```json", "").replace("```", "").strip()

            left = raw.find("{")
            right = raw.rfind("}") + 1
            if left == -1 or right == 0:
                raise ValueError("未找到JSON结构")

            result = json.loads(raw[left:right])
            return result["summary"].strip()

        except Exception as e:
            if attempt < max_retry:
                delay = sleep_interval * (2 ** (attempt - 1))
                print(f"    重试 {attempt}/{max_retry}，等待 {delay}s: {str(e)[:60]}")
                time.sleep(delay)
            else:
                print(f"    摘要生成失败: {str(e)[:60]}")
                return ""

def run_summary(input_file, output_file, api_key, api_url, model_name, max_retry, sleep_interval):
    """Generate problem summaries for each sample."""
    client = OpenAI(api_key=api_key, base_url=api_url)

    print(f"Reading: {input_file}")
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Total samples: {len(samples)}")

    # Resume
    finished_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    finished_ids.add(item.get("sample_id", ""))
        print(f"Resuming: {len(finished_ids)} already done")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    success = 0
    fail = 0

    with open(output_file, "a", encoding="utf-8") as fout:
        for idx, item in enumerate(samples, 1):
            sid = item.get("sample_id", f"sample_{idx}")
            if sid in finished_ids:
                continue

            problem = item.get("problem", "")
            print(f"\n[{idx}/{len(samples)}] {sid}")
            summary = generate_summary(client, problem, model_name, max_retry, sleep_interval)

            item["problem_summary"] = summary
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

            if summary:
                success += 1
                print(f"  ✓ {summary[:50]}...")
            else:
                fail += 1
                print(f"  ✗ failed")

            time.sleep(sleep_interval)

    print(f"Summary: success={success}, fail={fail} -> {output_file}")

# ===================== Task 2: Fingerprint Generation =====================
def extract_latex(text: str) -> str:
    pattern = r'\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\\((.*?)\\\)|(?<!\$)\$(.*?)\$'
    matches = re.findall(pattern, text, re.DOTALL)
    formulas = []
    for group in matches:
        for match in group:
            if match:
                formulas.append(match)
    return " ".join(formulas)

def get_structural_fingerprint(latex_str: str) -> dict:
    f = Counter()
    # Basic operators (cap=3)
    f['FRAC'] = len(re.findall(r'\\frac', latex_str))
    f['POWER'] = len(re.findall(r'\^', latex_str))
    f['SQRT'] = len(re.findall(r'\\sqrt', latex_str))
    f['SIN'] = len(re.findall(r'\\sin\b', latex_str))
    f['COS'] = len(re.findall(r'\\cos\b', latex_str))
    f['TAN'] = len(re.findall(r'\\tan\b', latex_str))
    f['LOG'] = len(re.findall(r'\\log', latex_str))
    f['LN'] = len(re.findall(r'\\ln', latex_str))
    f['EXP'] = len(re.findall(r'(e|\\mathrm\{e\})\^', latex_str))
    f['LIMIT'] = len(re.findall(r'\\lim', latex_str))
    f['INTEGRAL'] = len(re.findall(r'\\int', latex_str))
    f['DERIVATIVE'] = (
        len(re.findall(r"f'|y'", latex_str)) +
        len(re.findall(r'\\prime', latex_str)) +
        len(re.findall(r'\\frac\{d', latex_str))
    )
    f['SUM'] = len(re.findall(r'\\sum', latex_str))
    f['EQUAL'] = len(re.findall(r'(?<![<>])=', latex_str))
    f['INEQUALITY'] = len(re.findall(r'\\geq|\\leq|\\geqslant|\\leqslant|>=|<=|>|<', latex_str))

    # Object labels (0/1)
    if re.search(r'\\in|\\cap|\\cup|\\subset', latex_str):
        f['SET'] = 1
    if re.search(r'a_\{?n|S_\{?n', latex_str):
        f['SEQUENCE'] = 1
    if re.search(r'[fgh]\([a-zA-Z]\)', latex_str):
        f['FUNCTION'] = 1
    if re.search(r'\\vec|\\overrightarrow|\\boldsymbol', latex_str):
        f['VECTOR'] = 1
    if re.search(r'\\triangle|\\angle|\\circle|\\parallel', latex_str):
        f['GEOMETRY'] = 1
    if re.search(r'P\(|C_\{?n|A_\{?n|\\binom|\\mathrm\{C\}', latex_str):
        f['PROBABILITY'] = 1

    return {k: min(v, 3) for k, v in f.items() if v > 0}

def run_fingerprint(input_file, output_file):
    """Generate structural fingerprints for each sample."""
    print(f"Reading: {input_file}")
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Total samples: {len(samples)}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(samples, 1):
            problem = item.get("problem", "")
            latex = extract_latex(problem)
            fp = get_structural_fingerprint(latex)
            item["fingerprint"] = fp
            item["has_formula"] = len(latex.strip()) > 0
            item["fingerprint_dim"] = len(fp)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            if idx % 100 == 0:
                print(f"  Processed {idx} samples")

    print(f"Fingerprint: {len(samples)} samples -> {output_file}")

# ===================== Task 3: Error Prediction =====================
PREDICT_SYSTEM_PROMPT = """你是一名高中数学大语言模型错误分析专家。

你的任务不是解答数学题，而是预测一个数学大模型（LLM）在解决该数学问题时最可能出现的核心错误模式。

请根据题目的数学结构、考查知识点、常见大模型失败模式，预测模型最可能出现的一个错误风险，并生成与历史模型错误分析记录（error_reason）类似粒度的错误描述。

注意：
- 当前没有模型解答，因此你预测的是"潜在错误风险"，不是已经发生的错误。
- 不要假设模型已经执行了某个具体错误步骤。
- 不要描述正确解法。
- 不要列举多个可能错误，只选择一个最可能发生的核心错误。
- 输出应该描述"模型可能在哪个数学环节犯错"。

==============================
重要限制
==============================

【1. 只关注数学错误】
不要将以下非数学因素作为错误来源：
- Markdown符号
- LaTeX格式
- 排版异常
- OCR识别错误
- 字符重复或显示异常

预测应基于：数学概念、公式定理、条件理解、推理过程、计算执行。

【2. 不要虚构具体错误过程】
不要写："模型令x=1"、"模型使用了某某错误公式"
应该写："模型可能倾向于……"、"模型可能在……环节误用……"

【3. 错误类型判断规则】
- 数学知识/公式/定理理解错误 → CE
- 方法正确但计算错误 → CPE
- 知识正确但推理逻辑问题 → RE

==============================
错误类型体系
==============================
PUE — Problem Understanding Error (问题理解错误)
CE — Conceptual Error (概念错误)
RE — Reasoning Error (推理错误)
CPE — Computational Execution Error (计算执行错误)
COE — Counting & Enumeration Error (计数与枚举错误)
OE — Output Error / Hallucination (输出错误/幻觉)

==============================
error_reason生成要求
==============================
- 指出具体数学环节
- 指出错误本质
- 说明可能后果
- 长度约50-150字

==============================
当前题目
==============================
{problem}

请严格输出JSON：
{{
  "predicted_error_type": "PUE/CE/RE/CPE/COE/OE",
  "predicted_error_reason": "错误风险描述"
}}"""

def predict_error(client, problem_text, model_name, max_retry, sleep_interval):
    prompt_text = PREDICT_SYSTEM_PROMPT.format(problem=problem_text)
    messages = [{"role": "user", "content": prompt_text}]

    for attempt in range(1, max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
                stream=False
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.replace("```json", "").replace("```", "").strip()
            left = raw.find("{")
            right = raw.rfind("}") + 1
            if left == -1 or right == 0:
                raise ValueError("No JSON found")
            result = json.loads(raw[left:right])
            return result.get("predicted_error_type", ""), result.get("predicted_error_reason", ""), raw

        except Exception as e:
            if attempt < max_retry:
                delay = sleep_interval * (2 ** (attempt - 1))
                print(f"    重试 {attempt}/{max_retry}，等待 {delay}s: {str(e)[:60]}")
                time.sleep(delay)
            else:
                print(f"    预测失败: {str(e)[:60]}")
                return None, None, None

def run_predict_error(input_file, output_file, api_key, api_url, model_name, max_retry, sleep_interval):
    """Predict error types and causes for test samples."""
    client = OpenAI(api_key=api_key, base_url=api_url)

    print(f"Reading: {input_file}")
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Total samples: {len(samples)}")

    # Backup
    if os.path.exists(output_file):
        import shutil
        shutil.copy(output_file, output_file + ".bak")
        print(f"Backup created: {output_file}.bak")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    success = 0
    fail = 0

    # Load existing predictions to resume
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing = [json.loads(line.strip()) for line in f if line.strip()]
        # Map existing by problem (assuming unique)
        existing_map = {item.get("problem", ""): item for item in existing}
        print(f"Resuming from existing: {len(existing_map)} samples")
    else:
        existing_map = {}

    with open(output_file, "w", encoding="utf-8") as fout:
        # Write existing records first
        for item in existing_map.values():
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        for idx, item in enumerate(samples, 1):
            sid = item.get("sample_id", f"sample_{idx}")
            problem = item.get("problem", "")

            if problem in existing_map:
                print(f"[{idx}/{len(samples)}] {sid} already done, skipping")
                continue

            if not problem:
                print(f"[{idx}/{len(samples)}] {sid} missing problem, skipping")
                continue

            print(f"[{idx}/{len(samples)}] {sid}")
            error_type, error_reason, raw = predict_error(client, problem, model_name, max_retry, sleep_interval)

            if error_type:
                item["predicted_error_type"] = error_type
                item["predicted_error_reason"] = error_reason
                item["predicted_error_raw"] = raw
                success += 1
                print(f"  ✓ {error_type}")
            else:
                item["predicted_error_type"] = None
                item["predicted_error_reason"] = None
                item["predicted_error_raw"] = raw
                fail += 1
                print(f"  ✗ failed")

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()
            time.sleep(sleep_interval)

    print(f"Predict: success={success}, fail={fail} -> {output_file}")

# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare EAP data: summaries, fingerprints, and error predictions"
    )
    parser.add_argument("--task", type=str, required=True,
                        choices=["summary", "fingerprint", "predict_error", "all"],
                        help="Task to run: summary, fingerprint, predict_error, or all")
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL file (for summary/fingerprint/predict_error)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (for summary/fingerprint/predict_error)")
    parser.add_argument("--error_db", type=str, default=None,
                        help="Error database file (for 'all' task)")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Test file (for 'all' task)")
    parser.add_argument("--output_dir", type=str, default="./prepared_data",
                        help="Output directory (for 'all' task)")

    # API configs
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY,
                        help="DeepSeek API key (or set DEEPSEEK_API_KEY env)")
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL,
                        help="API base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name")
    parser.add_argument("--max_retry", type=int, default=DEFAULT_MAX_RETRY,
                        help="Max retries")
    parser.add_argument("--sleep_interval", type=float, default=DEFAULT_SLEEP_INTERVAL,
                        help="Sleep between API calls")

    args = parser.parse_args()

    # Validate API key for tasks that need it
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if args.task in ["summary", "predict_error", "all"] and not api_key:
        print("Error: API key required for summary and predict_error. Set DEEPSEEK_API_KEY env or use --api_key.")
        sys.exit(1)

    if args.task == "summary":
        if not args.input or not args.output:
            print("Error: --input and --output required for summary")
            sys.exit(1)
        run_summary(args.input, args.output, api_key, args.api_url, args.model,
                    args.max_retry, args.sleep_interval)

    elif args.task == "fingerprint":
        if not args.input or not args.output:
            print("Error: --input and --output required for fingerprint")
            sys.exit(1)
        run_fingerprint(args.input, args.output)

    elif args.task == "predict_error":
        if not args.input or not args.output:
            print("Error: --input and --output required for predict_error")
            sys.exit(1)
        run_predict_error(args.input, args.output, api_key, args.api_url, args.model,
                          args.max_retry, args.sleep_interval)

    elif args.task == "all":
        if not args.error_db or not args.test_file:
            print("Error: --error_db and --test_file required for 'all'")
            sys.exit(1)

        os.makedirs(args.output_dir, exist_ok=True)
        error_db_base = os.path.splitext(os.path.basename(args.error_db))[0]
        test_base = os.path.splitext(os.path.basename(args.test_file))[0]

        # Step 1: Summary on error_db
        error_summary = os.path.join(args.output_dir, f"{error_db_base}_with_summary.jsonl")
        print("\n" + "=" * 60)
        print("Step 1: Generating summaries for error database...")
        run_summary(args.error_db, error_summary, api_key, args.api_url, args.model,
                    args.max_retry, args.sleep_interval)

        # Step 2: Fingerprint on error_db (using summary file as input)
        error_fp = os.path.join(args.output_dir, f"{error_db_base}_with_fingerprint.jsonl")
        print("\n" + "=" * 60)
        print("Step 2: Generating fingerprints for error database...")
        run_fingerprint(error_summary, error_fp)

        # Step 3: Predict errors on test file
        test_pred = os.path.join(args.output_dir, f"{test_base}_with_pred.jsonl")
        print("\n" + "=" * 60)
        print("Step 3: Predicting errors for test samples...")
        run_predict_error(args.test_file, test_pred, api_key, args.api_url, args.model,
                          args.max_retry, args.sleep_interval)

        print("\n" + "=" * 60)
        print("All tasks completed!")
        print(f"Error database with summary: {error_summary}")
        print(f"Error database with fingerprint: {error_fp}")
        print(f"Test file with predictions: {test_pred}")
        print("=" * 60)

if __name__ == "__main__":
    main()