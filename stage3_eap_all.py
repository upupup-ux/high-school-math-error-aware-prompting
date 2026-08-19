#!/usr/bin/env python3
"""
Stage 3: Unified EAP Experiments on 400 Test Samples
Based on the official DeepSeek-Math EAP implementations.

Supports:
- Qwen2.5-Math-1.5B-Instruct (local)
- Qwen2.5-Math-7B-Instruct (local)
- DeepSeekMath-7B-Instruct (local)  - reference implementation
- DeepSeek-V3 (API via SiliconFlow)

Strategies:
- cot: Zero-shot CoT baseline
- problem: EAP-Problem (original problem embedding)
- summary: EAP-Summary (problem summary embedding)
- formula: EAP-Formula (structural fingerprint + Jaccard)
- errorcause: EAP-ErrorCause (predicted error cause embedding)
- fusion: EAP-Fusion (equally weighted fusion of all four)

Usage:
    export SILICONFLOW_API_KEY="your_api_key"   # for DeepSeek-V3
    python stage3_eap_all.py --model deepseek-math --strategy summary \
        --test_file /path/to/test.jsonl \
        --error_db /path/to/error_db.jsonl
"""

import os
import sys
import json
import re
import time
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

# ===================== Model Configurations =====================
MODEL_CONFIGS = {
    "qwen1.5b": {
        "type": "local",
        "model_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "max_new_tokens": 2048,
        "batch_size": 40,
        "use_chat_template": True,
        "output_suffix": "qwen1.5b"
    },
    "qwen7b": {
        "type": "local",
        "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
        "max_new_tokens": 2048,
        "batch_size": 20,
        "use_chat_template": True,
        "output_suffix": "qwen7b"
    },
    "deepseek-math": {
        "type": "local",
        "model_id": "deepseek-ai/deepseek-math-7b-instruct",
        "max_new_tokens": 2048,
        "batch_size": 2,
        "use_chat_template": False,          # 官方推荐纯文本
        "output_suffix": "deepseek-math",
        "garbled_pattern": r'[åĲĳéĩı]',
        "clean_up_tokenization_spaces": True
    },
    "deepseek-v3": {
        "type": "api",
        "model_id": "deepseek-ai/DeepSeek-V3",
        "max_new_tokens": 4096,
        "batch_size": 5,
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "output_suffix": "deepseek-v3",
        "temperature": 0.0,
        "sleep_interval": 1
    }
}

# ===================== Strategy Configurations =====================
STRATEGY_CONFIGS = {
    "cot": {
        "requires_error_db": False,
        "requires_embedding": False,
        "requires_fingerprint": False,
        "requires_pred_error": False,
        "output_suffix": "cot"
    },
    "problem": {
        "requires_error_db": True,
        "embedding_field": "problem_embedding",
        "output_suffix": "eap_problem"
    },
    "summary": {
        "requires_error_db": True,
        "embedding_field": "summary_embedding",
        "output_suffix": "eap_summary"
    },
    "formula": {
        "requires_error_db": True,
        "requires_fingerprint": True,
        "output_suffix": "eap_formula"
    },
    "errorcause": {
        "requires_error_db": True,
        "embedding_field": "error_reason_embedding",
        "output_suffix": "eap_errorcause"
    },
    "fusion": {
        "requires_error_db": True,
        "requires_embedding": True,
        "requires_fingerprint": True,
        "output_suffix": "eap_fusion"
    }
}

DEFAULT_MODEL = "deepseek-math"
DEFAULT_STRATEGY = "cot"
TOP_K = 3

# ===================== Error Type Full Names =====================
ERROR_TYPE_FULL_NAMES = {
    "PUE": "Problem Understanding Error",
    "CE": "Conceptual Error",
    "RE": "Reasoning Error",
    "CPE": "Computational Execution Error",
    "COE": "Counting & Enumeration Error",
    "OE": "Output Error / Hallucination"
}

# ===================== Utility Functions =====================
def make_sample_id(item, idx):
    typ = item.get("type", "UNKN")[:4].upper()
    level = item.get("level", "0")
    return f"{typ}-L{level}-{idx+1:03d}"

def extract_boxed(text: str) -> str:
    if not text:
        return ""
    marker = r'\boxed{'
    start_idx = text.rfind(marker)
    if start_idx == -1:
        return ""
    brace_start = start_idx + len(marker)
    depth = 1
    i = brace_start
    n = len(text)
    while i < n and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[brace_start : i-1].strip()
    match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    return match.group(1).strip() if match else ""

def is_garbled(text, pattern):
    if not pattern or not text:
        return False
    return bool(re.search(pattern, text))

def get_api_key(config):
    env_var = config.get("api_key_env")
    if not env_var:
        raise ValueError(f"API key environment variable not specified")
    key = os.environ.get(env_var)
    if not key:
        raise ValueError(f"Environment variable {env_var} not set.")
    return key

def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2))

def jaccard_similarity(fp1, fp2):
    if not fp1 or not fp2:
        return 0.0
    set1 = set(fp1.keys())
    set2 = set(fp2.keys())
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0

def load_model(config):
    if config["type"] == "local":
        print(f"Loading local model: {config['model_id']} ...")
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        return tokenizer, model, None
    elif config["type"] == "api":
        print(f"Initializing API client for {config['model_id']} ...")
        api_key = get_api_key(config)
        client = OpenAI(api_key=api_key, base_url=config["api_base"])
        return None, None, client
    else:
        raise ValueError(f"Unknown model type: {config['type']}")

# ===================== Retrieval Functions (based on DeepSeek-Math code) =====================
def retrieve_topk_problem(test_emb, error_pool, k=3):
    if test_emb is None:
        return []
    scores = []
    for item in error_pool:
        sim = cosine_similarity(test_emb, item["problem_embedding"])
        scores.append((sim, item["error_type"], item["error_reason"]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [{"error_type": et, "error_reason": reason, "similarity_score": round(sim, 4)}
            for sim, et, reason in scores[:k]]

def retrieve_topk_summary(test_emb, error_pool, k=3):
    if test_emb is None:
        return []
    scores = []
    for item in error_pool:
        sim = cosine_similarity(test_emb, item["summary_embedding"])
        scores.append((sim, item["error_type"], item["error_reason"]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [{"error_type": et, "error_reason": reason, "similarity_score": round(sim, 4)}
            for sim, et, reason in scores[:k]]

def retrieve_topk_formula(test_fp, error_pool, k=3):
    if not test_fp:
        return []
    scores = []
    for item in error_pool:
        sim = jaccard_similarity(test_fp, item["fingerprint"])
        scores.append((sim, item["error_type"], item["error_reason"]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [{"error_type": et, "error_reason": reason, "similarity_score": round(sim, 4)}
            for sim, et, reason in scores[:k]]

def retrieve_topk_errorcause(test_emb, error_pool, k=3):
    if test_emb is None:
        return []
    scores = []
    for item in error_pool:
        sim = cosine_similarity(test_emb, item["error_reason_embedding"])
        scores.append((sim, item["error_type"], item["error_reason"]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [{"error_type": et, "error_reason": reason, "similarity_score": round(sim, 4)}
            for sim, et, reason in scores[:k]]

def retrieve_topk_fusion(test_item, error_pool, k=3):
    test_problem_emb = test_item.get("problem_embedding")
    test_summary_emb = test_item.get("summary_embedding")
    test_fp = test_item.get("fingerprint")
    test_pred_emb = test_item.get("predicted_error_embedding")

    scores = []
    for err in error_pool:
        sim_problem = cosine_similarity(test_problem_emb, err.get("problem_embedding"))
        sim_summary = cosine_similarity(test_summary_emb, err.get("summary_embedding"))
        sim_formula = jaccard_similarity(test_fp, err.get("fingerprint"))
        sim_errorcause = cosine_similarity(test_pred_emb, err.get("error_reason_embedding"))

        # 有效相似度（非零？但原代码直接取全部，因为余弦可能为0，Jaccard可能为0）
        # 按照原DeepSeek-Math fusion代码，直接取全部四个，即使为0也参与平均。
        avg_sim = (sim_problem + sim_summary + sim_formula + sim_errorcause) / 4.0
        scores.append((avg_sim, err["error_type"], err["error_reason"]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [{"error_type": et, "error_reason": reason, "similarity_score": round(sim, 4)}
            for sim, et, reason in scores[:k]]

# ===================== Prompt Building (DeepSeek-Math style) =====================
def build_eap_prompt(problem_content, retrieved_errors):
    error_block = ""
    if retrieved_errors:
        lines = []
        for idx, info in enumerate(retrieved_errors, 1):
            et_abbr = info["error_type"].strip()
            et_full = ERROR_TYPE_FULL_NAMES.get(et_abbr, et_abbr)
            reason = info["error_reason"].strip()
            lines.append(f"{idx}. [{et_full}] {reason}")
        error_block = (
            "The following are error patterns identified from previous model "
            "solutions on similar mathematical problems. "
            "Use them as auxiliary references to identify potential pitfalls, "
            "but determine independently whether they are relevant to the current problem.\n\n"
            + "\n".join(lines)
            + "\n\n"
        )
    # 将错误块和问题、指令拼接到 user 消息
    user_content = error_block + problem_content + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    return user_content

def build_cot_prompt(problem_content):
    return problem_content + "\nPlease reason step by step, and put your final answer within \\boxed{}."

# ===================== Generation Functions =====================
def generate_local_batch(tokenizer, model, prompts, config):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    results = []
    prompt_len = inputs.input_ids.shape[1]
    for i in range(len(prompts)):
        resp_ids = outputs[i, prompt_len:]
        decoded = tokenizer.decode(
            resp_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=config.get("clean_up_tokenization_spaces", False)
        )
        results.append(decoded.strip())
    return results

def generate_api_batch(client, messages_list, config):
    results = []
    for messages in messages_list:
        try:
            response = client.chat.completions.create(
                model=config["model_id"],
                messages=messages,
                max_tokens=config["max_new_tokens"],
                temperature=config.get("temperature", 0.0),
                stream=False
            )
            results.append(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"  API error: {e}")
            results.append(None)
        time.sleep(config.get("sleep_interval", 0.5))
    return results

# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser(description="Stage 3: EAP experiments (DeepSeek-Math style)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use")
    parser.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY,
                        choices=list(STRATEGY_CONFIGS.keys()),
                        help="EAP strategy: cot, problem, summary, formula, errorcause, fusion")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--error_db", type=str, default=None,
                        help="Path to error database JSONL file (required for all except cot)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs (default: ./outputs)")
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model]
    strategy_config = STRATEGY_CONFIGS[args.strategy]

    if strategy_config["requires_error_db"] and not args.error_db:
        print(f"Error: --error_db is required for strategy '{args.strategy}'")
        sys.exit(1)

    output_file = os.path.join(args.output_dir, f"{model_config['output_suffix']}_{strategy_config['output_suffix']}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load error database ----
    error_pool = []
    if args.error_db:
        print(f"Loading error database: {args.error_db}")
        with open(args.error_db, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                error_pool.append({
                    "error_type": item.get("error_type", ""),
                    "error_reason": item.get("error_reason", ""),
                    "problem_embedding": item.get("problem_embedding"),
                    "summary_embedding": item.get("summary_embedding"),
                    "fingerprint": item.get("fingerprint"),
                    "error_reason_embedding": item.get("error_reason_embedding")
                })
        print(f"Error pool size: {len(error_pool)}")

    # ---- Load test samples ----
    print(f"Reading test file: {args.test_file}")
    test_samples = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    print(f"Total test samples: {len(test_samples)}")
    for idx, item in enumerate(test_samples):
        if "sample_id" not in item:
            item["sample_id"] = make_sample_id(item, idx)

    # ---- Load model ----
    global tokenizer
    tokenizer, model, client = load_model(model_config)

    # ---- Resume from current output ----
    finished_problems = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                finished_problems.add(item.get("problem", ""))
        print(f"Resuming: {len(finished_problems)} already completed")

    remaining = [item for item in test_samples if item["problem"] not in finished_problems]
    total_rem = len(remaining)
    if total_rem == 0:
        print("All samples already done.")
        return
    print(f"Remaining: {total_rem}")

    # ---- Processing loop ----
    with open(output_file, "a", encoding="utf-8") as fout:
        for batch_start in range(0, total_rem, model_config["batch_size"]):
            batch_items = remaining[batch_start: batch_start + model_config["batch_size"]]
            print(f"\nProcessing batch {batch_start+1}-{min(batch_start+len(batch_items), total_rem)} "
                  f"(remaining {total_rem - batch_start})")

            batch_retrieved = []
            batch_inputs = []

            for item in batch_items:
                # ---- Retrieve based on strategy ----
                if args.strategy == "cot":
                    retrieved = []
                elif args.strategy == "problem":
                    retrieved = retrieve_topk_problem(item.get("problem_embedding"), error_pool, TOP_K)
                elif args.strategy == "summary":
                    retrieved = retrieve_topk_summary(item.get("summary_embedding"), error_pool, TOP_K)
                elif args.strategy == "formula":
                    # 严格遵循DeepSeek-Math代码：仅当has_formula为True时检索
                    if item.get("has_formula", False):
                        retrieved = retrieve_topk_formula(item.get("fingerprint"), error_pool, TOP_K)
                    else:
                        retrieved = []
                elif args.strategy == "errorcause":
                    retrieved = retrieve_topk_errorcause(item.get("predicted_error_embedding"), error_pool, TOP_K)
                elif args.strategy == "fusion":
                    retrieved = retrieve_topk_fusion(item, error_pool, TOP_K)
                else:
                    retrieved = []

                batch_retrieved.append(retrieved)

                # ---- Build prompt (DeepSeek-Math style: pure user content) ----
                if model_config["type"] == "local":
                    if args.strategy == "cot":
                        prompt = build_cot_prompt(item["problem"])
                    else:
                        prompt = build_eap_prompt(item["problem"], retrieved)
                    # Qwen 使用 chat template，但我们的 prompt 已经是纯文本，需要用 apply_chat_template 包装
                    if model_config.get("use_chat_template", False):
                        messages = [{"role": "user", "content": prompt}]
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    # DeepSeek-Math 直接使用 prompt 文本
                    batch_inputs.append(prompt)
                else:  # API
                    if args.strategy == "cot":
                        messages = [{"role": "user", "content": build_cot_prompt(item["problem"])}]
                    else:
                        user_content = build_eap_prompt(item["problem"], retrieved)
                        messages = [{"role": "user", "content": user_content}]
                    batch_inputs.append(messages)

            # ---- Generate ----
            if model_config["type"] == "local":
                outputs = generate_local_batch(tokenizer, model, batch_inputs, model_config)
            else:
                outputs = generate_api_batch(client, batch_inputs, model_config)

            # ---- Write results ----
            for item, retrieved, output in zip(batch_items, batch_retrieved, outputs):
                if output is None:
                    item["model_output"] = "[生成失败，请重试]"
                    item["predicted_answer"] = ""
                else:
                    item["model_output"] = output
                    item["predicted_answer"] = extract_boxed(output)

                if args.strategy != "cot":
                    item["retrieved_top3_mistakes"] = retrieved

                if "garbled_pattern" in model_config:
                    if is_garbled(output, model_config["garbled_pattern"]) or is_garbled(item["predicted_answer"], model_config["garbled_pattern"]):
                        print(f"  Warning: garbled output for {item.get('sample_id')}")

                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"  ✓ {item.get('sample_id')} done")

            if model_config["type"] == "local":
                torch.cuda.empty_cache()

    print(f"\nAll done! Results saved to {output_file}")

if __name__ == "__main__":
    main()