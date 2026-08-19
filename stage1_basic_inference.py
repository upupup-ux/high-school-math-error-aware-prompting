#!/usr/bin/env python3
"""
Unified inference script for 960 high school math problems.
Supports:
- Qwen2.5-Math-1.5B-Instruct (local)
- Qwen2.5-Math-7B-Instruct (local)
- DeepSeekMath-7B-Instruct (local)
- DeepSeek-V3 (API via SiliconFlow)

Usage:
    export SILICONFLOW_API_KEY="your_api_key"  # only for DeepSeek-V3
    python run_inference.py --model qwen7b --test_file /path/to/test.jsonl
"""

import os
import json
import re
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

# ===================== 配置 =====================
MODEL_CONFIGS = {
    "qwen1.5b": {
        "type": "local",
        "model_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "max_new_tokens": 2048,
        "batch_size": 8,
        "use_chat_template": True,
        "system_prompt": "Please reason step by step, and put your final answer within \\boxed{}.",
        "output_suffix": "qwen1.5b_960_output"
    },
    "qwen7b": {
        "type": "local",
        "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
        "max_new_tokens": 2048,
        "batch_size": 8,
        "use_chat_template": True,
        "system_prompt": "Please reason step by step, and put your final answer within \\boxed{}.",
        "output_suffix": "qwen7b_960_output"
    },
    "deepseek-math": {
        "type": "local",
        "model_id": "deepseek-ai/deepseek-math-7b-instruct",
        "max_new_tokens": 2048,
        "batch_size": 2,
        "use_chat_template": False,
        "system_prompt": None,
        "output_suffix": "deepseek-math_960_output",
        "garbled_pattern": r'[åĲĳéĩı]',
        "clean_up_tokenization_spaces": True
    },
    "deepseek-v3": {
        "type": "api",
        "model_id": "deepseek-ai/DeepSeek-V3",
        "max_new_tokens": 4096,
        "batch_size": 5,
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY",  # 从环境变量读取
        "output_suffix": "deepseek-v3_960_output",
        "temperature": 0.0,
        "sleep_interval": 0.5
    }
}

DEFAULT_MODEL = "qwen7b"

# ===================== 工具函数 =====================
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
        raise ValueError(f"API key environment variable not specified for {config['model_id']}")
    key = os.environ.get(env_var)
    if not key:
        raise ValueError(f"Environment variable {env_var} not set. Please set it before running.")
    return key

# ===================== 模型加载 =====================
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

# ===================== 生成函数 =====================
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
            output = response.choices[0].message.content.strip()
            results.append(output)
        except Exception as e:
            print(f"  API error: {e}")
            results.append(None)
        time.sleep(config.get("sleep_interval", 0.5))
    return results

# ===================== 提示构建 =====================
def build_prompt_local(problem, config):
    if config.get("use_chat_template", False):
        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": problem}
        ]
        # tokenizer is global, accessed in main loop
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # DeepSeek-Math: pure text concatenation
        return problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."

def build_messages_api(problem, config):
    return [
        {"role": "user", "content": problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
    ]

# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(description="Run inference on 960 math problems")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    output_file = os.path.join(args.output_dir, f"{config['output_suffix']}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载测试集
    print(f"Reading test file: {args.test_file}")
    test_samples = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    print(f"Total samples: {len(test_samples)}")
    for idx, item in enumerate(test_samples):
        if "sample_id" not in item:
            item["sample_id"] = make_sample_id(item, idx)

    # 加载模型
    global tokenizer
    tokenizer, model, client = load_model(config)

    # 断点续跑
    finished_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        finished_ids.add(json.loads(line).get("sample_id", ""))
                    except:
                        pass
        print(f"Resuming: {len(finished_ids)} already completed")

    remaining = [i for i, it in enumerate(test_samples) if it["sample_id"] not in finished_ids]
    total_rem = len(remaining)
    if total_rem == 0:
        print("All samples already done.")
        return
    print(f"Remaining: {total_rem}")

    # 生成循环
    with open(output_file, "a", encoding="utf-8") as fout:
        for batch_start in range(0, total_rem, config["batch_size"]):
            batch_indices = remaining[batch_start: batch_start + config["batch_size"]]
            batch_items = [test_samples[i] for i in batch_indices]
            print(f"\nProcessing batch {batch_start+1}-{min(batch_start+len(batch_indices), total_rem)} "
                  f"(remaining {total_rem - batch_start})")

            if config["type"] == "local":
                # 构建 prompts
                prompts = [build_prompt_local(it["problem"], config) for it in batch_items]
                outputs = generate_local_batch(tokenizer, model, prompts, config)
                for item, out in zip(batch_items, outputs):
                    if out is None:
                        print(f"  Warning: generation failed for {item['sample_id']}")
                        continue
                    item["model_output"] = out
                    item["predicted_answer"] = extract_boxed(out)
                    # 可选：乱码检测（仅DeepSeek-Math）
                    if "garbled_pattern" in config:
                        if is_garbled(out, config["garbled_pattern"]) or is_garbled(item["predicted_answer"], config["garbled_pattern"]):
                            print(f"  Warning: garbled output for {item['sample_id']}")
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()
                    print(f"  ✓ {item['sample_id']} done")
            else:  # API
                messages_list = [build_messages_api(it["problem"], config) for it in batch_items]
                outputs = generate_api_batch(client, messages_list, config)
                for item, out in zip(batch_items, outputs):
                    if out is None:
                        print(f"  ✗ {item['sample_id']} failed")
                        continue
                    item["model_output"] = out
                    item["predicted_answer"] = extract_boxed(out)
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()
                    print(f"  ✓ {item['sample_id']} done")

            # 清理显存（本地模型）
            if config["type"] == "local":
                torch.cuda.empty_cache()

    print(f"\nAll done! Results saved to {output_file}")

if __name__ == "__main__":
    main()