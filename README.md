# high-school-math-error-aware-prompting
Code and dataset for "From Error Analysis to Error-Aware Prompting: Structured Error Knowledge for High School"

# Error-Aware Prompting (EAP) for High School Mathematical Reasoning

This repository contains the code and experimental framework for the paper:  
**"From Error Analysis to Error-Aware Prompting: Structured Error Knowledge for High School Mathematical Reasoning"**  

The project systematically investigates how structured error knowledge (error type–cause pairs) can be leveraged via Error-Aware Prompting (EAP) to improve LLM reasoning on high school math problems.

---

## Repository Structure

```
.
├── stage1_basic_inference.py      # 960-problem baseline CoT inference
├── stage2_annotate_errors.py      # Error annotation with six-category taxonomy
├── stage3_eap_all.py              # Unified EAP experiments (CoT + 5 strategies)
├── prepare_eap_data.py            # Data preparation: summaries, fingerprints, predictions
├── embed_vectors.py               # Generate normalized embeddings via BGE
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Requirements

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
openai>=1.0.0
transformers>=4.36.0
torch>=2.0.0
accelerate>=0.25.0
numpy>=1.24.0
sentence-transformers>=2.2.0
```

**GPU Recommendation:** For local inference, a GPU with at least 16 GB VRAM is recommended (tested on Kaggle dual T4).

---

## Data Preparation

Before running experiments, you need to prepare the following data files.

### 1. Raw Datasets

- **Error database:** JSONL file with fields: `problem`, `steps`, `model_output`, `answer`
- **Test set:** JSONL file with fields: `problem`, `answer`

### 2. Generate Problem Summaries (for EAP-Summary)

```bash
export DEEPSEEK_API_KEY="your-api-key"
python prepare_eap_data.py --task summary \
    --input error_db.jsonl \
    --output error_db_with_summary.jsonl
```

### 3. Generate Structural Fingerprints (for EAP-Formula)

```bash
python prepare_eap_data.py --task fingerprint \
    --input error_db_with_summary.jsonl \
    --output error_db_with_fingerprint.jsonl
```

### 4. Predict Error Types & Causes (for EAP-ErrorCause)

```bash
python prepare_eap_data.py --task predict_error \
    --input test_400.jsonl \
    --output test_400_with_pred.jsonl
```

### 5. Generate Normalized Semantic Embeddings

Only required for: **EAP-Problem, EAP-Summary, EAP-ErrorCause,** and **EAP-Fusion**.  
Not required for: **EAP-Formula** (uses Jaccard similarity on fingerprints).

All embeddings are generated using `BGE-large-zh-v1.5` and normalized to unit length.

Create `embed_vectors.py`:

```python
from sentence_transformers import SentenceTransformer
import json

def add_embeddings(input_file, output_file, text_field, embedding_field, model_path="BAAI/bge-large-zh-v1.5"):
    model = SentenceTransformer(model_path)
    model.eval()
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    texts = [item.get(text_field, '') for item in samples]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32)
    for item, emb in zip(samples, embeddings):
        item[embedding_field] = emb.tolist()
    with open(output_file, 'w') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Embeddings added to {output_file}")

if __name__ == "__main__":
    # Error database
    add_embeddings("error_db_with_fingerprint.jsonl", "error_db_embedded.jsonl",
                   "problem", "problem_embedding")
    add_embeddings("error_db_with_summary.jsonl", "error_db_embedded.jsonl",
                   "problem_summary", "summary_embedding")
    add_embeddings("error_db_embedded.jsonl", "error_db_embedded.jsonl",
                   "error_reason", "error_reason_embedding")

    # Test set
    add_embeddings("test_400_with_pred.jsonl", "test_400_embedded.jsonl",
                   "problem", "problem_embedding")
    add_embeddings("test_400_with_pred.jsonl", "test_400_embedded.jsonl",
                   "problem_summary", "summary_embedding")
    add_embeddings("test_400_with_pred.jsonl", "test_400_embedded.jsonl",
                   "predicted_error_reason", "predicted_error_embedding")
```

Run the script:

```bash
python embed_vectors.py
```

After this step, you will have:

- **Error database** with `problem_embedding`, `summary_embedding`, `error_reason_embedding`
- **Test set** with `problem_embedding`, `summary_embedding`, `predicted_error_embedding`

> **Note:** If you only plan to run EAP-Formula, you can skip the entire embedding step.

---

## Running the Experiments

### Stage 1: Baseline CoT on 960 Problems

```bash
python stage1_basic_inference.py --model qwen7b --test_file balance_960_test.jsonl
```

Available models: `qwen1.5b`, `qwen7b`, `deepseek-math`, `deepseek-v3`.

### Stage 2: Error Annotation

```bash
export DEEPSEEK_API_KEY="your-api-key"
python stage2_annotate_errors.py \
    --input_file error_samples.jsonl \
    --output_file error_samples_labeled.jsonl \
    --existing_label_file partial_labels.jsonl   # optional
```

### Stage 3: EAP Experiments on 400 Samples

```bash
# CoT baseline
python stage3_eap_all.py --model qwen7b --strategy cot \
    --test_file test_400_embedded.jsonl

# EAP-Problem
python stage3_eap_all.py --model qwen7b --strategy problem \
    --test_file test_400_embedded.jsonl \
    --error_db error_db_embedded.jsonl

# EAP-Summary
python stage3_eap_all.py --model qwen7b --strategy summary \
    --test_file test_400_embedded.jsonl \
    --error_db error_db_embedded.jsonl

# EAP-Formula
python stage3_eap_all.py --model qwen7b --strategy formula \
    --test_file test_400_embedded.jsonl \
    --error_db error_db_embedded.jsonl

# EAP-ErrorCause
python stage3_eap_all.py --model qwen7b --strategy errorcause \
    --test_file test_400_embedded.jsonl \
    --error_db error_db_embedded.jsonl

# EAP-Fusion
python stage3_eap_all.py --model qwen7b --strategy fusion \
    --test_file test_400_embedded.jsonl \
    --error_db error_db_embedded.jsonl
```

All outputs are JSONL files saved in `./outputs/` with fields `model_output`, `predicted_answer`, and (for EAP) `retrieved_top3_mistakes`.

---

## Citation

```bibtex
@inproceedings{ma2026eap,
  title={From Error Analysis to Error-Aware Prompting: Structured Error Knowledge for High School Mathematical Reasoning},
  author={Ma, Yingying and Sun, Chao},
  booktitle={Proceedings of the International Conference on Intelligent Education and Intelligent Research (IEIR)},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.
