# Error-Aware Prompting (EAP) for High School Mathematical Reasoning

This repository contains the code and experimental framework for the paper:  
**"From Error Analysis to Error-Aware Prompting: Structured Error Knowledge for High School Mathematical Reasoning"**

The project systematically investigates how structured error knowledge (error type–cause pairs) can be leveraged via Error-Aware Prompting (EAP) to improve LLM reasoning on high school math problems.

---

## Repository Structure

```
.
├── Data/
│   ├── balance_960_test.jsonl      # 960-problem benchmark
│   └── sampled_test_400.jsonl      # 400-problem EAP test set
├── stage1_basic_inference.py       # Stage 1: 960-problem baseline CoT inference
├── stage2_annotate_errors.py       # Stage 2: Error annotation (six-category taxonomy)
├── prepare_eap_data.py             # Data preparation: summaries, fingerprints, predictions
├── embed_vectors.py                # Generate normalized embeddings via BGE
├── stage3_eap_all.py               # Stage 3: Unified EAP experiments (CoT + 5 strategies)
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

- **GPU Recommendation:** For local inference, a GPU with at least 16 GB VRAM is recommended (tested on Kaggle dual T4).
- **Batch Size:** The default batch sizes are set in each script's `MODEL_CONFIGS` dictionary. You can adjust them (e.g., reduce if you encounter OOM) by modifying the `batch_size` value for your chosen model.

---

## Data Preparation

Before running experiments, you need to prepare the following data files.

### 1. Raw Datasets

- **Error database:** JSONL file with fields: `problem`, `steps`, `model_output`, `answer`
- **Test set:** JSONL file with fields: `problem`, `answer`

The provided `Data/balance_960_test.jsonl` and `Data/sampled_test_400.jsonl` are ready to use.

### 2. Generate Problem Summaries (for EAP-Summary)

```bash
export DEEPSEEK_API_KEY="your-api-key"
python prepare_eap_data.py --task summary \
    --input Data/error_db.jsonl \
    --output Data/error_db_with_summary.jsonl
```

### 3. Generate Structural Fingerprints (for EAP-Formula)

```bash
python prepare_eap_data.py --task fingerprint \
    --input Data/error_db_with_summary.jsonl \
    --output Data/error_db_with_fingerprint.jsonl
```

### 4. Predict Error Types & Causes (for EAP-ErrorCause)

```bash
python prepare_eap_data.py --task predict_error \
    --input Data/sampled_test_400.jsonl \
    --output Data/sampled_test_400_with_pred.jsonl
```

### 5. Generate Normalized Semantic Embeddings

Only required for: **EAP-Problem**, **EAP-Summary**, **EAP-ErrorCause**, and **EAP-Fusion**.  
Not required for: **EAP-Formula** (uses Jaccard similarity on fingerprints).

All embeddings are generated using `BGE-large-zh-v1.5` and normalized to unit length.

```bash
python embed_vectors.py
```

> **Note:** `embed_vectors.py` expects the input files to be in the `Data/` directory with the names shown above. Modify the script if your filenames differ.

After this step, you will have:

- Error database with `problem_embedding`, `summary_embedding`, `error_reason_embedding`
- Test set with `problem_embedding`, `summary_embedding`, `predicted_error_embedding`

> **Note:** If you only plan to run EAP-Formula, you can skip the entire embedding step.

---

## Running the Experiments

The three stages are designed to be run sequentially:

- **Stage 1** generates baseline CoT outputs on the 960-problem benchmark.
- **Stage 2** takes the incorrect samples from Stage 1 (where `predicted_answer != answer`) and annotates them with error types and reasons.
- **Stage 3** uses the annotated error database from Stage 2 to run EAP experiments on the 400-problem test set.

**Data flow:** Stage 1 → error samples → Stage 2 → annotated error database → Stage 3

### Stage 1: Baseline CoT on 960 Problems

```bash
python stage1_basic_inference.py --model qwen7b --test_file Data/balance_960_test.jsonl
```

- Available models: `qwen1.5b`, `qwen7b`, `deepseek-math`, `deepseek-v3`
- The output will be saved in `./outputs/` as `<model>_960_output.jsonl`

### Stage 2: Error Annotation

**Input:** The output file from Stage 1 (e.g., `./outputs/qwen7b_960_output.jsonl`).  
**What it does:** Identifies incorrect samples and annotates them with error types (PUE, CE, RE, CPE, COE, OE) and reasons.

```bash
export DEEPSEEK_API_KEY="your-api-key"
python stage2_annotate_errors.py \
    --input_file ./outputs/qwen7b_960_output.jsonl \
    --output_file Data/error_db.jsonl \
    --existing_label_file Data/partial_labels.jsonl   # optional
```

### Stage 3: EAP Experiments on 400 Samples

**Input:** The annotated error database from Stage 2 (`Data/error_db.jsonl` or `Data/error_db_embedded.jsonl`).

```bash
# CoT baseline
python stage3_eap_all.py --model qwen7b --strategy cot \
    --test_file Data/sampled_test_400_embedded.jsonl

# EAP-Problem
python stage3_eap_all.py --model qwen7b --strategy problem \
    --test_file Data/sampled_test_400_embedded.jsonl \
    --error_db Data/error_db_embedded.jsonl

# EAP-Summary
python stage3_eap_all.py --model qwen7b --strategy summary \
    --test_file Data/sampled_test_400_embedded.jsonl \
    --error_db Data/error_db_embedded.jsonl

# EAP-Formula
python stage3_eap_all.py --model qwen7b --strategy formula \
    --test_file Data/sampled_test_400_embedded.jsonl \
    --error_db Data/error_db_embedded.jsonl

# EAP-ErrorCause
python stage3_eap_all.py --model qwen7b --strategy errorcause \
    --test_file Data/sampled_test_400_embedded.jsonl \
    --error_db Data/error_db_embedded.jsonl

# EAP-Fusion
python stage3_eap_all.py --model qwen7b --strategy fusion \
    --test_file Data/sampled_test_400_embedded.jsonl \
    --error_db Data/error_db_embedded.jsonl
```

All outputs are JSONL files saved in `./outputs/` with fields `model_output`, `predicted_answer`, and (for EAP) `retrieved_top3_mistakes`.

---

## Adjusting Batch Size

If you encounter out-of-memory (OOM) errors, reduce the `batch_size` in the `MODEL_CONFIGS` dictionary of the corresponding script. For example, in `stage3_eap_all.py`:

```python
"qwen7b": {
    ...
    "batch_size": 20,   # change to 10 or lower
}
```

The same applies to `stage1_basic_inference.py` and `stage2_annotate_errors.py`.

---

## Citation

If you use this code, please cite the corresponding paper (details to be added).

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.
