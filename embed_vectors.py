#!/usr/bin/env python3
# embed_vectors.py
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
