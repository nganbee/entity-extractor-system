import os
import yaml
import numpy as np
import torch
import evaluate

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yml")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "multinerd_en")
MAX_LENGTH = 128
BATCH_SIZE = 16


def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[INFO] Loaded config from: {config_path}")
    return config


def load_model_and_tokenizer(repo_id: str):
    print(f"[INFO] Loading model and tokenizer from: {repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForTokenClassification.from_pretrained(repo_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    label_list = list(id2label.values())

    print(f"[INFO] Device: {device} | Labels: {len(label_list)}")
    return model, tokenizer, id2label, label_list, device


def load_dataset(data_path: str = DATA_PATH):
    print(f"[INFO] Loading dataset from: {data_path}")
    dataset = load_from_disk(data_path)
    print(f"[INFO] Splits: {list(dataset.keys())}")
    return dataset


def build_tokenize_fn(tokenizer, max_length: int = MAX_LENGTH):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
            add_special_tokens=True,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return tokenize_and_align_labels


def prepare_test_dataset(ds_reduced, tokenize_fn):
    print("[INFO] Tokenizing test split...")
    test_dataset = ds_reduced["test"].map(tokenize_fn, batched=True)

    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    columns_to_remove = [c for c in test_dataset.column_names if c not in columns_to_keep]
    test_dataset = test_dataset.remove_columns(columns_to_remove)
    test_dataset.set_format(type="torch")

    print(f"[INFO] Test samples: {len(test_dataset)}")
    return test_dataset


def build_dataloader(test_dataset, tokenizer, batch_size: int = BATCH_SIZE):
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)


# ─── 4. Metrics ───────────────────────────────────────────────────────────────

def build_compute_metrics_fn(label_list: list):
    seqeval_metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[pred] for pred, lbl in zip(prediction, label) if lbl != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[lbl] for pred, lbl in zip(prediction, label) if lbl != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval_metric.compute(
            predictions=true_predictions,
            references=true_labels,
            zero_division=0
        )
        return {
            "precision": results["overall_precision"],
            "recall":    results["overall_recall"],
            "f1":        results["overall_f1"],
            "accuracy":  results["overall_accuracy"],
        }

    return compute_metrics


def batch_evaluate(dataloader, model, compute_metrics, device="cuda"):
    model.eval()

    all_logits = []
    all_labels = []
    global_max_len = 0

    print("[INFO] Running inference over batches...")
    for batch in tqdm(dataloader, desc="Inference"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [batch, seq_len, num_labels]

        logits_np = logits.cpu().numpy()
        labels_np = labels.cpu().numpy()

        seq_len = logits_np.shape[1]
        if seq_len > global_max_len:
            global_max_len = seq_len

        all_logits.append(logits_np)
        all_labels.append(labels_np)

    print("[INFO] Padding and computing metrics...")
    padded_logits = []
    padded_labels = []

    for logit, label in tqdm(zip(all_logits, all_labels), total=len(all_logits), desc="Padding"):
        pad_length = global_max_len - logit.shape[1]
        padded_logits.append(np.pad(logit,  ((0, 0), (0, pad_length), (0, 0)), constant_values=0))
        padded_labels.append(np.pad(label, ((0, 0), (0, pad_length)),          constant_values=-100))

    final_logits = np.concatenate(padded_logits, axis=0)
    final_labels = np.concatenate(padded_labels, axis=0)

    metrics = compute_metrics((final_logits, final_labels))
    return metrics


def main():
    config = load_config()
    repo_id = config["hub"]["repo_id"]

    # Load model & tokenizer
    model, tokenizer, id2label, label_list, device = load_model_and_tokenizer(repo_id)

    # Load and prepare test dataset
    dataset = load_dataset()
    tokenize_fn = build_tokenize_fn(tokenizer)
    test_dataset = prepare_test_dataset(dataset, tokenize_fn)
    dataloader = build_dataloader(test_dataset, tokenizer)

    # Evaluate
    compute_metrics = build_compute_metrics_fn(label_list)
    metrics = batch_evaluate(dataloader, model, compute_metrics, device=str(device))

    # Print results
    print("\n" + "=" * 35)
    print("EVALUATION RESULTS")
    print("=" * 35)
    print(f"F1        : {metrics['f1']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
