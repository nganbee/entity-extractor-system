import os
import yaml
import numpy as np
import evaluate
from dotenv import load_dotenv, find_dotenv

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

load_dotenv(find_dotenv())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yml")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "multinerd_en")


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load training configuration from file YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[INFO] Loaded config from: {config_path}")
    return config


def parse_labels(config: dict) -> tuple[dict, dict, list]:

    label2id: dict = config["labels"]["label2id"]
    id2label: dict = {v: k for k, v in label2id.items()}
    label_list: list = list(label2id.keys())
    print(f"[INFO] Number of labels: {len(label_list)}")
    return label2id, id2label, label_list


def load_dataset(data_path: str = DATA_PATH):
    
    print(f"[INFO] Loading dataset from: {data_path}")
    dataset = load_from_disk(data_path)
    print(f"[INFO] Dataset splits: {list(dataset.keys())}")
    return dataset


def build_tokenize_fn(tokenizer, config: dict, label_list: list = None):

    max_length = config["model"]["max_length"]

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
                    # Special tokens (<s>, </s>, <pad>) -> bỏ qua
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # Sub-token đầu tiên của từ -> giữ nhãn gốc
                    label_ids.append(label[word_idx])
                else:
                    # Sub-token phụ của cùng một từ -> bỏ qua
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return tokenize_and_align_labels


def tokenize_dataset(dataset, tokenize_fn):
    """Áp dụng tokenize_fn lên toàn bộ dataset (batched)."""
    print("[INFO] Tokenizing dataset...")
    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    tokenized_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    print("[INFO] Tokenization complete.")
    return tokenized_ds


def load_tokenizer(model_id: str):

    print(f"[INFO] Loading tokenizer: {model_id}")
    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id: str, label2id: dict, id2label: dict):
    
    print(f"[INFO] Loading model: {model_id}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    return model

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
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics

def build_training_args(config: dict) -> TrainingArguments:
    """Xây dựng TrainingArguments từ config."""
    output_dir = config["training"].get("output_dir", "./xlm-roberta-results")
    output_dir = os.path.join(BASE_DIR, output_dir)

    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=200,
        save_strategy="steps",
        save_steps=200,
        
        learning_rate=config["training"].get("learning_rate", 1e-5),
        max_grad_norm=1.0,
        per_device_train_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        weight_decay=config["training"].get("weight_decay", 0.01),
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        
        fp16=False,
        gradient_accumulation_steps=2,
        dataloader_num_workers=2,
        report_to="none",
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
    )


def build_trainer(
    model,
    args: TrainingArguments,
    tokenized_ds,
    tokenizer,
    compute_metrics,
) -> Trainer:

    return Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )


def run_training(trainer: Trainer):

    print("[INFO] Starting training...")
    train_result = trainer.train()
    print(f"[INFO] Training finished.")
    return train_result


def push_to_hub(model, tokenizer, repo_id: str, hf_token: str | None = None):
    from huggingface_hub import login

    if not repo_id:
        print("[WARNING] Can not found HF_REPO_ID ")
        return

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[WARNING] Can not found HF_TOKEN ")
        return

    login(token=token)
    print(f"[INFO] Pushing model & tokenizer to Hub: {repo_id}")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print(f"[INFO] Successfully: https://huggingface.co/{repo_id}")



def main():
    # Load config
    config = load_config()
    label2id, id2label, label_list = parse_labels(config)
    model_id = config["model"]["model_checkpoint"]

    # Load data
    dataset = load_dataset()

    # Load tokenizer, prepare tokenized dataset
    tokenizer = load_tokenizer(model_id)
    tokenize_fn = build_tokenize_fn(tokenizer, config, label_list)
    tokenized_ds = tokenize_dataset(dataset, tokenize_fn)

    # Load model
    model = load_model(model_id, label2id, id2label)

    # Metrics
    compute_metrics = build_compute_metrics_fn(label_list)

    #  Training
    training_args = build_training_args(config)
    trainer = build_trainer(model, training_args, tokenized_ds, tokenizer, compute_metrics)
    run_training(trainer)

    repo_id = os.environ.get("HF_REPO_ID")
    push_to_hub(model, tokenizer, repo_id)


if __name__ == "__main__":
    main()
