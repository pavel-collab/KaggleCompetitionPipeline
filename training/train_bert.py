"""
Fine-tune BERT for competition classification.

Usage:
    python train_bert.py
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

from config import (
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    BERT_MODEL_NAME,
    BERT_OUTPUT_DIR,
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
    BERT_MAX_LENGTH,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
)


class CompetitionDataset(Dataset):
    """Dataset for competition classification."""

    def __init__(self, data: list[dict], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label_id"]),
        }


def load_data(filepath) -> list[dict]:
    """Load dataset from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }


def main():
    print("=" * 60)
    print("BERT Fine-tuning for Competition Classification")
    print("=" * 60)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"\nLoading model: {BERT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Load datasets
    print("\nLoading datasets...")
    train_data = load_data(TRAIN_FILE)
    val_data = load_data(VAL_FILE)
    test_data = load_data(TEST_FILE)

    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # Create datasets
    train_dataset = CompetitionDataset(train_data, tokenizer, BERT_MAX_LENGTH)
    val_dataset = CompetitionDataset(val_data, tokenizer, BERT_MAX_LENGTH)
    test_dataset = CompetitionDataset(test_data, tokenizer, BERT_MAX_LENGTH)

    # Training arguments
    BERT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(BERT_OUTPUT_DIR),
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        learning_rate=BERT_LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        report_to="none",  # Disable wandb/mlflow
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = [item["label_id"] for item in test_data]

    print(classification_report(
        true_labels,
        pred_labels,
        target_names=list(ID2LABEL.values()),
    ))

    # Save model
    print("\n" + "=" * 60)
    print(f"Saving model to: {BERT_OUTPUT_DIR}")
    print("=" * 60)

    trainer.save_model(str(BERT_OUTPUT_DIR))
    tokenizer.save_pretrained(str(BERT_OUTPUT_DIR))

    print("\nDone! Model saved.")
    print("\nTo use the model:")
    print("  from transformers import pipeline")
    print(f"  classifier = pipeline('text-classification', model='{BERT_OUTPUT_DIR}')")
    print("  result = classifier('Your competition text here')")


if __name__ == "__main__":
    main()
