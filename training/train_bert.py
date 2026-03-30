"""
Fine-tune BERT for competition classification.

Usage:
    python train_bert.py
"""

import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

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
    BERT_WARMUP_RATIO,
    BERT_EARLY_STOPPING_PATIENCE,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
)
from training_plots import plot_all_metrics


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


def compute_class_weights(train_data: list[dict], num_labels: int) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    labels = [item["label_id"] for item in train_data]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=labels,
    )
    return torch.tensor(class_weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Trainer with weighted cross-entropy loss for imbalanced classes."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Move class weights to same device as logits
        weights = self.class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


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

    # Show class distribution
    train_labels = [item["label"] for item in train_data]
    label_counts = Counter(train_labels)
    print("\n  Train class distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count} ({count/len(train_data)*100:.1f}%)")

    # Create datasets
    train_dataset = CompetitionDataset(train_data, tokenizer, BERT_MAX_LENGTH)
    val_dataset = CompetitionDataset(val_data, tokenizer, BERT_MAX_LENGTH)
    test_dataset = CompetitionDataset(test_data, tokenizer, BERT_MAX_LENGTH)

    # Compute class weights for imbalanced dataset
    print("\nComputing class weights for imbalanced classes...")
    class_weights = compute_class_weights(train_data, NUM_LABELS)
    print(f"  Class weights: {dict(zip(ID2LABEL.values(), class_weights.tolist()))}")

    # Training arguments
    BERT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(BERT_OUTPUT_DIR),
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        learning_rate=BERT_LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=BERT_WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",  # Disable wandb/mlflow
    )

    # Trainer with class weights
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=BERT_EARLY_STOPPING_PATIENCE)],
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # Generate training plots
    plot_all_metrics(trainer, BERT_OUTPUT_DIR)

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
