"""
Data preparation and utilities for BERT training pipeline.

This file is READ-ONLY during autonomous experimentation.
Agent should NOT modify this file - only train.py can be changed.

Usage:
    python prepare.py              # Prepare data from dataset_builder.py
    python prepare.py --refresh    # Re-run data pipeline and prepare

Inspired by: https://github.com/karpathy/autoresearch
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

# ==================== Configuration (FIXED - DO NOT MODIFY) ====================

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

TRAIN_FILE = PROCESSED_DATA_DIR / "train.json"
VAL_FILE = PROCESSED_DATA_DIR / "val.json"
TEST_FILE = PROCESSED_DATA_DIR / "test.json"

# Model
BASE_MODEL = "distilbert-base-uncased"
MAX_SEQ_LEN = 512

# Labels (FIXED)
LABELS = ["CLASSIC ML", "LLM/NLP", "CV", "Other"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# Training constraints
TIME_BUDGET = 300  # 5 minutes wall clock (excluding startup)
EVAL_SAMPLES = None  # Use all validation samples (None = all)

# ==================== Data Classes ====================


@dataclass
class Sample:
    """Single training sample with invariants."""
    text: str
    label: str
    label_id: int

    def __post_init__(self):
        assert self.text, "Sample text cannot be empty"
        assert self.label in LABELS, f"Invalid label: {self.label}"
        assert self.label_id == LABEL2ID[self.label], "Label ID mismatch"


# ==================== Dataset ====================


class CompetitionDataset(Dataset):
    """PyTorch Dataset for competition classification."""

    def __init__(
        self,
        samples: list[Sample],
        tokenizer: AutoTokenizer,
        max_length: int = MAX_SEQ_LEN,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample.label_id, dtype=torch.long),
        }


# ==================== Data Loading ====================


def load_samples(filepath: Path) -> list[Sample]:
    """Load samples from JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Run dataset_builder.py first!"
        )

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = Sample(
            text=item["text"],
            label=item["label"],
            label_id=item["label_id"],
        )
        samples.append(sample)

    return samples


def get_tokenizer() -> AutoTokenizer:
    """Get tokenizer for the base model."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    return tokenizer


def get_dataloaders(
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    tokenizer = get_tokenizer()

    train_samples = load_samples(TRAIN_FILE)
    val_samples = load_samples(VAL_FILE)
    test_samples = load_samples(TEST_FILE)

    train_dataset = CompetitionDataset(train_samples, tokenizer)
    val_dataset = CompetitionDataset(val_samples, tokenizer)
    test_dataset = CompetitionDataset(test_samples, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def compute_class_weights(samples: list[Sample]) -> torch.Tensor:
    """Compute class weights for balanced training."""
    labels = [s.label_id for s in samples]

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_LABELS),
        y=labels,
    )

    return torch.tensor(weights, dtype=torch.float32)


# ==================== Evaluation ====================


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    verbose: bool = True,
) -> dict:
    """
    Evaluate model predictions.

    Returns dict with:
        - accuracy: overall accuracy
        - f1_macro: macro F1 score (main metric)
        - f1_weighted: weighted F1 score
        - per_class_f1: F1 per class
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_per_class = f1_score(y_true, y_pred, average=None)

    results = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_f1": {LABELS[i]: f1_per_class[i] for i in range(NUM_LABELS)},
    }

    if verbose:
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"F1 Macro:    {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        print("\nPer-class F1:")
        for label, score in results["per_class_f1"].items():
            print(f"  {label}: {score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=LABELS))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    return results


def print_data_stats():
    """Print dataset statistics."""
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    for name, filepath in [("Train", TRAIN_FILE), ("Val", VAL_FILE), ("Test", TEST_FILE)]:
        if not filepath.exists():
            print(f"{name}: NOT FOUND")
            continue

        samples = load_samples(filepath)
        label_counts = {}
        for s in samples:
            label_counts[s.label] = label_counts.get(s.label, 0) + 1

        print(f"\n{name}: {len(samples)} samples")
        for label in LABELS:
            count = label_counts.get(label, 0)
            pct = count / len(samples) * 100 if samples else 0
            print(f"  {label}: {count} ({pct:.1f}%)")


# ==================== Main ====================


def main():
    """Prepare data for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for BERT training")
    parser.add_argument("--refresh", action="store_true", help="Re-run data pipeline")
    args = parser.parse_args()

    print("=" * 60)
    print("Data Preparation")
    print("=" * 60)

    # Check if data exists
    data_exists = all(f.exists() for f in [TRAIN_FILE, VAL_FILE, TEST_FILE])

    if args.refresh or not data_exists:
        print("\nRunning dataset_builder.py...")
        from dataset_builder import DatasetBuilder

        builder = DatasetBuilder(
            max_pages=15,
            augmentation_factor=3,
        )

        # If raw data exists, skip collection
        collect = not (RAW_DATA_DIR / "competitions.json").exists()

        # If labeled data exists, skip labeling
        label = not (PROCESSED_DATA_DIR / "labeled_competitions.json").exists()

        builder.build(
            collect=collect,
            label=label,
            augment=True,
            prepare=True,
        )
    else:
        print("\nData already exists. Use --refresh to re-run pipeline.")

    # Print stats
    print_data_stats()

    # Test tokenizer
    print("\nTokenizer test:")
    tokenizer = get_tokenizer()
    test_text = "Sample Competition [SEP] Tags: ml, classification [SEP] Description text"
    tokens = tokenizer(test_text, truncation=True, max_length=MAX_SEQ_LEN)
    print(f"  Input: {test_text[:50]}...")
    print(f"  Tokens: {len(tokens['input_ids'])}")

    # Test dataloader
    if data_exists:
        print("\nDataloader test:")
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
        batch = next(iter(train_loader))
        print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  Batch labels: {batch['labels'].tolist()}")

    print("\nPreparation complete!")


if __name__ == "__main__":
    main()
