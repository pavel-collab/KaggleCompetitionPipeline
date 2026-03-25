"""
Prepare dataset for training: split into train/val/test.

Usage:
    python prepare_dataset.py
"""

import json
import random
from collections import Counter

from config import (
    LABELED_DATA_FILE,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    PROCESSED_DATA_DIR,
    LABEL2ID,
)


def load_labeled_data() -> list[dict]:
    """Load labeled competitions."""
    if not LABELED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Labeled data not found: {LABELED_DATA_FILE}\n"
            "Run label_data.py first!"
        )

    with open(LABELED_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def format_text(item: dict) -> str:
    """Format competition as single text for model input."""
    parts = [item["title"]]

    if item.get("tags"):
        parts.append(f"Tags: {item['tags']}")

    if item.get("description"):
        # Truncate description to reasonable length
        desc = item["description"][:1500]
        parts.append(desc)

    return " [SEP] ".join(parts)


def prepare_samples(data: list[dict]) -> list[dict]:
    """Convert raw data to training samples."""
    samples = []

    for item in data:
        label = item["label"]

        # Skip items with unknown labels
        if label not in LABEL2ID:
            print(f"Warning: Unknown label '{label}', skipping")
            continue

        samples.append({
            "text": format_text(item),
            "label": label,
            "label_id": LABEL2ID[label],
        })

    return samples


def stratified_split(
    samples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split data with stratification by label."""
    random.seed(seed)

    # Group by label
    by_label = {}
    for sample in samples:
        label = sample["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(sample)

    train, val, test = [], [], []

    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # Shuffle final splits
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_split(data: list[dict], filepath) -> None:
    """Save split to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_stats(name: str, data: list[dict]) -> None:
    """Print statistics for a split."""
    labels = [item["label"] for item in data]
    counts = Counter(labels)

    print(f"\n{name} ({len(data)} samples):")
    for label, count in sorted(counts.items()):
        pct = count / len(data) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")


def main():
    print("=" * 60)
    print("Dataset Preparation")
    print("=" * 60)

    # Load and prepare
    raw_data = load_labeled_data()
    print(f"Loaded {len(raw_data)} labeled competitions")

    samples = prepare_samples(raw_data)
    print(f"Prepared {len(samples)} valid samples")

    # Split
    train, val, test = stratified_split(samples)

    # Print statistics
    print_stats("Train", train)
    print_stats("Validation", val)
    print_stats("Test", test)

    # Save
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_split(train, TRAIN_FILE)
    save_split(val, VAL_FILE)
    save_split(test, TEST_FILE)

    print("\n" + "=" * 60)
    print("Saved splits:")
    print(f"  Train: {TRAIN_FILE}")
    print(f"  Val:   {VAL_FILE}")
    print(f"  Test:  {TEST_FILE}")


if __name__ == "__main__":
    main()
