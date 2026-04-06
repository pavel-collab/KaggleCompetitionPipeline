"""
Evaluate trained models and compare with OpenRouter LLM baseline.

Evaluates:
- BERT classifier (if trained)
- LoRA classifier (if trained, requires GPU)
- OpenRouter LLM (baseline)

Usage:
    python evaluate.py
    python evaluate.py --bert-only
    python evaluate.py --llm-only
"""

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import torch
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from config import (
    BERT_OUTPUT_DIR,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    LORA_OUTPUT_DIR,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TEST_FILE,
)


class CompetitionLabel(BaseModel):
    """Structured output for competition classification."""

    type: Literal["CLASSIC ML", "LLM/NLP", "CV", "Other"] = Field(
        description="Competition type"
    )

#TODO: is that prompt has already been in the collection importing?
CLASSIFICATION_PROMPT = PromptTemplate.from_template(
    """You are an expert Kaggle competition evaluator.

Given a competition, determine its machine learning category based on description and tags.

Competition text:
{text}

Classify as one of:
- CLASSIC ML: Traditional ML (regression, classification, time series, tabular data)
- LLM/NLP: Natural Language Processing, text analysis, language models
- CV: Computer Vision, image classification, object detection
- Other: Anything else (optimization, simulation, etc.)

Return only the type."""
)

LORA_INFERENCE_TEMPLATE = """Below is a Kaggle competition. Classify it into one of these categories: {labels}

### Competition:
{text}

### Classification:
"""


def load_test_data() -> list[dict]:
    """Load test dataset."""
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_FILE}\n" "Run prepare_dataset.py first!"
        )

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_bert(test_data: list[dict]) -> dict | None:
    """Evaluate BERT classifier."""
    if not BERT_OUTPUT_DIR.exists():
        print("  BERT model not found, skipping...")
        return None

    print("  Loading BERT model...")
    classifier = pipeline(
        "text-classification",
        model=str(BERT_OUTPUT_DIR),
        device=0 if torch.cuda.is_available() else -1,
    )

    print("  Running inference...")
    predictions = []
    true_labels = []

    for i, item in enumerate(test_data):
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i + 1}/{len(test_data)}")

        result = classifier(item["text"], truncation=True)
        pred_label = result[0]["label"]
        predictions.append(LABEL2ID.get(pred_label, -1))
        true_labels.append(item["label_id"])

    return compute_metrics(true_labels, predictions, "BERT")


def evaluate_lora(test_data: list[dict]) -> dict | None:
    """Evaluate LoRA classifier."""
    if not LORA_OUTPUT_DIR.exists():
        print("  LoRA model not found, skipping...")
        return None

    if not torch.cuda.is_available():
        print("  LoRA requires GPU, skipping...")
        return None

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("  unsloth not installed, skipping LoRA evaluation...")
        return None

    print("  Loading LoRA model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(LORA_OUTPUT_DIR),
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print("  Running inference...")
    predictions = []
    true_labels = []

    for i, item in enumerate(test_data):
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i + 1}/{len(test_data)}")

        prompt = LORA_INFERENCE_TEMPLATE.format(
            labels=", ".join(LABELS),
            text=item["text"][:1500],
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_text = result.split("### Classification:")[-1].strip()

        # Map prediction to label
        pred_label = -1
        for label in LABELS:
            if label.lower() in pred_text.lower():
                pred_label = LABEL2ID[label]
                break

        predictions.append(pred_label)
        true_labels.append(item["label_id"])

    return compute_metrics(true_labels, predictions, "LoRA")


def evaluate_openrouter(test_data: list[dict]) -> dict | None:
    """Evaluate OpenRouter LLM (baseline)."""
    if not OPENAI_API_KEY:
        print("  OPENAI_API_KEY not set, skipping...")
        return None

    print(f"  Using model: {OPENAI_MODEL}")
    print(f"  API base: {OPENAI_API_BASE}")

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        temperature=0,
    )

    chain = CLASSIFICATION_PROMPT | llm.with_structured_output(CompetitionLabel)

    print("  Running inference...")
    predictions = []
    true_labels = []
    errors = 0

    for i, item in enumerate(test_data):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{len(test_data)}")

        try:
            result = chain.invoke({"text": item["text"][:2000]})
            pred_label = LABEL2ID.get(result.type, -1)
            predictions.append(pred_label)
            true_labels.append(item["label_id"])

            # Rate limiting
            time.sleep(0.3)

        except Exception as e:
            print(f"    Error on sample {i}: {e}")
            errors += 1
            predictions.append(-1)
            true_labels.append(item["label_id"])

    if errors > 0:
        print(f"  Total errors: {errors}")

    return compute_metrics(true_labels, predictions, "OpenRouter LLM")


def compute_metrics(
    true_labels: list[int], predictions: list[int], model_name: str
) -> dict:
    """Compute evaluation metrics."""
    # Filter out invalid predictions (-1)
    valid_mask = [p != -1 for p in predictions]
    valid_true = [t for t, m in zip(true_labels, valid_mask) if m]
    valid_pred = [p for p, m in zip(predictions, valid_mask) if m]

    if len(valid_pred) == 0:
        return {
            "model": model_name,
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "valid_samples": 0,
            "total_samples": len(true_labels),
        }

    return {
        "model": model_name,
        "accuracy": accuracy_score(valid_true, valid_pred),
        "f1_macro": f1_score(valid_true, valid_pred, average="macro"),
        "f1_weighted": f1_score(valid_true, valid_pred, average="weighted"),
        "valid_samples": len(valid_pred),
        "total_samples": len(true_labels),
        "true_labels": valid_true,
        "predictions": valid_pred,
    }


def plot_confusion_matrix(results: dict, output_path: Path) -> None:
    """Plot confusion matrix heatmap for a single model."""
    if "true_labels" not in results:
        return

    cm = confusion_matrix(results["true_labels"], results["predictions"])

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=[l[:10] for l in LABELS],
        yticklabels=[l[:10] for l in LABELS],
        ylabel='True Label',
        xlabel='Predicted Label',
        title=f'Confusion Matrix: {results["model"]}'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    model_name = results["model"].lower().replace(" ", "_").replace("/", "_")
    plt.savefig(output_path / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_path / f'confusion_matrix_{model_name}.png'}")


def plot_per_class_metrics(results: dict, output_path: Path) -> None:
    """Plot per-class precision, recall, and F1 for a single model."""
    if "true_labels" not in results:
        return

    report = classification_report(
        results["true_labels"],
        results["predictions"],
        target_names=list(ID2LABEL.values()),
        output_dict=True,
        zero_division=0,
    )

    # Extract per-class metrics
    classes = list(ID2LABEL.values())
    precision = [report[c]["precision"] for c in classes]
    recall = [report[c]["recall"] for c in classes]
    f1 = [report[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#9b59b6')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Metrics: {results["model"]}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c[:10] for c in classes], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    model_name = results["model"].lower().replace(" ", "_").replace("/", "_")
    plt.savefig(output_path / f"per_class_metrics_{model_name}.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_path / f'per_class_metrics_{model_name}.png'}")


def plot_model_comparison(results_list: list[dict], output_path: Path) -> None:
    """Plot comparison bar chart for all models."""
    if len(results_list) < 1:
        return

    models = [r["model"] for r in results_list]
    accuracy = [r["accuracy"] for r in results_list]
    f1_macro = [r["f1_macro"] for r in results_list]
    f1_weighted = [r["f1_weighted"] for r in results_list]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', color='#e74c3c')
    bars2 = ax.bar(x, f1_macro, width, label='F1 Macro', color='#3498db')
    bars3 = ax.bar(x + width, f1_weighted, width, label='F1 Weighted', color='#2ecc71')

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / "model_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'model_comparison.png'}")


def plot_evaluation_summary(results_list: list[dict], output_path: Path) -> None:
    """Create a summary plot with confusion matrices for all models."""
    valid_results = [r for r in results_list if "true_labels" in r]

    if not valid_results:
        return

    n_models = len(valid_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for idx, results in enumerate(valid_results):
        ax = axes[idx]
        cm = confusion_matrix(results["true_labels"], results["predictions"])

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

        ax.set_xticks(np.arange(len(LABELS)))
        ax.set_yticks(np.arange(len(LABELS)))
        ax.set_xticklabels([l[:8] for l in LABELS], rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([l[:8] for l in LABELS], fontsize=9)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center", fontsize=10,
                       color="white" if cm[i, j] > thresh else "black")

        ax.set_title(f'{results["model"]}\nAcc: {results["accuracy"]:.3f} | F1: {results["f1_macro"]:.3f}',
                    fontsize=11, fontweight='bold')

    plt.suptitle('Evaluation Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "evaluation_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'evaluation_summary.png'}")


def plot_all_evaluation_metrics(results_list: list[dict], output_path: Path) -> None:
    """Generate all evaluation plots."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating evaluation plots...")
    print("=" * 60)

    # Individual model plots
    for results in results_list:
        plot_confusion_matrix(results, output_path)
        plot_per_class_metrics(results, output_path)

    # Comparison plots
    plot_model_comparison(results_list, output_path)
    plot_evaluation_summary(results_list, output_path)

    print("=" * 60)
    print(f"All plots saved to: {output_path}")
    print("=" * 60)


def print_results(results: list[dict]) -> None:
    """Print comparison table and detailed reports."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Summary table
    print("\n{:<20} {:>10} {:>12} {:>12} {:>10}".format(
        "Model", "Accuracy", "F1 (macro)", "F1 (weighted)", "Samples"
    ))
    print("-" * 70)

    for r in results:
        print("{:<20} {:>10.4f} {:>12.4f} {:>12.4f} {:>10}".format(
            r["model"],
            r["accuracy"],
            r["f1_macro"],
            r["f1_weighted"],
            f"{r['valid_samples']}/{r['total_samples']}",
        ))

    # Detailed reports
    for r in results:
        if "true_labels" not in r:
            continue

        print("\n" + "=" * 70)
        print(f"Classification Report: {r['model']}")
        print("=" * 70)

        print(classification_report(
            r["true_labels"],
            r["predictions"],
            target_names=list(ID2LABEL.values()),
            zero_division=0,
        ))

        print("Confusion Matrix:")
        cm = confusion_matrix(r["true_labels"], r["predictions"])
        print(f"{'':>12}", end="")
        for label in LABELS:
            print(f"{label[:8]:>10}", end="")
        print()

        for i, row in enumerate(cm):
            print(f"{LABELS[i][:10]:>12}", end="")
            for val in row:
                print(f"{val:>10}", end="")
            print()

    # Comparison with baseline
    if len(results) >= 2:
        baseline = None
        for r in results:
            if "OpenRouter" in r["model"]:
                baseline = r
                break

        if baseline:
            print("\n" + "=" * 70)
            print("COMPARISON WITH BASELINE (OpenRouter LLM)")
            print("=" * 70)

            for r in results:
                if r["model"] == baseline["model"]:
                    continue

                diff_acc = r["accuracy"] - baseline["accuracy"]
                diff_f1 = r["f1_macro"] - baseline["f1_macro"]

                sign_acc = "+" if diff_acc >= 0 else ""
                sign_f1 = "+" if diff_f1 >= 0 else ""

                print(f"\n{r['model']} vs {baseline['model']}:")
                print(f"  Accuracy:  {sign_acc}{diff_acc:.4f} ({sign_acc}{diff_acc*100:.2f}%)")
                print(f"  F1 macro:  {sign_f1}{diff_f1:.4f} ({sign_f1}{diff_f1*100:.2f}%)")


def save_results(results: list[dict], output_file: str = "evaluation_results.json") -> None:
    """Save results to JSON file."""
    from config import BASE_DIR

    output_path = BASE_DIR / output_file

    # Remove numpy arrays for JSON serialization
    clean_results = []
    for r in results:
        clean = {k: v for k, v in r.items() if k not in ["true_labels", "predictions"]}
        if "true_labels" in r:
            clean["true_labels"] = [int(x) for x in r["true_labels"]]
            clean["predictions"] = [int(x) for x in r["predictions"]]
        clean_results.append(clean)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--bert-only", action="store_true", help="Evaluate only BERT")
    parser.add_argument("--lora-only", action="store_true", help="Evaluate only LoRA")
    parser.add_argument("--llm-only", action="store_true", help="Evaluate only OpenRouter LLM")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--no-plots", action="store_true", help="Don't generate plots")
    args = parser.parse_args()

    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data()
    print(f"Loaded {len(test_data)} test samples")

    results = []

    # Evaluate models based on arguments
    evaluate_all = not (args.bert_only or args.lora_only or args.llm_only)

    if evaluate_all or args.bert_only:
        print("\n" + "-" * 40)
        print("Evaluating BERT classifier...")
        print("-" * 40)
        bert_results = evaluate_bert(test_data)
        if bert_results:
            results.append(bert_results)

    if evaluate_all or args.lora_only:
        print("\n" + "-" * 40)
        print("Evaluating LoRA classifier...")
        print("-" * 40)
        lora_results = evaluate_lora(test_data)
        if lora_results:
            results.append(lora_results)

    if evaluate_all or args.llm_only:
        print("\n" + "-" * 40)
        print("Evaluating OpenRouter LLM (baseline)...")
        print("-" * 40)
        llm_results = evaluate_openrouter(test_data)
        if llm_results:
            results.append(llm_results)

    if not results:
        print("\nNo models were evaluated. Make sure models are trained or API keys are set.")
        return

    # Print results
    print_results(results)

    # Generate plots
    if not args.no_plots:
        from config import BASE_DIR
        plots_dir = BASE_DIR / "evaluation_plots"
        plot_all_evaluation_metrics(results, plots_dir)

    # Save results
    if not args.no_save:
        save_results(results)


if __name__ == "__main__":
    main()
