"""
Visualization utilities for training metrics.

Creates plots for:
- Training and validation loss over epochs/steps
- Accuracy and F1 scores over epochs
- Learning rate schedule
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots


def extract_metrics_from_log_history(log_history: list[dict]) -> dict:
    """
    Extract metrics from HuggingFace Trainer log history.

    Args:
        log_history: List of log entries from trainer.state.log_history

    Returns:
        Dictionary with extracted metrics organized by type
    """
    metrics = {
        "train_loss": {"steps": [], "values": []},
        "eval_loss": {"epochs": [], "values": []},
        "eval_accuracy": {"epochs": [], "values": []},
        "eval_f1_macro": {"epochs": [], "values": []},
        "eval_f1_weighted": {"epochs": [], "values": []},
        "learning_rate": {"steps": [], "values": []},
    }

    for entry in log_history:
        # Training loss (logged every N steps)
        if "loss" in entry and "eval_loss" not in entry:
            step = entry.get("step", len(metrics["train_loss"]["steps"]))
            metrics["train_loss"]["steps"].append(step)
            metrics["train_loss"]["values"].append(entry["loss"])

            if "learning_rate" in entry:
                metrics["learning_rate"]["steps"].append(step)
                metrics["learning_rate"]["values"].append(entry["learning_rate"])

        # Evaluation metrics (logged every epoch)
        if "eval_loss" in entry:
            epoch = entry.get("epoch", len(metrics["eval_loss"]["epochs"]) + 1)
            metrics["eval_loss"]["epochs"].append(epoch)
            metrics["eval_loss"]["values"].append(entry["eval_loss"])

            if "eval_accuracy" in entry:
                metrics["eval_accuracy"]["epochs"].append(epoch)
                metrics["eval_accuracy"]["values"].append(entry["eval_accuracy"])

            if "eval_f1_macro" in entry:
                metrics["eval_f1_macro"]["epochs"].append(epoch)
                metrics["eval_f1_macro"]["values"].append(entry["eval_f1_macro"])

            if "eval_f1_weighted" in entry:
                metrics["eval_f1_weighted"]["epochs"].append(epoch)
                metrics["eval_f1_weighted"]["values"].append(entry["eval_f1_weighted"])

    return metrics


def plot_training_loss(metrics: dict, output_path: Path) -> None:
    """Plot training loss over steps."""
    if not metrics["train_loss"]["values"]:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        metrics["train_loss"]["steps"],
        metrics["train_loss"]["values"],
        'b-',
        linewidth=2,
        label='Training Loss'
    )

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "training_loss.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'training_loss.png'}")


def plot_eval_loss(metrics: dict, output_path: Path) -> None:
    """Plot evaluation loss over epochs."""
    if not metrics["eval_loss"]["values"]:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        metrics["eval_loss"]["epochs"],
        metrics["eval_loss"]["values"],
        'r-o',
        linewidth=2,
        markersize=8,
        label='Validation Loss'
    )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set integer x-ticks for epochs
    if metrics["eval_loss"]["epochs"]:
        ax.set_xticks(range(1, int(max(metrics["eval_loss"]["epochs"])) + 1))

    plt.tight_layout()
    plt.savefig(output_path / "validation_loss.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'validation_loss.png'}")


def plot_loss_combined(metrics: dict, output_path: Path) -> None:
    """Plot training and validation loss together."""
    has_train = bool(metrics["train_loss"]["values"])
    has_eval = bool(metrics["eval_loss"]["values"])

    if not has_train and not has_eval:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if has_train:
        ax.plot(
            metrics["train_loss"]["steps"],
            metrics["train_loss"]["values"],
            'b-',
            linewidth=1.5,
            alpha=0.7,
            label='Training Loss'
        )

    if has_eval and has_train:
        # Convert epochs to approximate steps for overlay
        total_steps = max(metrics["train_loss"]["steps"]) if metrics["train_loss"]["steps"] else 1
        total_epochs = max(metrics["eval_loss"]["epochs"]) if metrics["eval_loss"]["epochs"] else 1
        steps_per_epoch = total_steps / total_epochs

        eval_steps = [e * steps_per_epoch for e in metrics["eval_loss"]["epochs"]]
        ax.plot(
            eval_steps,
            metrics["eval_loss"]["values"],
            'r-o',
            linewidth=2,
            markersize=8,
            label='Validation Loss'
        )

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "loss_combined.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'loss_combined.png'}")


def plot_accuracy(metrics: dict, output_path: Path) -> None:
    """Plot accuracy over epochs."""
    if not metrics["eval_accuracy"]["values"]:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        metrics["eval_accuracy"]["epochs"],
        metrics["eval_accuracy"]["values"],
        'g-o',
        linewidth=2,
        markersize=8,
        label='Accuracy'
    )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    if metrics["eval_accuracy"]["epochs"]:
        ax.set_xticks(range(1, int(max(metrics["eval_accuracy"]["epochs"])) + 1))

    plt.tight_layout()
    plt.savefig(output_path / "accuracy.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'accuracy.png'}")


def plot_f1_scores(metrics: dict, output_path: Path) -> None:
    """Plot F1 scores over epochs."""
    has_macro = bool(metrics["eval_f1_macro"]["values"])
    has_weighted = bool(metrics["eval_f1_weighted"]["values"])

    if not has_macro and not has_weighted:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if has_macro:
        ax.plot(
            metrics["eval_f1_macro"]["epochs"],
            metrics["eval_f1_macro"]["values"],
            'purple',
            linestyle='-',
            marker='o',
            linewidth=2,
            markersize=8,
            label='F1 Macro'
        )

    if has_weighted:
        ax.plot(
            metrics["eval_f1_weighted"]["epochs"],
            metrics["eval_f1_weighted"]["values"],
            'orange',
            linestyle='-',
            marker='s',
            linewidth=2,
            markersize=8,
            label='F1 Weighted'
        )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Scores Over Epochs', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    epochs = metrics["eval_f1_macro"]["epochs"] or metrics["eval_f1_weighted"]["epochs"]
    if epochs:
        ax.set_xticks(range(1, int(max(epochs)) + 1))

    plt.tight_layout()
    plt.savefig(output_path / "f1_scores.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'f1_scores.png'}")


def plot_learning_rate(metrics: dict, output_path: Path) -> None:
    """Plot learning rate schedule."""
    if not metrics["learning_rate"]["values"]:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        metrics["learning_rate"]["steps"],
        metrics["learning_rate"]["values"],
        'c-',
        linewidth=2,
        label='Learning Rate'
    )

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(output_path / "learning_rate.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'learning_rate.png'}")


def plot_metrics_summary(metrics: dict, output_path: Path) -> None:
    """Create a summary plot with all key metrics."""
    # Determine which metrics are available
    has_train_loss = bool(metrics["train_loss"]["values"])
    has_eval_loss = bool(metrics["eval_loss"]["values"])
    has_accuracy = bool(metrics["eval_accuracy"]["values"])
    has_f1 = bool(metrics["eval_f1_macro"]["values"])

    n_plots = sum([has_train_loss or has_eval_loss, has_accuracy, has_f1])

    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Loss plot
    if has_train_loss or has_eval_loss:
        ax = axes[idx]
        if has_train_loss:
            ax.plot(
                metrics["train_loss"]["steps"],
                metrics["train_loss"]["values"],
                'b-', alpha=0.7, linewidth=1.5, label='Train'
            )
        if has_eval_loss and has_train_loss:
            total_steps = max(metrics["train_loss"]["steps"])
            total_epochs = max(metrics["eval_loss"]["epochs"])
            steps_per_epoch = total_steps / total_epochs
            eval_steps = [e * steps_per_epoch for e in metrics["eval_loss"]["epochs"]]
            ax.plot(eval_steps, metrics["eval_loss"]["values"], 'r-o', linewidth=2, markersize=6, label='Val')
        elif has_eval_loss:
            ax.plot(metrics["eval_loss"]["epochs"], metrics["eval_loss"]["values"], 'r-o', linewidth=2, markersize=6, label='Val')
        ax.set_xlabel('Step' if has_train_loss else 'Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        idx += 1

    # Accuracy plot
    if has_accuracy:
        ax = axes[idx]
        ax.plot(
            metrics["eval_accuracy"]["epochs"],
            metrics["eval_accuracy"]["values"],
            'g-o', linewidth=2, markersize=6
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        idx += 1

    # F1 plot
    if has_f1:
        ax = axes[idx]
        ax.plot(
            metrics["eval_f1_macro"]["epochs"],
            metrics["eval_f1_macro"]["values"],
            'purple', linestyle='-', marker='o', linewidth=2, markersize=6, label='Macro'
        )
        if metrics["eval_f1_weighted"]["values"]:
            ax.plot(
                metrics["eval_f1_weighted"]["epochs"],
                metrics["eval_f1_weighted"]["values"],
                'orange', linestyle='-', marker='s', linewidth=2, markersize=6, label='Weighted'
            )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Training Metrics Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "metrics_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'metrics_summary.png'}")


def plot_all_metrics(trainer, output_path: Path) -> None:
    """
    Generate all training plots from a HuggingFace Trainer.

    Args:
        trainer: HuggingFace Trainer instance after training
        output_path: Directory to save plots
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating training plots...")
    print("=" * 60)

    # Extract metrics from trainer log history
    log_history = trainer.state.log_history
    metrics = extract_metrics_from_log_history(log_history)

    # Generate individual plots
    plot_training_loss(metrics, output_path)
    plot_eval_loss(metrics, output_path)
    plot_loss_combined(metrics, output_path)
    plot_accuracy(metrics, output_path)
    plot_f1_scores(metrics, output_path)
    plot_learning_rate(metrics, output_path)

    # Generate summary plot
    plot_metrics_summary(metrics, output_path)

    print("=" * 60)
    print(f"All plots saved to: {output_path}")
    print("=" * 60)
